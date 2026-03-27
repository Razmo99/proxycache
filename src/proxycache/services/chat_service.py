# -*- coding: utf-8 -*-

"""High-level request handling for the proxy service."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from proxycache.cache.metadata import CacheStore
from proxycache.clients.llama import LlamaClient
from proxycache.config import Settings
from proxycache.observability.otel import (
    add_cache_attributes,
    add_input_attributes,
    add_lifecycle_event,
    add_llm_attributes,
    add_response_attributes,
    add_timing_attributes,
    set_error,
    start_inference_span,
)
from proxycache.services.cache_lifecycle import CacheLifecycleManager
from proxycache.services.policies import (
    extract_timings,
    request_has_multimodal_workload,
    slot_management_mode,
)
from proxycache.services.slots import GSlot, SlotManager
from proxycache.services.streaming import StreamingCoordinator

log = logging.getLogger(__name__)
ACQUIRE_TIMEOUT = 300.0


class ProxyService:
    """Coordinates slot management, cache persistence, and upstream requests."""

    def __init__(
        self,
        settings: Settings,
        clients: list[LlamaClient],
        slot_manager: SlotManager,
        cache_store: CacheStore,
    ) -> None:
        self.settings = settings
        self.clients = clients
        self.slot_manager = slot_manager
        self.cache_store = cache_store
        self.lifecycle = CacheLifecycleManager(settings, slot_manager, cache_store)
        self.streaming = StreamingCoordinator(self.lifecycle)

    async def models(self) -> dict[str, list[dict[str, str]]]:
        return {"data": [{"id": self.settings.model_id}]}

    async def proxy_upstream_request(self, request: Request, path: str) -> Response:
        client = self.clients[0]
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length"}
        }
        upstream_path = "/" + path.lstrip("/")
        if upstream_path == "//":
            upstream_path = "/"

        upstream_request = client.client.build_request(
            request.method,
            upstream_path,
            params=request.query_params,
            headers=headers,
        )
        response = await client.client.send(upstream_request, stream=True)
        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"content-length", "transfer-encoding", "connection"}
        }
        body = b"" if request.method.upper() == "HEAD" else response.aiter_raw()
        return StreamingResponse(
            body,
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type"),
            background=BackgroundTask(response.aclose),
        )

    async def chat(self, request: Request) -> Response:
        started_at = time.time()
        payload = await request.json()
        messages = payload.get("messages") or []
        stream = bool(payload.get("stream", False))
        client_model = payload.get("model") or self.settings.model_id
        client = self.clients[0]
        current_span = start_inference_span(client.base_url, client_model, payload)
        add_input_attributes(current_span, payload)

        try:
            response = await self._handle_chat_request(
                payload=payload,
                messages=messages,
                stream=stream,
                client_model=client_model,
                current_span=current_span,
                started_at=started_at,
            )
            return response
        except Exception as exc:
            set_error(current_span, exc.__class__.__name__, str(exc))
            current_span.end()
            log.exception("chat_setup_error: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

    async def _handle_chat_request(
        self,
        payload: dict[str, Any],
        messages: list[dict[str, Any]],
        stream: bool,
        client_model: str,
        current_span,
        started_at: float,
    ) -> Response:
        backend_id = 0
        client = self.clients[backend_id]
        backend_model_id, backend_capabilities = await client.get_model_info()
        backend_is_multimodal = await client.is_multimodal()
        request_has_media = request_has_multimodal_workload(messages)
        prefix = self.cache_store.raw_prefix(messages)
        key = self.cache_store.prefix_key_sha256(f"{backend_model_id}\n{prefix}")
        blocks = self.cache_store.block_hashes_from_text(prefix, self.settings.words_per_block)
        n_words = len(self.cache_store.words_from_text(prefix))
        is_big = n_words > self.settings.big_threshold_words

        restore_candidate = (
            self.cache_store.find_best_restore_candidate(
                blocks,
                self.settings.words_per_block,
                self.settings.lcp_threshold,
                backend_model_id,
            )
            if is_big
            else None
        )
        restore_key: str | None = None
        if restore_candidate:
            restore_key, ratio = restore_candidate
            add_lifecycle_event(
                current_span,
                "proxycache.cache.restore.candidate.selected",
                cache_key_prefix=restore_key[:16],
                match_ratio=round(ratio, 6),
                request_words=n_words,
            )
            log.info("restore_candidate basename=%s ratio=%.3f", restore_key[:16], ratio)
        elif is_big:
            add_lifecycle_event(current_span, "proxycache.cache.restore.candidate.miss", request_words=n_words)
            log.info("restore_candidate none")
        else:
            add_lifecycle_event(
                current_span,
                "proxycache.cache.strategy.small_request",
                request_words=n_words,
                threshold_words=self.settings.big_threshold_words,
            )
            log.info("small_request n_words=%d threshold=%d", n_words, self.settings.big_threshold_words)

        if backend_is_multimodal != self.slot_manager.is_multimodal_backend(backend_id):
            await self.slot_manager.set_multimodal_backend(backend_id, backend_is_multimodal)

        add_lifecycle_event(
            current_span,
            "proxycache.request.modalities.inspected",
            backend_id=backend_id,
            backend_caps=backend_capabilities,
            request_has_media=request_has_media,
            backend_model_id=backend_model_id,
        )

        bypass_slots, bypass_reason = slot_management_mode(
            backend_is_multimodal=backend_is_multimodal,
            backend_model_id=backend_model_id,
            client_model=client_model,
            slots_supported=client.slots_supported(),
        )

        add_llm_attributes(current_span, client_model, response_model=backend_model_id)
        body = dict(payload)
        body["model"] = client_model

        if bypass_slots:
            return await self._handle_bypass_request(
                client=client,
                body=body,
                stream=stream,
                backend_model_id=backend_model_id,
                bypass_reason=bypass_reason,
                request_has_media=request_has_media,
                current_span=current_span,
                started_at=started_at,
            )

        try:
            slot, _, restored = await asyncio.wait_for(
                self.slot_manager.acquire_for_request(restore_key if is_big else None),
                timeout=ACQUIRE_TIMEOUT,
            )
        except TimeoutError:
            log.error("acquire_timeout is_big=%s restore_key=%s", is_big, restore_key[:16] if restore_key else None)
            set_error(current_span, "slot_acquire_timeout", "all slots busy")
            add_lifecycle_event(
                current_span,
                "proxycache.slot.acquire.timeout",
                is_big=is_big,
                restore_key_prefix=restore_key[:16] if restore_key else None,
            )
            current_span.end()
            return JSONResponse({"error": "all slots busy, please retry later"}, status_code=503)

        add_lifecycle_event(
            current_span,
            "proxycache.slot.acquired",
            backend_id=slot[0],
            slot_id=slot[1],
            restored=bool(restored),
            restore_key_prefix=restore_key[:16] if restore_key else None,
        )
        self.lifecycle.record_restore_outcome(
            current_span,
            slot,
            restore_key if is_big else None,
            restored,
        )

        backend_id, slot_id = slot
        client = self.clients[backend_id]
        add_cache_attributes(
            current_span,
            bool(restored),
            slot_id,
            self.slot_manager.locked_slot_count(),
        )

        body["cache_prompt"] = bool(is_big)
        body["n_keep"] = -1
        options = dict(body.get("options") or {})
        options.update({"slot_id": slot_id, "id_slot": slot_id, "n_keep": -1, "cache_prompt": bool(is_big)})
        body["options"] = options

        if stream:
            return await self._handle_streaming_slot_request(
                client=client,
                body=body,
                slot=slot,
                key=key,
                prefix=prefix,
                blocks=blocks,
                backend_model_id=backend_model_id,
                restore_key=restore_key if is_big else None,
                restored=restored,
                persist_cache=is_big,
                current_span=current_span,
            )

        return await self._handle_json_slot_request(
            client=client,
            body=body,
            slot=slot,
            key=key,
            prefix=prefix,
            blocks=blocks,
            backend_model_id=backend_model_id,
            restore_key=restore_key if is_big else None,
            restored=restored,
            is_big=is_big,
            current_span=current_span,
            started_at=started_at,
        )

    async def _handle_bypass_request(
        self,
        client: LlamaClient,
        body: dict[str, Any],
        stream: bool,
        backend_model_id: str,
        bypass_reason: str,
        request_has_media: bool,
        current_span,
        started_at: float,
    ) -> Response:
        add_cache_attributes(current_span, False)
        current_span.set_attribute("proxycache.slot.management.bypass", True)
        current_span.set_attribute("proxycache.slot.management.bypass_reason", bypass_reason)
        add_lifecycle_event(
            current_span,
            "proxycache.slot.management.bypassed",
            reason=bypass_reason,
            request_has_media=request_has_media,
            backend_model_id=backend_model_id,
        )
        add_lifecycle_event(current_span, "proxycache.cache.save.skipped", reason=bypass_reason)

        if stream:
            response = await client.chat_completions(body, slot_id=None, stream=True)
            if not isinstance(response, httpx.Response):
                set_error(
                    current_span,
                    "provider_non_stream_response",
                    "provider returned invalid streaming response",
                )
                current_span.end()
                return JSONResponse(
                    {"error": "provider returned invalid streaming response"},
                    status_code=502,
                )
            if response.status_code != 200:
                error_text = await response.aread()
                await response.aclose()
                set_error(current_span, f"upstream_http_{response.status_code}", "upstream returned non-200 status")
                current_span.end()
                return JSONResponse({"error": error_text.decode("utf-8", "ignore")}, status_code=response.status_code)
            generator = await self.streaming.start_multimodal_stream(
                response,
                backend_model_id,
                current_span,
            )
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        output = await client.chat_completions(body, slot_id=None, stream=False)
        if not isinstance(output, dict):
            set_error(current_span, "provider_non_json_body", "provider returned non-JSON body")
            current_span.end()
            return JSONResponse({"error": "provider non-JSON body"}, status_code=502)
        add_response_attributes(current_span, output)
        add_timing_attributes(current_span, extract_timings(output))
        current_span.end()
        log.info("multimodal_json_done dur_ms=%d", int((time.time() - started_at) * 1000))
        return JSONResponse(content=output, status_code=200)

    async def _handle_streaming_slot_request(
        self,
        client: LlamaClient,
        body: dict[str, Any],
        slot: GSlot,
        key: str,
        prefix: str,
        blocks: list[str],
        backend_model_id: str,
        restore_key: str | None,
        restored: bool | None,
        persist_cache: bool,
        current_span,
    ) -> Response:
        response: httpx.Response | None = None
        try:
            response = await client.chat_completions(body, slot_id=slot[1], stream=True)
            if not isinstance(response, httpx.Response):
                self.lifecycle.release_slot(slot, current_span)
                set_error(
                    current_span,
                    "provider_non_stream_response",
                    "provider returned invalid streaming response",
                )
                current_span.end()
                return JSONResponse(
                    {"error": "provider returned invalid streaming response"},
                    status_code=502,
                )
            if response.status_code != 200:
                error_text = await response.aread()
                await response.aclose()
                self.lifecycle.release_slot(slot, current_span)
                set_error(current_span, f"upstream_http_{response.status_code}", "upstream returned non-200 status")
                current_span.end()
                return JSONResponse({"error": error_text.decode("utf-8", "ignore")}, status_code=response.status_code)

            generator = await self.streaming.start_stream_task(
                response,
                slot,
                key,
                prefix,
                blocks,
                backend_model_id,
                current_span,
                restore_key,
                restored,
                persist_cache,
            )
        except Exception:
            if response is not None:
                with suppress(Exception):
                    await response.aclose()
            self.lifecycle.release_slot(slot, current_span)
            raise

        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def _handle_json_slot_request(
        self,
        client: LlamaClient,
        body: dict[str, Any],
        slot: GSlot,
        key: str,
        prefix: str,
        blocks: list[str],
        backend_model_id: str,
        restore_key: str | None,
        restored: bool | None,
        is_big: bool,
        current_span,
        started_at: float,
    ) -> Response:
        saved = False
        try:
            output = await client.chat_completions(body, slot_id=slot[1], stream=False)
            if not isinstance(output, dict):
                set_error(current_span, "provider_non_json_body", "provider returned non-JSON body")
                return JSONResponse({"error": "provider non-JSON body"}, status_code=502)

            timings = extract_timings(output)
            add_response_attributes(current_span, output)
            add_timing_attributes(current_span, timings)
            self.lifecycle.maybe_poison_restore(
                restore_key,
                restored,
                backend_model_id,
                timings,
                current_span,
            )

            if is_big:
                saved = await self.lifecycle.save_cache_artifacts(
                    slot=slot,
                    key=key,
                    prefix=prefix,
                    blocks=blocks,
                    model_id=backend_model_id,
                    span=current_span,
                )
            else:
                add_lifecycle_event(
                    current_span,
                    "proxycache.cache.save.skipped",
                    backend_id=slot[0],
                    slot_id=slot[1],
                    cache_key_prefix=key[:16],
                    reason="small_request",
                )

            log.info(
                "json_done g=%s key=%s saved=%s is_big=%s dur_ms=%d",
                slot,
                key[:16],
                saved,
                is_big,
                int((time.time() - started_at) * 1000),
            )
            return JSONResponse(content=output, status_code=200)
        finally:
            self.lifecycle.release_slot(slot, current_span)
            current_span.end()
