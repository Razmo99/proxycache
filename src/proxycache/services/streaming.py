# -*- coding: utf-8 -*-

"""Streaming response coordination."""

from __future__ import annotations

import asyncio
import logging
from contextlib import nullcontext, suppress
from typing import Any

import httpx
from opentelemetry import trace

from proxycache.observability.otel import (
    add_lifecycle_event,
    add_restore_attributes,
    add_timing_attributes,
    set_error,
)
from proxycache.services.cache_lifecycle import CacheLifecycleManager
from proxycache.services.policies import consume_sse_timings
from proxycache.services.slots import GSlot

log = logging.getLogger(__name__)
STREAM_QUEUE_SIZE = 16


class StreamingCoordinator:
    """Owns background streaming tasks and slot cleanup."""

    def __init__(self, lifecycle: CacheLifecycleManager) -> None:
        self.lifecycle = lifecycle

    async def start_stream_task(
        self,
        response: httpx.Response,
        slot: GSlot,
        key: str,
        prefix: str,
        blocks: list[str],
        model_id: str,
        span=None,
        restore_key: str | None = None,
        restore_match_ratio: float | None = None,
        restored: bool | None = None,
        persist_cache: bool = False,
        system_fingerprint: str = "",
        tools_fingerprint: str = "",
    ):
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=STREAM_QUEUE_SIZE)

        async def reader() -> None:
            with self._span_scope(span):
                sse_buffer = b""
                last_timings: dict[str, Any] | None = None
                saved = False
                try:
                    add_lifecycle_event(
                        span,
                        "proxycache.stream.started",
                        backend_id=slot[0],
                        slot_id=slot[1],
                        cache_key_prefix=key[:16],
                    )
                    async for chunk in response.aiter_raw():
                        if not chunk:
                            continue
                        sse_buffer += chunk
                        sse_buffer, last_timings = consume_sse_timings(
                            sse_buffer,
                            last_timings,
                        )
                        await queue.put(chunk)
                except asyncio.CancelledError:
                    set_error(span, "cancelled", "stream reader cancelled")
                    raise
                except Exception as exc:
                    set_error(span, exc.__class__.__name__, str(exc))
                    log.exception("stream_reader_error g=%s key=%s: %s", slot, key[:16], exc)
                finally:
                    with suppress(Exception):
                        await response.aclose()
                    restore_assessment = self.lifecycle.assess_restore_quality(
                        restore_key,
                        restored,
                        model_id,
                        last_timings,
                        match_ratio=restore_match_ratio,
                        span=span,
                    )
                    if restore_assessment is not None:
                        add_restore_attributes(
                            span,
                            actual_ratio=restore_assessment.actual_ratio,
                            degraded=restore_assessment.degraded,
                        )
                    if persist_cache:
                        saved = await self.lifecycle.save_cache_artifacts(
                            slot,
                            key,
                            prefix,
                            blocks,
                            model_id,
                            system_fingerprint=system_fingerprint,
                            tools_fingerprint=tools_fingerprint,
                            span=span,
                        )
                    else:
                        add_lifecycle_event(
                            span,
                            "proxycache.cache.save.skipped",
                            backend_id=slot[0],
                            slot_id=slot[1],
                            cache_key_prefix=key[:16],
                            reason="small_request",
                        )
                    add_timing_attributes(span, last_timings)
                    self.lifecycle.release_slot(slot, span)
                    log.info("stream_reader_done g=%s key=%s saved=%s", slot, key[:16], saved)
                    if span is not None:
                        span.end()
                    with suppress(Exception):
                        await queue.put(None)

        asyncio.create_task(reader())
        return self._generator(queue)

    async def start_multimodal_stream(
        self,
        response: httpx.Response,
        model_id: str,
        span=None,
    ):
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=STREAM_QUEUE_SIZE)

        async def reader() -> None:
            with self._span_scope(span):
                last_timings: dict[str, Any] | None = None
                sse_buffer = b""
                try:
                    add_lifecycle_event(
                        span,
                        "proxycache.stream.started",
                        mode="multimodal_passthrough",
                        model_id=model_id,
                    )
                    async for chunk in response.aiter_raw():
                        if not chunk:
                            continue
                        sse_buffer += chunk
                        sse_buffer, last_timings = consume_sse_timings(
                            sse_buffer,
                            last_timings,
                        )
                        await queue.put(chunk)
                except asyncio.CancelledError:
                    set_error(span, "cancelled", "multimodal stream reader cancelled")
                    raise
                except Exception as exc:
                    set_error(span, exc.__class__.__name__, str(exc))
                    log.exception(
                        "multimodal_stream_reader_error model_id=%s: %s",
                        model_id[:16],
                        exc,
                    )
                finally:
                    with suppress(Exception):
                        await response.aclose()
                    add_timing_attributes(span, last_timings)
                    if span is not None:
                        span.end()
                    with suppress(Exception):
                        await queue.put(None)

        asyncio.create_task(reader())
        return self._generator(queue)

    @staticmethod
    async def _generator(queue: asyncio.Queue[bytes | None]):
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    @staticmethod
    def _span_scope(span):
        return trace.use_span(span, end_on_exit=False) if span is not None else nullcontext()
