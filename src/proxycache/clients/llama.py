# -*- coding: utf-8 -*-

"""HTTP client for llama.cpp-compatible backends."""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


def _capabilities_from_model_payload(payload: dict[str, Any]) -> list[str]:
    for collection_name in ("models", "data"):
        models = payload.get(collection_name) or []
        if not models or not isinstance(models[0], dict):
            continue
        capabilities = models[0].get("capabilities") or []
        if isinstance(capabilities, list):
            return [str(item) for item in capabilities if item]
    return []


class LlamaClient:
    """Async client for chat, slot management, and model discovery."""

    def __init__(self, base_url: str, request_timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=request_timeout,
            limits=limits,
        )
        self._is_multimodal_cached: bool | None = None
        self._model_id_cached: str | None = None
        self._capabilities_cached: list[str] | None = None
        self._slots_supported_cached: bool | None = None
        log.info("client_init url=%s httpx_version=%s", base_url, httpx.__version__)

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def _with_slot_id(body: dict[str, Any], slot_id: int | None) -> tuple[dict[str, Any], dict[str, int]]:
        if slot_id is None:
            return body, {}

        updated = dict(body)
        updated["_slot_id"] = slot_id
        updated["slot_id"] = slot_id
        updated["id_slot"] = slot_id

        options = dict(updated.get("options") or {})
        options["slot_id"] = slot_id
        options["id_slot"] = slot_id
        updated["options"] = options

        return updated, {"slot_id": slot_id, "id_slot": slot_id}

    async def chat_completions(
        self,
        body: dict[str, Any],
        slot_id: int | None = None,
        stream: bool = False,
    ) -> httpx.Response | dict[str, Any]:
        payload, query = self._with_slot_id(body, slot_id)
        if stream:
            request = self.client.build_request(
                "POST",
                "/v1/chat/completions",
                json=payload,
                params=query,
            )
            return await self.client.send(request, stream=True)

        response = await self.client.post(
            "/v1/chat/completions",
            json=payload,
            params=query,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            raw = response.text or ""
            log.error("non_stream_non_json content_type=%s raw_len=%d", content_type, len(raw))
            return {
                "object": "error",
                "message": "provider returned non-JSON",
                "raw": raw[:2048],
            }
        try:
            return response.json()
        except Exception as exc:
            raw = response.text or ""
            log.error(
                "non_stream_json_parse_error status=%d raw_len=%d err=%s",
                response.status_code,
                len(raw),
                exc,
            )
            return {
                "object": "error",
                "message": "invalid json from provider",
                "raw": raw[:2048],
            }

    async def save_slot(self, slot_id: int, basename: str) -> bool:
        response = await self.client.post(
            f"/slots/{slot_id}",
            params={"action": "save"},
            json={"filename": basename},
        )
        if response.status_code in {404, 405, 501}:
            self.mark_slots_unsupported("save_not_supported", response.status_code)
            return False
        if response.status_code == 500:
            log.warning("save_slot_500 slot=%d basename=%s", slot_id, basename[:16])
            return False
        response.raise_for_status()
        return True

    async def restore_slot(self, slot_id: int, basename: str) -> bool:
        response = await self.client.post(
            f"/slots/{slot_id}",
            params={"action": "restore"},
            json={"filename": basename},
        )
        if response.status_code in {404, 405, 501}:
            self.mark_slots_unsupported("restore_not_supported", response.status_code)
            return False
        if response.status_code != 200:
            log.warning(
                "restore_slot_status=%d slot=%d basename=%s",
                response.status_code,
                slot_id,
                basename[:16],
            )
            return False
        return True

    async def get_model_info(self) -> tuple[str, list[str]]:
        if self._model_id_cached is not None and self._capabilities_cached is not None:
            return self._model_id_cached, list(self._capabilities_cached)

        try:
            response = await self.client.get("/v1/models")
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data") or []
            if models and isinstance(models[0], dict):
                model_id = str(models[0].get("id") or "unknown")
            else:
                model_id = "unknown"
            capabilities = _capabilities_from_model_payload(payload)
            self._model_id_cached = model_id
            self._capabilities_cached = capabilities
            self._is_multimodal_cached = "multimodal" in capabilities
            log.debug(
                "get_model_info base_url=%s id=%s capabilities=%s",
                self.base_url,
                model_id,
                capabilities,
            )
            return model_id, capabilities
        except Exception as exc:
            log.warning("get_model_info_fail base_url=%s err=%s", self.base_url, exc)
            return self._model_id_cached or "unknown", list(self._capabilities_cached or [])

    async def is_multimodal(self) -> bool:
        if self._is_multimodal_cached is not None:
            return self._is_multimodal_cached
        _, capabilities = await self.get_model_info()
        is_multimodal = "multimodal" in capabilities
        self._is_multimodal_cached = is_multimodal
        log.info(
            "multimodal_detected_by_models base_url=%s is_mm=%s capabilities=%s",
            self.base_url,
            is_multimodal,
            capabilities,
        )
        return is_multimodal

    def slots_supported(self) -> bool:
        return self._slots_supported_cached is not False

    def mark_slots_unsupported(self, reason: str, status_code: int | None = None) -> None:
        if self._slots_supported_cached is False:
            return
        self._slots_supported_cached = False
        log.warning(
            "slots_unsupported_detected base_url=%s reason=%s status_code=%s",
            self.base_url,
            reason,
            status_code,
        )
