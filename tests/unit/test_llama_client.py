# -*- coding: utf-8 -*-

from __future__ import annotations

import httpx
import pytest

from proxycache.clients.llama import LlamaClient, _capabilities_from_model_payload


@pytest.mark.unit
def test_capabilities_from_model_payload_supports_models_key() -> None:
    payload = {"models": [{"capabilities": ["multimodal", "tool-use"]}]}

    assert _capabilities_from_model_payload(payload) == ["multimodal", "tool-use"]


@pytest.mark.unit
def test_with_slot_id_populates_body_and_query() -> None:
    body, query = LlamaClient._with_slot_id({"options": {"temperature": 0}}, 7)

    assert query == {"slot_id": 7, "id_slot": 7}
    assert body["slot_id"] == 7
    assert body["id_slot"] == 7
    assert body["options"]["slot_id"] == 7


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_completions_returns_error_payload_for_non_json_body(llama_client: LlamaClient, mock_backend) -> None:
    mock_backend.chat_content_type = "text/plain"

    result = await llama_client.chat_completions({"messages": []}, stream=False)

    assert result["object"] == "error"
    assert result["message"] == "provider returned non-JSON"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_model_info_caches_first_successful_response(llama_client: LlamaClient, mock_backend) -> None:
    mock_backend.model_id = "model-a"
    mock_backend.capabilities = ["multimodal"]

    first = await llama_client.get_model_info()
    mock_backend.model_id = "model-b"
    mock_backend.capabilities = []
    second = await llama_client.get_model_info()

    assert first == ("model-a", ["multimodal"])
    assert second == ("model-a", ["multimodal"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_model_info_returns_unknown_when_request_fails_without_cache(llama_client: LlamaClient) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom")

    await llama_client.client.aclose()
    llama_client.client = httpx.AsyncClient(base_url=llama_client.base_url, transport=httpx.MockTransport(handler))

    assert await llama_client.get_model_info() == ("unknown", [])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_restore_slot_marks_slots_unsupported_on_404(llama_client: LlamaClient, mock_backend) -> None:
    mock_backend.slot_restore_status_code = 404

    restored = await llama_client.restore_slot(1, "cache-key")

    assert restored is False
    assert llama_client.slots_supported() is False
