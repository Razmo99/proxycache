from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from proxycache.clients.llama import LlamaClient
from proxycache.services.policies import (
    looks_like_multimodal_model_name,
    slot_management_mode,
)


def test_model_name_heuristic_detects_qwen_vl() -> None:
    assert looks_like_multimodal_model_name("qwen3-vl-30b-instruct-bf16")
    assert not looks_like_multimodal_model_name("qwen3-4b-128k")


def test_slot_management_uses_heuristic_when_capabilities_missing() -> None:
    bypass, reason = slot_management_mode(
        backend_is_multimodal=False,
        backend_model_id="unknown",
        client_model="qwen3-vl-30b-instruct-bf16",
        slots_supported=True,
    )
    assert bypass is True
    assert reason == "model_name_heuristic"


@pytest.mark.asyncio
async def test_save_slot_501_marks_backend_slots_unsupported() -> None:
    client = LlamaClient("http://example.test", request_timeout=1.0)
    real_client = client.client
    client.client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(status_code=501)))
    try:
        ok = await client.save_slot(0, "test-key")
    finally:
        client.client = real_client
        await real_client.aclose()

    assert ok is False
    assert client.slots_supported() is False
