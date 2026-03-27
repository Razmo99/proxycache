# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_small_json_request_skips_cache_save(async_test_client, mock_backend, messages_factory) -> None:
    response = await async_test_client.post(
        "/v1/chat/completions",
        json={"model": "llama.cpp", "messages": messages_factory("small"), "stream": False},
    )

    assert response.status_code == 200
    chat_request = next(request for request in mock_backend.requests if request.url.path == "/v1/chat/completions")
    payload = json.loads(chat_request.content)
    assert payload["cache_prompt"] is False
    assert payload["options"]["cache_prompt"] is False
    assert payload["options"]["slot_id"] == 0
    assert not any(request.url.path.startswith("/slots/") for request in mock_backend.requests)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_big_json_request_restores_and_records_metadata(async_test_client, mock_backend, messages_factory) -> None:
    service = async_test_client.proxycache_app.state.service
    prefix = service.cache_store.raw_prefix(messages_factory("big"))
    blocks = service.cache_store.block_hashes_from_text(prefix, service.settings.words_per_block)
    service.cache_store.write_meta("restore-key", prefix, blocks, service.settings.words_per_block, "llama.cpp")

    response = await async_test_client.post(
        "/v1/chat/completions",
        json={"model": "llama.cpp", "messages": messages_factory("big"), "stream": False},
    )

    assert response.status_code == 200
    slot_actions = [(request.url.path, request.url.params.get("action")) for request in mock_backend.requests if request.url.path.startswith("/slots/")]
    assert ("/slots/0", "restore") in slot_actions
    assert ("/slots/0", "save") in slot_actions
    saved_key = service.cache_store.prefix_key_sha256("llama.cpp\n" + prefix)
    assert (service.cache_store.meta_dir / f"{saved_key}.meta.json").exists()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multimodal_request_bypasses_slot_management(async_test_client, mock_backend, messages_factory) -> None:
    mock_backend.model_id = "qwen3-vl-30b"

    response = await async_test_client.post(
        "/v1/chat/completions",
        json={"model": "qwen3-vl-30b", "messages": messages_factory("multimodal"), "stream": False},
    )

    assert response.status_code == 200
    assert [request.url.path for request in mock_backend.requests].count("/v1/chat/completions") == 1
    assert not any(request.url.path.startswith("/slots/") for request in mock_backend.requests)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multimodal_streaming_request_bypasses_slots_and_forwards_sse(
    async_test_client,
    mock_backend,
    messages_factory,
) -> None:
    mock_backend.model_id = "qwen3-vl-30b"
    mock_backend.stream_chunks = [
        b'data: {"choices":[{"delta":{"content":"vision"}}]}\n\n',
        b'data: {"timings":{"prompt_n":9,"cache_n":0,"prompt_ms":6.0}}\n\n',
        b"data: [DONE]\n\n",
    ]

    async with async_test_client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "qwen3-vl-30b", "messages": messages_factory("multimodal"), "stream": True},
    ) as response:
        body = b"".join([chunk async for chunk in response.aiter_raw()])

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b'"content":"vision"' in body
    assert b"data: [DONE]" in body
    assert [request.url.path for request in mock_backend.requests].count("/v1/chat/completions") == 1
    assert not any(request.url.path.startswith("/slots/") for request in mock_backend.requests)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slot_acquire_timeout_returns_503(async_test_client, messages_factory) -> None:
    service = async_test_client.proxycache_app.state.service
    service.slot_manager.acquire_for_request = AsyncMock(side_effect=TimeoutError)

    response = await async_test_client.post(
        "/v1/chat/completions",
        json={"model": "llama.cpp", "messages": messages_factory("big"), "stream": False},
    )

    assert response.status_code == 503
    assert response.json() == {"error": "all slots busy, please retry later"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streaming_request_forwards_sse_and_releases_slot(async_test_client, mock_backend, messages_factory) -> None:
    mock_backend.stream_chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b'data: {"timings":{"prompt_n":12,"cache_n":6,"prompt_ms":8.0}}\n\n',
        b"data: [DONE]\n\n",
    ]
    service = async_test_client.proxycache_app.state.service

    async with async_test_client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "llama.cpp", "messages": messages_factory("big"), "stream": True},
    ) as response:
        body = b"".join([chunk async for chunk in response.aiter_raw()])

    assert response.status_code == 200
    assert b'"content":"hello"' in body
    assert service.slot_manager.locked_slot_count() == 0
