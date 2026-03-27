# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_models_endpoint(async_test_client) -> None:
    response = await async_test_client.get("/v1/models")

    assert response.status_code == 200


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_non_stream_chat(async_test_client, messages_factory) -> None:
    response = await async_test_client.post(
        "/v1/chat/completions",
        json={"model": "llama.cpp", "messages": messages_factory("small"), "stream": False},
    )

    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_stream_chat(async_test_client, mock_backend, messages_factory) -> None:
    mock_backend.stream_chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    async with async_test_client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "llama.cpp", "messages": messages_factory("big"), "stream": True},
    ) as response:
        body = b"".join([chunk async for chunk in response.aiter_raw()])

    assert response.status_code == 200
    assert b"hello" in body
