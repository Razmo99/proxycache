# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio

import httpx
import pytest
from starlette.requests import Request

from proxycache.cache.metadata import CacheStore
from proxycache.services.chat_service import ProxyService
from proxycache.services.slots import SlotManager


class _AsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_proxy_upstream_request_head_returns_empty_body(make_settings) -> None:
    settings = make_settings(name="chat-service")
    request_sent: list[httpx.Request] = []

    async def send(request: httpx.Request, stream: bool = False) -> httpx.Response:
        request_sent.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-length": "99", "x-upstream": "ok"},
            stream=_AsyncStream([b"ignored"]),
        )

    fake_httpx_client = type(
        "FakeHTTPXClient",
        (),
        {
            "build_request": staticmethod(lambda method, path, params=None, headers=None: httpx.Request(method, f"http://backend.test{path}", params=params, headers=headers)),
            "send": staticmethod(send),
        },
    )()
    fake_client = type("FakeClient", (), {"client": fake_httpx_client, "close": staticmethod(lambda: asyncio.sleep(0))})()
    slot_manager = SlotManager(settings.backends)
    cache_store = CacheStore(settings.meta_dir, settings.slot_save_path, settings.words_per_block, settings.max_saved_caches)
    service = ProxyService(settings, [fake_client], slot_manager, cache_store)
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "HEAD",
            "path": "/",
            "raw_path": b"/",
            "scheme": "http",
            "query_string": b"",
            "headers": [(b"host", b"testserver"), (b"x-test", b"1")],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        }
    )

    response = await service.proxy_upstream_request(request, "/")
    messages: list[dict[str, object]] = []
    receive_events = [
        {"type": "http.request", "body": b"", "more_body": False},
        {"type": "http.disconnect"},
    ]

    async def receive() -> dict[str, object]:
        return receive_events.pop(0)

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    await response(request.scope, receive, send)
    body_messages = [message for message in messages if message["type"] == "http.response.body"]

    assert response.status_code == 200
    assert body_messages == []
    assert response.headers["x-upstream"] == "ok"
    assert "content-length" not in response.headers
    assert request_sent[0].method == "HEAD"
