# -*- coding: utf-8 -*-

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from proxycache.api.app import create_app
from proxycache.cache.metadata import CacheStore
from proxycache.clients.llama import LlamaClient
from proxycache.config import BackendSettings, Settings
from proxycache.services.slots import SlotManager


@dataclass
class SpanStub:
    ended: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def end(self) -> None:
        self.ended = True


@dataclass
class MockBackend:
    model_id: str = "llama.cpp"
    capabilities: list[str] = field(default_factory=list)
    chat_json: dict[str, Any] = field(
        default_factory=lambda: {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            "timings": {"prompt_n": 8, "cache_n": 4, "prompt_ms": 12.5},
        }
    )
    chat_status_code: int = 200
    chat_content_type: str = "application/json"
    stream_chunks: list[bytes] = field(default_factory=list)
    slot_save_status_code: int = 200
    slot_restore_status_code: int = 200
    requests: list[httpx.Request] = field(default_factory=list)
    closed: bool = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)

        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": self.model_id, "capabilities": self.capabilities}]},
            )

        if request.url.path == "/v1/chat/completions":
            if request.headers.get("accept") == "text/event-stream" or self.stream_chunks:
                return httpx.Response(
                    self.chat_status_code,
                    headers={"content-type": "text/event-stream"},
                    stream=StaticAsyncStream(self.stream_chunks),
                )
            if self.chat_content_type == "application/json":
                return httpx.Response(
                    self.chat_status_code,
                    headers={"content-type": self.chat_content_type},
                    json=self.chat_json,
                )
            return httpx.Response(
                self.chat_status_code,
                headers={"content-type": self.chat_content_type},
                text="non-json-body",
            )

        if request.url.path.startswith("/slots/"):
            action = request.url.params.get("action")
            status_code = self.slot_save_status_code if action == "save" else self.slot_restore_status_code
            return httpx.Response(status_code, json={"ok": status_code == 200})

        body = [] if request.method == "HEAD" else [f"upstream:{request.method}:{request.url.path}".encode()]
        return httpx.Response(200, stream=StaticAsyncStream(body))


class StaticAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    def _make_settings(**overrides: Any) -> Settings:
        root = tmp_path / overrides.pop("name", "runtime")
        meta_dir = root / "meta"
        slot_save_path = root / "slots"
        settings = Settings(
            backends=overrides.pop(
                "backends",
                (BackendSettings(url="http://backend.test", n_slots=2),),
            ),
            words_per_block=overrides.pop("words_per_block", 2),
            big_threshold_words=overrides.pop("big_threshold_words", 3),
            lcp_threshold=overrides.pop("lcp_threshold", 0.6),
            meta_dir=overrides.pop("meta_dir", meta_dir),
            slot_save_path=overrides.pop("slot_save_path", slot_save_path),
            request_timeout=overrides.pop("request_timeout", 5.0),
            model_id=overrides.pop("model_id", "llama.cpp"),
            port=overrides.pop("port", 8081),
            log_level=overrides.pop("log_level", "INFO"),
            max_saved_caches=overrides.pop("max_saved_caches", 2),
        )
        if overrides:
            unexpected = ", ".join(sorted(overrides))
            raise AssertionError(f"Unexpected settings overrides: {unexpected}")
        settings.ensure_directories()
        return settings

    return _make_settings


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend()


@pytest.fixture
async def llama_client(mock_backend: MockBackend) -> LlamaClient:
    client = LlamaClient("http://backend.test", request_timeout=5.0)
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        base_url=client.base_url,
        transport=httpx.MockTransport(mock_backend.handler),
    )
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
def cache_store(make_settings: Callable[..., Settings]) -> CacheStore:
    settings = make_settings(name="cache-store")
    return CacheStore(
        meta_dir=settings.meta_dir,
        cache_dir=settings.slot_save_path,
        words_per_block=settings.words_per_block,
        max_saved_caches=settings.max_saved_caches,
    )


@pytest.fixture
def slot_manager(make_settings: Callable[..., Settings]) -> SlotManager:
    settings = make_settings(name="slot-manager")
    return SlotManager(settings.backends)


@pytest.fixture
def span_stub() -> SpanStub:
    return SpanStub()


@pytest.fixture
def messages_factory() -> Callable[[str], list[dict[str, Any]]]:
    def _messages(kind: str) -> list[dict[str, Any]]:
        if kind == "small":
            return [{"role": "user", "content": "tiny prompt"}]
        if kind == "big":
            return [{"role": "user", "content": "one two three four five six"}]
        if kind == "multimodal":
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "http://example.test/img.png"}},
                    ],
                }
            ]
        raise AssertionError(f"Unknown message kind: {kind}")

    return _messages


@pytest.fixture
def test_app(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: Callable[..., Settings],
) -> Callable[[MockBackend | None], Any]:
    def _build_app(backend: MockBackend | None = None):
        active_backend = backend or MockBackend()
        monkeypatch.setattr("proxycache.api.app.init_otel", lambda app, httpx_client: None)
        monkeypatch.setattr("proxycache.api.app.shutdown_otel", lambda: None)
        monkeypatch.setattr(
            "proxycache.api.app.LlamaClient",
            lambda url, request_timeout: _build_llama_client(active_backend, url, request_timeout),
        )
        return create_app(make_settings(name="app"))

    return _build_app


def _build_llama_client(backend: MockBackend, url: str, request_timeout: float) -> LlamaClient:
    client = LlamaClient(url, request_timeout=request_timeout)
    client.client = httpx.AsyncClient(base_url=client.base_url, transport=httpx.MockTransport(backend.handler))
    return client


@pytest_asyncio.fixture
async def async_test_client(test_app: Callable[[MockBackend | None], Any], mock_backend: MockBackend):
    app = test_app(mock_backend)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            client.proxycache_app = app
            yield client
