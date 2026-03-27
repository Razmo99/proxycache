# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from proxycache.services.chat_service import ProxyService


class SpanStub:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.ended = False
        self.events: list[tuple[str, dict[str, object]]] = []
        self.status = None

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, object]) -> None:
        self.events.append((name, attributes))

    def set_status(self, status) -> None:
        self.status = status

    def end(self) -> None:
        self.ended = True

    def is_recording(self) -> bool:
        return True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_bypass_request_returns_502_for_invalid_stream_response(make_settings, cache_store) -> None:
    settings = make_settings(name="bypass-stream-invalid")
    service = ProxyService(settings, [], SimpleNamespace(), cache_store)
    client = SimpleNamespace(chat_completions=AsyncMock(return_value={"not": "response"}))
    span = SpanStub()

    response = await service._handle_bypass_request(
        client=client,
        body={"messages": []},
        stream=True,
        backend_model_id="model-a",
        bypass_reason="models_capability",
        request_has_media=True,
        current_span=span,
        started_at=time.time(),
    )

    assert response.status_code == 502
    assert response.body == b'{"error":"provider returned invalid streaming response"}'
    assert span.ended is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_bypass_request_returns_upstream_error_for_non_200_stream(make_settings, cache_store) -> None:
    settings = make_settings(name="bypass-stream-error")
    service = ProxyService(settings, [], SimpleNamespace(), cache_store)
    response = httpx.Response(503, content=b"upstream down")
    client = SimpleNamespace(chat_completions=AsyncMock(return_value=response))
    span = SpanStub()

    result = await service._handle_bypass_request(
        client=client,
        body={"messages": []},
        stream=True,
        backend_model_id="model-a",
        bypass_reason="models_capability",
        request_has_media=True,
        current_span=span,
        started_at=time.time(),
    )

    assert result.status_code == 503
    assert result.body == b'{"error":"upstream down"}'
    assert span.ended is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_streaming_slot_request_returns_502_for_invalid_response(make_settings, cache_store) -> None:
    settings = make_settings(name="slot-stream-invalid")
    service = ProxyService(settings, [], SimpleNamespace(), cache_store)
    released: list[tuple[int, int]] = []
    service.lifecycle.release_slot = lambda slot, span=None: released.append(slot)
    client = SimpleNamespace(chat_completions=AsyncMock(return_value={"not": "response"}))
    span = SpanStub()

    result = await service._handle_streaming_slot_request(
        client=client,
        body={"messages": []},
        slot=(0, 1),
        key="cache-key",
        prefix="prefix",
        blocks=["a"],
        backend_model_id="model-a",
        restore_key=None,
        restored=None,
        persist_cache=False,
        current_span=span,
    )

    assert result.status_code == 502
    assert span.ended is True
    assert released == [(0, 1)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_streaming_slot_request_releases_slot_on_exception(make_settings, cache_store) -> None:
    settings = make_settings(name="slot-stream-exc")
    service = ProxyService(settings, [], SimpleNamespace(), cache_store)
    released: list[tuple[int, int]] = []
    service.lifecycle.release_slot = lambda slot, span=None: released.append(slot)
    response = httpx.Response(200)
    client = SimpleNamespace(chat_completions=AsyncMock(return_value=response))
    service.streaming.start_stream_task = AsyncMock(side_effect=RuntimeError("stream failed"))

    with pytest.raises(RuntimeError, match="stream failed"):
        await service._handle_streaming_slot_request(
            client=client,
            body={"messages": []},
            slot=(0, 1),
            key="cache-key",
            prefix="prefix",
            blocks=["a"],
            backend_model_id="model-a",
            restore_key=None,
            restored=None,
            persist_cache=False,
            current_span=SpanStub(),
        )

    assert released == [(0, 1)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_json_slot_request_returns_502_for_non_dict_payload(make_settings, cache_store) -> None:
    settings = make_settings(name="json-slot-invalid")
    service = ProxyService(settings, [], SimpleNamespace(), cache_store)
    released: list[tuple[int, int]] = []
    service.lifecycle.release_slot = lambda slot, span=None: released.append(slot)
    client = SimpleNamespace(chat_completions=AsyncMock(return_value="not-a-dict"))
    span = SpanStub()

    result = await service._handle_json_slot_request(
        client=client,
        body={"messages": []},
        slot=(0, 1),
        key="cache-key",
        prefix="prefix",
        blocks=["a"],
        backend_model_id="model-a",
        restore_key=None,
        restored=None,
        is_big=False,
        current_span=span,
        started_at=time.time(),
    )

    assert result.status_code == 502
    assert released == [(0, 1)]
    assert span.ended is True
