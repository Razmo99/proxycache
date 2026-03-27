# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest

from proxycache.services.streaming import StreamingCoordinator


class FakeResponse:
    def __init__(self, chunks: list[bytes], exc: Exception | None = None) -> None:
        self._chunks = chunks
        self._exc = exc
        self.closed = False

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk
        if self._exc is not None:
            raise self._exc

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class LifecycleStub:
    poisoned: list[tuple[Any, ...]] = field(default_factory=list)
    released: list[tuple[int, int]] = field(default_factory=list)
    saved: AsyncMock = field(default_factory=lambda: AsyncMock(return_value=True))

    def maybe_poison_restore(self, *args: Any) -> None:
        self.poisoned.append(args)

    async def save_cache_artifacts(self, *args: Any) -> bool:
        return await self.saved(*args)

    def release_slot(self, slot: tuple[int, int], span=None) -> None:
        self.released.append(slot)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_stream_task_releases_slot_and_saves_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("proxycache.services.streaming.add_lifecycle_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("proxycache.services.streaming.add_timing_attributes", lambda *args, **kwargs: None)
    monkeypatch.setattr("proxycache.services.streaming.set_error", lambda *args, **kwargs: None)
    lifecycle = LifecycleStub()
    coordinator = StreamingCoordinator(lifecycle)
    response = FakeResponse(
        [
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
            b'data: {"timings":{"prompt_n":10,"cache_n":2,"prompt_ms":4.5}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    generator = await coordinator.start_stream_task(
        response=response,
        slot=(0, 1),
        key="cache-key",
        prefix="prefix",
        blocks=["a"],
        model_id="model-a",
        restore_key="restore-key",
        restored=True,
        persist_cache=True,
    )
    chunks = [chunk async for chunk in generator]

    assert len(chunks) == 3
    assert response.closed is True
    assert lifecycle.released == [(0, 1)]
    lifecycle.saved.assert_awaited_once()
    assert lifecycle.poisoned[0][:3] == ("restore-key", True, "model-a")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_multimodal_stream_closes_response_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("proxycache.services.streaming.add_lifecycle_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("proxycache.services.streaming.add_timing_attributes", lambda *args, **kwargs: None)
    monkeypatch.setattr("proxycache.services.streaming.set_error", lambda *args, **kwargs: None)
    lifecycle = LifecycleStub()
    coordinator = StreamingCoordinator(lifecycle)
    response = FakeResponse([b'data: {"choices":[]}\n\n'], exc=RuntimeError("stream failed"))

    generator = await coordinator.start_multimodal_stream(response, "model-a")
    chunks = [chunk async for chunk in generator]

    assert chunks == [b'data: {"choices":[]}\n\n']
    assert response.closed is True
    assert lifecycle.released == []
