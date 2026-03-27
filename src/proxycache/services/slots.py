# -*- coding: utf-8 -*-

"""Slot allocation and lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from proxycache.config import BackendSettings

log = logging.getLogger(__name__)
GSlot = tuple[int, int]


@dataclass(slots=True)
class BackendRuntime:
    """Runtime backend state."""

    identifier: int
    client: Any
    n_slots: int


class SlotManager:
    """LRU slot manager across configured backends."""

    def __init__(self, backends: tuple[BackendSettings, ...]) -> None:
        self.backends = [
            BackendRuntime(identifier=index, client=None, n_slots=backend.n_slots)
            for index, backend in enumerate(backends)
        ]
        self._all_slots: list[GSlot] = [
            (backend.identifier, slot_id)
            for backend in self.backends
            for slot_id in range(backend.n_slots)
        ]
        self._last_used: dict[GSlot, float] = {slot: 0.0 for slot in self._all_slots}
        self._locks: dict[GSlot, asyncio.Lock] = {slot: asyncio.Lock() for slot in self._all_slots}
        self._is_multimodal_backends: dict[int, bool] = {}
        total_slots = sum(backend.n_slots for backend in self.backends)
        log.info("slot_manager n_backends=%d total_slots=%d", len(self.backends), total_slots)

    def set_clients(self, clients: list[Any]) -> None:
        for index, client in enumerate(clients):
            self.backends[index].client = client

    async def set_multimodal_backend(self, backend_id: int, is_multimodal: bool) -> None:
        self._is_multimodal_backends[backend_id] = is_multimodal
        log.info("multimodal_backend_set be_id=%d is_mm=%s", backend_id, is_multimodal)

    def is_multimodal_backend(self, backend_id: int) -> bool:
        return self._is_multimodal_backends.get(backend_id, False)

    def locked_slot_count(self) -> int:
        return sum(1 for lock in self._locks.values() if lock.locked())

    async def acquire_for_request(
        self,
        restore_key: str | None = None,
    ) -> tuple[GSlot, asyncio.Lock, bool | None]:
        slot, lock = self._get_free_or_oldest()
        await lock.acquire()
        restored: bool | None = None
        if restore_key:
            client = self.backends[slot[0]].client
            restored = await client.restore_slot(slot[1], restore_key)
            log.info(
                "restore_before_chat g=%s key=%s ok=%s",
                slot,
                restore_key[:16],
                restored,
            )
        return slot, lock, restored

    async def save_after(self, slot: GSlot, key: str) -> bool:
        client = self.backends[slot[0]].client
        saved = await client.save_slot(slot[1], key)
        self._last_used[slot] = time.time()
        return saved

    def release(self, slot: GSlot) -> None:
        lock = self._locks[slot]
        if lock.locked():
            lock.release()

    def _is_free(self, slot: GSlot) -> bool:
        return self._last_used.get(slot, 0.0) == 0.0

    def _get_free_or_oldest(self) -> tuple[GSlot, asyncio.Lock]:
        free_slots = [slot for slot in self._all_slots if self._is_free(slot)]
        if free_slots:
            slot = free_slots[0]
            return slot, self._locks[slot]
        slot = min(self._all_slots, key=lambda item: self._last_used.get(item, 0.0))
        return slot, self._locks[slot]
