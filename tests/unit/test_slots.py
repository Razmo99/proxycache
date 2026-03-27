# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from proxycache.config import BackendSettings
from proxycache.services.slots import SlotManager


@pytest.mark.unit
@pytest.mark.asyncio
async def test_acquire_for_request_uses_free_slot_first() -> None:
    manager = SlotManager((BackendSettings(url="http://one.test", n_slots=2),))
    manager.set_clients([SimpleNamespace(restore_slot=None, save_slot=None)])

    slot, _, restored = await manager.acquire_for_request()
    manager.release(slot)

    assert slot == (0, 0)
    assert restored is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_acquire_for_request_restores_when_restore_key_present() -> None:
    restore_calls: list[tuple[int, str]] = []

    async def restore_slot(slot_id: int, key: str) -> bool:
        restore_calls.append((slot_id, key))
        return True

    manager = SlotManager((BackendSettings(url="http://one.test", n_slots=1),))
    manager.set_clients([SimpleNamespace(restore_slot=restore_slot, save_slot=None)])

    slot, _, restored = await manager.acquire_for_request("restore-key")
    manager.release(slot)

    assert slot == (0, 0)
    assert restored is True
    assert restore_calls == [(0, "restore-key")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_acquire_for_request_waits_until_locked_slot_is_released() -> None:
    manager = SlotManager((BackendSettings(url="http://one.test", n_slots=1),))
    manager.set_clients([SimpleNamespace(restore_slot=None, save_slot=None)])

    first_slot, _, first_restored = await manager.acquire_for_request()
    second_task = asyncio.create_task(manager.acquire_for_request())
    await asyncio.sleep(0)

    assert second_task.done() is False
    assert manager.locked_slot_count() == 1
    assert first_restored is None

    manager.release(first_slot)
    second_slot, _, second_restored = await second_task
    manager.release(second_slot)

    assert second_slot == first_slot
    assert second_restored is None


@pytest.mark.unit
def test_get_free_or_oldest_prefers_oldest_used_slot() -> None:
    manager = SlotManager((BackendSettings(url="http://one.test", n_slots=2),))
    manager._last_used[(0, 0)] = 200.0
    manager._last_used[(0, 1)] = 100.0

    slot, _ = manager._get_free_or_oldest()

    assert slot == (0, 1)


@pytest.mark.unit
def test_locked_slot_count_tracks_lock_state() -> None:
    manager = SlotManager((BackendSettings(url="http://one.test", n_slots=1),))
    lock = manager._locks[(0, 0)]
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(lock.acquire())
        assert manager.locked_slot_count() == 1
    finally:
        manager.release((0, 0))
        loop.close()
