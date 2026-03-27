# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from proxycache.services.cache_lifecycle import CacheLifecycleManager


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_cache_artifacts_records_meta_and_prunes(make_settings, cache_store) -> None:
    settings = make_settings(name="lifecycle")
    slot_manager = SimpleNamespace(save_after=AsyncMock(return_value=True), release=lambda slot: None)
    lifecycle = CacheLifecycleManager(settings, slot_manager, cache_store)
    cache_store.prune_saved_caches = lambda model_id, keep: 3

    ok = await lifecycle.save_cache_artifacts((0, 1), "cache-key", "prefix", ["a"], "model-a")

    assert ok is True
    assert (cache_store.meta_dir / "cache-key.meta.json").exists()
    slot_manager.save_after.assert_awaited_once_with((0, 1), "cache-key")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_save_cache_artifacts_returns_false_when_save_raises(make_settings, cache_store) -> None:
    settings = make_settings(name="lifecycle-save-fail")
    slot_manager = SimpleNamespace(save_after=AsyncMock(side_effect=RuntimeError("save failed")), release=lambda slot: None)
    lifecycle = CacheLifecycleManager(settings, slot_manager, cache_store)

    ok = await lifecycle.save_cache_artifacts((0, 1), "cache-key", "prefix", ["a"], "model-a")

    assert ok is False
    assert not (cache_store.meta_dir / "cache-key.meta.json").exists()


@pytest.mark.unit
def test_maybe_poison_restore_only_marks_failed_cache_reuse(make_settings, cache_store) -> None:
    settings = make_settings(name="poison")
    slot_manager = SimpleNamespace(save_after=None, release=lambda slot: None)
    lifecycle = CacheLifecycleManager(settings, slot_manager, cache_store)
    cache_store.write_meta("cache-key", "prefix", ["a"], 2, "model-a")

    lifecycle.maybe_poison_restore(
        "cache-key",
        True,
        "model-a",
        {"prompt_n": 20, "cache_n": 0, "prompt_ms": 55.0},
    )

    assert cache_store.is_restore_poisoned("cache-key") is True


@pytest.mark.unit
def test_release_slot_delegates_to_slot_manager(make_settings, cache_store) -> None:
    released: list[tuple[int, int]] = []
    settings = make_settings(name="release")
    slot_manager = SimpleNamespace(save_after=None, release=lambda slot: released.append(slot))
    lifecycle = CacheLifecycleManager(settings, slot_manager, cache_store)

    lifecycle.release_slot((0, 1))

    assert released == [(0, 1)]

