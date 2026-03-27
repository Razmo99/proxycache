# -*- coding: utf-8 -*-

"""Cache lifecycle helpers for save, restore, and retention."""

from __future__ import annotations

import logging
from typing import Any

from proxycache.cache.metadata import CacheStore
from proxycache.config import Settings
from proxycache.observability.otel import add_lifecycle_event, set_error
from proxycache.services.slots import GSlot, SlotManager

log = logging.getLogger(__name__)


class CacheLifecycleManager:
    """Encapsulates cache save/restore side effects."""

    def __init__(
        self,
        settings: Settings,
        slot_manager: SlotManager,
        cache_store: CacheStore,
    ) -> None:
        self.settings = settings
        self.slot_manager = slot_manager
        self.cache_store = cache_store

    async def save_cache_artifacts(
        self,
        slot: GSlot,
        key: str,
        prefix: str,
        blocks: list[str],
        model_id: str,
        span=None,
    ) -> bool:
        add_lifecycle_event(
            span,
            "proxycache.cache.save.started",
            backend_id=slot[0],
            slot_id=slot[1],
            cache_key_prefix=key[:16],
            model_id=model_id,
        )
        try:
            saved = await self.slot_manager.save_after(slot, key)
        except Exception as exc:
            set_error(span, exc.__class__.__name__, str(exc))
            add_lifecycle_event(
                span,
                "proxycache.cache.save.failed",
                backend_id=slot[0],
                slot_id=slot[1],
                cache_key_prefix=key[:16],
                error_type=exc.__class__.__name__,
            )
            log.warning("save_after_exception g=%s key=%s: %s", slot, key[:16], exc)
            return False

        add_lifecycle_event(
            span,
            "proxycache.cache.save.completed",
            backend_id=slot[0],
            slot_id=slot[1],
            cache_key_prefix=key[:16],
            saved=saved,
        )
        if not saved:
            return False

        self.cache_store.clear_restore_poison(key)
        try:
            self.cache_store.write_meta(
                key,
                prefix,
                blocks,
                self.settings.words_per_block,
                model_id,
            )
            add_lifecycle_event(
                span,
                "proxycache.cache.metadata.recorded",
                cache_key_prefix=key[:16],
                model_id=model_id,
            )
        except Exception as exc:
            set_error(span, exc.__class__.__name__, str(exc))
            add_lifecycle_event(
                span,
                "proxycache.cache.metadata.record.failed",
                cache_key_prefix=key[:16],
                model_id=model_id,
                error_type=exc.__class__.__name__,
            )
            log.warning("record_saved_cache_exception key=%s: %s", key[:16], exc)
            return True

        log.info(
            "record_saved_cache key=%s model_id=%s keep=%d",
            key[:16],
            model_id,
            self.settings.max_saved_caches,
        )
        pruned = self.cache_store.prune_saved_caches(
            model_id,
            keep=self.settings.max_saved_caches,
        )
        if pruned:
            log.info(
                "retention_pruned_artifacts key=%s model_id=%s deleted=%d keep=%d",
                key[:16],
                model_id,
                pruned,
                self.settings.max_saved_caches,
            )
        return True

    def release_slot(self, slot: GSlot, span=None) -> None:
        self.slot_manager.release(slot)
        add_lifecycle_event(
            span,
            "proxycache.slot.released",
            backend_id=slot[0],
            slot_id=slot[1],
        )

    @staticmethod
    def record_restore_outcome(
        span,
        slot: GSlot,
        restore_key: str | None,
        restored: bool | None,
    ) -> None:
        if not restore_key:
            return
        add_lifecycle_event(
            span,
            "proxycache.cache.restore.completed"
            if restored
            else "proxycache.cache.restore.failed",
            backend_id=slot[0],
            slot_id=slot[1],
            cache_key_prefix=restore_key[:16],
        )

    def maybe_poison_restore(
        self,
        restore_key: str | None,
        restored: bool | None,
        model_id: str,
        timings: dict[str, Any] | None,
        span=None,
    ) -> None:
        if not restore_key or not restored or not timings:
            return
        prompt_n = int(timings.get("prompt_n") or 0)
        cache_n = int(timings.get("cache_n") or 0)
        prompt_ms = float(timings.get("prompt_ms") or 0.0)
        if prompt_n > 0 and cache_n == 0:
            try:
                self.cache_store.poison_restore_key(
                    restore_key,
                    model_id,
                    prompt_n=prompt_n,
                    cache_n=cache_n,
                    prompt_ms=prompt_ms,
                )
                add_lifecycle_event(
                    span,
                    "proxycache.cache.restore.poisoned",
                    cache_key_prefix=restore_key[:16],
                    model_id=model_id,
                    prompt_tokens=prompt_n,
                    cache_read_tokens=cache_n,
                    reason="no_cache_reuse_after_restore",
                )
            except Exception as exc:
                log.warning("poison_restore_exception key=%s err=%s", restore_key[:16], exc)
