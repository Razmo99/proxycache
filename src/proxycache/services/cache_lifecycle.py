# -*- coding: utf-8 -*-

"""Cache lifecycle helpers for save, restore, and retention."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from proxycache.cache.metadata import CacheStore, LOW_CACHE_REUSE_RATIO_THRESHOLD
from proxycache.config import Settings
from proxycache.observability.otel import add_lifecycle_event, set_error
from proxycache.services.slots import GSlot, SlotManager

log = logging.getLogger(__name__)
LOW_CACHE_REUSE_GAP_THRESHOLD = 0.25


@dataclass(frozen=True, slots=True)
class RestoreAssessment:
    """Observed restore quality derived from llama.cpp timings."""

    prompt_n: int
    cache_n: int
    prompt_ms: float
    actual_ratio: float
    degraded: bool
    poison_reason: str | None = None


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
        system_fingerprint: str = "",
        tools_fingerprint: str = "",
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
                system_fingerprint=system_fingerprint,
                tools_fingerprint=tools_fingerprint,
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

    def assess_restore_quality(
        self,
        restore_key: str | None,
        restored: bool | None,
        model_id: str,
        timings: dict[str, Any] | None,
        match_ratio: float | None = None,
        span=None,
    ) -> RestoreAssessment | None:
        if not restore_key or not restored or not timings:
            return None
        prompt_n = int(timings.get("prompt_n") or 0)
        cache_n = int(timings.get("cache_n") or 0)
        prompt_ms = float(timings.get("prompt_ms") or 0.0)
        actual_ratio = cache_n / prompt_n if prompt_n > 0 else 0.0
        poisoned_reason: str | None = None
        degraded = prompt_n > 0 and (
            cache_n == 0
            or (
                match_ratio is not None
                and actual_ratio < LOW_CACHE_REUSE_RATIO_THRESHOLD
                and actual_ratio + LOW_CACHE_REUSE_GAP_THRESHOLD < match_ratio
            )
        )

        add_lifecycle_event(
            span,
            "proxycache.cache.restore.quality.evaluated",
            cache_key_prefix=restore_key[:16],
            model_id=model_id,
            prompt_tokens=prompt_n,
            cache_read_tokens=cache_n,
            actual_cache_read_ratio=round(actual_ratio, 6),
            predicted_cache_read_ratio=round(match_ratio, 6) if match_ratio is not None else None,
            degraded=degraded,
        )

        if degraded:
            poisoned_reason = (
                "no_cache_reuse_after_restore"
                if cache_n == 0
                else "low_cache_reuse_after_restore"
            )
            try:
                self.cache_store.poison_restore_key(
                    restore_key,
                    model_id,
                    prompt_n=prompt_n,
                    cache_n=cache_n,
                    prompt_ms=prompt_ms,
                    reason=poisoned_reason,
                )
                add_lifecycle_event(
                    span,
                    "proxycache.cache.restore.poisoned",
                    cache_key_prefix=restore_key[:16],
                    model_id=model_id,
                    prompt_tokens=prompt_n,
                    cache_read_tokens=cache_n,
                    actual_cache_read_ratio=round(actual_ratio, 6),
                    predicted_cache_read_ratio=round(match_ratio, 6) if match_ratio is not None else None,
                    reason=poisoned_reason,
                )
                log.warning(
                    "restore_degraded key=%s actual_ratio=%.3f predicted_ratio=%s prompt_n=%d cache_n=%d",
                    restore_key[:16],
                    actual_ratio,
                    f"{match_ratio:.3f}" if match_ratio is not None else "n/a",
                    prompt_n,
                    cache_n,
                )
            except Exception as exc:
                log.warning("poison_restore_exception key=%s err=%s", restore_key[:16], exc)
        return RestoreAssessment(
            prompt_n=prompt_n,
            cache_n=cache_n,
            prompt_ms=prompt_ms,
            actual_ratio=actual_ratio,
            degraded=degraded,
            poison_reason=poisoned_reason,
        )
