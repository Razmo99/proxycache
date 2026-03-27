# -*- coding: utf-8 -*-

"""KV cache metadata management and similarity helpers."""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
POISON_SUFFIX = ".poison.json"


@dataclass(frozen=True, slots=True)
class CacheMetadata:
    """Metadata persisted for one saved KV cache."""

    key: str
    model_id: str
    prefix_len: int
    wpb: int
    blocks: list[str]
    timestamp: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CacheMetadata:
        return cls(
            key=str(payload["key"]),
            model_id=str(payload["model_id"]),
            prefix_len=int(payload["prefix_len"]),
            wpb=int(payload["wpb"]),
            blocks=[str(block) for block in payload.get("blocks", [])],
            timestamp=float(payload.get("timestamp", 0.0)),
        )


class CacheStore:
    """Handles cache hashing, metadata, and retention."""

    def __init__(
        self,
        meta_dir: Path,
        cache_dir: Path,
        words_per_block: int,
        max_saved_caches: int,
    ) -> None:
        self.meta_dir = meta_dir
        self.cache_dir = cache_dir
        self.words_per_block = words_per_block
        self.max_saved_caches = max_saved_caches

    def raw_prefix(self, messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages or []:
            content = message.get("content", "")
            rendered = content.strip() if isinstance(content, str) else str(content).strip()
            if rendered:
                parts.append(rendered)
        text = "\n\n".join(parts).strip()
        log.debug("raw_prefix len_chars=%d", len(text))
        return text

    @staticmethod
    def words_from_text(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def block_hashes_from_text(self, text: str, wpb: int | None = None) -> list[str]:
        block_size = wpb or self.words_per_block
        words = self.words_from_text(text)
        hashes: list[str] = []
        for index in range(0, len(words), block_size):
            block = " ".join(words[index : index + block_size])
            hashes.append(hashlib.sha256(block.encode("utf-8")).hexdigest())
        log.debug("block_hashes n_blocks=%d wpb=%d", len(hashes), block_size)
        return hashes

    @staticmethod
    def lcp_blocks(blocks1: list[str], blocks2: list[str]) -> int:
        limit = min(len(blocks1), len(blocks2))
        index = 0
        while index < limit and blocks1[index] == blocks2[index]:
            index += 1
        return index

    @staticmethod
    def prefix_key_sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def scan_all_meta(self, meta_dir: Path | None = None) -> list[CacheMetadata]:
        target_dir = meta_dir or self.meta_dir
        files = sorted(
            glob.glob(str(target_dir / "*.meta.json")),
            key=os.path.getmtime,
            reverse=True,
        )
        metas: list[CacheMetadata] = []
        for path in files:
            try:
                with open(path, encoding="utf-8") as handle:
                    metas.append(CacheMetadata.from_dict(json.load(handle)))
            except Exception as exc:
                log.warning("scan_meta_fail %s: %s", path, exc)
        return metas

    def find_best_restore_candidate(
        self,
        req_blocks: list[str],
        wpb: int,
        threshold: float,
        model_id: str,
    ) -> tuple[str, float] | None:
        best_key: str | None = None
        best_ratio = 0.0

        for meta in self.scan_all_meta():
            if meta.model_id != model_id or meta.wpb != wpb:
                continue
            if self.is_restore_poisoned(meta.key):
                continue

            lcp = self.lcp_blocks(req_blocks, meta.blocks)
            denom = max(1, min(len(req_blocks), len(meta.blocks)))
            ratio = lcp / denom
            if ratio >= threshold and ratio > best_ratio:
                best_ratio = ratio
                best_key = meta.key

        return (best_key, best_ratio) if best_key else None

    def write_meta(
        self,
        key: str,
        prefix_text: str,
        blocks: list[str],
        wpb: int,
        model_id: str,
    ) -> None:
        payload = CacheMetadata(
            key=key,
            model_id=model_id,
            prefix_len=len(prefix_text),
            wpb=wpb,
            blocks=blocks,
            timestamp=time.time(),
        )
        with self._meta_path(key).open("w", encoding="utf-8") as handle:
            json.dump(payload.__dict__, handle, indent=2, ensure_ascii=False)

    def touch_meta(self, key: str) -> None:
        path = self._meta_path(key)
        try:
            with path.open("r+", encoding="utf-8") as handle:
                payload = json.load(handle)
                payload["timestamp"] = time.time()
                handle.seek(0)
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.truncate()
        except FileNotFoundError:
            log.warning("touch_meta_missing key=%s", key[:16])
        except Exception as exc:
            log.warning("touch_meta_fail key=%s: %s", key[:16], exc)

    def prune_saved_caches(
        self,
        model_id: str,
        keep: int | None = None,
        meta_dir: Path | None = None,
        cache_dir: Path | None = None,
    ) -> int:
        keep_count = max(0, self.max_saved_caches if keep is None else keep)
        target_meta_dir = meta_dir or self.meta_dir
        target_cache_dir = cache_dir or self.cache_dir

        metas = [
            meta
            for meta in self.scan_all_meta(target_meta_dir)
            if meta.model_id == model_id and meta.key
        ]
        log.info(
            "retention_scan model_id=%s meta_dir=%s cache_dir=%s keep=%d found=%d",
            model_id,
            target_meta_dir,
            target_cache_dir,
            keep_count,
            len(metas),
        )

        deleted = 0
        for meta in metas[keep_count:]:
            blob_path = target_cache_dir / meta.key
            for path in (
                blob_path,
                self._meta_path(meta.key, target_meta_dir),
                self._poison_path(meta.key, target_meta_dir),
            ):
                try:
                    path.unlink()
                    deleted += 1
                except FileNotFoundError:
                    continue
                except Exception as exc:
                    log.warning(
                        "retention_delete_fail key=%s path=%s err=%s",
                        meta.key[:16],
                        path,
                        exc,
                    )
            log.info(
                "retention_pruned key=%s model_id=%s cache_dir=%s",
                meta.key[:16],
                model_id,
                target_cache_dir,
            )
        return deleted

    def cleanup_restore_poisons(self) -> int:
        deleted = 0
        for path in self.meta_dir.glob(f"*{POISON_SUFFIX}"):
            key = path.name[: -len(POISON_SUFFIX)]
            was_poisoned = path.exists()
            still_poisoned = self.is_restore_poisoned(key)
            if was_poisoned and not still_poisoned:
                deleted += 1
        return deleted

    def poison_restore_key(
        self,
        key: str,
        model_id: str,
        prompt_n: int,
        cache_n: int,
        prompt_ms: float,
        reason: str = "no_cache_reuse_after_restore",
    ) -> None:
        payload = {
            "key": key,
            "model_id": model_id,
            "reason": reason,
            "prompt_n": prompt_n,
            "cache_n": cache_n,
            "prompt_ms": prompt_ms,
            "timestamp": time.time(),
        }
        with self._poison_path(key).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        log.warning(
            "restore_poisoned key=%s prompt_n=%d cache_n=%d prompt_ms=%.2f",
            key[:16],
            prompt_n,
            cache_n,
            prompt_ms,
        )

    def clear_restore_poison(self, key: str) -> None:
        self._delete_poison_file(key, "cleared")

    def is_restore_poisoned(self, key: str) -> bool:
        path = self._poison_path(key)
        if not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            log.warning("restore_poison_read_fail key=%s: %s", key[:16], exc)
            self._delete_poison_file(key, "unreadable")
            return False

        if not self._meta_path(key).exists():
            self._delete_poison_file(key, "orphaned")
            return False

        prompt_n = int(payload.get("prompt_n") or 0)
        cache_n = int(payload.get("cache_n") or 0)
        if prompt_n > 0 and cache_n == 0:
            return True

        self._delete_poison_file(key, "inactive")
        return False

    def _delete_poison_file(self, key: str, reason: str) -> None:
        path = self._poison_path(key)
        try:
            path.unlink()
            log.info("restore_poison_deleted key=%s reason=%s", key[:16], reason)
        except FileNotFoundError:
            return
        except Exception as exc:
            log.warning(
                "restore_poison_delete_fail key=%s reason=%s err=%s",
                key[:16],
                reason,
                exc,
            )

    def _meta_path(self, key: str, meta_dir: Path | None = None) -> Path:
        return (meta_dir or self.meta_dir) / f"{key}.meta.json"

    def _poison_path(self, key: str, meta_dir: Path | None = None) -> Path:
        return (meta_dir or self.meta_dir) / f"{key}{POISON_SUFFIX}"
