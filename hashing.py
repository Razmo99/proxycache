# hashing.py

# -*- coding: utf-8 -*-

"""
Raw-хэширование: raw_prefix без ролей, только контент, разделённый двойным переводом строки.

Блоки по 100 слов, LCP по полным SHA256-хэшам.
Key = sha256(model_id + "\\n" + raw_prefix), т.е. модель включена в ключ.

Метафайлы содержат:
- key
- model_id
- prefix_len
- wpb
- blocks
- timestamp
"""

import os
import json
import hashlib
import re
import time
import glob
import logging
from typing import List, Dict, Optional, Tuple

from config import MAX_SAVED_CACHES, META_DIR, SLOT_SAVE_PATH, WORDS_PER_BLOCK

log = logging.getLogger(__name__)
POISON_SUFFIX = ".poison.json"


def raw_prefix(messages: List[Dict]) -> str:
    parts = []
    for msg in messages or []:
        content = msg.get("content", "")
        if isinstance(content, str):
            content = content.strip()
        else:
            content = str(content).strip()
        if content:
            parts.append(content)
    text = "\n\n".join(parts).strip()
    log.debug("raw_prefix len_chars=%d", len(text))
    return text


def words_from_text(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def block_hashes_from_text(text: str, wpb: int = WORDS_PER_BLOCK) -> List[str]:
    words = words_from_text(text)
    hashes: List[str] = []
    for i in range(0, len(words), wpb):
        block = " ".join(words[i:i + wpb])
        h = hashlib.sha256(block.encode("utf-8")).hexdigest()
        hashes.append(h)
    log.debug("block_hashes n_blocks=%d wpb=%d", len(hashes), wpb)
    return hashes


def lcp_blocks(blocks1: List[str], blocks2: List[str]) -> int:
    n = min(len(blocks1), len(blocks2))
    i = 0
    while i < n and blocks1[i] == blocks2[i]:
        i += 1
    return i


def prefix_key_sha256(text: str) -> str:
    """
    Базовая SHA256-обёртка; для кеша в неё передаём model_id + "\\n" + raw_prefix.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan_all_meta(meta_dir: str = META_DIR) -> List[Dict]:
    files = sorted(
        glob.glob(os.path.join(meta_dir, "*.meta.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    metas: List[Dict] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fd:
                meta = json.load(fd)
                metas.append(meta)
        except Exception as e:
            log.warning("scan_meta_fail %s: %s", f, e)
    log.debug("scan_meta n_found=%d", len(metas))
    return metas


def find_best_restore_candidate(
    req_blocks: List[str],
    wpb: int,
    th: float,
    model_id: str,
) -> Optional[Tuple[str, float]]:
    """
    Ищет лучший кандидат для restore среди мета-файлов ТОЛЬКО текущей модели.

    Фильтруем по:
    - meta["model_id"] == model_id
    - meta["wpb"] == wpb
    """
    metas = scan_all_meta()
    best_key: Optional[str] = None
    best_ratio = 0.0

    for meta in metas:
        if meta.get("model_id") != model_id:
            continue
        if int(meta.get("wpb") or 0) != wpb:
            continue
        key = meta.get("key")
        if not key or is_restore_poisoned(key):
            continue

        cand_blocks = meta.get("blocks") or []
        lcp = lcp_blocks(req_blocks, cand_blocks)
        denom = max(1, min(len(req_blocks), len(cand_blocks)))
        ratio = lcp / denom

        if ratio >= th and ratio > best_ratio:
            best_ratio = ratio
            best_key = key

    return (best_key, best_ratio) if best_key else None


def write_meta(
    key: str,
    prefix_text: str,
    blocks: List[str],
    wpb: int,
    model_id: str,
) -> None:
    """
    Записывает/перезаписывает meta-файл для key, привязанный к конкретной модели.
    """
    meta = {
        "key": key,
        "model_id": model_id,
        "prefix_len": len(prefix_text),
        "wpb": wpb,
        "blocks": blocks,
        "timestamp": time.time(),
    }
    path = os.path.join(META_DIR, f"{key}.meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def touch_meta(key: str) -> None:
    """
    Обновляет timestamp в существующем meta-файле key.meta.json.
    """
    path = os.path.join(META_DIR, f"{key}.meta.json")
    try:
        with open(path, "r+", encoding="utf-8") as f:
            try:
                meta = json.load(f)
            except Exception as e:
                log.warning("touch_meta_read_fail key=%s: %s", key[:16], e)
                return
            meta["timestamp"] = time.time()
            f.seek(0)
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.truncate()
        log.debug("touch_meta_ok key=%s", key[:16])
    except FileNotFoundError:
        log.warning("touch_meta_missing key=%s", key[:16])
    except Exception as e:
        log.warning("touch_meta_fail key=%s: %s", key[:16], e)


def _poison_path(key: str) -> str:
    return os.path.join(META_DIR, f"{key}{POISON_SUFFIX}")


def _meta_path(key: str) -> str:
    return os.path.join(META_DIR, f"{key}.meta.json")


def _meta_path_in_dir(key: str, meta_dir: str) -> str:
    return os.path.join(meta_dir, f"{key}.meta.json")


def _poison_path_in_dir(key: str, meta_dir: str) -> str:
    return os.path.join(meta_dir, f"{key}{POISON_SUFFIX}")


def _delete_poison_file(key: str, reason: str) -> None:
    path = _poison_path(key)
    try:
        os.remove(path)
        log.info("restore_poison_deleted key=%s reason=%s", key[:16], reason)
    except FileNotFoundError:
        return
    except Exception as e:
        log.warning("restore_poison_delete_fail key=%s reason=%s err=%s", key[:16], reason, e)


def is_restore_poisoned(key: str) -> bool:
    path = _poison_path(key)
    if not os.path.exists(path):
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        log.warning("restore_poison_read_fail key=%s: %s", key[:16], e)
        _delete_poison_file(key, "unreadable")
        return False

    if not os.path.exists(_meta_path(key)):
        _delete_poison_file(key, "orphaned")
        return False

    prompt_n = int(payload.get("prompt_n") or 0)
    cache_n = int(payload.get("cache_n") or 0)
    if prompt_n > 0 and cache_n == 0:
        return True

    _delete_poison_file(key, "inactive")
    return False


def clear_restore_poison(key: str) -> None:
    _delete_poison_file(key, "cleared")


def prune_saved_caches(
    model_id: str,
    keep: int = MAX_SAVED_CACHES,
    meta_dir: str = META_DIR,
    cache_dir: str = SLOT_SAVE_PATH,
) -> int:
    """
    Keep only the newest saved caches for this proxy instance/model.

    Retention is driven by the local meta directory, while actual KV blobs live
    under cache_dir and are addressed by basename == meta["key"].
    """
    if keep < 0:
        keep = 0

    metas = []
    for meta in scan_all_meta(meta_dir):
        if meta.get("model_id") != model_id:
            continue
        key = meta.get("key")
        if not key:
            continue
        metas.append(meta)

    log.info(
        "retention_scan model_id=%s meta_dir=%s cache_dir=%s keep=%d found=%d",
        model_id,
        meta_dir,
        cache_dir,
        keep,
        len(metas),
    )

    deleted = 0
    for meta in metas[keep:]:
        key = str(meta["key"])

        blob_path = os.path.join(cache_dir, key)
        for path in (
            blob_path,
            _meta_path_in_dir(key, meta_dir),
            _poison_path_in_dir(key, meta_dir),
        ):
            try:
                os.remove(path)
                deleted += 1
            except FileNotFoundError:
                continue
            except Exception as e:
                log.warning("retention_delete_fail key=%s path=%s err=%s", key[:16], path, e)

        log.info(
            "retention_pruned key=%s model_id=%s cache_dir=%s",
            key[:16],
            model_id,
            cache_dir,
        )

    return deleted


def cleanup_restore_poisons() -> int:
    deleted = 0
    for path in glob.glob(os.path.join(META_DIR, f"*{POISON_SUFFIX}")):
        name = os.path.basename(path)
        if not name.endswith(POISON_SUFFIX):
            continue
        key = name[: -len(POISON_SUFFIX)]
        was_poisoned = os.path.exists(path)
        still_poisoned = is_restore_poisoned(key)
        if was_poisoned and not still_poisoned:
            deleted += 1
    return deleted


def poison_restore_key(
    key: str,
    model_id: str,
    prompt_n: int,
    cache_n: int,
    prompt_ms: float,
    reason: str = "no_cache_reuse_after_restore",
) -> None:
    path = _poison_path(key)
    payload = {
        "key": key,
        "model_id": model_id,
        "reason": reason,
        "prompt_n": prompt_n,
        "cache_n": cache_n,
        "prompt_ms": prompt_ms,
        "timestamp": time.time(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.warning(
        "restore_poisoned key=%s prompt_n=%d cache_n=%d prompt_ms=%.2f",
        key[:16],
        prompt_n,
        cache_n,
        prompt_ms,
    )
