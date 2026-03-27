# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from proxycache.cache.metadata import CacheStore


@pytest.mark.unit
def test_raw_prefix_joins_non_empty_content(cache_store: CacheStore) -> None:
    prefix = cache_store.raw_prefix(
        [
            {"role": "system", "content": "  alpha  "},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ["not", "text"]},
        ]
    )

    assert prefix == "alpha\n\n['not', 'text']"


@pytest.mark.unit
def test_find_best_restore_candidate_skips_poisoned_entries(
    cache_store: CacheStore,
    tmp_path: Path,
) -> None:
    req_text = "one two three four"
    req_blocks = cache_store.block_hashes_from_text(req_text, 2)
    cache_store.write_meta("good-key", req_text, req_blocks, 2, "model-a")
    cache_store.write_meta("poison-key", req_text, req_blocks, 2, "model-a")
    cache_store.poison_restore_key("poison-key", "model-a", prompt_n=10, cache_n=0, prompt_ms=20.0)

    result = cache_store.find_best_restore_candidate(req_blocks, 2, 0.5, "model-a")

    assert result == ("good-key", 1.0)


@pytest.mark.unit
def test_is_restore_poisoned_clears_orphaned_file(cache_store: CacheStore) -> None:
    poison_path = cache_store.meta_dir / "orphan.poison.json"
    poison_path.write_text(
        json.dumps({"prompt_n": 5, "cache_n": 0}),
        encoding="utf-8",
    )

    assert cache_store.is_restore_poisoned("orphan") is False
    assert not poison_path.exists()


@pytest.mark.unit
def test_cleanup_restore_poisons_counts_deleted_inactive_entries(cache_store: CacheStore) -> None:
    key = "inactive-key"
    cache_store.write_meta(key, "one two", ["block"], 2, "model-a")
    (cache_store.meta_dir / f"{key}.poison.json").write_text(
        json.dumps({"prompt_n": 4, "cache_n": 1}),
        encoding="utf-8",
    )

    deleted = cache_store.cleanup_restore_poisons()

    assert deleted == 1
    assert not (cache_store.meta_dir / f"{key}.poison.json").exists()


@pytest.mark.unit
def test_scan_all_meta_skips_invalid_json(cache_store: CacheStore) -> None:
    valid_key = "valid"
    cache_store.write_meta(valid_key, "one two", ["block"], 2, "model-a")
    broken_path = cache_store.meta_dir / "broken.meta.json"
    broken_path.write_text("{not-json", encoding="utf-8")
    os.utime(broken_path, None)

    metas = cache_store.scan_all_meta()

    assert [meta.key for meta in metas] == [valid_key]

