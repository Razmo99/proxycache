import json
import os
import tempfile
from pathlib import Path

from proxycache.cache.metadata import CacheStore


def test_prune_saved_caches_keeps_newest_keys_for_model() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        meta_dir = root / "meta"
        cache_dir = root / "slots"
        meta_dir.mkdir()
        cache_dir.mkdir()

        store = CacheStore(
            meta_dir=meta_dir,
            cache_dir=cache_dir,
            words_per_block=100,
            max_saved_caches=2,
        )

        model_id = "qwen3-4b-128k"
        other_model = "different-model"

        def write_meta_file(key: str, model: str, timestamp: int) -> None:
            payload = {
                "key": key,
                "model_id": model,
                "prefix_len": 10,
                "wpb": 100,
                "blocks": ["x"],
                "timestamp": timestamp,
            }
            path = meta_dir / f"{key}.meta.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.utime(path, (timestamp, timestamp))
            (cache_dir / key).write_text(key, encoding="utf-8")

        write_meta_file("old", model_id, 100)
        write_meta_file("mid", model_id, 200)
        write_meta_file("new", model_id, 300)
        write_meta_file("other", other_model, 400)
        (meta_dir / "old.poison.json").write_text("{}", encoding="utf-8")

        deleted = store.prune_saved_caches(model_id, keep=2, meta_dir=meta_dir, cache_dir=cache_dir)

        assert deleted >= 2
        assert not (meta_dir / "old.meta.json").exists()
        assert not (cache_dir / "old").exists()
        assert not (meta_dir / "old.poison.json").exists()
        assert (meta_dir / "mid.meta.json").exists()
        assert (meta_dir / "new.meta.json").exists()
        assert (cache_dir / "mid").exists()
        assert (cache_dir / "new").exists()
        assert (meta_dir / "other.meta.json").exists()
        assert (cache_dir / "other").exists()
