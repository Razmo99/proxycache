import os
import sys
import tempfile
import json
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hashing as hs


class RetentionTests(unittest.TestCase):
    def test_prune_saved_caches_keeps_newest_keys_for_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = os.path.join(tmpdir, "meta")
            cache_dir = os.path.join(tmpdir, "slots")
            os.makedirs(meta_dir, exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)

            model_id = "qwen3-4b-128k"
            other_model = "different-model"

            def write_meta_file(key: str, model: str, ts: int) -> None:
                path = os.path.join(meta_dir, f"{key}.meta.json")
                payload = {
                    "key": key,
                    "model_id": model,
                    "prefix_len": 10,
                    "wpb": 100,
                    "blocks": ["x"],
                    "timestamp": ts,
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                os.utime(path, (ts, ts))
                with open(os.path.join(cache_dir, key), "w", encoding="utf-8") as f:
                    f.write(key)

            write_meta_file("old", model_id, 100)
            write_meta_file("mid", model_id, 200)
            write_meta_file("new", model_id, 300)
            write_meta_file("other", other_model, 400)

            with open(os.path.join(meta_dir, "old.poison.json"), "w", encoding="utf-8") as f:
                f.write("{}")

            deleted = hs.prune_saved_caches(
                model_id,
                keep=2,
                meta_dir=meta_dir,
                cache_dir=cache_dir,
            )

            self.assertGreaterEqual(deleted, 2)
            self.assertFalse(os.path.exists(os.path.join(meta_dir, "old.meta.json")))
            self.assertFalse(os.path.exists(os.path.join(cache_dir, "old")))
            self.assertFalse(os.path.exists(os.path.join(meta_dir, "old.poison.json")))
            self.assertTrue(os.path.exists(os.path.join(meta_dir, "mid.meta.json")))
            self.assertTrue(os.path.exists(os.path.join(meta_dir, "new.meta.json")))
            self.assertTrue(os.path.exists(os.path.join(cache_dir, "mid")))
            self.assertTrue(os.path.exists(os.path.join(cache_dir, "new")))
            self.assertTrue(os.path.exists(os.path.join(meta_dir, "other.meta.json")))
            self.assertTrue(os.path.exists(os.path.join(cache_dir, "other")))


if __name__ == "__main__":
    unittest.main()
