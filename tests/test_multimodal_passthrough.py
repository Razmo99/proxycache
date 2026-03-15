import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import _looks_like_multimodal_model_name, _slot_management_mode
from llama_client import LlamaClient


class MultimodalPassthroughTests(unittest.IsolatedAsyncioTestCase):
    def test_model_name_heuristic_detects_qwen_vl(self):
        self.assertTrue(_looks_like_multimodal_model_name("qwen3-vl-30b-instruct-bf16"))
        self.assertFalse(_looks_like_multimodal_model_name("qwen3-4b-128k"))

    def test_slot_management_uses_heuristic_when_capabilities_missing(self):
        bypass, reason = _slot_management_mode(
            backend_is_multimodal=False,
            backend_model_id="unknown",
            client_model="qwen3-vl-30b-instruct-bf16",
            slots_supported=True,
        )
        self.assertTrue(bypass)
        self.assertEqual(reason, "model_name_heuristic")

    async def test_save_slot_501_marks_backend_slots_unsupported(self):
        client = LlamaClient("http://example.test")
        real_client = client.client
        fake_response = SimpleNamespace(status_code=501)
        client.client = SimpleNamespace(post=AsyncMock(return_value=fake_response))
        try:
            ok = await client.save_slot(0, "test-key")
        finally:
            client.client = real_client
            await real_client.aclose()

        self.assertFalse(ok)
        self.assertFalse(client.slots_supported())


if __name__ == "__main__":
    unittest.main()
