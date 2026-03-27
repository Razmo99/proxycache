# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest

from proxycache.services.policies import (
    consume_sse_timings,
    message_has_multimodal_content,
    request_has_multimodal_workload,
    slot_management_mode,
)


@pytest.mark.unit
def test_message_has_multimodal_content_detects_image_part() -> None:
    message = {"content": [{"type": "text", "text": "look"}, {"type": "input_image", "image_url": "x"}]}

    assert message_has_multimodal_content(message) is True


@pytest.mark.unit
def test_request_has_multimodal_workload_returns_false_for_text_messages() -> None:
    assert request_has_multimodal_workload([{"content": "plain text"}]) is False


@pytest.mark.unit
def test_slot_management_mode_prefers_backend_heuristic() -> None:
    bypass, reason = slot_management_mode(
        backend_is_multimodal=False,
        backend_model_id="llava-1.6",
        client_model="llama.cpp",
        slots_supported=True,
    )

    assert bypass is True
    assert reason == "backend_model_name_heuristic"


@pytest.mark.unit
def test_consume_sse_timings_handles_fragmented_payloads() -> None:
    first = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n' b'data: {"timings":{"prompt_n":7,'
    second = b'"cache_n":3,"prompt_ms":9.5}}\n\n' b"data: [DONE]\n\n"

    buffer, last_timings = consume_sse_timings(first, None)
    buffer, last_timings = consume_sse_timings(buffer + second, last_timings)

    assert buffer == b""
    assert last_timings == {"prompt_n": 7, "cache_n": 3, "prompt_ms": 9.5}

