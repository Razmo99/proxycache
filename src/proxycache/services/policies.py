# -*- coding: utf-8 -*-

"""Request policy helpers."""

from __future__ import annotations

import json
import re
from typing import Any

MULTIMODAL_MODEL_PATTERNS = (
    "llava",
    "vision",
    "qwen-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "minicpm-v",
    "minicpmo",
    "internvl",
    "gemma-3",
)


def message_has_multimodal_content(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "")
        if part_type in {
            "image_url",
            "input_image",
            "audio_url",
            "input_audio",
            "video_url",
            "input_video",
            "file",
            "input_file",
        }:
            return True
    return False


def request_has_multimodal_workload(messages: list[dict[str, Any]]) -> bool:
    return any(message_has_multimodal_content(message) for message in messages or [])


def looks_like_multimodal_model_name(model_id: str | None) -> bool:
    if not model_id:
        return False
    lowered = model_id.lower()
    if any(pattern in lowered for pattern in MULTIMODAL_MODEL_PATTERNS):
        return True
    return re.search(r"(^|[-_/])(?:vl|vision|omni|audio)($|[-_/])", lowered) is not None


def slot_management_mode(
    backend_is_multimodal: bool,
    backend_model_id: str,
    client_model: str,
    slots_supported: bool,
) -> tuple[bool, str]:
    if not slots_supported:
        return True, "slots_unsupported"
    if backend_is_multimodal:
        return True, "models_capability"
    if looks_like_multimodal_model_name(backend_model_id):
        return True, "backend_model_name_heuristic"
    if looks_like_multimodal_model_name(client_model):
        return True, "model_name_heuristic"
    return False, "slots_enabled"


def extract_timings(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    timings = payload.get("timings")
    return timings if isinstance(timings, dict) else None


def consume_sse_timings(
    buffer: bytes,
    last_timings: dict[str, Any] | None,
) -> tuple[bytes, dict[str, Any] | None]:
    while b"\n\n" in buffer:
        raw_event, buffer = buffer.split(b"\n\n", 1)
        data_lines = [
            line[5:].lstrip() for line in raw_event.splitlines() if line.startswith(b"data:")
        ]
        if not data_lines:
            continue
        data = b"\n".join(data_lines).strip()
        if not data or data == b"[DONE]":
            continue
        payload: dict[str, Any] | None = None
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            pass
        if payload is None:
            continue
        timings = extract_timings(payload)
        if timings:
            last_timings = timings
    return buffer, last_timings
