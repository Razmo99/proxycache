# -*- coding: utf-8 -*-

"""Application configuration."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path


def configure_logging(level: str) -> None:
    """Configure process-wide logging once."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@dataclass(frozen=True, slots=True)
class BackendSettings:
    """Configuration for one llama.cpp backend."""

    url: str
    n_slots: int


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings loaded from the environment."""

    backends: tuple[BackendSettings, ...]
    words_per_block: int
    big_threshold_words: int
    lcp_threshold: float
    meta_dir: Path
    slot_save_path: Path
    request_timeout: float
    model_id: str
    port: int
    log_level: str
    max_saved_caches: int

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables."""
        backends = _load_backends()
        meta_dir = Path.cwd() / os.getenv("META_DIR", "kv_meta")

        env_slot_path = os.getenv("SLOT_SAVE_PATH")
        if env_slot_path:
            slot_save_path = Path(env_slot_path)
        else:
            slot_save_path = _default_slot_save_path(meta_dir)

        settings = cls(
            backends=backends,
            words_per_block=_env_int("WORDS_PER_BLOCK", 100),
            big_threshold_words=_env_int("BIG_THRESHOLD_WORDS", 500),
            lcp_threshold=_env_float("LCP_TH", 0.6),
            meta_dir=meta_dir.resolve(),
            slot_save_path=slot_save_path.resolve(),
            request_timeout=_env_float("REQUEST_TIMEOUT", 600.0),
            model_id=os.getenv("MODEL_ID", "llama.cpp"),
            port=_env_int("PORT", 8081),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_saved_caches=_env_int(
                "MAX_SAVED_CACHES",
                _default_max_saved_caches(backends),
            ),
        )
        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        """Create required directories."""
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.slot_save_path.mkdir(parents=True, exist_ok=True)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {raw}") from exc


def _load_backends() -> tuple[BackendSettings, ...]:
    backends_raw = os.getenv("BACKENDS")
    if not backends_raw:
        return (
            BackendSettings(
                url=os.getenv("LLAMA_URL", "http://127.0.0.1:8000"),
                n_slots=_env_int("N_SLOTS", 2),
            ),
        )

    try:
        payload = json.loads(backends_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("BACKENDS must be valid JSON") from exc

    if not isinstance(payload, list) or not payload:
        raise ValueError("BACKENDS must be a non-empty JSON array")

    backends: list[BackendSettings] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"BACKENDS[{index}] must be an object")
        url = str(item.get("url") or "").strip()
        if not url:
            raise ValueError(f"BACKENDS[{index}].url is required")
        try:
            n_slots = int(item.get("n_slots", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"BACKENDS[{index}].n_slots must be an integer") from exc
        if n_slots < 1:
            raise ValueError(f"BACKENDS[{index}].n_slots must be >= 1")
        backends.append(BackendSettings(url=url, n_slots=n_slots))
    return tuple(backends)


def _default_max_saved_caches(backends: tuple[BackendSettings, ...]) -> int:
    n_slots_raw = os.getenv("N_SLOTS")
    if n_slots_raw:
        try:
            return max(0, int(n_slots_raw))
        except ValueError:
            n_slots_raw = None
    if backends:
        return max(0, backends[0].n_slots)
    return 0


def _default_slot_save_path(meta_dir: Path) -> Path:
    meta_name = meta_dir.name.rstrip(os.sep)
    if meta_name == "meta":
        return meta_dir.parent.parent
    return meta_dir.parent
