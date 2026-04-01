# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pytest

from proxycache.config import Settings, _default_slot_save_path


@pytest.mark.unit
def test_settings_from_env_uses_llama_url_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("BACKENDS", raising=False)
    monkeypatch.setenv("LLAMA_URL", "http://fallback.test")
    monkeypatch.setenv("N_SLOTS", "3")
    monkeypatch.setenv("META_DIR", str(tmp_path / "meta"))

    settings = Settings.from_env()

    assert settings.backends[0].url == "http://fallback.test"
    assert settings.backends[0].n_slots == 3
    assert settings.max_saved_caches == 3


@pytest.mark.unit
def test_settings_reject_invalid_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LCP_TH", "bad-float")

    with pytest.raises(ValueError, match="Invalid float for LCP_TH"):
        Settings.from_env()


@pytest.mark.unit
def test_default_slot_save_path_uses_parent_parent_for_meta_dir_named_meta(
    tmp_path: Path,
) -> None:
    meta_dir = tmp_path / "kv" / "meta"

    assert _default_slot_save_path(meta_dir) == tmp_path


@pytest.mark.unit
def test_ensure_directories_creates_missing_directories(tmp_path: Path) -> None:
    from proxycache.config import Settings

    meta_dir = tmp_path / "slots" / "model" / "meta"
    settings = Settings(
        backends=(),
        words_per_block=100,
        big_threshold_words=500,
        lcp_threshold=0.6,
        meta_dir=meta_dir,
        slot_save_path=tmp_path / "slots",
        request_timeout=600.0,
        model_id="test",
        port=8080,
        log_level="INFO",
        max_saved_caches=4,
    )

    assert not meta_dir.exists()
    assert not (tmp_path / "slots").exists()

    settings.ensure_directories()

    assert meta_dir.exists()
    assert (tmp_path / "slots").exists()


@pytest.mark.unit
def test_ensure_directories_skips_existing_directories(tmp_path: Path) -> None:
    from proxycache.config import Settings

    meta_dir = tmp_path / "slots" / "model" / "meta"
    meta_dir.mkdir(parents=True)

    settings = Settings(
        backends=(),
        words_per_block=100,
        big_threshold_words=500,
        lcp_threshold=0.6,
        meta_dir=meta_dir,
        slot_save_path=tmp_path / "slots",
        request_timeout=600.0,
        model_id="test",
        port=8080,
        log_level="INFO",
        max_saved_caches=4,
    )

    settings.ensure_directories()

    assert meta_dir.exists()
    assert (tmp_path / "slots").exists()
