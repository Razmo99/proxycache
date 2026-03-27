from pathlib import Path

import pytest

from proxycache.config import Settings


def test_settings_from_env_parses_backends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "BACKENDS",
        '[{"url": "http://one.test", "n_slots": 2}, {"url": "http://two.test", "n_slots": 1}]',
    )
    monkeypatch.setenv("META_DIR", str(tmp_path / "meta"))
    monkeypatch.setenv("SLOT_SAVE_PATH", str(tmp_path / "slots"))

    settings = Settings.from_env()

    assert len(settings.backends) == 2
    assert settings.backends[0].url == "http://one.test"
    assert settings.backends[1].n_slots == 1
    assert settings.meta_dir == (tmp_path / "meta").resolve()
    assert settings.slot_save_path == (tmp_path / "slots").resolve()


def test_settings_reject_invalid_backend_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKENDS", '{"url": "broken"}')

    with pytest.raises(ValueError, match="BACKENDS must be a non-empty JSON array"):
        Settings.from_env()
