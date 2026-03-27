# -*- coding: utf-8 -*-

from __future__ import annotations

import runpy

import pytest

import proxycache
from proxycache.main import main


@pytest.mark.unit
def test_resolve_version_falls_back_to_pyproject(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxycache, "version", lambda name: (_ for _ in ()).throw(proxycache.PackageNotFoundError()))

    assert proxycache._resolve_version() == "1.0.1"


@pytest.mark.unit
def test_main_runs_uvicorn_with_settings(monkeypatch: pytest.MonkeyPatch, make_settings) -> None:
    settings = make_settings(name="entry-main", port=9090, log_level="DEBUG")
    run_calls: list[dict[str, object]] = []

    monkeypatch.setattr("proxycache.main.Settings.from_env", lambda: settings)
    monkeypatch.setattr("proxycache.main.create_app", lambda active_settings: {"settings": active_settings})
    monkeypatch.setattr(
        "proxycache.main.uvicorn.run",
        lambda app, host, port, log_level: run_calls.append(
            {"app": app, "host": host, "port": port, "log_level": log_level}
        ),
    )

    main()

    assert run_calls == [
        {
            "app": {"settings": settings},
            "host": "0.0.0.0",
            "port": 9090,
            "log_level": "debug",
        }
    ]


@pytest.mark.unit
def test_module_entrypoint_invokes_main(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr("proxycache.main.main", lambda: called.append("called"))

    runpy.run_module("proxycache", run_name="__main__")

    assert called == ["called"]

