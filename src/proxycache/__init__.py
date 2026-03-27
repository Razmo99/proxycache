# -*- coding: utf-8 -*-

"""Proxycache package."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _resolve_version() -> str:
    try:
        return version("proxycache")
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject_path.open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])


__version__ = _resolve_version()

__all__ = ["__version__"]
