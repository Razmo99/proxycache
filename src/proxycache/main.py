# -*- coding: utf-8 -*-

"""CLI entrypoint for running the service."""

from __future__ import annotations

import uvicorn

from proxycache.api.app import create_app
from proxycache.config import Settings


def main() -> None:
    """Run the proxy server with uvicorn."""
    settings = Settings.from_env()
    uvicorn.run(
        create_app(settings),
        host="0.0.0.0",  # nosec B104
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
