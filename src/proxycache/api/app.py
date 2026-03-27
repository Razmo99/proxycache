# -*- coding: utf-8 -*-

"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from proxycache import __version__
from proxycache.cache.metadata import CacheStore
from proxycache.clients.llama import LlamaClient
from proxycache.config import Settings, configure_logging
from proxycache.observability.otel import init_otel, shutdown_otel
from proxycache.services.chat_service import ProxyService
from proxycache.services.slots import SlotManager

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app and its runtime dependencies."""
    runtime_settings = settings or Settings.from_env()
    configure_logging(runtime_settings.log_level)

    clients = [
        LlamaClient(backend.url, request_timeout=runtime_settings.request_timeout)
        for backend in runtime_settings.backends
    ]
    slot_manager = SlotManager(runtime_settings.backends)
    slot_manager.set_clients(clients)
    cache_store = CacheStore(
        meta_dir=runtime_settings.meta_dir,
        cache_dir=runtime_settings.slot_save_path,
        words_per_block=runtime_settings.words_per_block,
        max_saved_caches=runtime_settings.max_saved_caches,
    )
    service = ProxyService(runtime_settings, clients, slot_manager, cache_store)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        deleted_poison_files = cache_store.cleanup_restore_poisons()
        init_otel(app, clients[0].client if clients else None)
        log.info(
            "app_start version=%s n_backends=%d port=%d",
            __version__,
            len(runtime_settings.backends),
            runtime_settings.port,
        )
        log.info(
            "retention_config max_saved_caches=%d meta_dir=%s slot_save_path=%s",
            runtime_settings.max_saved_caches,
            runtime_settings.meta_dir,
            runtime_settings.slot_save_path,
        )
        if deleted_poison_files:
            log.info("startup_deleted_restore_poisons n=%d", deleted_poison_files)
        yield
        await asyncio.gather(*(client.close() for client in clients))
        shutdown_otel()

    app = FastAPI(title="Proxycache", lifespan=lifespan)
    app.state.service = service

    @app.get("/v1/models")
    async def models():
        return await service.models()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await service.chat(request)

    @app.api_route("/", methods=["GET", "HEAD"])
    async def upstream_root(request: Request):
        return await service.proxy_upstream_request(request, "/")

    @app.api_route("/{path:path}", methods=["GET", "HEAD"])
    async def upstream_passthrough(path: str, request: Request):
        return await service.proxy_upstream_request(request, path)

    return app
