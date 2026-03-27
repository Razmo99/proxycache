# -*- coding: utf-8 -*-

from __future__ import annotations

import json

import pytest

from proxycache.api.app import create_app
from proxycache.clients.llama import LlamaClient


def _patched_llama_client(mock_backend, url: str, request_timeout: float) -> LlamaClient:
    import httpx

    client = LlamaClient(url, request_timeout=request_timeout)
    client.client = httpx.AsyncClient(base_url=client.base_url, transport=httpx.MockTransport(mock_backend.handler))
    return client


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_app_lifespan_cleans_inactive_restore_poisons(
    monkeypatch: pytest.MonkeyPatch,
    make_settings,
    mock_backend,
) -> None:
    settings = make_settings(name="lifespan")
    key = "inactive-key"
    poison_path = settings.meta_dir / f"{key}.poison.json"
    meta_path = settings.meta_dir / f"{key}.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "key": key,
                "model_id": "llama.cpp",
                "prefix_len": 4,
                "wpb": settings.words_per_block,
                "blocks": ["a"],
                "timestamp": 1.0,
            }
        ),
        encoding="utf-8",
    )
    poison_path.write_text(json.dumps({"prompt_n": 4, "cache_n": 1}), encoding="utf-8")
    monkeypatch.setattr("proxycache.api.app.init_otel", lambda app, httpx_client: None)
    monkeypatch.setattr("proxycache.api.app.shutdown_otel", lambda: None)
    monkeypatch.setattr(
        "proxycache.api.app.LlamaClient",
        lambda url, request_timeout: _patched_llama_client(mock_backend, url, request_timeout),
    )

    app = create_app(settings)

    async with app.router.lifespan_context(app):
        assert not poison_path.exists()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_models_route_returns_proxy_model_id(async_test_client) -> None:
    response = await async_test_client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {"data": [{"id": "llama.cpp"}]}
