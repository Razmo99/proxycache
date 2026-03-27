# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_passthrough_route_forwards_get_request(async_test_client, mock_backend) -> None:
    response = await async_test_client.get("/healthz", params={"full": "1"}, headers={"x-test-header": "yes"})

    assert response.status_code == 200
    assert response.text == "upstream:GET:/healthz"
    upstream_request = next(request for request in mock_backend.requests if request.url.path == "/healthz")
    assert upstream_request.url.params["full"] == "1"
    assert upstream_request.headers["x-test-header"] == "yes"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_passthrough_root_get_request_uses_root_route(async_test_client, mock_backend) -> None:
    response = await async_test_client.get("/")

    assert response.status_code == 200
    assert response.content == b"upstream:GET:/"
    upstream_request = next(request for request in mock_backend.requests if request.url.path == "/")
    assert upstream_request.method == "GET"
