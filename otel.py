# otel.py
# -*- coding: utf-8 -*-

"""
OpenTelemetry instrumentation for proxycache.
Uses automatic instrumentation with manual span enrichment.
"""

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.trace import SpanKind, Status, StatusCode

from config import OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME


def init_otel(app, httpx_client) -> None:
    resource = Resource(attributes={SERVICE_NAME: OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = OTEL_EXPORTER_OTLP_ENDPOINT.strip()
    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(
        app,
        server_request_hook=_server_request_hook,
        client_request_hook=_client_request_hook,
        client_response_hook=_client_response_hook,
    )

    HTTPXClientInstrumentor().instrument()


def _server_request_hook(span, request):
    if span is None:
        return
    span.kind = SpanKind.SERVER


def _client_request_hook(span, request):
    if span is None:
        return
    span.kind = SpanKind.CLIENT
    span.set_attribute("http.request.method", request.method)


def _client_response_hook(span, response):
    if span is None:
        return
    if response.status_code:
        span.set_attribute("http.response.status_code", response.status_code)


def add_cache_attributes(
    span, cache_hit: bool, cache_key: str, slot_id: int = None, n_used: int = None
) -> None:
    span.set_attribute("cache.hit", cache_hit)
    span.set_attribute("cache.key", cache_key)
    if slot_id is not None:
        span.set_attribute("slot.id", slot_id)
    if n_used is not None:
        span.set_attribute("slot.n_used", n_used)


def add_llm_attributes(
    span, model: str, request_tokens: int = None, response_tokens: int = None
) -> None:
    span.set_attribute("llm.model", model)
    if request_tokens is not None:
        span.set_attribute("llm.request.tokens", request_tokens)
    if response_tokens is not None:
        span.set_attribute("llm.response.tokens", response_tokens)


def set_error(span, message: str) -> None:
    span.set_status(Status(StatusCode.ERROR, message))
