# otel.py
# -*- coding: utf-8 -*-

"""
OpenTelemetry instrumentation for proxycache.
Uses automatic instrumentation with manual span enrichment.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.trace import Status, StatusCode

import os
from version import __version__


def init_otel(app, httpx_client) -> None:
    model_id = os.getenv("MODEL_ID", "").strip() or "unknown-model"
    hostname = os.getenv("HOSTNAME", "").strip() or "unknown-host"

    attributes = {
        "service.name": "proxycache",
        "service.version": __version__,
        "service.instance.id": f"{model_id}@{hostname}",
    }
    provider = TracerProvider(resource=Resource(attributes=attributes))

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)

    if httpx_client is not None:
        HTTPXClientInstrumentor.instrument_client(httpx_client, tracer_provider=provider)

def add_cache_attributes(
    span,
    cache_hit: bool,
    cache_key: str,
    slot_id: int | None = None,
    n_used: int | None = None,
) -> None:
    if span is None or not span.is_recording():
        return
    span.set_attribute("cache.hit", cache_hit)
    span.set_attribute("cache.key", cache_key)
    if slot_id is not None:
        span.set_attribute("slot.id", slot_id)
    if n_used is not None:
        span.set_attribute("slot.n_used", n_used)


def add_llm_attributes(
    span,
    model: str,
    request_tokens: int | None = None,
    response_tokens: int | None = None,
) -> None:
    if span is None or not span.is_recording():
        return
    span.set_attribute("llm.model", model)
    if request_tokens is not None:
        span.set_attribute("llm.request.tokens", request_tokens)
    if response_tokens is not None:
        span.set_attribute("llm.response.tokens", response_tokens)


def add_timing_attributes(span, timings: dict | None) -> None:
    if span is None or not span.is_recording() or not timings:
        return

    prompt_n = timings.get("prompt_n")
    predicted_n = timings.get("predicted_n")
    cache_n = timings.get("cache_n")
    prompt_ms = timings.get("prompt_ms")
    predicted_ms = timings.get("predicted_ms")

    if prompt_n is not None:
        span.set_attribute("llm.request.tokens", int(prompt_n))
    if predicted_n is not None:
        span.set_attribute("llm.response.tokens", int(predicted_n))
    if cache_n is not None:
        span.set_attribute("llm.cache.tokens", int(cache_n))
    if prompt_ms is not None:
        span.set_attribute("llm.prompt.duration_ms", float(prompt_ms))
    if predicted_ms is not None:
        span.set_attribute("llm.completion.duration_ms", float(predicted_ms))


def set_error(span, message: str) -> None:
    if span is None or not span.is_recording():
        return
    span.set_status(Status(StatusCode.ERROR, message))
