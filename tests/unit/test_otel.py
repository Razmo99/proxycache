# -*- coding: utf-8 -*-

from __future__ import annotations

import logging

import pytest
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_OUTPUT_TYPE,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM_INSTRUCTIONS,
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv._incubating.attributes.host_attributes import HOST_NAME
from opentelemetry.semconv.attributes.server_attributes import (
    SERVER_ADDRESS,
    SERVER_PORT,
)
from opentelemetry.semconv.attributes.service_attributes import (
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from opentelemetry.trace import StatusCode

from proxycache.observability import otel


class SpanProbe:
    def __init__(self, recording: bool = True) -> None:
        self.recording = recording
        self.attributes: dict[str, object] = {}
        self.events: list[tuple[str, dict[str, object]]] = []
        self.name: str | None = None
        self.status = None

    def is_recording(self) -> bool:
        return self.recording

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def update_name(self, value: str) -> None:
        self.name = value

    def add_event(self, name: str, attributes: dict[str, object]) -> None:
        self.events.append((name, attributes))

    def set_status(self, status) -> None:
        self.status = status


@pytest.mark.unit
def test_otlp_exporter_kwargs_marks_http_endpoint_insecure() -> None:
    assert otel._otlp_exporter_kwargs("http://collector.test") == {
        "endpoint": "http://collector.test",
        "insecure": True,
    }
    assert otel._otlp_exporter_kwargs("https://collector.test") == {"endpoint": "https://collector.test"}


@pytest.mark.unit
def test_resource_attributes_use_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "proxycache-test")
    monkeypatch.setenv("OTEL_SERVICE_VERSION", "9.9.9")
    monkeypatch.setenv("OTEL_HOST_NAME", "otel-host")
    monkeypatch.setenv("OTEL_SERVICE_INSTANCE_ID", "instance-123")

    attrs = otel._resource_attributes()

    assert attrs[SERVICE_NAME] == "proxycache-test"
    assert attrs[SERVICE_VERSION] == "9.9.9"
    assert attrs[HOST_NAME] == "otel-host"
    assert attrs[SERVICE_INSTANCE_ID] == "instance-123"


@pytest.mark.unit
def test_request_attributes_extracts_openai_fields() -> None:
    attrs = otel._request_attributes(
        {
            "max_tokens": 256,
            "stop": "DONE",
            "options": {"top_k": 40},
            "response_format": {"type": "json_schema"},
        }
    )

    assert attrs[GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert attrs[GEN_AI_REQUEST_STOP_SEQUENCES] == ["DONE"]
    assert attrs["gen_ai.request.top_k"] == 40
    assert attrs[GEN_AI_OUTPUT_TYPE] == "json"


@pytest.mark.unit
def test_json_attribute_respects_max_content_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(otel, "_MAX_CONTENT_CHARS", 8)

    rendered = otel._json_attribute({"content": "very-long"})

    assert len(rendered) == 8


@pytest.mark.unit
def test_part_from_content_item_covers_supported_modalities() -> None:
    assert otel._part_from_content_item("hello") == [{"type": "text", "content": "hello"}]
    assert otel._part_from_content_item(123) == [{"type": "unknown", "content": "123"}]
    assert otel._part_from_content_item({"type": "image_url", "image_url": {"url": "http://img"}}) == [
        {"type": "uri", "modality": "image", "uri": "http://img"}
    ]
    assert otel._part_from_content_item({"type": "input_audio", "audio": {"url": "http://aud"}}) == [
        {"type": "uri", "modality": "audio", "uri": "http://aud"}
    ]
    assert otel._part_from_content_item({"type": "input_video", "video": {"url": "http://vid"}}) == [
        {"type": "uri", "modality": "video", "uri": "http://vid"}
    ]
    assert otel._part_from_content_item({"type": "file", "file_id": "file-1"}) == [
        {"type": "file", "modality": "image", "file_id": "file-1"}
    ]
    assert otel._part_from_content_item({"type": "custom", "content": "x"}) == [
        {"type": "custom", "content": "x"}
    ]


@pytest.mark.unit
def test_message_parts_includes_tool_calls_and_default_text() -> None:
    parts = otel._message_parts(
        {
            "content": [{"type": "text", "text": "hello"}],
            "tool_calls": [{"id": "call-1", "function": {"name": "lookup", "arguments": "{}"}}],
        }
    )
    empty_parts = otel._message_parts({})

    assert parts[0] == {"type": "text", "content": "hello"}
    assert parts[1]["type"] == "tool_call"
    assert empty_parts == [{"type": "text", "content": ""}]


@pytest.mark.unit
def test_add_input_response_timing_and_error_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(otel, "_CAPTURE_CONTENT", True)
    span = SpanProbe()

    otel.add_input_attributes(
        span,
        {
            "messages": [
                {"role": "system", "content": "system rules"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "image_url", "image_url": {"url": "http://img"}},
                    ],
                },
            ],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
    )
    otel.add_cache_attributes(span, True, slot_id=3, n_used=2)
    otel.add_llm_attributes(span, "model-a", response_model="model-b")
    otel.add_response_attributes(
        span,
        {
            "id": "resp-1",
            "model": "model-b",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
        },
    )
    otel.add_timing_attributes(
        span,
        {"prompt_n": 10, "predicted_n": 4, "cache_n": 6, "prompt_ms": 8.5, "predicted_ms": 2.0},
    )
    otel.set_error(span, "timeout", "request timed out")
    otel.add_lifecycle_event(span, "proxycache.test", retry=1, payload={"k": "v"}, ignored=None)

    assert GEN_AI_SYSTEM_INSTRUCTIONS in span.attributes
    assert GEN_AI_INPUT_MESSAGES in span.attributes
    assert GEN_AI_TOOL_DEFINITIONS in span.attributes
    assert span.attributes["proxycache.cache.hit"] is True
    assert span.attributes["proxycache.slot.id"] == 3
    assert span.attributes["proxycache.slot.in_use_count"] == 2
    assert span.attributes[GEN_AI_REQUEST_MODEL] == "model-a"
    assert span.attributes[GEN_AI_RESPONSE_MODEL] == "model-b"
    assert span.name == "chat model-a"
    assert span.attributes[GEN_AI_RESPONSE_ID] == "resp-1"
    assert span.attributes[GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]
    assert GEN_AI_OUTPUT_MESSAGES in span.attributes
    assert span.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 4
    assert span.attributes[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 6
    assert span.attributes["proxycache.timing.prompt_ms"] == 8.5
    assert span.attributes["proxycache.timing.completion_ms"] == 2.0
    assert span.attributes[ERROR_TYPE] == "timeout"
    assert span.status.status_code is StatusCode.ERROR
    assert span.events == [("proxycache.test", {"retry": 1, "payload": "{'k': 'v'}"})]


@pytest.mark.unit
def test_start_inference_span_uses_server_and_request_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class TracerProbe:
        def start_span(self, name: str, kind, attributes: dict[str, object]):
            captured["name"] = name
            captured["kind"] = kind
            captured["attributes"] = attributes
            return "span"

    monkeypatch.setattr(otel, "get_tracer", lambda: TracerProbe())
    monkeypatch.setattr(otel, "_DEFAULT_GEN_AI_PROVIDER", "proxycache-test")

    span = otel.start_inference_span(
        "http://backend.test:8080",
        "model-x",
        {"max_tokens": 16, "stop": ["DONE"]},
    )

    assert span == "span"
    assert captured["name"] == "chat model-x"
    attrs = captured["attributes"]
    assert attrs[GEN_AI_PROVIDER_NAME] == "proxycache-test"
    assert attrs[GEN_AI_REQUEST_MODEL] == "model-x"
    assert attrs[SERVER_ADDRESS] == "backend.test"
    assert attrs[SERVER_PORT] == 8080


@pytest.mark.unit
def test_init_and_shutdown_otel_manage_instrumentation(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTracerProvider:
        def __init__(self, resource=None) -> None:
            self.resource = resource
            self.processors: list[object] = []
            self.flushed = False

        def add_span_processor(self, processor: object) -> None:
            self.processors.append(processor)

        def force_flush(self) -> None:
            self.flushed = True

    class FakeLoggerProvider:
        def __init__(self, resource=None) -> None:
            self.resource = resource
            self.processors: list[object] = []
            self.flushed = False

        def add_log_record_processor(self, processor: object) -> None:
            self.processors.append(processor)

        def force_flush(self) -> None:
            self.flushed = True

    class FakeLoggingHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET, logger_provider=None) -> None:
            super().__init__(level)
            self.logger_provider = logger_provider

    trace_state = {"provider": object()}
    logger_state = {"provider": object()}
    fastapi_calls: list[tuple[str, object]] = []
    httpx_calls: list[tuple[str, object]] = []

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.test")
    monkeypatch.setattr(otel, "TracerProvider", FakeTracerProvider)
    monkeypatch.setattr(otel, "LoggerProvider", FakeLoggerProvider)
    monkeypatch.setattr(otel, "LoggingHandler", FakeLoggingHandler)
    monkeypatch.setattr(otel, "BatchSpanProcessor", lambda exporter: ("span-processor", exporter))
    monkeypatch.setattr(otel, "BatchLogRecordProcessor", lambda exporter: ("log-processor", exporter))
    monkeypatch.setattr(otel, "OTLPSpanExporter", lambda **kwargs: ("span-exporter", kwargs))
    monkeypatch.setattr(otel, "OTLPLogExporter", lambda **kwargs: ("log-exporter", kwargs))
    monkeypatch.setattr(otel.trace, "get_tracer_provider", lambda: trace_state["provider"])
    monkeypatch.setattr(otel.trace, "set_tracer_provider", lambda provider: trace_state.__setitem__("provider", provider))
    monkeypatch.setattr(otel, "get_logger_provider", lambda: logger_state["provider"])
    monkeypatch.setattr(otel, "set_logger_provider", lambda provider: logger_state.__setitem__("provider", provider))
    monkeypatch.setattr(
        otel.FastAPIInstrumentor,
        "instrument_app",
        lambda app, tracer_provider=None: fastapi_calls.append(("instrument", app)),
    )
    monkeypatch.setattr(
        otel.FastAPIInstrumentor,
        "uninstrument_app",
        lambda app: fastapi_calls.append(("uninstrument", app)),
    )
    monkeypatch.setattr(
        otel.HTTPXClientInstrumentor,
        "instrument_client",
        lambda client, tracer_provider=None: httpx_calls.append(("instrument", client)),
    )
    monkeypatch.setattr(
        otel.HTTPXClientInstrumentor,
        "uninstrument_client",
        lambda client: httpx_calls.append(("uninstrument", client)),
    )
    monkeypatch.setattr(otel, "_TRACER_PROVIDER", None)
    monkeypatch.setattr(otel, "_LOGGER_PROVIDER", None)
    monkeypatch.setattr(otel, "_LOGGING_HANDLER", None)
    monkeypatch.setattr(otel, "_INSTRUMENTED_APP", None)
    monkeypatch.setattr(otel, "_INSTRUMENTED_HTTPX_CLIENT", None)

    app = object()
    httpx_client = object()
    otel.init_otel(app, httpx_client)

    assert isinstance(otel._TRACER_PROVIDER, FakeTracerProvider)
    assert isinstance(otel._LOGGER_PROVIDER, FakeLoggerProvider)
    assert isinstance(otel._LOGGING_HANDLER, FakeLoggingHandler)
    assert fastapi_calls == [("instrument", app)]
    assert httpx_calls == [("instrument", httpx_client)]

    otel.shutdown_otel()

    assert fastapi_calls[-1] == ("uninstrument", app)
    assert httpx_calls[-1] == ("uninstrument", httpx_client)
    assert otel._INSTRUMENTED_APP is None
    assert otel._INSTRUMENTED_HTTPX_CLIENT is None
