# otel.py
# -*- coding: utf-8 -*-

"""
OpenTelemetry instrumentation for proxycache.
Uses automatic HTTP instrumentation plus manual GenAI client spans.
"""

from urllib.parse import urlparse
import json

from opentelemetry import trace
from opentelemetry._logs import get_logger_provider, set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.semconv.attributes.server_attributes import (
    SERVER_ADDRESS,
    SERVER_PORT,
)
from opentelemetry.semconv.attributes.service_attributes import (
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from opentelemetry.semconv._incubating.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_OUTPUT_TYPE,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_CHOICE_COUNT,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_SEED,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_K,
    GEN_AI_REQUEST_TOP_P,
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
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.trace import SpanKind, Status, StatusCode

import os
import logging
import socket
from version import __version__


_TRACER_NAME = "proxycache.genai"
_DEFAULT_GEN_AI_PROVIDER = os.getenv("OTEL_GEN_AI_PROVIDER", "").strip() or "openai"
_CAPTURE_CONTENT = os.getenv("OTEL_GEN_AI_CAPTURE_CONTENT", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_MAX_CONTENT_CHARS = max(0, int(os.getenv("OTEL_GEN_AI_MAX_CONTENT_CHARS", "16384")))
_LOGGER_PROVIDER: LoggerProvider | None = None
_LOGGING_HANDLER: LoggingHandler | None = None
_TRACER_PROVIDER: TracerProvider | None = None
_INSTRUMENTED_APP = None
_INSTRUMENTED_HTTPX_CLIENT = None


def _otlp_exporter_kwargs(endpoint: str) -> dict[str, object]:
    kwargs: dict[str, object] = {"endpoint": endpoint}
    if endpoint.startswith("http://"):
        kwargs["insecure"] = True
    return kwargs


def _configure_otel_logging(resource: Resource, endpoint: str) -> None:
    global _LOGGER_PROVIDER
    global _LOGGING_HANDLER

    logger_provider = get_logger_provider()
    if isinstance(logger_provider, LoggerProvider):
        _LOGGER_PROVIDER = logger_provider
    elif _LOGGER_PROVIDER is None:
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(**_otlp_exporter_kwargs(endpoint)))
        )
        set_logger_provider(logger_provider)
        current_provider = get_logger_provider()
        if isinstance(current_provider, LoggerProvider):
            _LOGGER_PROVIDER = current_provider

    if _LOGGER_PROVIDER is None:
        return

    if _LOGGING_HANDLER is None:
        _LOGGING_HANDLER = LoggingHandler(
            level=logging.NOTSET,
            logger_provider=_LOGGER_PROVIDER,
        )

    root_logger = logging.getLogger()
    if _LOGGING_HANDLER not in root_logger.handlers:
        root_logger.addHandler(_LOGGING_HANDLER)


def _resource_attributes() -> dict[str, str]:
    service_name = os.getenv("OTEL_SERVICE_NAME", "").strip() or "proxycache"
    service_version = os.getenv("OTEL_SERVICE_VERSION", "").strip() or __version__
    host_name = (
        os.getenv("OTEL_HOST_NAME", "").strip()
        or os.getenv("HOSTNAME", "").strip()
        or socket.gethostname().strip()
        or "unknown-host"
    )
    instance_id = (
        os.getenv("OTEL_SERVICE_INSTANCE_ID", "").strip()
        or f"{service_name}@{host_name}"
    )

    return {
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        SERVICE_INSTANCE_ID: instance_id,
        HOST_NAME: host_name,
    }


def init_otel(app, httpx_client) -> None:
    global _TRACER_PROVIDER
    global _INSTRUMENTED_APP
    global _INSTRUMENTED_HTTPX_CLIENT

    resource = Resource.create(_resource_attributes())

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    otlp_logs_endpoint = (
        os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "").strip() or otlp_endpoint
    )

    tracer_provider = trace.get_tracer_provider()
    if isinstance(tracer_provider, TracerProvider):
        _TRACER_PROVIDER = tracer_provider
    elif _TRACER_PROVIDER is None:
        provider = TracerProvider(resource=resource)
        if otlp_endpoint:
            exporter = OTLPSpanExporter(**_otlp_exporter_kwargs(otlp_endpoint))
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        current_provider = trace.get_tracer_provider()
        if isinstance(current_provider, TracerProvider):
            _TRACER_PROVIDER = current_provider

    if otlp_logs_endpoint:
        _configure_otel_logging(resource, otlp_logs_endpoint)

    if _INSTRUMENTED_APP is not app:
        FastAPIInstrumentor.instrument_app(app, tracer_provider=_TRACER_PROVIDER)
        _INSTRUMENTED_APP = app

    if httpx_client is not None and _INSTRUMENTED_HTTPX_CLIENT is not httpx_client:
        HTTPXClientInstrumentor.instrument_client(
            httpx_client,
            tracer_provider=_TRACER_PROVIDER,
        )
        _INSTRUMENTED_HTTPX_CLIENT = httpx_client


def shutdown_otel() -> None:
    global _INSTRUMENTED_APP
    global _INSTRUMENTED_HTTPX_CLIENT
    global _LOGGER_PROVIDER
    global _LOGGING_HANDLER

    if _INSTRUMENTED_HTTPX_CLIENT is not None:
        HTTPXClientInstrumentor.uninstrument_client(_INSTRUMENTED_HTTPX_CLIENT)
        _INSTRUMENTED_HTTPX_CLIENT = None

    if _INSTRUMENTED_APP is not None:
        FastAPIInstrumentor.uninstrument_app(_INSTRUMENTED_APP)
        _INSTRUMENTED_APP = None

    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, "force_flush"):
        tracer_provider.force_flush()

    if _LOGGER_PROVIDER is not None:
        if _LOGGING_HANDLER is not None:
            logging.getLogger().removeHandler(_LOGGING_HANDLER)
        if hasattr(_LOGGER_PROVIDER, "force_flush"):
            _LOGGER_PROVIDER.force_flush()


def get_tracer():
    return trace.get_tracer(_TRACER_NAME, __version__)


def _server_attributes(base_url: str) -> dict[str, object]:
    parsed = urlparse(base_url)
    attrs: dict[str, object] = {}

    if parsed.hostname:
        attrs[SERVER_ADDRESS] = parsed.hostname
    if parsed.port is not None:
        attrs[SERVER_PORT] = parsed.port

    return attrs


def _request_attributes(payload: dict | None) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}

    attrs: dict[str, object] = {}

    for key, attr_name in (
        ("max_tokens", GEN_AI_REQUEST_MAX_TOKENS),
        ("n", GEN_AI_REQUEST_CHOICE_COUNT),
        ("temperature", GEN_AI_REQUEST_TEMPERATURE),
        ("top_p", GEN_AI_REQUEST_TOP_P),
        ("frequency_penalty", GEN_AI_REQUEST_FREQUENCY_PENALTY),
        ("presence_penalty", GEN_AI_REQUEST_PRESENCE_PENALTY),
        ("seed", GEN_AI_REQUEST_SEED),
    ):
        value = payload.get(key)
        if value is not None:
            attrs[attr_name] = value

    stop_sequences = payload.get("stop")
    if isinstance(stop_sequences, list) and stop_sequences:
        attrs[GEN_AI_REQUEST_STOP_SEQUENCES] = stop_sequences
    elif isinstance(stop_sequences, str) and stop_sequences:
        attrs[GEN_AI_REQUEST_STOP_SEQUENCES] = [stop_sequences]

    options = payload.get("options")
    if isinstance(options, dict):
        top_k = options.get("top_k")
        if top_k is not None:
            attrs[GEN_AI_REQUEST_TOP_K] = top_k

    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        fmt_type = str(response_format.get("type") or "").lower()
        if fmt_type in {"json_object", "json_schema"}:
            attrs[GEN_AI_OUTPUT_TYPE] = "json"
        elif fmt_type in {"text", "json", "image", "speech"}:
            attrs[GEN_AI_OUTPUT_TYPE] = fmt_type

    return attrs


def _json_attribute(value: object) -> str:
    rendered = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    if _MAX_CONTENT_CHARS and len(rendered) > _MAX_CONTENT_CHARS:
        return rendered[:_MAX_CONTENT_CHARS]
    return rendered


def _text_part(content: str) -> dict[str, object]:
    return {"type": "text", "content": content}


def _generic_part(part_type: str, **extra: object) -> dict[str, object]:
    payload = {"type": part_type}
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def _part_from_content_item(item: object) -> list[dict[str, object]]:
    if isinstance(item, str):
        return [_text_part(item)]
    if not isinstance(item, dict):
        return [_generic_part("unknown", content=str(item))]

    item_type = str(item.get("type") or "unknown")
    if item_type in {"text", "input_text"}:
        return [_text_part(str(item.get("text") or item.get("content") or ""))]
    if item_type in {"image_url", "input_image"}:
        image_value = item.get("image_url") or item.get("image") or {}
        url = image_value.get("url") if isinstance(image_value, dict) else image_value
        if isinstance(url, str) and url:
            return [{"type": "uri", "modality": "image", "uri": url}]
        return [_generic_part(item_type)]
    if item_type in {"audio_url", "input_audio"}:
        audio_value = item.get("audio_url") or item.get("audio") or {}
        url = audio_value.get("url") if isinstance(audio_value, dict) else audio_value
        if isinstance(url, str) and url:
            return [{"type": "uri", "modality": "audio", "uri": url}]
        return [_generic_part(item_type)]
    if item_type in {"video_url", "input_video"}:
        video_value = item.get("video_url") or item.get("video") or {}
        url = video_value.get("url") if isinstance(video_value, dict) else video_value
        if isinstance(url, str) and url:
            return [{"type": "uri", "modality": "video", "uri": url}]
        return [_generic_part(item_type)]
    if item_type in {"file", "input_file"}:
        file_id = item.get("file_id") or item.get("id")
        if isinstance(file_id, str) and file_id:
            return [{"type": "file", "modality": "image", "file_id": file_id}]
        return [_generic_part(item_type)]
    return [_generic_part(item_type, content=item.get("content"))]


def _message_parts(message: dict) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    content = message.get("content")

    if isinstance(content, list):
        for item in content:
            parts.extend(_part_from_content_item(item))
    elif isinstance(content, str) and content:
        parts.append(_text_part(content))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            parts.append(
                {
                    "type": "tool_call",
                    "id": tool_call.get("id"),
                    "name": function.get("name") or tool_call.get("name") or "tool",
                    "arguments": function.get("arguments"),
                }
            )

    if not parts:
        parts.append(_text_part(""))

    return parts


def add_input_attributes(span, payload: dict | None) -> None:
    if not _CAPTURE_CONTENT or span is None or not span.is_recording():
        return
    if not isinstance(payload, dict):
        return

    messages = payload.get("messages")
    if isinstance(messages, list):
        input_messages: list[dict[str, object]] = []
        system_instructions: list[dict[str, object]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user")
            parts = _message_parts(message)
            if role == "system":
                system_instructions.extend(parts)
            else:
                input_messages.append({"role": role, "parts": parts})

        if system_instructions:
            span.set_attribute(
                GEN_AI_SYSTEM_INSTRUCTIONS,
                _json_attribute(system_instructions),
            )
        if input_messages:
            span.set_attribute(GEN_AI_INPUT_MESSAGES, _json_attribute(input_messages))

    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        span.set_attribute(GEN_AI_TOOL_DEFINITIONS, _json_attribute(tools))


def start_inference_span(base_url: str, model: str, payload: dict | None = None):
    attrs: dict[str, object] = {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_PROVIDER_NAME: _DEFAULT_GEN_AI_PROVIDER,
        GEN_AI_REQUEST_MODEL: model,
    }
    attrs.update(_server_attributes(base_url))
    attrs.update(_request_attributes(payload))
    span_name = f"chat {model}".strip()
    return get_tracer().start_span(
        span_name,
        kind=SpanKind.CLIENT,
        attributes=attrs,
    )


def add_cache_attributes(
    span,
    cache_hit: bool,
    slot_id: int | None = None,
    n_used: int | None = None,
) -> None:
    if span is None or not span.is_recording():
        return
    span.set_attribute("proxycache.cache.hit", cache_hit)
    if slot_id is not None:
        span.set_attribute("proxycache.slot.id", slot_id)
    if n_used is not None:
        span.set_attribute("proxycache.slot.in_use_count", n_used)


def add_llm_attributes(
    span,
    model: str,
    response_model: str | None = None,
) -> None:
    if span is None or not span.is_recording():
        return
    span.set_attribute(GEN_AI_REQUEST_MODEL, model)
    span.update_name(f"chat {model}".strip())
    if response_model:
        span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)


def add_response_attributes(span, payload: dict | None) -> None:
    if span is None or not span.is_recording() or not isinstance(payload, dict):
        return

    response_id = payload.get("id")
    if response_id:
        span.set_attribute(GEN_AI_RESPONSE_ID, str(response_id))

    response_model = payload.get("model")
    if response_model:
        span.set_attribute(GEN_AI_RESPONSE_MODEL, str(response_model))

    choices = payload.get("choices")
    if isinstance(choices, list):
        finish_reasons = [
            str(choice.get("finish_reason"))
            for choice in choices
            if isinstance(choice, dict) and choice.get("finish_reason")
        ]
        if finish_reasons:
            span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)
        if _CAPTURE_CONTENT:
            output_messages: list[dict[str, object]] = []
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message") or {}
                if not isinstance(message, dict):
                    continue
                output_messages.append(
                    {
                        "role": str(message.get("role") or "assistant"),
                        "parts": _message_parts(message),
                        "finish_reason": str(choice.get("finish_reason") or "stop"),
                    }
                )
            if output_messages:
                span.set_attribute(
                    GEN_AI_OUTPUT_MESSAGES,
                    _json_attribute(output_messages),
                )


def add_timing_attributes(span, timings: dict | None) -> None:
    if span is None or not span.is_recording() or not timings:
        return

    prompt_n = timings.get("prompt_n")
    predicted_n = timings.get("predicted_n")
    cache_n = timings.get("cache_n")
    prompt_ms = timings.get("prompt_ms")
    predicted_ms = timings.get("predicted_ms")

    if prompt_n is not None:
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, int(prompt_n))
    if predicted_n is not None:
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, int(predicted_n))
    if cache_n is not None:
        span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, int(cache_n))
    if prompt_ms is not None:
        span.set_attribute("proxycache.timing.prompt_ms", float(prompt_ms))
    if predicted_ms is not None:
        span.set_attribute("proxycache.timing.completion_ms", float(predicted_ms))


def set_error(span, error_type: str, description: str | None = None) -> None:
    if span is None or not span.is_recording():
        return
    span.set_attribute(ERROR_TYPE, error_type)
    span.set_status(Status(StatusCode.ERROR, description or error_type))


def add_lifecycle_event(span, name: str, **attributes: object) -> None:
    if span is None or not span.is_recording():
        return

    event_attributes: dict[str, object] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            event_attributes[key] = value
        else:
            event_attributes[key] = str(value)

    span.add_event(name, event_attributes)
