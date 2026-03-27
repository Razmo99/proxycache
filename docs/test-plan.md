# Proxycache Test Plan

## Goals

- Build reliable unit coverage around the core cache, slot, policy, and client logic.
- Add integration coverage for the FastAPI app and the request orchestration path.
- Keep a thin smoke layer that validates startup plus one JSON and one streaming flow.

## Test Layers

### Unit

- `config`: environment parsing, defaults, and directory setup.
- `cache.metadata`: prefix extraction, hashing, restore candidate selection, retention, poison lifecycle.
- `services.policies`: multimodal detection, slot bypass policy, SSE timing extraction.
- `services.slots`: free-slot preference, oldest-slot fallback, restore invocation, release behavior.
- `clients.llama`: request shaping, JSON parsing fallbacks, model capability caching, slot support detection.
- `services.cache_lifecycle`: save, metadata recording, prune, poison-on-restore behavior.
- `services.streaming`: streaming cleanup, timing capture, save-or-skip decisions, slot release.

### Integration

- `api.app`: app factory wiring, lifespan startup/shutdown, poison cleanup.
- `services.chat_service`: small vs big request policy, restore selection, slot acquisition, save/skip paths, timeout/error handling, passthrough routes.
- JSON and streaming behavior against mocked upstream llama.cpp endpoints.

### Smoke

- App boots with isolated temp directories.
- `/v1/models` works.
- One non-stream chat request succeeds.
- One stream request succeeds and finishes cleanly.

## Fixtures

- `make_settings`: isolated `Settings` object rooted in `tmp_path`.
- `mock_backend`: HTTPX `MockTransport` handler plus request recording.
- `llama_client`: real `LlamaClient` with a mocked transport.
- `cache_store` and `slot_manager`: real instances for service-level tests.
- `test_app`, `test_client`, `async_test_client`: shared app/client fixtures.
- `messages_factory`: reusable small, large, and multimodal payloads.
- `span_stub`: lightweight fake span for service unit tests.

## Execution Strategy

1. Land fixture infrastructure first.
2. Expand unit coverage around deterministic logic.
3. Add integration coverage for orchestration and app boundaries.
4. Add smoke tests.
5. Run the full suite after each layer and tighten any uncovered behavior gaps found during execution.
6. Enforce a coverage floor in pytest and CI; the suite currently enforces 80% line coverage and should be raised further as weaker modules are covered.
