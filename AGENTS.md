# AGENTS.md

## Project Overview

This is a Python-based proxy service for llama.cpp that manages KV cache slots, enables context reuse, and restores cached contexts from disk to accelerate long-context chat and IDE workflows. The service implements an OpenAI-compatible Chat Completions API.

### Multimodal Support

The proxy detects multimodal models (like LLaVA, Qwen-VL, Gemini) from the backend `/v1/models` response by checking model capabilities. When a backend reports `multimodal`, the proxy switches that backend to passthrough mode and skips slot save/restore for subsequent requests.

## Build & Run Commands

```bash
# Install dependencies
pip install -e .[dev]

# Start the proxy server
python -m proxycache
# or
uvicorn proxycache.api.app:create_app --factory --host 0.0.0.0 --port 8081
```

**Environment variables required:**
- `BACKENDS` (JSON): Backend configuration `[{"url": "...", "n_slots": N}]`
- `LLAMA_URL`: Fallback llama.cpp server URL (default: http://127.0.0.1:8000)
- `N_SLOTS`: Number of slots per backend (default: 2)
- `PORT`: Proxy port (default: 8081)
- `MODEL_ID`: Model identifier (default: "llama.cpp")
- `META_DIR`: Cache metadata directory (default: "kv_meta")

## Code Style Guidelines

### General
- Use UTF-8 encoding (`# -*- coding: utf-8 -*-` header in all files)
- Follow PEP 8 conventions
- Use type hints for all function signatures
- Prefer explicit over implicit

### Imports
- Group imports: standard library → third-party → local
- Use absolute imports
- Local imports: `from config import ...`, `import hashing as hs`

### Naming Conventions
- Classes: PascalCase (`SlotManager`, `LlamaClient`)
- Functions/variables: snake_case (`acquire_for_request`, `backend_model_id`)
- Constants: UPPER_SNAKE_CASE (`BACKENDS`, `BIG_THRESHOLD_WORDS`)
- Module-level variables: prefix with underscore for internal use (`_all_slots`, `_locks`)

### Error Handling
- Use try/except with specific exception types
- Log exceptions with context using `log.exception()`
- Return JSON error responses for HTTP endpoints with appropriate status codes
- Always release locks in finally blocks or via context managers

### Async/Await
- All I/O operations use async/await
- Use `asyncio.Lock` for slot-level concurrency control
- Implement timeouts for critical operations (`asyncio.wait_for(..., timeout=300.0)`)
- Background tasks for streaming responses with proper cleanup

### Logging
- Use `logging.getLogger(__name__)` per module
- Include context in log messages (slot ID, key prefix, ratios)
- Use structured logging format: `log.info("msg g=%s key=%s saved=%s", g, key[:16], ok)`

### File Organization
- `src/proxycache/api/`: FastAPI app factory and HTTP routes
- `src/proxycache/services/`: orchestration, streaming, slot management, request policy
- `src/proxycache/clients/`: HTTP clients for llama.cpp backends
- `src/proxycache/cache/`: cache metadata, hashing, retention logic
- `src/proxycache/config.py`: centralized configuration from environment variables
- `src/proxycache/main.py`: CLI entry point

### Type Hints
- Use `Optional[T]` for nullable types
- Use `List[T]`, `Dict[K, V]`, `Tuple[...]` for collections
- Define type aliases: `GSlot = Tuple[int, int]`
- Annotate async generators: `AsyncGenerator[bytes, None]`

### Streaming Responses
- Use `asyncio.Queue` for producer-consumer pattern
- For multimodal backends: reader task just passes through without save/restore
- For regular backends: reader task handles cleanup (save, write meta, release)
- Sentinel value (`None`) signals end of stream
- Always close responses in finally blocks

## Testing

No formal test suite exists. Manual testing via curl or OpenAI-compatible clients:

```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama.cpp", "messages": [{"role": "user", "content": "test"}], "stream": false}'
```

## Commit and PR Conventions

- Use Conventional Commits such as `fix:`, `feat:`, `docs:`, `refactor:`.
- Local commit messages are validated with `pre-commit` using `Commitizen`.
- Pull request titles are validated in GitHub Actions and should follow the same format.

## Deployment Notes

- Ensure `--slot-save-path` directory exists and is writable
- Match `N_SLOTS` to llama.cpp `-np` parameter
- Meta files stored in `META_DIR/*.meta.json`
- KV cache files managed by llama.cpp under its `--slot-save-path`
