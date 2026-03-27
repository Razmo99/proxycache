# Coding Standards

## Architecture

- Use a `src` layout with one real import package: `src/proxycache/`.
- Keep HTTP wiring in `api/`, backend adapters in `clients/`, cache persistence in `cache/`, and orchestration/business logic in `services/`.
- Build the FastAPI app through an app factory. Avoid putting runtime state or environment parsing directly in route modules.

## Python Standards

- Target Python `3.11+`.
- Use absolute imports only.
- Require type hints on public functions and methods.
- Prefer dataclasses for configuration and small state carriers.
- Keep files under roughly 500 lines; split by responsibility when a module starts mixing transport, persistence, and policy logic.
- Do not use `assert` for runtime validation or control flow.
- Avoid import-time side effects beyond constant definitions and lightweight app construction.
- Use structured logging with stable keys instead of ad hoc prose logs.

## FastAPI Standards

- Route handlers should stay thin and delegate policy/IO orchestration to services.
- Use lifespan hooks instead of deprecated startup/shutdown decorators.
- Keep OpenAI-compatible request/response shaping at the HTTP boundary.

## Testing Standards

- Use `pytest` as the default test runner.
- Prefer focused unit tests around policy, config, and cache logic.
- Add async tests with `pytest-asyncio` only where needed.
- Test imports through the installed/package layout, not by mutating `sys.path` inside test modules.

## Smell Checklist

- Mixed concerns in one file.
- Hidden environment coupling.
- Runtime `assert` statements.
- Overly broad exception handling without context.
- Tests depending on repository-relative import hacks.
