# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_LIVE_SMOKE_PATH = Path(__file__).with_name("test_live_proxy_cache.py")
_LIVE_SMOKE_SPEC = importlib.util.spec_from_file_location("proxycache_live_smoke", _LIVE_SMOKE_PATH)
if _LIVE_SMOKE_SPEC is None or _LIVE_SMOKE_SPEC.loader is None:
    raise RuntimeError(f"unable to load live smoke helpers from {_LIVE_SMOKE_PATH}")
_LIVE_SMOKE = importlib.util.module_from_spec(_LIVE_SMOKE_SPEC)
sys.modules[_LIVE_SMOKE_SPEC.name] = _LIVE_SMOKE
_LIVE_SMOKE_SPEC.loader.exec_module(_LIVE_SMOKE)

pytestmark = pytest.mark.real_smoke

_OPENCODE_LONG_CTX_SIZE = 65536
_OPENCODE_LONG_SYSTEM_PROMPT_MIN_CHARS = 65536
_OPENCODE_BASIC_MAX_PROMPT_MS_RATIO = 0.85
_OPENCODE_LONG_MAX_PROMPT_MS_RATIO = 0.9
_OPENCODE_MIN_CACHE_READ = 1


@pytest.fixture
def real_smoke_stack_factory():
    if shutil.which("docker") is None:
        pytest.skip("docker is required for real smoke tests")

    from contextlib import contextmanager
    from uuid import uuid4

    @contextmanager
    def _factory(*, model_preset_name: str, backend_ctx_size: int | None = None):
        suffix = uuid4().hex[:8]
        stack = _LIVE_SMOKE.RealSmokeStack(
            repo_root=Path(__file__).resolve().parents[2],
            model_preset_name=model_preset_name,
            proxy_port=_LIVE_SMOKE._free_port(),
            backend_port=_LIVE_SMOKE._free_port(),
            network_name=f"proxycache-real-smoke-net-{suffix}",
            proxy_name=f"proxycache-real-smoke-proxy-{suffix}",
            backend_name=f"proxycache-real-smoke-llama-{suffix}",
            proxy_image=f"proxycache-real-smoke:{suffix}",
            backend_image=f"proxycache-real-smoke-backend:{suffix}",
            meta_volume=f"proxycache-real-smoke-meta-{suffix}",
            cache_volume=f"proxycache-real-smoke-cache-{suffix}",
            model_port=_LIVE_SMOKE._env_int("REAL_SMOKE_MODEL_PORT", 5800),
            backend_ctx_size=backend_ctx_size
            if backend_ctx_size is not None
            else _LIVE_SMOKE._env_int("REAL_SMOKE_BACKEND_CTX_SIZE", 4096),
            backend_extra_args="",
        )
        stack.start()
        try:
            yield stack
        finally:
            stack.stop()

    return _factory


def _run_opencode_sdk_smoke(
    *,
    stack: object,
    source_script_path: Path,
    opencode_port: int,
    model: str,
    scenario: str,
) -> tuple[dict[str, object], str]:
    sdk_dir = Path(tempfile.mkdtemp(prefix="proxycache-opencode-sdk-"))
    try:
        script_path = sdk_dir / "opencode_sdk_smoke.mjs"
        shutil.copy2(source_script_path, script_path)
        subprocess.run(["npm", "init", "-y"], cwd=sdk_dir, check=True, capture_output=True, text=True)
        subprocess.run(
            ["npm", "install", "@opencode-ai/sdk"],
            cwd=sdk_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        env = os.environ.copy()
        env.update(
            {
                "OPENCODE_SMOKE_PROXY_URL": stack.proxy_url,
                "OPENCODE_SMOKE_MODEL": model,
                "OPENCODE_SMOKE_SCENARIO": scenario,
                "OPENCODE_SMOKE_REPEAT": str(_LIVE_SMOKE._env_int("REAL_SMOKE_PREFIX_REPEAT", 96)),
                "OPENCODE_SMOKE_PORT": str(opencode_port),
                "OPENCODE_SMOKE_REPO_ROOT": str(Path(__file__).resolve().parents[2]),
            }
        )
        try:
            result = subprocess.run(
                ["node", str(script_path)],
                cwd=sdk_dir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=_LIVE_SMOKE._env_int("OPENCODE_SMOKE_TIMEOUT_SECONDS", 180),
            )
        except subprocess.TimeoutExpired as exc:
            pytest.fail(
                f"OpenCode SDK smoke timed out for scenario={scenario}\n\n"
                f"stdout:\n{exc.stdout or ''}\n\nstderr:\n{exc.stderr or ''}\n\n{stack.logs()}"
            )
        if result.returncode != 0:
            pytest.fail(
                f"OpenCode SDK smoke failed for scenario={scenario}\n\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n\n{stack.logs()}"
            )
        return json.loads(result.stdout.strip()), stack.logs()
    finally:
        subprocess.run(
            ["pkill", "-f", f"opencode serve --hostname=127.0.0.1 --port={opencode_port}"],
            check=False,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(sdk_dir, ignore_errors=True)


@pytest.mark.real_smoke
def test_opencode_sdk_revert_flow_reuses_proxycache(real_smoke_stack_factory) -> None:
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("node and npm are required for the OpenCode SDK smoke test")

    model_preset_name = _LIVE_SMOKE._primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    source_script_path = Path(__file__).with_name("opencode_sdk_smoke.mjs")
    opencode_port = _LIVE_SMOKE._free_port()

    with real_smoke_stack_factory(model_preset_name=model_preset_name) as stack:
        payload, proxy_logs = _run_opencode_sdk_smoke(
            stack=stack,
            source_script_path=source_script_path,
            opencode_port=opencode_port,
            model=model,
            scenario="basic",
        )

    assert payload["first"].get("messageID"), payload
    assert payload["second"].get("messageID") == payload["revertedMessageID"], payload
    assert payload["afterRevert"].get("messageID"), payload
    assert "restore_before_chat" in proxy_logs, f"payload={payload}\n\n{proxy_logs}"
    if payload["afterRevert"]["cacheRead"] is not None:
        assert payload["afterRevert"]["cacheRead"] >= _OPENCODE_MIN_CACHE_READ, payload
    if payload["first"].get("promptMs") is not None and payload["afterRevert"].get("promptMs") is not None:
        ratio = payload["afterRevert"]["promptMs"] / max(payload["first"]["promptMs"], 1)
        assert ratio <= _OPENCODE_BASIC_MAX_PROMPT_MS_RATIO, payload
    assert payload["messageCount"] < 3, payload


@pytest.mark.real_smoke
def test_opencode_sdk_long_context_revert_reuses_proxycache(real_smoke_stack_factory) -> None:
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("node and npm are required for the OpenCode SDK smoke test")

    model_preset_name = _LIVE_SMOKE._primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    source_script_path = Path(__file__).with_name("opencode_sdk_smoke.mjs")
    opencode_port = _LIVE_SMOKE._free_port()

    with real_smoke_stack_factory(
        model_preset_name=model_preset_name,
        backend_ctx_size=_OPENCODE_LONG_CTX_SIZE,
    ) as stack:
        payload, proxy_logs = _run_opencode_sdk_smoke(
            stack=stack,
            source_script_path=source_script_path,
            opencode_port=opencode_port,
            model=model,
            scenario="long_context_revert",
        )

    assert payload["scenario"] == "long_context_revert", payload
    assert payload["systemPromptChars"] >= _OPENCODE_LONG_SYSTEM_PROMPT_MIN_CHARS, payload
    assert payload["first"].get("messageID"), payload
    assert payload["second"].get("messageID") == payload["revertedMessageID"], payload
    assert payload["afterRevert"].get("messageID"), payload
    assert "restore_before_chat" in proxy_logs, f"payload={payload}\n\n{proxy_logs}"
    assert payload["afterRevert"]["cacheRead"] >= _OPENCODE_MIN_CACHE_READ, payload
    if payload["first"].get("promptMs") is not None and payload["afterRevert"].get("promptMs") is not None:
        ratio = payload["afterRevert"]["promptMs"] / max(payload["first"]["promptMs"], 1)
        assert ratio <= _OPENCODE_LONG_MAX_PROMPT_MS_RATIO, payload
    if payload["second"].get("promptMs") is not None and payload["afterRevert"].get("promptMs") is not None:
        assert payload["afterRevert"]["promptMs"] < payload["second"]["promptMs"], payload
    assert payload["messageCount"] < 3, payload


@pytest.mark.real_smoke
def test_opencode_sdk_branch_thrash_reuses_prefix_without_corruption(real_smoke_stack_factory) -> None:
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("node and npm are required for the OpenCode SDK smoke test")

    model_preset_name = _LIVE_SMOKE._primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    source_script_path = Path(__file__).with_name("opencode_sdk_smoke.mjs")
    opencode_port = _LIVE_SMOKE._free_port()

    with real_smoke_stack_factory(
        model_preset_name=model_preset_name,
        backend_ctx_size=_OPENCODE_LONG_CTX_SIZE,
    ) as stack:
        payload, proxy_logs = _run_opencode_sdk_smoke(
            stack=stack,
            source_script_path=source_script_path,
            opencode_port=opencode_port,
            model=model,
            scenario="long_context_branch_thrash",
        )

        assert payload["scenario"] == "long_context_branch_thrash", payload
        assert payload["systemPromptChars"] >= _OPENCODE_LONG_SYSTEM_PROMPT_MIN_CHARS, payload
        assert payload["first"].get("messageID"), payload
        assert payload["second"].get("messageID") == payload["revertedMessageID"], payload
        assert payload["branchOne"].get("messageID") == payload["secondBranchRevertedMessageID"], payload
        assert payload["afterRevert"]["cacheRead"] >= _OPENCODE_MIN_CACHE_READ, payload
        assert payload["afterSecondRevert"]["cacheRead"] >= _OPENCODE_MIN_CACHE_READ, payload
        assert payload["finalBranch"]["cacheRead"] >= _OPENCODE_MIN_CACHE_READ, payload
        assert proxy_logs.count("restore_before_chat") >= 2, f"payload={payload}\n\n{proxy_logs}"
        assert not stack.poison_files(), f"payload={payload}\n\n{stack.logs()}"
        if payload["second"].get("promptMs") is not None and payload["afterRevert"].get("promptMs") is not None:
            assert payload["afterRevert"]["promptMs"] < payload["second"]["promptMs"], payload
        if payload["branchOne"].get("promptMs") is not None and payload["afterSecondRevert"].get("promptMs") is not None:
            assert payload["afterSecondRevert"]["promptMs"] < payload["branchOne"]["promptMs"], payload
        if payload["first"].get("promptMs") is not None and payload["finalBranch"].get("promptMs") is not None:
            assert payload["finalBranch"]["promptMs"] < payload["first"]["promptMs"], payload
        assert payload["messageCount"] < 3, payload
