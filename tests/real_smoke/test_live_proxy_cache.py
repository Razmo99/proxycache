# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

pytestmark = pytest.mark.real_smoke

if os.getenv("CI"):
    pytest.skip("real smoke tests are disabled in CI", allow_module_level=True)

if os.getenv("REAL_SMOKE") != "1":
    pytest.skip("set REAL_SMOKE=1 to run real smoke tests", allow_module_level=True)


MODEL_PRESETS = {
    "qwen_0_5b_instruct": {
        "label": "Qwen2.5 0.5B Instruct",
    },
    "qwen_coder_0_5b_instruct": {
        "label": "Qwen2.5 Coder 0.5B Instruct",
    },
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        text=True,
        capture_output=True,
    )


def _wait_for_http(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: str | None = None
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                response = client.get(url)
                if response.status_code == 200:
                    return
                last_error = f"unexpected status {response.status_code}"
            except Exception as exc:  # pragma: no cover - best-effort wait loop
                last_error = str(exc)
            time.sleep(1.0)
    raise AssertionError(f"service at {url} did not become ready: {last_error}")


def _selected_presets() -> list[str]:
    raw = os.getenv("REAL_SMOKE_MODEL_PRESETS", "qwen_0_5b_instruct")
    if raw == "all":
        return list(MODEL_PRESETS)
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in MODEL_PRESETS]
    if unknown:
        raise AssertionError(f"unknown REAL_SMOKE_MODEL_PRESETS entries: {', '.join(unknown)}")
    return selected


def _primary_preset() -> str:
    override = os.getenv("REAL_SMOKE_PRIMARY_PRESET")
    if override:
        if override not in MODEL_PRESETS:
            raise AssertionError(f"unknown REAL_SMOKE_PRIMARY_PRESET: {override}")
        return override
    return _selected_presets()[0]


def _build_large_prompt(label: str, suffix: str, repeat: int) -> str:
    vocab = {
        "atlas": [
            "orion",
            "nebula",
            "quasar",
            "pulsar",
            "cosmos",
            "galaxy",
            "meteor",
            "zenith",
        ],
        "zephyr": [
            "monsoon",
            "tempest",
            "gust",
            "squall",
            "cyclone",
            "breeze",
            "jetstream",
            "barometer",
        ],
        "meridian": [
            "lathe",
            "sprocket",
            "torque",
            "gearbox",
            "camshaft",
            "bearing",
            "flywheel",
            "ratchet",
        ],
        "delta": [
            "fjord",
            "estuary",
            "tributary",
            "lagoon",
            "shoal",
            "tideline",
            "headwater",
            "marshland",
        ],
    }
    words = vocab.get(label, [label] * 8)
    blocks = [
        f"{label} {words[0]} {words[1]} {words[2]} {words[3]} {words[4]} {words[5]} {words[6]} {words[7]} marker{index:03d}"
        for index in range(repeat)
    ]
    return "\n".join(blocks) + f"\n\nReply with exactly: {suffix}"


def _conversation_messages(
    *,
    system_tag: str,
    user_labels: list[str],
    suffix: str,
    repeat: int,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": _build_large_prompt(system_tag, f"{suffix} SYSTEM", max(8, repeat // 4)),
        }
    ]
    for index, label in enumerate(user_labels, start=1):
        messages.append(
            {
                "role": "user",
                "content": _build_large_prompt(label, f"{suffix} USER {index}", repeat),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"ACK {suffix} {label} {index}",
            }
        )
    messages.append(
        {
            "role": "user",
            "content": f"Respond with exactly: {suffix}",
        }
    )
    return messages


def _extract_message_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _timings(payload: dict[str, object]) -> dict[str, object]:
    timings = payload.get("timings")
    if not isinstance(timings, dict):
        raise AssertionError(f"missing timings: {payload}")
    return timings


def _usage(payload: dict[str, object]) -> dict[str, object]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise AssertionError(f"missing usage: {payload}")
    return usage


def _cache_n(payload: dict[str, object]) -> int:
    return int(_timings(payload).get("cache_n") or 0)


def _prompt_ms(payload: dict[str, object]) -> float:
    return float(_timings(payload).get("prompt_ms") or 0.0)


def _prompt_tokens(payload: dict[str, object]) -> int:
    return int(_usage(payload).get("prompt_tokens") or 0)


def _post_chat(
    client: httpx.Client,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, object]:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "stream": False,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": messages,
        },
    )
    response.raise_for_status()
    payload = response.json()
    assert _extract_message_text(payload)
    return payload


@dataclass
class RealSmokeStack:
    repo_root: Path
    model_preset_name: str
    proxy_port: int
    backend_port: int
    network_name: str
    proxy_name: str
    backend_name: str
    proxy_image: str
    backend_image: str
    meta_volume: str
    cache_volume: str
    model_port: int
    backend_ctx_size: int
    backend_extra_args: str

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self.proxy_port}"

    @property
    def ingress_url(self) -> str:
        return f"http://127.0.0.1:{self.backend_port}/upstream/{self.model_preset_name}"

    def start(self) -> None:
        _docker("network", "create", self.network_name)
        _docker("volume", "create", self.meta_volume)
        _docker("volume", "create", self.cache_volume)
        proxy_image_override = os.getenv("REAL_SMOKE_PROXY_IMAGE")
        if proxy_image_override:
            self.proxy_image = proxy_image_override
        else:
            _docker("build", "-t", self.proxy_image, str(self.repo_root))

        backend_image_override = os.getenv("REAL_SMOKE_BACKEND_IMAGE")
        if backend_image_override:
            self.backend_image = backend_image_override
        else:
            _docker(
                "build",
                "-t",
                self.backend_image,
                str(self.repo_root / "docker" / "real-smoke-llama-swap"),
            )

        self._start_backend_container()

        backends = json.dumps(
            [
                {
                    "url": f"http://llama-swap:{self.model_port}",
                    "n_slots": 1,
                }
            ]
        )
        _docker(
            "run",
            "-d",
            "--name",
            self.proxy_name,
            "--network",
            self.network_name,
            "-p",
            f"{self.proxy_port}:8081",
            "--network-alias",
            "proxycache",
            "-e",
            f"BACKENDS={backends}",
            "-e",
            "PORT=8081",
            "-e",
            "LOG_LEVEL=INFO",
            "-e",
            f"MODEL_ID={os.getenv('REAL_SMOKE_MODEL_ID', self.model_preset_name)}",
            "-e",
            f"WORDS_PER_BLOCK={_env_int('REAL_SMOKE_WORDS_PER_BLOCK', 8)}",
            "-e",
            f"BIG_THRESHOLD_WORDS={_env_int('REAL_SMOKE_BIG_THRESHOLD_WORDS', 64)}",
            "-e",
            f"LCP_TH={_env_float('REAL_SMOKE_LCP_THRESHOLD', 0.8)}",
            "-e",
            f"REQUEST_TIMEOUT={_env_float('REAL_SMOKE_REQUEST_TIMEOUT', 300.0)}",
            "-e",
            f"MAX_SAVED_CACHES={_env_int('REAL_SMOKE_MAX_SAVED_CACHES', 8)}",
            "-e",
            "META_DIR=/meta",
            "-e",
            "SLOT_SAVE_PATH=/cache",
            "-v",
            f"{self.meta_volume}:/meta",
            "-v",
            f"{self.cache_volume}:/cache",
            self.proxy_image,
        )
        _wait_for_http(
            f"{self.proxy_url}/v1/models",
            timeout_s=_env_float("REAL_SMOKE_PROXY_READY_TIMEOUT", 120.0),
        )
        _wait_for_http(
            f"{self.ingress_url}/v1/models",
            timeout_s=_env_float("REAL_SMOKE_PROXY_READY_TIMEOUT", 120.0),
        )
        self._wait_for_model_http(
            f"http://127.0.0.1:{self.model_port}/v1/models",
            timeout_s=_env_float("REAL_SMOKE_BACKEND_READY_TIMEOUT", 300.0),
        )

    def _start_backend_container(self) -> None:
        _docker(
            "run",
            "-d",
            "--name",
            self.backend_name,
            "--network",
            self.network_name,
            "--network-alias",
            "llama-swap",
            "-p",
            f"{self.backend_port}:8080",
            "-e",
            f"REAL_SMOKE_CTX_SIZE={self.backend_ctx_size}",
            "-e",
            f"REAL_SMOKE_PARALLEL={_env_int('REAL_SMOKE_BACKEND_PARALLEL', 1)}",
            "-e",
            f"REAL_SMOKE_EXTRA_ARGS={self.backend_extra_args}",
            "-v",
            f"{self.cache_volume}:/cache",
            self.backend_image,
        )
        _wait_for_http(
            f"http://127.0.0.1:{self.backend_port}/health",
            timeout_s=_env_float("REAL_SMOKE_BACKEND_READY_TIMEOUT", 300.0),
        )

    def stop(self) -> None:
        for name in (self.proxy_name, self.backend_name):
            _docker("rm", "-f", name, check=False)
        for volume in (self.meta_volume, self.cache_volume):
            _docker("volume", "rm", "-f", volume, check=False)
        _docker("network", "rm", self.network_name, check=False)

    def restart_backend(self, *, backend_ctx_size: int | None = None, backend_extra_args: str | None = None) -> None:
        if backend_ctx_size is not None:
            self.backend_ctx_size = backend_ctx_size
        if backend_extra_args is not None:
            self.backend_extra_args = backend_extra_args.strip()
        _docker("rm", "-f", self.backend_name, check=False)
        self._start_backend_container()
        _wait_for_http(
            f"{self.ingress_url}/v1/models",
            timeout_s=_env_float("REAL_SMOKE_PROXY_READY_TIMEOUT", 120.0),
        )
        self._wait_for_model_http(
            f"http://127.0.0.1:{self.model_port}/v1/models",
            timeout_s=_env_float("REAL_SMOKE_BACKEND_READY_TIMEOUT", 300.0),
        )

    def logs(self) -> str:
        chunks: list[str] = []
        for name in (self.backend_name, self.proxy_name):
            result = _docker("logs", name, check=False)
            chunks.append(f"== {name} ==\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        return "\n\n".join(chunks)

    def _wait_for_model_http(self, url: str, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        last_output = ""
        command = (
            "curl -sS -o /tmp/real-smoke-model-ready.json -w '%{http_code}' "
            + shlex.quote(url)
        )
        while time.time() < deadline:
            result = self._exec_in_backend(command)
            status = result.stdout.strip()
            if result.returncode == 0 and status == "200":
                return
            body = self._exec_in_backend("cat /tmp/real-smoke-model-ready.json 2>/dev/null || true")
            last_output = (
                f"code={result.returncode} status={status} "
                f"stderr={result.stderr.strip()} body={body.stdout.strip()}"
            )
            time.sleep(1.0)
        raise AssertionError(f"service at {url} did not become ready: {last_output}")

    def meta_files(self) -> list[str]:
        result = _docker(
            "exec",
            self.proxy_name,
            "sh",
            "-lc",
            "find /meta -maxdepth 1 -name '*.meta.json' -printf '%f\n' | sort",
            check=False,
        )
        if result.returncode != 0:
            return []
        return [line for line in result.stdout.splitlines() if line.strip()]

    def _exec_in_proxy(self, command: str) -> subprocess.CompletedProcess[str]:
        return _docker("exec", self.proxy_name, "sh", "-lc", command, check=False)

    def _exec_in_backend(self, command: str) -> subprocess.CompletedProcess[str]:
        return _docker("exec", self.backend_name, "sh", "-lc", command, check=False)

    def latest_meta(self) -> tuple[str, dict[str, object]]:
        files = self.meta_files()
        if not files:
            raise AssertionError(f"no meta files found\n\n{self.logs()}")
        latest = files[-1]
        result = self._exec_in_proxy(f"cat /meta/{shlex.quote(latest)}")
        if result.returncode != 0:
            raise AssertionError(f"unable to read meta file {latest}\n\n{self.logs()}")
        return latest, json.loads(result.stdout)

    def meta_payloads(self) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for name in self.meta_files():
            result = self._exec_in_proxy(f"cat /meta/{shlex.quote(name)}")
            if result.returncode == 0:
                payloads.append(json.loads(result.stdout))
        return payloads

    def cache_blob_path(self, key: str) -> str:
        return f"/cache/{key}"

    def corrupt_cache_blob(self, key: str) -> None:
        command = (
            f"test -f {shlex.quote(self.cache_blob_path(key))} && "
            f"truncate -s 1 {shlex.quote(self.cache_blob_path(key))}"
        )
        result = self._exec_in_backend(command)
        if result.returncode != 0:
            raise AssertionError(f"unable to corrupt cache blob for {key}\n\n{self.logs()}")

    def cache_blob_exists(self, key: str) -> bool:
        result = self._exec_in_backend(f"test -f {shlex.quote(self.cache_blob_path(key))}")
        return result.returncode == 0

    def write_poison_file(self, key: str, model_id: str, *, prompt_n: int, cache_n: int) -> None:
        payload = json.dumps(
            {
                "key": key,
                "model_id": model_id,
                "reason": "live_smoke_forced_poison",
                "prompt_n": prompt_n,
                "cache_n": cache_n,
                "prompt_ms": 0.0,
                "timestamp": time.time(),
            },
            separators=(",", ":"),
        )
        command = (
            f"cat <<'EOF' > /meta/{shlex.quote(f'{key}.poison.json')}\n"
            f"{payload}\n"
            "EOF"
        )
        result = self._exec_in_proxy(command)
        if result.returncode != 0:
            raise AssertionError(f"unable to write poison file for {key}\n\n{self.logs()}")

    def poison_files(self) -> list[str]:
        result = self._exec_in_proxy(
            "find /meta -maxdepth 1 -name '*.poison.json' -printf '%f\n' | sort"
        )
        if result.returncode != 0:
            return []
        return [line for line in result.stdout.splitlines() if line.strip()]

    def backend_post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        quoted_payload = shlex.quote(json.dumps(payload, separators=(",", ":")))
        command = (
            "curl -sS -X POST "
            f"{shlex.quote(f'http://127.0.0.1:{self.model_port}{path}')}"
            " -H 'Content-Type: application/json' "
            f" -d {quoted_payload}"
        )
        result = self._exec_in_backend(command)
        if result.returncode != 0:
            raise AssertionError(f"backend POST failed for {path}\n\n{self.logs()}")
        return json.loads(result.stdout)


@pytest.fixture
def real_smoke_stack_factory():
    if shutil.which("docker") is None:
        pytest.skip("docker is required for real smoke tests")

    @contextmanager
    def _factory(
        *,
        model_preset_name: str,
        backend_ctx_size: int | None = None,
        backend_extra_args: str = "",
    ):
        suffix = uuid4().hex[:8]
        stack = RealSmokeStack(
            repo_root=Path(__file__).resolve().parents[2],
            model_preset_name=model_preset_name,
            proxy_port=_free_port(),
            backend_port=_free_port(),
            network_name=f"proxycache-real-smoke-net-{suffix}",
            proxy_name=f"proxycache-real-smoke-proxy-{suffix}",
            backend_name=f"proxycache-real-smoke-llama-{suffix}",
            proxy_image=f"proxycache-real-smoke:{suffix}",
            backend_image=f"proxycache-real-smoke-backend:{suffix}",
            meta_volume=f"proxycache-real-smoke-meta-{suffix}",
            cache_volume=f"proxycache-real-smoke-cache-{suffix}",
            model_port=_env_int("REAL_SMOKE_MODEL_PORT", 5800),
            backend_ctx_size=backend_ctx_size or _env_int("REAL_SMOKE_BACKEND_CTX_SIZE", 4096),
            backend_extra_args=backend_extra_args.strip(),
        )
        stack.start()
        try:
            yield stack
        finally:
            stack.stop()

    return _factory


def _exercise_basic_restore(
    stack: RealSmokeStack,
    *,
    model: str,
    repeat: int,
    max_tokens: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    prompt_a_first = _build_large_prompt("atlas", "READY", repeat)
    prompt_b = _build_large_prompt("zephyr", "SWITCH", repeat)
    prompt_a_second = _build_large_prompt("atlas", "READY AGAIN", repeat)
    with httpx.Client(base_url=stack.proxy_url, timeout=_env_float("REAL_SMOKE_HTTP_TIMEOUT", 300.0)) as client:
        payload_a_first = _post_chat(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt_a_first}],
            max_tokens=max_tokens,
        )
        payload_b = _post_chat(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt_b}],
            max_tokens=max_tokens,
        )
        payload_a_second = _post_chat(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt_a_second}],
            max_tokens=max_tokens,
        )
    return payload_a_first, payload_b, payload_a_second


@pytest.mark.real_smoke
@pytest.mark.parametrize("model_preset_name", _selected_presets())
def test_large_prefix_restore_reuses_cache_with_measurable_gain(
    real_smoke_stack_factory,
    model_preset_name: str,
) -> None:
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    repeat = _env_int("REAL_SMOKE_PREFIX_REPEAT", 96)
    max_ratio = _env_float("REAL_SMOKE_MAX_PROMPT_MS_RATIO", 0.9)
    min_cache_n = _env_int("REAL_SMOKE_MIN_CACHE_N", 1)
    max_tokens = _env_int("REAL_SMOKE_MAX_TOKENS", 16)

    with real_smoke_stack_factory(model_preset_name=model_preset_name) as stack:
        try:
            payload_a_first, payload_b, payload_a_second = _exercise_basic_restore(
                stack,
                model=model,
                repeat=repeat,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            pytest.fail(
                f"real smoke request sequence failed for {model_preset_name}: {exc}\n\n{stack.logs()}"
            )

        first_prompt_ms = _prompt_ms(payload_a_first)
        second_prompt_ms = _prompt_ms(payload_a_second)
        second_cache_n = _cache_n(payload_a_second)

        assert _extract_message_text(payload_b)
        assert len(stack.meta_files()) >= 2
        assert second_cache_n >= min_cache_n, payload_a_second
        assert first_prompt_ms > 0.0, payload_a_first
        assert second_prompt_ms > 0.0, payload_a_second
        assert second_prompt_ms <= first_prompt_ms * max_ratio, (
            f"preset={model_preset_name} expected restored prompt_ms <= "
            f"{first_prompt_ms * max_ratio:.2f}, got {second_prompt_ms:.2f}. "
            f"first={first_prompt_ms:.2f}, cache_n={second_cache_n}\n\n{stack.logs()}"
        )


@pytest.mark.real_smoke
def test_system_prompt_toggle_invalidates_then_restores(real_smoke_stack_factory) -> None:
    model_preset_name = _primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    repeat = _env_int("REAL_SMOKE_CONVERSATION_REPEAT", 48)
    max_tokens = _env_int("REAL_SMOKE_MAX_TOKENS", 16)
    min_cache_n = _env_int("REAL_SMOKE_MIN_CACHE_N", 1)
    invalidated_cache_max = _env_int("REAL_SMOKE_INVALIDATED_CACHE_MAX", 32)

    base_messages = _conversation_messages(
        system_tag="atlas",
        user_labels=["atlas", "meridian"],
        suffix="MCP OFF",
        repeat=repeat,
    )
    mcp_messages = _conversation_messages(
        system_tag="zephyr",
        user_labels=["atlas", "meridian"],
        suffix="MCP ON",
        repeat=repeat,
    )
    evict_messages = _conversation_messages(
        system_tag="delta",
        user_labels=["delta", "zephyr"],
        suffix="MCP EVICT",
        repeat=repeat,
    )

    with real_smoke_stack_factory(model_preset_name=model_preset_name) as stack:
        try:
            with httpx.Client(base_url=stack.proxy_url, timeout=_env_float("REAL_SMOKE_HTTP_TIMEOUT", 300.0)) as client:
                payload_base_first = _post_chat(client, model=model, messages=base_messages, max_tokens=max_tokens)
                payload_evict = _post_chat(client, model=model, messages=evict_messages, max_tokens=max_tokens)
                payload_mcp_changed = _post_chat(client, model=model, messages=mcp_messages, max_tokens=max_tokens)
                payload_evict_again = _post_chat(client, model=model, messages=evict_messages, max_tokens=max_tokens)
                payload_base_restored = _post_chat(client, model=model, messages=base_messages, max_tokens=max_tokens)
        except Exception as exc:
            pytest.fail(f"system prompt toggle scenario failed: {exc}\n\n{stack.logs()}")

        assert _extract_message_text(payload_evict)
        assert _extract_message_text(payload_evict_again)
        assert _cache_n(payload_mcp_changed) <= invalidated_cache_max, payload_mcp_changed
        assert _cache_n(payload_base_restored) >= min_cache_n, payload_base_restored
        assert _prompt_ms(payload_base_restored) < _prompt_ms(payload_mcp_changed), (
            f"restored base prompt should be faster than MCP-modified prompt. "
            f"restored={_prompt_ms(payload_base_restored):.2f}, "
            f"mcp_changed={_prompt_ms(payload_mcp_changed):.2f}\n\n{stack.logs()}"
        )
        assert _prompt_ms(payload_base_restored) <= _prompt_ms(payload_base_first), (
            f"restored base prompt should not be slower than initial base prompt. "
            f"first={_prompt_ms(payload_base_first):.2f}, restored={_prompt_ms(payload_base_restored):.2f}\n\n{stack.logs()}"
        )


@pytest.mark.real_smoke
def test_near_match_early_system_change_does_not_false_restore(real_smoke_stack_factory) -> None:
    model_preset_name = _primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    repeat = _env_int("REAL_SMOKE_CONVERSATION_REPEAT", 48)
    max_tokens = _env_int("REAL_SMOKE_MAX_TOKENS", 16)
    min_cache_n = _env_int("REAL_SMOKE_MIN_CACHE_N", 1)
    false_restore_cache_max = _env_int("REAL_SMOKE_FALSE_RESTORE_CACHE_MAX", 32)

    shared_prefix = _build_large_prompt("atlas", "NEAR MATCH SHARED", repeat)
    base_messages = [
        {"role": "system", "content": f"profile=base\n{shared_prefix}"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "ACK base"},
        {"role": "user", "content": "Reply with exactly: BASE"},
    ]
    near_match_messages = [
        {"role": "system", "content": f"profile=impostor\n{shared_prefix}"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "ACK base"},
        {"role": "user", "content": "Reply with exactly: IMPOSTOR"},
    ]
    evict_messages = _conversation_messages(
        system_tag="delta",
        user_labels=["delta", "zephyr"],
        suffix="NEAR MATCH EVICT",
        repeat=repeat,
    )

    with real_smoke_stack_factory(model_preset_name=model_preset_name) as stack:
        try:
            with httpx.Client(base_url=stack.proxy_url, timeout=_env_float("REAL_SMOKE_HTTP_TIMEOUT", 300.0)) as client:
                _post_chat(client, model=model, messages=base_messages, max_tokens=max_tokens)
                payload_evict = _post_chat(client, model=model, messages=evict_messages, max_tokens=max_tokens)
                payload_near_match = _post_chat(client, model=model, messages=near_match_messages, max_tokens=max_tokens)
                payload_evict_again = _post_chat(client, model=model, messages=evict_messages, max_tokens=max_tokens)
                payload_base_restored = _post_chat(client, model=model, messages=base_messages, max_tokens=max_tokens)
        except Exception as exc:
            pytest.fail(f"near-match restore trap failed: {exc}\n\n{stack.logs()}")

        assert _extract_message_text(payload_evict)
        assert _extract_message_text(payload_evict_again)
        assert _cache_n(payload_near_match) <= false_restore_cache_max, payload_near_match
        assert _cache_n(payload_base_restored) >= min_cache_n, payload_base_restored
        assert _prompt_ms(payload_base_restored) < _prompt_ms(payload_near_match), (
            f"base restore should be cheaper than near-match miss. "
            f"base_restored={_prompt_ms(payload_base_restored):.2f}, "
            f"near_match={_prompt_ms(payload_near_match):.2f}\n\n{stack.logs()}"
        )


@pytest.mark.real_smoke
@pytest.mark.parametrize(
    ("label", "base_messages_factory", "variant_messages_factory"),
    [
        (
            "early_system_line_change",
            lambda repeat: [
                {"role": "system", "content": f"profile=base\n{_build_large_prompt('atlas', 'MATCH BASE', repeat)}"},
                {"role": "user", "content": "Reply with exactly: BASE"},
            ],
            lambda repeat: [
                {"role": "system", "content": f"profile=variant\n{_build_large_prompt('atlas', 'MATCH BASE', repeat)}"},
                {"role": "user", "content": "Reply with exactly: VARIANT"},
            ],
        ),
        (
            "message_shape_change",
            lambda repeat: [{"role": "user", "content": _build_large_prompt("atlas", "SHAPE BASE", repeat)}],
            lambda repeat: [
                {"role": "system", "content": _build_large_prompt("atlas", "SHAPE SEGMENTED", max(8, repeat // 3))},
                {"role": "user", "content": _build_large_prompt("atlas", "SHAPE SEGMENTED", max(8, repeat // 3))},
                {"role": "assistant", "content": "ACK segmented"},
                {"role": "user", "content": "Reply with exactly: SEGMENTED"},
            ],
        ),
        (
            "reordered_middle_history",
            lambda repeat: _conversation_messages(
                system_tag="atlas",
                user_labels=["atlas", "meridian", "delta"],
                suffix="ORDERED",
                repeat=repeat,
            ),
            lambda repeat: [
                _conversation_messages(
                    system_tag="atlas",
                    user_labels=["atlas", "meridian", "delta"],
                    suffix="ORDERED",
                    repeat=repeat,
                )[0],
                _conversation_messages(
                    system_tag="atlas",
                    user_labels=["atlas", "meridian", "delta"],
                    suffix="ORDERED",
                    repeat=repeat,
                )[3],
                _conversation_messages(
                    system_tag="atlas",
                    user_labels=["atlas", "meridian", "delta"],
                    suffix="ORDERED",
                    repeat=repeat,
                )[4],
                _conversation_messages(
                    system_tag="atlas",
                    user_labels=["atlas", "meridian", "delta"],
                    suffix="ORDERED",
                    repeat=repeat,
                )[1],
                _conversation_messages(
                    system_tag="atlas",
                    user_labels=["atlas", "meridian", "delta"],
                    suffix="ORDERED",
                    repeat=repeat,
                )[2],
                *_conversation_messages(
                    system_tag="atlas",
                    user_labels=["atlas", "meridian", "delta"],
                    suffix="ORDERED",
                    repeat=repeat,
                )[5:],
            ],
        ),
    ],
)
def test_restore_matching_stress_matrix_avoids_false_positive_candidates(
    real_smoke_stack_factory,
    label: str,
    base_messages_factory,
    variant_messages_factory,
) -> None:
    model_preset_name = _primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    repeat = _env_int("REAL_SMOKE_CONVERSATION_REPEAT", 48)
    max_tokens = _env_int("REAL_SMOKE_MAX_TOKENS", 16)
    min_cache_n = _env_int("REAL_SMOKE_MIN_CACHE_N", 1)
    false_restore_cache_max = _env_int("REAL_SMOKE_FALSE_RESTORE_CACHE_MAX", 32)
    base_messages = base_messages_factory(repeat)
    variant_messages = variant_messages_factory(repeat)
    evict_messages = _conversation_messages(
        system_tag="delta",
        user_labels=["delta", "zephyr"],
        suffix=f"STRESS {label}",
        repeat=repeat,
    )

    with real_smoke_stack_factory(model_preset_name=model_preset_name) as stack:
        try:
            with httpx.Client(base_url=stack.proxy_url, timeout=_env_float("REAL_SMOKE_HTTP_TIMEOUT", 300.0)) as client:
                _post_chat(client, model=model, messages=base_messages, max_tokens=max_tokens)
                _post_chat(client, model=model, messages=evict_messages, max_tokens=max_tokens)
                payload_variant = _post_chat(client, model=model, messages=variant_messages, max_tokens=max_tokens)
                _post_chat(client, model=model, messages=evict_messages, max_tokens=max_tokens)
                payload_base_restored = _post_chat(client, model=model, messages=base_messages, max_tokens=max_tokens)
        except Exception as exc:
            pytest.fail(f"restore matching stress case {label} failed: {exc}\n\n{stack.logs()}")

        assert _cache_n(payload_variant) <= false_restore_cache_max, f"label={label} payload={payload_variant}"
        assert _cache_n(payload_base_restored) >= min_cache_n, f"label={label} payload={payload_base_restored}"
        assert _prompt_ms(payload_base_restored) < _prompt_ms(payload_variant), (
            f"label={label} expected exact restore to beat false match. "
            f"variant={_prompt_ms(payload_variant):.2f}, restored={_prompt_ms(payload_base_restored):.2f}\n\n{stack.logs()}"
        )


@pytest.mark.real_smoke
def test_restore_prefers_exact_candidate_among_many_similar_saved_prefixes(real_smoke_stack_factory) -> None:
    model_preset_name = _primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    repeat = _env_int("REAL_SMOKE_CONVERSATION_REPEAT", 48)
    max_tokens = _env_int("REAL_SMOKE_MAX_TOKENS", 16)
    min_cache_n = _env_int("REAL_SMOKE_MIN_CACHE_N", 1)
    false_restore_cache_max = _env_int("REAL_SMOKE_FALSE_RESTORE_CACHE_MAX", 32)
    shared = _build_large_prompt("atlas", "SHARED", repeat)

    def _messages(tag: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": f"repo={tag}\n{shared}"},
            {"role": "user", "content": f"Reply with exactly: {tag}"},
        ]

    base_messages = _messages("BETA")
    sibling_messages = _messages("ALPHA")
    cousin_messages = _messages("GAMMA")
    impostor_messages = _messages("DELTA")

    with real_smoke_stack_factory(model_preset_name=model_preset_name) as stack:
        try:
            with httpx.Client(base_url=stack.proxy_url, timeout=_env_float("REAL_SMOKE_HTTP_TIMEOUT", 300.0)) as client:
                _post_chat(client, model=model, messages=sibling_messages, max_tokens=max_tokens)
                _post_chat(client, model=model, messages=base_messages, max_tokens=max_tokens)
                _post_chat(client, model=model, messages=cousin_messages, max_tokens=max_tokens)
                payload_impostor = _post_chat(client, model=model, messages=impostor_messages, max_tokens=max_tokens)
                payload_base_restored = _post_chat(client, model=model, messages=base_messages, max_tokens=max_tokens)
        except Exception as exc:
            pytest.fail(f"many-candidate restore ranking failed: {exc}\n\n{stack.logs()}")

        assert len(stack.meta_files()) >= 3
        assert _cache_n(payload_base_restored) >= min_cache_n, payload_base_restored
        assert _cache_n(payload_impostor) <= false_restore_cache_max, payload_impostor
        assert _prompt_ms(payload_base_restored) < _prompt_ms(payload_impostor), (
            f"exact candidate should outperform unseen impostor. "
            f"base_restored={_prompt_ms(payload_base_restored):.2f}, "
            f"impostor={_prompt_ms(payload_impostor):.2f}\n\n{stack.logs()}"
        )


@pytest.mark.real_smoke
@pytest.mark.parametrize(
    ("backend_extra_args", "label"),
    [
        ("", "default"),
        ("--swa-full", "swa_full"),
    ],
)
def test_restore_reuse_survives_swa_full_backend_mode(
    real_smoke_stack_factory,
    backend_extra_args: str,
    label: str,
) -> None:
    model_preset_name = _primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    repeat = _env_int("REAL_SMOKE_PREFIX_REPEAT", 96)
    max_tokens = _env_int("REAL_SMOKE_MAX_TOKENS", 16)
    min_cache_n = _env_int("REAL_SMOKE_MIN_CACHE_N", 1)

    with real_smoke_stack_factory(
        model_preset_name=model_preset_name,
        backend_extra_args=backend_extra_args,
    ) as stack:
        try:
            payload_a_first, _, payload_a_second = _exercise_basic_restore(
                stack,
                model=model,
                repeat=repeat,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            pytest.fail(f"swa mode {label} failed during request sequence: {exc}\n\n{stack.logs()}")

        assert len(stack.meta_files()) >= 2
        assert _cache_n(payload_a_second) >= min_cache_n, (
            f"swa mode {label} did not show cache reuse. "
            f"first_prompt_ms={_prompt_ms(payload_a_first):.2f}, "
            f"second_prompt_ms={_prompt_ms(payload_a_second):.2f}\n\n{stack.logs()}"
        )


@pytest.mark.real_smoke
def test_restore_reuse_works_for_six_k_token_context(real_smoke_stack_factory) -> None:
    model_preset_name = _primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    repeat = _env_int("REAL_SMOKE_LONG_PREFIX_REPEAT", 384)
    max_tokens = _env_int("REAL_SMOKE_MAX_TOKENS", 16)
    min_cache_n = _env_int("REAL_SMOKE_MIN_CACHE_N", 1)
    target_prompt_tokens = _env_int("REAL_SMOKE_LONG_TARGET_TOKENS", 6000)
    ctx_size = _env_int("REAL_SMOKE_LONG_CTX_SIZE", 12288)
    max_ratio = _env_float("REAL_SMOKE_LONG_MAX_PROMPT_MS_RATIO", 0.92)

    with real_smoke_stack_factory(
        model_preset_name=model_preset_name,
        backend_ctx_size=ctx_size,
    ) as stack:
        try:
            payload_a_first, payload_b, payload_a_second = _exercise_basic_restore(
                stack,
                model=model,
                repeat=repeat,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            pytest.fail(f"6k context scenario failed: {exc}\n\n{stack.logs()}")

        assert _extract_message_text(payload_b)
        assert _prompt_tokens(payload_a_first) >= target_prompt_tokens, payload_a_first
        assert _cache_n(payload_a_second) >= min_cache_n, payload_a_second
        assert _prompt_ms(payload_a_second) <= _prompt_ms(payload_a_first) * max_ratio, (
            f"6k restore did not improve prompt_ms enough. "
            f"first={_prompt_ms(payload_a_first):.2f}, "
            f"restored={_prompt_ms(payload_a_second):.2f}\n\n{stack.logs()}"
        )


@pytest.mark.real_smoke
def test_concurrent_same_prefix_requests_do_not_deadlock_and_leave_reusable_state(
    real_smoke_stack_factory,
) -> None:
    model_preset_name = _primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    repeat = _env_int("REAL_SMOKE_PREFIX_REPEAT", 96)
    max_tokens = _env_int("REAL_SMOKE_MAX_TOKENS", 16)
    min_cache_n = _env_int("REAL_SMOKE_MIN_CACHE_N", 1)
    prompt = _build_large_prompt("atlas", "CONCURRENT", repeat)
    evict_prompt = _build_large_prompt("delta", "CONCURRENT EVICT", repeat)

    def _post_once(base_url: str, payload_messages: list[dict[str, str]]) -> dict[str, object]:
        with httpx.Client(base_url=base_url, timeout=_env_float("REAL_SMOKE_HTTP_TIMEOUT", 300.0)) as client:
            return _post_chat(client, model=model, messages=payload_messages, max_tokens=max_tokens)

    with real_smoke_stack_factory(model_preset_name=model_preset_name) as stack:
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(_post_once, stack.proxy_url, [{"role": "user", "content": prompt}])
                    for _ in range(2)
                ]
                concurrent_payloads = [future.result(timeout=_env_float("REAL_SMOKE_HTTP_TIMEOUT", 300.0)) for future in futures]

            with httpx.Client(base_url=stack.proxy_url, timeout=_env_float("REAL_SMOKE_HTTP_TIMEOUT", 300.0)) as client:
                payload_evict = _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": evict_prompt}],
                    max_tokens=max_tokens,
                )
                payload_restored = _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
        except Exception as exc:
            pytest.fail(f"concurrent same-prefix scenario failed: {exc}\n\n{stack.logs()}")

        assert len(concurrent_payloads) == 2
        assert all(_extract_message_text(payload) for payload in concurrent_payloads), concurrent_payloads
        assert _extract_message_text(payload_evict)
        assert len(stack.meta_files()) >= 1
        assert _cache_n(payload_restored) >= min_cache_n, payload_restored


@pytest.mark.real_smoke
def test_poisoned_restore_candidate_is_skipped(real_smoke_stack_factory) -> None:
    model_preset_name = _primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    repeat = _env_int("REAL_SMOKE_PREFIX_REPEAT", 96)
    max_tokens = _env_int("REAL_SMOKE_MAX_TOKENS", 16)
    poisoned_cache_max = _env_int("REAL_SMOKE_POISONED_CACHE_MAX", 32)
    poisoned_ratio_min = _env_float("REAL_SMOKE_POISONED_PROMPT_RATIO_MIN", 0.95)

    prompt_a = _build_large_prompt("atlas", "POISON BASE", repeat)
    prompt_b = _build_large_prompt("zephyr", "POISON EVICT", repeat)

    with real_smoke_stack_factory(model_preset_name=model_preset_name) as stack:
        try:
            with httpx.Client(base_url=stack.proxy_url, timeout=_env_float("REAL_SMOKE_HTTP_TIMEOUT", 300.0)) as client:
                payload_first = _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_a}],
                    max_tokens=max_tokens,
                )
                _, meta_payload = stack.latest_meta()
                key = str(meta_payload["key"])
                backend_model_id = str(meta_payload["model_id"])
                stack.write_poison_file(
                    key,
                    backend_model_id,
                    prompt_n=max(1, _prompt_tokens(payload_first)),
                    cache_n=0,
                )
                assert f"{key}.poison.json" in stack.poison_files()
                payload_evict = _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_b}],
                    max_tokens=max_tokens,
                )
                payload_poisoned_retry = _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_a}],
                    max_tokens=max_tokens,
                )
        except Exception as exc:
            pytest.fail(f"poison scenario failed: {exc}\n\n{stack.logs()}")

        assert _extract_message_text(payload_evict)
        assert _cache_n(payload_poisoned_retry) <= poisoned_cache_max, payload_poisoned_retry
        assert _prompt_ms(payload_poisoned_retry) >= _prompt_ms(payload_first) * poisoned_ratio_min, (
            f"poisoned retry unexpectedly reused too much cache. "
            f"first={_prompt_ms(payload_first):.2f}, retry={_prompt_ms(payload_poisoned_retry):.2f}\n\n{stack.logs()}"
        )


@pytest.mark.real_smoke
def test_corrupted_saved_cache_blob_forces_restore_failure_or_poison(real_smoke_stack_factory) -> None:
    model_preset_name = _primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    repeat = _env_int("REAL_SMOKE_PREFIX_REPEAT", 96)
    max_tokens = _env_int("REAL_SMOKE_MAX_TOKENS", 16)
    unhappy_cache_max = _env_int("REAL_SMOKE_UNHAPPY_RESTORE_CACHE_MAX", 32)

    prompt_base = _build_large_prompt("atlas", "CORRUPT BASE", repeat)
    prompt_evict = _build_large_prompt("zephyr", "CORRUPT EVICT", repeat)

    with real_smoke_stack_factory(model_preset_name=model_preset_name) as stack:
        try:
            with httpx.Client(base_url=stack.proxy_url, timeout=_env_float("REAL_SMOKE_HTTP_TIMEOUT", 300.0)) as client:
                payload_seed = _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_base}],
                    max_tokens=max_tokens,
                )
                _, meta_payload = stack.latest_meta()
                key = str(meta_payload["key"])
                assert stack.cache_blob_exists(key)
                stack.corrupt_cache_blob(key)
                _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_evict}],
                    max_tokens=max_tokens,
                )
                payload_retry = _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_base}],
                    max_tokens=max_tokens,
                )
                _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_evict}],
                    max_tokens=max_tokens,
                )
                payload_second_retry = _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_base}],
                    max_tokens=max_tokens,
                )
        except Exception as exc:
            pytest.fail(f"corrupted cache blob scenario failed: {exc}\n\n{stack.logs()}")

        logs = stack.logs()
        poison_file = f"{key}.poison.json"
        poison_exists = poison_file in stack.poison_files()
        assert _cache_n(payload_retry) <= unhappy_cache_max, payload_retry
        assert (
            "restore_before_chat" in logs and ("ok=False" in logs or poison_exists or "restore_poisoned" in logs)
        ), f"expected corrupted restore to fail or poison\n\n{logs}"
        if poison_exists:
            assert _cache_n(payload_second_retry) <= unhappy_cache_max, payload_second_retry
        else:
            assert _cache_n(payload_second_retry) > unhappy_cache_max, payload_second_retry
        if poison_exists:
            assert _prompt_ms(payload_second_retry) >= _prompt_ms(payload_seed) * 0.8, (
                f"poisoned retry should stay cold. "
                f"seed={_prompt_ms(payload_seed):.2f}, second_retry={_prompt_ms(payload_second_retry):.2f}\n\n{logs}"
            )
        else:
            assert _prompt_ms(payload_second_retry) < _prompt_ms(payload_seed), (
                f"corrupted artifact should recover after a clean recompute/save cycle. "
                f"seed={_prompt_ms(payload_seed):.2f}, second_retry={_prompt_ms(payload_second_retry):.2f}\n\n{logs}"
            )


@pytest.mark.real_smoke
def test_backend_ctx_mismatch_makes_restore_unhappy_and_can_poison(real_smoke_stack_factory) -> None:
    model_preset_name = _primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    repeat = _env_int("REAL_SMOKE_LONG_PREFIX_REPEAT", 384)
    max_tokens = _env_int("REAL_SMOKE_MAX_TOKENS", 16)
    target_prompt_tokens = _env_int("REAL_SMOKE_LONG_TARGET_TOKENS", 6000)

    prompt_base = _build_large_prompt("atlas", "CTX MISMATCH BASE", repeat)
    prompt_evict = _build_large_prompt("delta", "CTX MISMATCH EVICT", repeat)

    with real_smoke_stack_factory(model_preset_name=model_preset_name, backend_ctx_size=12288) as stack:
        try:
            with httpx.Client(base_url=stack.proxy_url, timeout=_env_float("REAL_SMOKE_HTTP_TIMEOUT", 300.0)) as client:
                payload_seed = _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_base}],
                    max_tokens=max_tokens,
                )
                assert _prompt_tokens(payload_seed) >= target_prompt_tokens, payload_seed
                _, meta_payload = stack.latest_meta()
                key = str(meta_payload["key"])
                _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_evict}],
                    max_tokens=max_tokens,
                )
                stack.restart_backend(backend_ctx_size=4096)
                response_retry = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "stream": False,
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt_base}],
                    },
                )
                response_second_retry = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "stream": False,
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt_base}],
                    },
                )
        except Exception as exc:
            pytest.fail(f"backend ctx mismatch scenario failed: {exc}\n\n{stack.logs()}")

        logs = stack.logs()
        poison_file = f"{key}.poison.json"
        poison_exists = poison_file in stack.poison_files()
        assert response_retry.status_code >= 400, response_retry.text
        assert (
            "restore_before_chat" in logs and ("ok=False" in logs or poison_exists or "restore_poisoned" in logs)
        ), f"expected ctx mismatch restore to fail or poison\n\n{logs}"
        assert response_second_retry.status_code >= 400, response_second_retry.text


@pytest.mark.real_smoke
def test_backend_slot_save_restore_round_trip_supports_reuse(real_smoke_stack_factory) -> None:
    model_preset_name = _primary_preset()
    model = os.getenv("REAL_SMOKE_MODEL_ID", model_preset_name)
    repeat = _env_int("REAL_SMOKE_PREFIX_REPEAT", 96)
    max_tokens = _env_int("REAL_SMOKE_MAX_TOKENS", 16)
    min_cache_n = _env_int("REAL_SMOKE_MIN_CACHE_N", 1)

    prompt_seed = _build_large_prompt("atlas", "SLOT SEED", repeat)
    prompt_evict = _build_large_prompt("delta", "SLOT EVICT", repeat)
    prompt_restore = _build_large_prompt("atlas", "SLOT RESTORE", repeat)
    slot_filename = "manual-live-slot.bin"

    with real_smoke_stack_factory(model_preset_name=model_preset_name) as stack:
        try:
            with httpx.Client(base_url=stack.proxy_url, timeout=_env_float("REAL_SMOKE_HTTP_TIMEOUT", 300.0)) as client:
                payload_seed = _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_seed}],
                    max_tokens=max_tokens,
                )
                save_payload = stack.backend_post(
                    "/slots/0?action=save",
                    {"filename": slot_filename},
                )
                payload_evict = _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_evict}],
                    max_tokens=max_tokens,
                )
                restore_payload = stack.backend_post(
                    "/slots/0?action=restore",
                    {"filename": slot_filename},
                )
                payload_restored = _post_chat(
                    client,
                    model=model,
                    messages=[{"role": "user", "content": prompt_restore}],
                    max_tokens=max_tokens,
                )
        except Exception as exc:
            pytest.fail(f"slot round-trip scenario failed: {exc}\n\n{stack.logs()}")

        assert _extract_message_text(payload_evict)
        assert save_payload.get("id_slot") == 0, save_payload
        assert str(save_payload.get("filename") or "") == slot_filename, save_payload
        assert int(restore_payload.get("n_restored") or 0) > 0, restore_payload
        assert _cache_n(payload_restored) >= min_cache_n, payload_restored
        assert _prompt_ms(payload_restored) < _prompt_ms(payload_seed), (
            f"restored slot prompt should be faster than seeded cold run. "
            f"seed={_prompt_ms(payload_seed):.2f}, restored={_prompt_ms(payload_restored):.2f}\n\n{stack.logs()}"
        )
