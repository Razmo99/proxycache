# app.py

# -*- coding: utf-8 -*-

"""
Simple KV Proxy (бронебойный):

- Большие: LCP→restore, затем чат строго в этот же слот, потом save+meta.
- Малые: свободный/старый слот, без restore и без дискового save/meta.
- Пин slota дублируется в root/options/query (через клиента).

Дополнительно:

- acquire_for_request обёрнут в таймаут, чтобы не висеть бесконечно, если слот не отпускается.
- Для stream:
    * чтение из llama.cpp идёт в отдельной фоновой задаче (reader);
    * reader пушит чанки в asyncio.Queue;
    * в своём finally reader всегда делает save_after + write_meta + release(g),
      и кладёт в очередь sentinel None;
    * StreamingResponse читает из очереди и никак не влияет на release слота.
"""

import asyncio
import json
import time
import re
import logging
from typing import List, Dict, AsyncGenerator, Optional
from opentelemetry import trace

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from starlette.background import BackgroundTask

from config import (
    BACKENDS,
    BIG_THRESHOLD_WORDS,
    LCP_TH,
    MAX_SAVED_CACHES,
    MODEL_ID,
    META_DIR,
    SLOT_SAVE_PATH,
    WORDS_PER_BLOCK,
    PORT,
)
import hashing as hs
from llama_client import LlamaClient
from slot_manager import SlotManager, GSlot

log = logging.getLogger(__name__)

ACQUIRE_TIMEOUT = 300.0
STREAM_QUEUE_SIZE = 16

app = FastAPI(title="Simple KV Proxy")

MULTIMODAL_MODEL_PATTERNS = (
    "llava",
    "vision",
    "qwen-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "minicpm-v",
    "minicpmo",
    "internvl",
    "gemma-3",
)


@app.on_event("startup")
async def startup():
    clients = [LlamaClient(be["url"]) for be in BACKENDS]
    sm = SlotManager()
    sm.set_clients(clients)
    app.state.clients = clients
    app.state.sm = sm
    deleted_poison_files = hs.cleanup_restore_poisons()

    from otel import init_otel
    from version import __version__

    init_otel(app, clients[0].client if clients else None)
    log.info(
        "app_start version=%s n_backends=%d port=%d", __version__, len(BACKENDS), PORT
    )
    log.info(
        "retention_config max_saved_caches=%d meta_dir=%s slot_save_path=%s",
        MAX_SAVED_CACHES,
        META_DIR,
        SLOT_SAVE_PATH,
    )
    if deleted_poison_files:
        log.info("startup_deleted_restore_poisons n=%d", deleted_poison_files)


@app.on_event("shutdown")
async def shutdown():
    clients: List[LlamaClient] = getattr(app.state, "clients", [])
    if clients:
        await asyncio.gather(*(c.close() for c in clients))


@app.get("/v1/models")
async def models():
    return {"data": [{"id": MODEL_ID}]}


def _message_has_multimodal_content(message: Dict) -> bool:
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "")
            if part_type in {
                "image_url",
                "input_image",
                "audio_url",
                "input_audio",
                "video_url",
                "input_video",
                "file",
                "input_file",
            }:
                return True
    return False


def _request_has_multimodal_workload(messages: List[Dict]) -> bool:
    return any(_message_has_multimodal_content(msg) for msg in messages or [])


def _looks_like_multimodal_model_name(model_id: Optional[str]) -> bool:
    if not model_id:
        return False

    lowered = model_id.lower()
    if any(pattern in lowered for pattern in MULTIMODAL_MODEL_PATTERNS):
        return True

    return re.search(r"(^|[-_/])(?:vl|vision|omni|audio)($|[-_/])", lowered) is not None


def _slot_management_mode(
    backend_is_multimodal: bool,
    backend_model_id: str,
    client_model: str,
    slots_supported: bool,
) -> tuple[bool, str]:
    if not slots_supported:
        return True, "slots_unsupported"

    if backend_is_multimodal:
        return True, "models_capability"

    if _looks_like_multimodal_model_name(backend_model_id):
        return True, "backend_model_name_heuristic"

    if _looks_like_multimodal_model_name(client_model):
        return True, "model_name_heuristic"

    return False, "slots_enabled"


def _record_saved_cache(
    key: str,
    prefix: str,
    blocks: List[str],
    model_id: str,
) -> None:
    hs.write_meta(key, prefix, blocks, WORDS_PER_BLOCK, model_id)
    log.info(
        "record_saved_cache key=%s model_id=%s keep=%d",
        key[:16],
        model_id,
        MAX_SAVED_CACHES,
    )
    pruned = hs.prune_saved_caches(model_id, keep=MAX_SAVED_CACHES)
    if pruned:
        log.info(
            "retention_pruned_artifacts key=%s model_id=%s deleted=%d keep=%d",
            key[:16],
            model_id,
            pruned,
            MAX_SAVED_CACHES,
        )


async def proxy_upstream_request(req: Request, path: str) -> Response:
    clients: List[LlamaClient] = app.state.clients
    client = clients[0]

    # Host/content-length belong to the incoming hop, not the upstream one.
    headers = {
        key: value
        for key, value in req.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    upstream_path = "/" + path.lstrip("/")
    if upstream_path == "//":
        upstream_path = "/"

    upstream_req = client.client.build_request(
        req.method,
        upstream_path,
        params=req.query_params,
        headers=headers,
    )
    resp = await client.client.send(upstream_req, stream=True)

    excluded_headers = {"content-length", "transfer-encoding", "connection"}
    response_headers = {
        key: value
        for key, value in resp.headers.items()
        if key.lower() not in excluded_headers
    }

    if req.method.upper() == "HEAD":
        body = b""
    else:
        body = resp.aiter_raw()

    return StreamingResponse(
        body,
        status_code=resp.status_code,
        headers=response_headers,
        media_type=resp.headers.get("content-type"),
        background=BackgroundTask(resp.aclose),
    )


async def start_stream_task(
    resp: httpx.Response,
    g: GSlot,
    key: str,
    prefix: str,
    blocks: List[str],
    model_id: str,
    sm: SlotManager,
    restore_key: Optional[str] = None,
    restored: Optional[bool] = None,
) -> AsyncGenerator[bytes, None]:
    queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=STREAM_QUEUE_SIZE)

    async def reader():
        sse_buffer = b""
        last_timings: Optional[Dict] = None
        try:
            log.info("stream_reader_start g=%s key=%s", g, key[:16])
            async for chunk in resp.aiter_raw():
                if not chunk:
                    continue
                sse_buffer += chunk
                sse_buffer, last_timings = _consume_sse_timings(
                    sse_buffer, last_timings
                )
                try:
                    await queue.put(chunk)
                except asyncio.CancelledError:
                    log.warning("stream_reader_cancelled_put g=%s key=%s", g, key[:16])
                    raise
        except asyncio.CancelledError:
            log.warning("stream_reader_cancelled g=%s key=%s", g, key[:16])
            raise
        except Exception as e:
            log.exception("stream_reader_error g=%s key=%s: %s", g, key[:16], e)
        finally:
            try:
                await resp.aclose()
            except Exception:
                pass
            _maybe_poison_restore(restore_key, restored, model_id, last_timings)
            ok = False
            try:
                ok = await sm.save_after(g, key)
            except Exception as e:
                log.warning("save_after_exception g=%s key=%s: %s", g, key[:16], e)
            if ok:
                hs.clear_restore_poison(key)
                try:
                    _record_saved_cache(key, prefix, blocks, model_id)
                except Exception as e:
                    log.warning("record_saved_cache_exception key=%s: %s", key[:16], e)
            sm.release(g)
            log.info("stream_reader_done g=%s key=%s saved=%s", g, key[:16], ok)
            try:
                await queue.put(None)
            except Exception:
                pass

    asyncio.create_task(reader())

    async def gen() -> AsyncGenerator[bytes, None]:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return gen()


async def start_stream_task_multimodal(
    resp: httpx.Response,
    model_id: str,
) -> AsyncGenerator[bytes, None]:
    """
    Stream task for multimodal backends (no slot management).
    Just passes through the response without save/restore.
    """
    queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=STREAM_QUEUE_SIZE)

    async def reader():
        try:
            log.info("multimodal_stream_reader_start model_id=%s", model_id[:16])
            async for chunk in resp.aiter_raw():
                if not chunk:
                    continue
                try:
                    await queue.put(chunk)
                except asyncio.CancelledError:
                    log.warning(
                        "multimodal_stream_reader_cancelled_put model_id=%s",
                        model_id[:16],
                    )
                    raise
        except asyncio.CancelledError:
            log.warning("multimodal_stream_reader_cancelled model_id=%s", model_id[:16])
            raise
        except Exception as e:
            log.exception(
                "multimodal_stream_reader_error model_id=%s: %s", model_id[:16], e
            )
        finally:
            try:
                await resp.aclose()
            except Exception:
                pass
            log.info("multimodal_stream_reader_done model_id=%s", model_id[:16])
            try:
                await queue.put(None)
            except Exception:
                pass

    asyncio.create_task(reader())

    async def gen() -> AsyncGenerator[bytes, None]:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return gen()


def _extract_timings(payload: Optional[Dict]) -> Optional[Dict]:
    if not isinstance(payload, dict):
        return None
    timings = payload.get("timings")
    return timings if isinstance(timings, dict) else None


def _current_recording_span():
    span = trace.get_current_span()
    return span if span and span.is_recording() else None


def _consume_sse_timings(
    buffer: bytes,
    last_timings: Optional[Dict],
) -> tuple[bytes, Optional[Dict]]:
    while b"\n\n" in buffer:
        raw_event, buffer = buffer.split(b"\n\n", 1)
        data_lines = []
        for line in raw_event.splitlines():
            if line.startswith(b"data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            continue
        data = b"\n".join(data_lines).strip()
        if not data or data == b"[DONE]":
            continue
        try:
            payload = json.loads(data)
        except Exception:
            continue
        timings = _extract_timings(payload)
        if timings:
            last_timings = timings
    return buffer, last_timings


def _maybe_poison_restore(
    restore_key: Optional[str],
    restored: Optional[bool],
    model_id: str,
    timings: Optional[Dict],
) -> None:
    if not restore_key or not restored or not timings:
        return

    prompt_n = int(timings.get("prompt_n") or 0)
    cache_n = int(timings.get("cache_n") or 0)
    prompt_ms = float(timings.get("prompt_ms") or 0.0)

    # Only poison when restore claimed success but llama.cpp reused none of the
    # prompt tokens. Short incremental restores can legitimately have prompt_n
    # close to cache_n, so keep this check intentionally conservative.
    if prompt_n > 0 and cache_n == 0:
        try:
            hs.poison_restore_key(
                restore_key,
                model_id,
                prompt_n=prompt_n,
                cache_n=cache_n,
                prompt_ms=prompt_ms,
                reason="no_cache_reuse_after_restore",
            )
        except Exception as e:
            log.warning(
                "poison_restore_exception key=%s err=%s",
                restore_key[:16],
                e,
            )


@app.post("/v1/chat/completions")
async def chat(req: Request):
    sm: SlotManager = app.state.sm
    clients: List[LlamaClient] = app.state.clients

    t0 = time.time()
    data = await req.json()

    messages: List[Dict] = data.get("messages") or []
    stream = bool(data.get("stream", False))
    client_model = data.get("model") or MODEL_ID

    be_id = 0
    client = clients[be_id]

    # model identity and multimodal capability come from /v1/models.
    backend_model_id, backend_caps = await client.get_model_info()
    backend_is_multimodal = await client.is_multimodal()
    request_has_media = _request_has_multimodal_workload(messages)

    prefix = hs.raw_prefix(messages)
    full_for_key = backend_model_id + "\n" + prefix
    key = hs.prefix_key_sha256(full_for_key)
    blocks = hs.block_hashes_from_text(prefix, WORDS_PER_BLOCK)
    n_words = len(hs.words_from_text(prefix))
    is_big = n_words > BIG_THRESHOLD_WORDS

    restore_key: Optional[str] = None
    if is_big:
        cand = hs.find_best_restore_candidate(
            blocks,
            WORDS_PER_BLOCK,
            LCP_TH,
            backend_model_id,
        )
        if cand:
            restore_key, ratio = cand
            log.info(
                "restore_candidate basename=%s ratio=%.3f",
                restore_key[:16],
                ratio,
            )
        else:
            log.info("restore_candidate none")
    else:
        log.info(
            "small_request n_words=%d threshold=%d",
            n_words,
            BIG_THRESHOLD_WORDS,
        )

    log.info(
        "before_acquire is_big=%s restore_key=%s",
        is_big,
        restore_key[:16] if restore_key else None,
    )

    if backend_is_multimodal != sm.is_multimodal_backend(be_id):
        await sm.set_multimodal_backend(be_id, backend_is_multimodal)
        log.info(
            "multimodal_backend_detected be_id=%d via=models is_mm=%s",
            be_id,
            backend_is_multimodal,
        )

    log.info(
        "request_modalities be=%d backend_caps=%s request_has_media=%s",
        be_id,
        backend_caps,
        request_has_media,
    )

    bypass_slots, bypass_reason = _slot_management_mode(
        backend_is_multimodal=backend_is_multimodal,
        backend_model_id=backend_model_id,
        client_model=client_model,
        slots_supported=client.slots_supported(),
    )

    # llama.cpp disables slot save/restore entirely when multimodal support is
    # enabled for the backend, so the proxy must bypass slot management here.
    if bypass_slots:
        from otel import add_cache_attributes, add_llm_attributes, add_timing_attributes, set_error

        current_span = _current_recording_span()
        add_cache_attributes(current_span, False, key[:32] if key else "")
        add_llm_attributes(current_span, backend_model_id)
        if current_span is not None and bypass_reason:
            current_span.set_attribute("proxycache.slot_bypass", True)
            current_span.set_attribute("proxycache.slot_bypass_reason", bypass_reason)

        log.info(
            "slot_passthrough model_id=%s reason=%s request_has_media=%s",
            backend_model_id,
            bypass_reason,
            request_has_media,
        )
        body = dict(data)
        body["model"] = client_model
        if stream:
            resp = await client.chat_completions(body, slot_id=None, stream=True)
            if resp.status_code != 200:
                err_txt = await resp.aread()
                await resp.aclose()
                set_error(current_span, f"upstream_http_{resp.status_code}")
                return JSONResponse(
                    {"error": err_txt.decode("utf-8", "ignore")},
                    status_code=resp.status_code,
                )
            gen = await start_stream_task_multimodal(resp, backend_model_id)
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            return StreamingResponse(
                gen, media_type="text/event-stream", headers=headers
            )
        else:
            out = await client.chat_completions(body, slot_id=None, stream=False)
            if not isinstance(out, dict):
                set_error(current_span, "provider_non_json_body")
                return JSONResponse(
                    {"error": "provider non-JSON body"}, status_code=502
                )
            add_timing_attributes(current_span, _extract_timings(out))
            log.info("multimodal_json_done dur_ms=%d", int((time.time() - t0) * 1000))
            return JSONResponse(content=out, status_code=200)

    try:
        g, lock, restored = await asyncio.wait_for(
            sm.acquire_for_request(restore_key if is_big else None),
            timeout=ACQUIRE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.error(
            "acquire_timeout is_big=%s restore_key=%s",
            is_big,
            restore_key[:16] if restore_key else None,
        )
        return JSONResponse(
            {"error": "all slots busy, please retry later"},
            status_code=503,
        )

    log.info("after_acquire g=%s restored=%s", g, restored)

    be_id, slot_id = g
    client = clients[be_id]

    from otel import add_cache_attributes, add_llm_attributes, add_timing_attributes, set_error

    current_span = _current_recording_span()
    n_used = sum(1 for lock in sm._locks.values() if lock.locked())
    add_cache_attributes(
        current_span, bool(restored), key[:32] if key else "", slot_id, n_used
    )
    add_llm_attributes(current_span, backend_model_id)

    body = dict(data)
    body["model"] = client_model
    body["cache_prompt"] = bool(is_big)
    body["n_keep"] = -1

    opts = dict(body.get("options") or {})
    opts["slot_id"] = slot_id
    opts["id_slot"] = slot_id
    opts["n_keep"] = -1
    opts["cache_prompt"] = bool(is_big)
    body["options"] = opts

    log.info(
        "dispatch be=%d slot=%d is_big=%s (restore_target=%s restored=%s model_id=%s)",
        be_id,
        slot_id,
        is_big,
        restore_key[:16] if restore_key else None,
        restored,
        backend_model_id,
    )

    try:
        if stream:
            resp = await client.chat_completions(
                body,
                slot_id=slot_id,
                stream=True,
            )
            if resp.status_code != 200:
                err_txt = await resp.aread()
                await resp.aclose()
                sm.release(g)
                set_error(current_span, f"upstream_http_{resp.status_code}")
                return JSONResponse(
                    {"error": err_txt.decode("utf-8", "ignore")},
                    status_code=resp.status_code,
                )

            gen = await start_stream_task(
                resp,
                g,
                key,
                prefix,
                blocks,
                backend_model_id,
                sm,
                restore_key=restore_key if is_big else None,
                restored=restored,
            )

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers=headers,
            )

        else:
            out = await client.chat_completions(
                body,
                slot_id=slot_id,
                stream=False,
            )
            if not isinstance(out, dict):
                sm.release(g)
                set_error(current_span, "provider_non_json_body")
                return JSONResponse(
                    {"error": "provider non-JSON body"},
                    status_code=502,
                )
            add_timing_attributes(current_span, _extract_timings(out))

            ok = False
            try:
                _maybe_poison_restore(
                    restore_key if is_big else None,
                    restored,
                    backend_model_id,
                    _extract_timings(out),
                )
                if is_big:
                    ok = await sm.save_after(g, key)
                    if ok:
                        hs.clear_restore_poison(key)
                        _record_saved_cache(
                            key,
                            prefix,
                            blocks,
                            backend_model_id,
                        )
            finally:
                sm.release(g)

            log.info(
                "json_done g=%s key=%s saved=%s is_big=%s dur_ms=%d",
                g,
                key[:16],
                ok,
                is_big,
                int((time.time() - t0) * 1000),
            )
            return JSONResponse(content=out, status_code=200)

    except Exception as e:
        sm.release(g)
        set_error(current_span, str(e))
        log.exception("chat_error g=%s key=%s: %s", g, key[:16], e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.api_route("/", methods=["GET", "HEAD"])
async def upstream_root(req: Request):
    return await proxy_upstream_request(req, "/")


@app.api_route("/{path:path}", methods=["GET", "HEAD"])
async def upstream_passthrough(path: str, req: Request):
    return await proxy_upstream_request(req, path)
