<img width="1000"  alt="image_" src="https://github.com/user-attachments/assets/0d966dde-f1d8-432f-bad0-aa79a5ccf396" />

### What this service is

This service is a proxy in front of llama.cpp that makes long‑context chat and IDE workflows much faster by managing llama.cpp slots, reusing cached context, and restoring saved caches from disk when needed. It speaks an OpenAI‑compatible Chat Completions API, so existing clients can connect without changes, including both streaming (SSE) and non‑stream responses depending on request settings.

### Why it’s needed

llama.cpp provides “slots,” each holding a conversation’s KV cache so repeated requests with the same or very similar prefix can skip recomputing the whole prompt and continue from the first mismatching token, which dramatically cuts latency for large prompts. In real teams the number of users can easily exceed the number of available slots (e.g., 20 developers but only 4 slots), so naive routing causes random slot reuse and cache overwrites that waste time and GPU/CPU cycles. This proxy solves that by steering requests to the right slot, saving evicted caches to disk, and restoring them on demand, so long prompts don’t need to be recomputed from scratch each time.

### How requests are balanced and slots are chosen

- Slots and heat: When a request lands in a slot and its cache is valid for reuse, the slot is considered “hot,” and new requests won’t overwrite it if other options exist, preserving useful KV for future reuse.
- Similarity matching: The proxy computes a fast, word‑block prefix similarity between the incoming conversation and existing hot slots, and only reuses a hot slot if the similarity meets a single ratio threshold (e.g., 85% of the shorter sequence), otherwise it rejects reuse to avoid polluting the hot cache with a weakly related prompt.
- Free and cold first: If reuse is rejected, the proxy sends the request to a free slot or a cold slot (one not currently carrying a valuable hot cache), protecting high‑value contexts from accidental overwrites under load.
- Oldest when full: If there are no free or cold slots, the proxy picks the least‑recently used slot and saves its current KV cache to disk before assigning the new request, ensuring nothing valuable is lost when the pool is exhausted.
- Restore on demand: When a new request matches a cache that was previously saved, the proxy restores that cache into a free/cold/oldest slot and routes the request there, which takes seconds versus minutes for full prompt recomputation on long contexts, especially in IDE scenarios with 30–60k tokens.
- Concurrency safety: Each slot is guarded with an async lock; if all are busy, the request waits for the first LRU slot to free, preventing race conditions and unintended cache overwrites during concurrent generation.

### Save and restore from disk

llama.cpp’s HTTP server exposes slot save/restore; saving writes a cache file to the directory provided by --slot‑save‑path, and restore loads by file basename (e.g., slotcache_`<key>`.bin), which is exactly how this proxy persists and revives caches across requests and restarts. The proxy keeps small local .meta files describing cached prefixes for fast lookup, while llama.cpp owns the actual KV .bin files under --slot‑save‑path for correctness and performance.

### Quick start

1) Start llama.cpp ( https://github.com/ggml-org/llama.cpp ) with slots and a cache directory:

```bash
llama-server -m ./model.gguf -np 4 --slot-save-path /var/kvcache --host 0.0.0.0 --port 8080 --swa-full
```

This enables the OpenAI‑compatible HTTP server, a pool of 4 slots, and a directory where slot KV caches are saved and restored by basename.

2) Run the proxy next to it:

```bash
git clone https://github.com/airnsk/proxycache.git
cd proxycache
python3 -m venv venv && source venv/bin/activate && pip install -e .[dev]
python -m proxycache  # or: uvicorn proxycache.api.app:create_app --factory --host 0.0.0.0 --port 8081
```

Your clients should call the proxy’s /v1/chat/completions endpoint; the proxy will handle similarity, slot selection, save/restore, and streaming vs non‑streaming automatically.

If you run into issues using gpt-oss-20b with an IDE like Cline, follow these instructions: https://www.reddit.com/r/CLine/comments/1mtcj2v/making_gptoss_20b_and_cline_work_together/

### Parameters

- LLAMA_SERVER_URL: The llama.cpp server base URL, e.g., http://127.0.0.1:8080, which must expose the OpenAI‑compatible chat completions endpoint.
- SLOTS_COUNT: The number of server slots (should match llama.cpp -np) so the proxy can track and plan reuse/restore correctly under load.
- SIMILARITY_MIN_RATIO: One similarity threshold (e.g., 0.85) controlling both active reuse and disk restore; if a match is below this ratio, the proxy will prefer a free/cold slot or restore instead of overwriting a hot slot.
- MIN_PREFIX_* (chars/words/blocks): Requests below this size are treated as “small” and steered to free/cold/oldest slots to avoid disturbing valuable hot caches used by large, long‑running prompts.
- LOCAL_META_DIR and --slot-save-path: The proxy stores small .meta descriptors locally for fast candidate lookup, while llama.cpp reads/writes the real KV cache files under --slot‑save-path using basename in the HTTP API.

### Observability

- `OTEL_EXPORTER_OTLP_ENDPOINT`: Enables OTLP trace export for the FastAPI server spans, HTTPX client spans, and the proxy’s GenAI client spans.
- `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`: Optional separate OTLP logs endpoint. If unset, logs use `OTEL_EXPORTER_OTLP_ENDPOINT`.
- `OTEL_GEN_AI_PROVIDER`: Provider flavor for GenAI semantic conventions. Set this explicitly for your deployment instead of relying on the default.
- `OTEL_GEN_AI_CAPTURE_CONTENT=1`: Opt-in capture for `gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.output.messages`, and `gen_ai.tool.definitions`.
- `OTEL_GEN_AI_MAX_CONTENT_CHARS`: Max serialized size for each captured content attribute before truncation. Default: `16384`.

When OTLP logging is enabled, stdlib logs are exported with trace/span correlation so logs emitted during a request can be joined to the active trace in your backend.

Cache lifecycle decisions are emitted as span events so slot behavior is visible without reading logs. The main event names are:

- `proxycache.cache.restore.candidate.selected`
- `proxycache.cache.restore.candidate.miss`
- `proxycache.cache.restore.completed`
- `proxycache.cache.restore.failed`
- `proxycache.cache.restore.poisoned`
- `proxycache.cache.save.started`
- `proxycache.cache.save.completed`
- `proxycache.cache.save.failed`
- `proxycache.cache.save.skipped`
- `proxycache.cache.metadata.recorded`
- `proxycache.slot.management.bypassed`
- `proxycache.slot.acquired`
- `proxycache.slot.released`

### Releases

- `main` is the stable release branch.
- `develop` is the prerelease branch and publishes `alpha` versions.
- GitHub Releases and `CHANGELOG.md` are generated by `python-semantic-release`.
- GHCR containers are published automatically for stable and prerelease releases.

Use Conventional Commits for release automation:

- `fix:` for patch releases
- `feat:` for minor releases
- `feat!:` or `BREAKING CHANGE:` for major releases

### Commit and PR checks

- Local commit messages are enforced with `pre-commit` + `Commitizen`.
- Pull request titles are enforced in GitHub Actions.
- Dependabot opens dependency update PRs against `develop` using Conventional Commit-style prefixes.

Setup:

```bash
pip install -e .[dev]
pre-commit install
pre-commit install --hook-type commit-msg
```

Valid examples:

- `feat: add prerelease workflow`
- `fix(ci): correct ghcr tagging`

See [contributing.md](/home/razmo/project/proxycache/docs/contributing.md) for the accepted commit types and examples.

### Real-model smoke test

There is an opt-in live smoke test that uses Docker containers to launch a custom `llama-swap` backend image plus a local build of the proxy, then verifies that proxy-managed cache restore is measurably effective.

It is disabled by default and does not run in CI.

Required environment:

```bash
export REAL_SMOKE=1
export REAL_SMOKE_MODEL_PRESETS=qwen_0_5b_instruct
```

The backend image bakes in these small GGUF presets:

- `qwen_0_5b_instruct`
- `qwen_coder_0_5b_instruct`

To run both:

```bash
export REAL_SMOKE=1
export REAL_SMOKE_MODEL_PRESETS=all
```

Optional environment:

```bash
export REAL_SMOKE_BACKEND_IMAGE=proxycache-real-smoke-backend:local
export REAL_SMOKE_MODEL_ID=qwen_0_5b_instruct
export REAL_SMOKE_PRIMARY_PRESET=qwen_0_5b_instruct
export REAL_SMOKE_PREFIX_REPEAT=96
export REAL_SMOKE_MAX_PROMPT_MS_RATIO=0.9
export REAL_SMOKE_MIN_CACHE_N=1
export REAL_SMOKE_LONG_CTX_SIZE=12288
export REAL_SMOKE_LONG_TARGET_TOKENS=6000
```

Run it with:

```bash
pytest -q --no-cov -m real_smoke tests/real_smoke/test_live_proxy_cache.py
```

What it does:

- builds a custom `llama-swap` image with both small Qwen GGUF files baked into `/models`
- starts the backend via a small wrapper so llama.cpp args like `--swa-full` and larger `--ctx-size` can be varied per smoke test without rebuilding the image
- starts that backend container and routes the selected preset through `/upstream/<model>`
- builds the local proxy Docker image and runs it in a second container
- runs a base save/evict/restore smoke on both baked-in Qwen presets
- runs a system-prompt toggle scenario that simulates enabling an MCP and checks that the changed system prompt invalidates reuse while the original prompt restores again later
- runs the same restore scenario with and without llama.cpp `--swa-full`
- runs a long-context scenario that asserts restore still works on a prompt of at least 6k tokens

There is also an OpenCode-specific real smoke in [test_opencode_sdk_smoke.py](/home/razmo/project/proxycache/tests/real_smoke/test_opencode_sdk_smoke.py). It uses the Node OpenCode SDK to manage session history and revert/undo behavior, then measures cache reuse by replaying the SDK session’s user messages through the proxy. Run it with:

```bash
REAL_SMOKE=1 pytest -q --no-cov -m real_smoke tests/real_smoke/test_opencode_sdk_smoke.py
```

That OpenCode smoke now includes:

- a basic revert/undo cache-reuse flow
- a fixed-default 64K backend scenario that loads a large OpenCode-style workspace prompt, sends `Hi`, sends `Goodbye`, reverts the second turn, and checks that the reverted branch restores efficiently
- a 64K branch-thrash scenario that repeatedly reverts and forks new turns to check that restore candidates stay valid without poisoning or stale-branch corruption

The container smoke in [test_live_proxy_cache.py](/home/razmo/project/proxycache/tests/real_smoke/test_live_proxy_cache.py) also includes adversarial cases such as:

- a near-match early system-prompt change intended to catch false-positive restores
- a matcher stress matrix covering early prefix mutations, message-shape changes, reordered history, and many-similar-candidate ranking
- concurrent identical long-prefix requests against a single-slot backend to catch deadlocks and bad slot-state cleanup
- corrupted saved-cache and backend-ctx-mismatch scenarios that probe how the proxy behaves when llama.cpp is unhappy with a restore

### Why this boosts IDE and long‑context productivity

For 30–60k‑token contexts typical in project‑wide IDE assistants, recomputing a full prompt can take minutes, whereas restoring a previously cached context and continuing from the first mismatching token typically takes seconds on llama.cpp, dramatically improving iteration speed for large teams with limited slots.
