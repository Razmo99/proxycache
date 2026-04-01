#!/bin/sh

set -eu

model_path="$1"
port="$2"
ctx_size="${REAL_SMOKE_CTX_SIZE:-4096}"
parallel="${REAL_SMOKE_PARALLEL:-1}"
extra_args="${REAL_SMOKE_EXTRA_ARGS:-}"

exec /app/llama-server \
  --host 0.0.0.0 \
  --port "$port" \
  --model "$model_path" \
  -ngl 999 \
  -fa 1 \
  --no-mmap \
  --mlock \
  --jinja \
  --cache-ram -1 \
  --parallel "$parallel" \
  --slot-save-path /cache \
  --swa-full \
  --metrics \
  --no-warmup \
  --ctx-size "$ctx_size" \
  $extra_args
