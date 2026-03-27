FROM ghcr.io/mostlygeek/llama-swap:cpu

ARG QWEN_INSTRUCT_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
ARG QWEN_CODER_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"

RUN mkdir -p /models /cache
RUN curl -L --fail --retry 5 --retry-delay 5 "$QWEN_INSTRUCT_URL" -o /models/qwen2.5-0.5b-instruct-q4_k_m.gguf
RUN curl -L --fail --retry 5 --retry-delay 5 "$QWEN_CODER_URL" -o /models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf

COPY config.yaml /app/config.yaml
COPY run-model.sh /app/run-model.sh
RUN chmod +x /app/run-model.sh
