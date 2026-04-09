# ── builder: compile llama-cpp-python and all hub deps ─────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ cmake && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY lib/ ./lib/
RUN pip install --no-cache-dir --prefix=/install .

# ── runtime ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

COPY --from=builder /install /usr/local

RUN useradd -m -u 1000 appuser && \
    mkdir -p /data && chown appuser:appuser /data

WORKDIR /app
COPY lib/ ./lib/

USER appuser

EXPOSE 8000

# Defaults suitable for containerised use.
# SESSION_MEMORY_MODE=cutoff skips the HuggingFace model download at startup.
# Override to "aggressive" or "balanced" to enable conversation compression.
ENV SESSION_DB=/data/sessions.db \
    SESSION_MEMORY_MODE=cutoff \
    HF_HOME=/home/appuser/.cache/huggingface

CMD ["uvicorn", "lib.hub.server:app", "--host", "0.0.0.0", "--port", "8000"]
