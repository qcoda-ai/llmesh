"""
In-process session compression using a locally downloaded GGUF model.
summarize() returns a SummarizeResult namedtuple with text, token counts, and duration_ms
so the hub can log compression events as inference_events with is_compression=1.

The model is downloaded from HuggingFace on first startup (cached in
~/.cache/huggingface/hub/) and loaded into memory via llama-cpp-python.
No external servers or Ollama required.

Config env vars (all optional):
  COMPRESS_MODEL_REPO   HuggingFace repo ID  (default: Qwen/Qwen2.5-0.5B-Instruct-GGUF)
  COMPRESS_MODEL_FILE   GGUF filename        (default: qwen2.5-0.5b-instruct-q4_k_m.gguf)
  COMPRESS_MODEL_CTX    Context window size  (default: 4096)
  COMPRESS_N_THREADS    CPU thread count     (default: os.cpu_count())

Set SESSION_MEMORY_MODE=cutoff to disable compression entirely — the model
will not be downloaded.
"""

import asyncio
import json
import logging
import os
import time
from typing import NamedTuple

logger = logging.getLogger("llmesh.hub.compressor")


class SummarizeResult(NamedTuple):
    text: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float

COMPRESS_MODEL_REPO = os.getenv("COMPRESS_MODEL_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
COMPRESS_MODEL_FILE = os.getenv("COMPRESS_MODEL_FILE", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
COMPRESS_MODEL_CTX  = int(os.getenv("COMPRESS_MODEL_CTX", "4096"))
# os.cpu_count() returns None in some container environments; clamp to at least 1
COMPRESS_N_THREADS  = int(os.getenv("COMPRESS_N_THREADS", str(max(os.cpu_count() or 1, 1))))

_SUMMARIZE_SYSTEM = (
    "You are a conversation summarizer. Create a dense, accurate summary of the "
    "following conversation that preserves all important context, decisions, and "
    "information. Be concise but complete."
)

_llm = None          # llama_cpp.Llama instance once loaded
_infer_sem = None    # asyncio.Semaphore — serialises concurrent compression calls
_ready = False


def _download_and_load() -> None:
    """Blocking: download GGUF (cached after first run) and load into memory."""
    global _llm
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("huggingface-hub is required for in-process compression. pip install huggingface-hub")

    try:
        from llama_cpp import Llama
    except ImportError:
        raise RuntimeError("llama-cpp-python is required for in-process compression. pip install llama-cpp-python")

    logger.info("Downloading %s/%s ...", COMPRESS_MODEL_REPO, COMPRESS_MODEL_FILE)
    model_path = hf_hub_download(
        repo_id=COMPRESS_MODEL_REPO,
        filename=COMPRESS_MODEL_FILE,
    )
    logger.info("Loading model from %s ...", model_path)
    _llm = Llama(
        model_path=model_path,
        n_ctx=COMPRESS_MODEL_CTX,
        n_threads=COMPRESS_N_THREADS,
        n_gpu_layers=0,   # CPU only — guarantees compatibility on all hardware
        verbose=False,
    )
    logger.info("Compression model ready.")


async def ensure_ready() -> bool:
    """
    Download and load the compression model if not already done.
    Awaited at hub startup — server does not accept requests until this returns.
    Returns True on success, False if loading failed (compression falls back to cutoff).
    """
    global _infer_sem, _ready
    _infer_sem = asyncio.Semaphore(1)

    mode = os.getenv("SESSION_MEMORY_MODE", "aggressive")
    if mode == "cutoff":
        logger.info("SESSION_MEMORY_MODE=cutoff — skipping model download.")
        return False

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _download_and_load)
        _ready = True
        return True
    except Exception as exc:
        logger.warning("Could not load compression model (%s). Falling back to cutoff mode.", exc)
        return False


async def summarize(messages: list[dict]) -> SummarizeResult | None:
    """
    Summarize a list of chat messages in-process.
    Returns SummarizeResult(text, prompt_tokens, completion_tokens, duration_ms),
    or None if the model is not ready or inference fails.
    Concurrent calls are serialised — llama.cpp is not thread-safe.
    """
    if not _ready or _llm is None or _infer_sem is None:
        return None

    prompt_messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": f"Summarize this conversation:\n\n{json.dumps(messages)}"},
    ]

    def _infer() -> SummarizeResult:
        t0 = time.time()
        response = _llm.create_chat_completion(
            messages=prompt_messages,
            max_tokens=512,
            temperature=0.0,
        )
        duration_ms = (time.time() - t0) * 1000
        usage = response.get("usage", {})
        return SummarizeResult(
            text=response["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            duration_ms=duration_ms,
        )

    loop = asyncio.get_event_loop()
    async with _infer_sem:
        try:
            return await loop.run_in_executor(None, _infer)
        except Exception as exc:
            logger.error("Summarization failed: %s", exc)
            return None
