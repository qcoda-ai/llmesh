import hashlib
import json
import logging
import os
import random
import re
import secrets
import sys
import psutil
import platform
import socket
import httpx
import asyncio
from pathlib import Path
from dotenv import load_dotenv

try:
    AGENT_VERSION = (Path(__file__).resolve().parents[2] / "VERSION").read_text().strip()
except Exception:
    AGENT_VERSION = "unknown"

# Line-buffer stdout/stderr so any residual non-logging writes (third-party
# libs, traceback printers) reach journalctl/docker logs immediately without
# requiring `PYTHONUNBUFFERED=1`. Agent diagnostics now flow via the
# `logging` module (D017 + D051), which writes to stderr by default; this
# shim is belt-and-braces for the stdout side.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

logger = logging.getLogger("llmesh.agent")

# Load .env explicitly from repo root so launchd-launched agents (where CWD
# is not the repo) still pick up project-level `.env` files. System env vars
# still take precedence over .env per python-dotenv default behaviour.
_repo_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _repo_env_path.exists():
    load_dotenv(_repo_env_path)
else:
    load_dotenv()  # fallback: CWD-walk

HUB_URL = os.getenv("HUB_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("LLMESH_API_KEY")
if not API_KEY:
    raise ValueError("LLMESH_API_KEY environment variable is required")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
VLLM_HOST = os.getenv("VLLM_HOST")          # None unless explicitly set
# Optional bearer token for auth-protected vLLM-compatible endpoints. The
# vanilla vLLM OpenAI server is unauthenticated, but production deployments
# often sit behind a proxy (LiteLLM, hardened reverse proxy, etc.) that
# requires `Authorization: Bearer <key>` on every request. Set this env var
# and the agent will attach the header to /health, /v1/models, and the
# inference call. Leave unset for plain local vLLM. See
# docs/integrations/litellm.md.
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
# Path the agent probes to decide whether the vLLM backend is up. Default
# `/health` works for vanilla vLLM. LiteLLM users should set this to
# `/health/liveliness` because LiteLLM's `/health` performs a real model
# probe and is much heavier than a liveness check.
VLLM_HEALTH_PATH = os.getenv("VLLM_HEALTH_PATH", "/health")
# Optional explicit override for the vLLM context window the agent reports
# to the hub at registration. By default the agent auto-detects this from
# `max_model_len` in vLLM's `/v1/models` response (vLLM exposes it as a
# non-standard extension field on the OpenAI model card). Set this when
# running behind a proxy that strips the field — e.g. some LiteLLM
# configurations — or when you want to clamp the advertised window below
# the model's actual capability. See decisions.md D015.
VLLM_MAX_CONTEXT = int(os.getenv("VLLM_MAX_CONTEXT", "0")) or None
# Master gate for vLLM real per-token streaming (D040). Default ON per D044
# after manual real-vLLM verification on operator hardware (token-by-token
# rendering, usage counts correct, batcher auto-tune working). Set to "false"
# / "0" / "no" to fall back to the blocking + D018 bridge path.
VLLM_STREAMING_ENABLED = os.getenv("VLLM_STREAMING_ENABLED", "true").lower() in ("true", "1", "yes")
MLX_HOST = os.getenv("MLX_HOST")              # None unless explicitly set
# MLX real per-token streaming (D059, default flipped ON per D060 after LAB-003
# graduated 2026-05-28 — automated 6/6 + hub round-trip + STREAM_BATCH_FIXED=1
# parity + client-disconnect 410 cancel all observed). Set to "false" / "0" /
# "no" to fall back to the blocking + D018 bridge path.
MLX_STREAMING_ENABLED = os.getenv("MLX_STREAMING_ENABLED", "true").lower() in ("true", "1", "yes")

# D066: process-level state for vLLM `stream_options.include_usage` support.
# vLLM versions older than ~0.4.1 (and proxies that strip stream_options) silently
# ignore `include_usage` — the streaming response has no usage chunk and the agent
# falls through to HP-1 delta-count estimation. The per-task warning fires every
# request, which clutters logs; instead we log a one-shot capability summary on
# first detection, then a one-shot recovery when usage starts appearing again.
# Set None until the first vLLM stream completes; then True/False sticky for the
# session. Operators see the capability state in the agent log once per process.
_vllm_usage_chunk_supported: bool | None = None

# Allow direct-script invocation (`python lib/agent/client.py`) in addition
# to module invocation (`python -m lib.agent.client`). Direct invocation
# puts `lib/agent/` on sys.path, not the repo root; absolute imports below
# fail with ModuleNotFoundError. Inject repo root before the absolute imports.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.agent.streaming_batcher import (
    StreamBatcher,
    StreamBatcherAborted,
    resolve_batcher_config,
)
from lib.agent.toolcall_parse import extract_tool_calls, should_normalize


class _StreamCancelled(Exception):
    """Hub returned 410 Gone — consumer disconnected, abort streaming."""


def _vllm_headers() -> dict:
    """Bearer-auth header for vLLM-path requests, or empty if no key set."""
    return {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}


async def _iter_sse_events(byte_stream):
    """Async generator over httpx aiter_bytes() yielding SSE event payloads.

    Buffers bytes until the `\\n\\n` event terminator (CF-1 — single SSE
    event can span multiple lines, and `httpx.aiter_lines()` returns one
    line at a time with no event reassembly). For each complete event,
    concatenates `data:` lines and yields the resulting payload string.

    Skips heartbeat comment lines (`: …`), `event:`, `id:`, `retry:` lines.
    Returns the literal string `"[DONE]"` for the SSE terminator sentinel
    so callers can string-compare without attempting `json.loads` on it.
    """
    buf = b""
    async for chunk in byte_stream:
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            data_lines = []
            for line in event.split(b"\n"):
                if not line or line.startswith(b":"):
                    continue
                if line.startswith(b"data:"):
                    data_lines.append(line[5:].lstrip())
            if data_lines:
                yield b"\n".join(data_lines).decode("utf-8", errors="replace")

def _load_or_create_node_salt() -> str:
    """Persistent random salt, created on first run and reused forever.

    Mixed into the node fingerprint so two machines with identical
    hostname/OS/arch/cpu_count cannot collide (the prior scheme hashed only
    those four fields, which would alias freshly-imaged VMs or containers
    onto the same fingerprint and let the second registration steal the
    first node's token). See decisions.md D021.

    Stored at $LLMESH_STATE_DIR/node_salt, defaulting to ~/.llmesh/node_salt.
    The file is 0600 so other local users can't read it. If the directory is
    not writable (read-only rootfs, no HOME, etc.) we fall back to an
    in-memory salt for the life of the process and log a warning — in that
    case the fingerprint is stable within one run but not across restarts,
    which is strictly no worse than the pre-D021 behavior.
    """
    state_dir = Path(os.environ.get("LLMESH_STATE_DIR", Path.home() / ".llmesh"))
    salt_path = state_dir / "node_salt"
    try:
        if salt_path.exists():
            salt = salt_path.read_text().strip()
            if salt:
                return salt
        state_dir.mkdir(parents=True, exist_ok=True)
        salt = secrets.token_hex(16)
        salt_path.write_text(salt)
        try:
            salt_path.chmod(0o600)
        except OSError:
            pass  # chmod is best-effort (e.g. Windows, noexec mounts)
        return salt
    except OSError as exc:
        logger.warning(
            "could not persist node salt to %s (%s). Falling back to an in-memory "
            "salt for this process; the node fingerprint will change on restart. "
            "Set LLMESH_STATE_DIR to a writable directory to fix.",
            salt_path, exc,
        )
        return secrets.token_hex(16)


_NODE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.\-]{0,63}$")


def _resolve_operator_node_id() -> str | None:
    """If LLMESH_NODE_ID is set and valid, return it. Otherwise None.

    Per D048 — operators may override the auto-generated GUID-style fingerprint
    with a human-readable label (e.g. hostname) so the dashboard, logs, and
    URL paths show which physical machine is which. Validation: must match
    `^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$` so the value is safe to embed in
    `/tasks/{node_id}/...` URL paths and in log lines without quoting.

    Operator owns collision avoidance: setting the same value on two agents
    pointing at the same hub causes the second registration to inherit the
    first's token (existing D021 hub behavior). Document, don't prevent.
    """
    raw = os.getenv("LLMESH_NODE_ID")
    if not raw:
        return None
    if not _NODE_ID_RE.match(raw):
        logger.warning(
            "⚠️  LLMESH_NODE_ID=%r ignored — must match "
            "[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}; falling back to fingerprint",
            raw,
        )
        return None
    return raw


def compute_node_fingerprint() -> str:
    """Return the agent's node identifier.

    If `LLMESH_NODE_ID` is set and validates, use it verbatim (D048).
    Otherwise compute the salted hash fingerprint (D021).
    """
    operator_id = _resolve_operator_node_id()
    if operator_id is not None:
        return operator_id
    salt = _load_or_create_node_salt()
    raw = (
        f"{socket.gethostname()}|{platform.system()}|{platform.machine()}"
        f"|{psutil.cpu_count(logical=False)}|{salt}"
    )
    return "node_" + hashlib.sha256(raw.encode()).hexdigest()[:16]

def check_ollama_available() -> bool:
    try:
        resp = httpx.get("http://localhost:11434/", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False

def get_ollama_models() -> list[str]:
    if not check_ollama_available():
        return []
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            return [model["name"] for model in resp.json().get("models", [])]
    except Exception:
        pass
    return []


# Embedding model detection (D028).
# Allowlist of known-embedding model id prefixes — name-based heuristic plus
# an explicit set so commonly-used models classify even when their name does
# not contain "embed". Strip trailing tag (e.g. ":latest", ":v1.5") before
# comparing to the allowlist.
_EMBED_NAME_RE = re.compile(r"(embed|bge-|e5-|gte-)", re.IGNORECASE)
_EMBED_ALLOWLIST = {
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    "snowflake-arctic-embed",
    "snowflake-arctic-embed2",
    "paraphrase-multilingual",
}


def _is_embedding_model(name: str) -> bool:
    base = name.split(":", 1)[0]
    if base in _EMBED_ALLOWLIST:
        return True
    return bool(_EMBED_NAME_RE.search(name))


def _ollama_model_context(model: str) -> int | None:
    """Hit Ollama /api/show for one model and return its context length.

    Ollama exposes context window under different keys depending on version
    and model family (`model_info.<arch>.context_length`, `parameters`,
    etc.). Best-effort lookup — returns None when nothing parseable is found,
    in which case the caller falls back to the node-level scalar."""
    try:
        resp = httpx.post(
            "http://localhost:11434/api/show",
            json={"name": model},
            timeout=3.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Newer Ollama returns model_info with arch-prefixed keys.
        info = data.get("model_info", {})
        for k, v in info.items():
            if k.endswith(".context_length") and isinstance(v, int) and v > 0:
                return v
        # Older responses include a `parameters` text blob — best effort.
        params = data.get("parameters", "")
        if isinstance(params, str):
            for line in params.splitlines():
                if line.strip().startswith("num_ctx"):
                    try:
                        return int(line.strip().split()[-1])
                    except ValueError:
                        pass
    except Exception:
        return None
    return None

def check_vllm_available() -> bool:
    if not VLLM_HOST:
        return False
    try:
        resp = httpx.get(f"{VLLM_HOST}{VLLM_HEALTH_PATH}", timeout=2.0, headers=_vllm_headers())
        return resp.status_code == 200
    except Exception:
        return False

def _query_vllm_models() -> tuple[list[str], int | None]:
    """Hit `VLLM_HOST/v1/models` once and return both the model id list and
    the largest `max_model_len` advertised across the returned model cards.

    vLLM exposes `max_model_len` as a non-standard extension field on each
    entry in `data[]`. Stock OpenAI servers and some proxies (notably some
    LiteLLM configurations) do not. When the field is absent on every
    entry, the second tuple element is None and the caller is expected to
    fall back to the explicit `VLLM_MAX_CONTEXT` env var or treat the
    backend as having unknown context capability.
    """
    if not VLLM_HOST:
        return [], None
    try:
        resp = httpx.get(f"{VLLM_HOST}/v1/models", timeout=2.0, headers=_vllm_headers())
        if resp.status_code != 200:
            logger.warning("vLLM /v1/models returned %s", resp.status_code)
            return [], None
        data = resp.json().get("data", [])
        models = [m["id"] for m in data]
        max_lens = [
            m.get("max_model_len")
            for m in data
            if isinstance(m.get("max_model_len"), int) and m.get("max_model_len") > 0
        ]
        max_ctx = max(max_lens) if max_lens else None
        return models, max_ctx
    except Exception as e:
        logger.warning("vLLM model listing failed: %s", e)
        return [], None


def get_vllm_models() -> list[str]:
    models, _ = _query_vllm_models()
    if models:
        logger.info("vLLM models found: %s", models)
    return models


def resolve_vllm_max_context() -> int | None:
    """Resolve the vLLM context window the agent should advertise.

    Priority (highest first):
      1. `VLLM_MAX_CONTEXT` env var — explicit operator override
      2. `max_model_len` auto-detected from `/v1/models`
      3. `None` — caller decides fallback (typically OLLAMA_NUM_CTX, or
         the agent reports an unknown-vLLM-context state)
    """
    if VLLM_MAX_CONTEXT:
        return VLLM_MAX_CONTEXT
    _, detected = _query_vllm_models()
    return detected

def check_mlx_available() -> bool:
    if not MLX_HOST:
        return False
    try:
        resp = httpx.get(f"{MLX_HOST}/", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False

def get_mlx_models() -> list[str]:
    if not MLX_HOST:
        return []
    try:
        resp = httpx.get(f"{MLX_HOST}/v1/models", timeout=2.0)
        if resp.status_code == 200:
            models = [m["id"] for m in resp.json().get("data", [])]
            logger.info("MLX models found: %s", models)
            return models
        logger.warning("MLX /v1/models returned %s", resp.status_code)
    except Exception as e:
        logger.warning("MLX model listing failed: %s", e)
    return []

def compute_safe_parallel_slots() -> int:
    """
    Detect how many concurrent Ollama inference slots this machine can safely run.

    Two constraints are computed and the minimum is taken:

      RAM  — (total_ram_gb - 4 GB OS reserve) / (largest_model_gb + kv_cache_gb)
              largest_model_gb comes from Ollama /api/tags file sizes (GGUF size ≈ RAM usage).
              kv_cache_gb is scaled from ~1 GB at 8192 tokens per OLLAMA_NUM_CTX.
              Falls back to 5 GB per slot when no models are installed yet.

      CPU  — logical_core_count // 4
              llama.cpp uses all available threads; fewer than ~4 cores per slot
              produces noticeably degraded throughput.

    The result is capped at 4 regardless of hardware to stay conservative.
    Set OLLAMA_PARALLEL_SLOTS to override completely.
    """
    # Hard override via env var
    override = os.getenv("OLLAMA_PARALLEL_SLOTS")
    if override:
        return max(1, int(override))

    # --- RAM constraint ---
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    available_ram_gb = total_ram_gb - 4.0  # reserve for OS + system overhead
    if available_ram_gb <= 0:
        return 1

    largest_model_gb = 0.0
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            for m in resp.json().get("models", []):
                size_gb = m.get("size", 0) / (1024 ** 3)
                largest_model_gb = max(largest_model_gb, size_gb)
    except Exception:
        pass

    if largest_model_gb == 0:
        largest_model_gb = 5.0  # conservative fallback: assume a mid-size 7B model

    # KV cache scales with context size — roughly 1 GB at 8192 tokens
    kv_cache_gb = OLLAMA_NUM_CTX / 8192.0
    per_slot_gb = largest_model_gb + kv_cache_gb
    ram_slots = max(1, int(available_ram_gb / per_slot_gb))

    # --- CPU constraint ---
    # Minimum 4 logical cores per slot for acceptable throughput
    cpu_slots = max(1, psutil.cpu_count(logical=True) // 4)

    slots = min(ram_slots, cpu_slots, 4)
    return slots

def _resolve_node_context_size(
    ollama_active: bool, vllm_active: bool, vllm_max_context: int | None
) -> int:
    """Resolve the `context_size` value the agent advertises to the hub.

    The hub treats this as the node's maximum context capability across all
    active backends and uses it for routing decisions (D010). When a node
    runs multiple backends with different windows, this returns the largest
    one — the per-request `num_ctx` priority chain handles per-task
    selection downstream (and is currently Ollama-only; see D015).

    Resolution order:
      - vLLM only, with a known window      → vllm_max_context
      - vLLM only, with unknown window      → OLLAMA_NUM_CTX (conservative)
      - Ollama only                         → OLLAMA_NUM_CTX
      - Both active, vLLM window known      → max(OLLAMA_NUM_CTX, vllm_max_context)
      - Both active, vLLM window unknown    → OLLAMA_NUM_CTX
      - Neither active                      → OLLAMA_NUM_CTX (legacy default)
    """
    if vllm_active and vllm_max_context:
        if ollama_active:
            return max(OLLAMA_NUM_CTX, vllm_max_context)
        return vllm_max_context
    return OLLAMA_NUM_CTX


def _image_capability_probe() -> tuple[bool, list[str], float]:
    """D064: detect mflux + installed image models + UMA-available memory.

    Returns (image_available, image_models, vram_gb). All best-effort:
    failures (mflux missing, no models installed, non-Apple-Silicon) collapse
    to (False, [], 0.0) and the agent advertises no image capability.

    Failure reasons logged at WARNING so operators can debug missing
    image-gen capability without raising the log level (D078)."""
    try:
        from lib.agent import image_driver_mflux as _img_drv  # type: ignore
        if not _img_drv.mflux_available():
            logger.warning("image probe: mflux import failed — install with `pip install mflux`")
            return (False, [], 0.0)
        installed = _img_drv.discover_installed_models()
        if not installed:
            logger.warning(
                "image probe: mflux OK but no installed models at %s — "
                "check LLMESH_IMAGE_MODELS_DIR (default ~/.llmesh/models/image), "
                "filesystem permissions (launchd-spawned agents need TCC access "
                "for external volumes), and that all registry repo_files are present.",
                _img_drv._models_dir(),
            )
            return (False, [], 0.0)
        return (True, installed, _img_drv.estimate_uma_gb())
    except Exception as exc:
        logger.warning("image probe: exception during capability probe: %r", exc)
        return (False, [], 0.0)


def gather_resources() -> dict:
    raw_ollama = get_ollama_models()
    # Split Ollama tags into chat vs embedding (D028). Embedding-only models
    # do not pollute the chat model list — `/v1/embeddings` filters on
    # `embedding_models` and `/v1/chat/completions` on the chat lists.
    ollama_chat = [m for m in raw_ollama if not _is_embedding_model(m)]
    ollama_embed = [m for m in raw_ollama if _is_embedding_model(m)]

    vllm_up = check_vllm_available()
    vllm_models, vllm_detected_ctx = _query_vllm_models() if vllm_up else ([], None)
    vllm_max_context = VLLM_MAX_CONTEXT or vllm_detected_ctx
    mlx_up = check_mlx_available()
    mlx_models = get_mlx_models() if mlx_up else []
    slots = compute_safe_parallel_slots()
    context_size = _resolve_node_context_size(
        ollama_active=len(raw_ollama) > 0,
        vllm_active=vllm_up,
        vllm_max_context=vllm_max_context,
    )

    # Per-model context map (D030). Best-effort: only populates entries we
    # can confidently determine. Hub falls back to `context_size` for any
    # missing model.
    model_context: dict[str, int] = {}
    for m in raw_ollama:
        ctx = _ollama_model_context(m)
        if ctx:
            model_context[m] = ctx
    if vllm_max_context:
        for m in vllm_models:
            model_context[m] = vllm_max_context
    # MLX does not expose per-model context; leave to fallback.

    if vllm_up:
        if vllm_max_context:
            src = "VLLM_MAX_CONTEXT" if VLLM_MAX_CONTEXT else "/v1/models max_model_len"
            logger.info("vLLM context window: %s (source: %s)", vllm_max_context, src)
        else:
            logger.warning(
                "vLLM context window: unknown — `/v1/models` did not expose "
                "`max_model_len` and `VLLM_MAX_CONTEXT` is unset. The hub will "
                "see this node as a %s-token node. Set `VLLM_MAX_CONTEXT` to "
                "fix routing decisions.",
                OLLAMA_NUM_CTX,
            )
    if ollama_embed:
        logger.info("Embedding models detected: %s", ollama_embed)
    image_available, image_models, vram_gb = _image_capability_probe()
    if image_available:
        logger.info("Image gen available — mflux + models: %s (UMA ~%sGB)",
                    image_models, vram_gb)

    return {
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024.**3), 2),
        "os_name": platform.system(),
        "ollama_available": len(raw_ollama) > 0,
        "ollama_models": ollama_chat,
        "embedding_models": ollama_embed,
        "vllm_available": vllm_up,
        "vllm_models": vllm_models,
        "mlx_available": mlx_up,
        "mlx_models": mlx_models,
        "image_available": image_available,
        "image_models": image_models,
        "vram_gb": vram_gb,
        "parallel_slots": slots,
        "streaming_capable": True,
        "context_size": context_size,
        "model_context": model_context,
        "agent_version": AGENT_VERSION,
    }

class AppState:
    def __init__(self):
        self.node_id = None
        self.node_token = None
        self.is_connected = False
        self.consecutive_errors = 0
        self.parallel_slots = 1  # updated after registration with the detected value
        self.vllm_models: list = []
        self.mlx_models: list = []
        self.embedding_models: list = []
        self.model_context: dict = {}

async def register_with_hub(state: AppState):
    resources = gather_resources()
    state.parallel_slots = resources["parallel_slots"]
    state.vllm_models = resources["vllm_models"]
    state.mlx_models = resources["mlx_models"]
    state.embedding_models = resources["embedding_models"]
    state.model_context = resources["model_context"]
    logger.info("Gathered local resources: %s", resources)
    logger.info("Parallel inference slots: %s", state.parallel_slots)

    payload = {
        "api_key": API_KEY,
        "resources": resources,
        "node_fingerprint": compute_node_fingerprint(),
    }

    try:
        async with httpx.AsyncClient() as client:
            logger.info("Registering with Hub at %s/register...", HUB_URL)
            response = await client.post(f"{HUB_URL}/register", json=payload)
            response.raise_for_status()
            node_data = response.json()
            state.node_id = node_data['node_id']
            state.node_token = node_data['node_token']
            state.is_connected = True
            state.consecutive_errors = 0
            logger.info("✅ Successfully registered! Node ID: %s", state.node_id)
            return True
    except Exception as e:
        logger.error("❌ Failed to register with Hub: %s", e)
        return False

async def heartbeat_loop(state: AppState):
    psutil.cpu_percent(interval=None)
    last_latency_ms = 0.0
    prev_ollama = None
    prev_vllm = None
    prev_mlx = None
    prev_image = None

    while True:
        # Uniform jitter on the 5s heartbeat cadence to avoid thundering-herd
        # alignment when many agents share a hub clock edge (D036 partial).
        await asyncio.sleep(5 + random.uniform(0, 1.0))

        if not state.is_connected or not state.node_id:
            continue

        try:
            cpu_load = psutil.cpu_percent(interval=None)
            not_overloaded = cpu_load < 80.0

            ollama_up = check_ollama_available() and not_overloaded
            vllm_up = check_vllm_available() and not_overloaded
            mlx_up = check_mlx_available() and not_overloaded

            # D064: image_available toggles per heartbeat — mflux import +
            # installed-model state may change while agent runs.
            image_up_h, _img_models_h, _vram_h = _image_capability_probe()

            # Re-register when any backend transitions back to available — refreshes model lists
            if prev_ollama is not None:
                if (ollama_up and not prev_ollama) or (vllm_up and not prev_vllm) or (mlx_up and not prev_mlx) or (image_up_h and not prev_image):
                    logger.info("🔄 Backend came back online — refreshing registration...")
                    await register_with_hub(state)

            prev_ollama = ollama_up
            prev_vllm = vllm_up
            prev_mlx = mlx_up
            prev_image = image_up_h
            payload = {
                "ollama_available": ollama_up,
                "vllm_available": vllm_up,
                "mlx_available": mlx_up,
                "image_available": image_up_h,
                "cpu_load": cpu_load,
                "latency_ms": round(last_latency_ms, 2)
            }

            import time
            async with httpx.AsyncClient() as client:
                start_time = time.time()
                response = await client.post(
                    f"{HUB_URL}/heartbeat/{state.node_id}",
                    json=payload,
                    headers={"Authorization": f"Bearer {state.node_token}"},
                )
                if response.status_code == 200:
                    last_latency_ms = (time.time() - start_time) * 1000
                    state.consecutive_errors = 0
                else:
                    state.consecutive_errors += 1
        except Exception as e:
            state.consecutive_errors += 1
            if state.consecutive_errors == 1:
                logger.warning("⚠️ Heartbeat failed: %s", e)

        if state.consecutive_errors >= 3:
            logger.error("🛑 Lost connection to Hub (failed 3 heartbeats). Halting operations to reconnect...")
            state.is_connected = False
            state.node_id = None
            state.node_token = None

def _normalize_tool_call_args_for_ollama(messages: list) -> list:
    """D-009 Phase 2 — coerce assistant-turn tool_calls[].function.arguments
    from JSON-string (OpenAI wire shape sent by callers) to dict (Ollama 0.30.7
    requirement for multi-turn). Non-destructive: returns a new list of dicts
    so we don't mutate caller state."""
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            out.append(m)
            continue
        tcs = m.get("tool_calls")
        if not tcs:
            out.append(m)
            continue
        new_tcs = []
        for tc in tcs:
            if not isinstance(tc, dict):
                new_tcs.append(tc); continue
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            new_tcs.append({**tc, "function": {**fn, "arguments": args}})
        out.append({**m, "tool_calls": new_tcs})
    return out


async def _run_streaming_ollama(client: httpx.AsyncClient, state: AppState, task: dict):
    """Stream Ollama inference token-by-token, batching POSTs to the hub via
    `StreamBatcher` (D041, refactor per D067).

    Was 1-POST-per-token in the original implementation — on fast Ollama
    models that meant per-second hub POST volume tracking the model's
    tokens/sec. The refactor adopts the same `StreamBatcher` adaptive
    coalescing the vLLM (D040) and MLX (D059) paths already use, with the
    same `STREAM_BATCH_FIXED=N` operator escape hatch for per-token mode.

    Ollama particulars vs the OpenAI-SSE backends:
    - Wire is JSONL (one JSON object per line), not SSE. `aiter_lines()` is
      sufficient; no event reassembly buffer needed.
    - Token field is `message.content` (chat path, `/api/chat`) or
      `response` (legacy generate path, `/api/generate`).
    - `prompt_eval_count` + `eval_count` arrive on the final `done:true`
      object. Ollama always emits usage — no HP-1 fallback needed and no
      D066-style capability flag.
    - Same invariants as vLLM/MLX preserved: CF-5 (done-chunk piggyback —
      batcher's done flush carries the last token content), CF-6 (streaming
      path never calls `/complete`; mid-stream errors emit error frame on
      `/stream` with done sentinel).
    """
    task_id = task["task_id"]
    model = task.get("model", "llama3")
    messages = task.get("messages", [])
    hub_stream_url = f"{HUB_URL}/tasks/{state.node_id}/{task_id}/stream"
    auth = {"Authorization": f"Bearer {state.node_token}"}

    endpoint = "http://localhost:11434/api/chat" if messages else "http://localhost:11434/api/generate"
    req_num_ctx = task.get("num_ctx") or OLLAMA_NUM_CTX
    payload = {"model": model, "stream": True, "options": {"num_ctx": req_num_ctx}}
    if messages:
        # D-009 Phase 2 — same un-coerce as non-stream path.
        messages = _normalize_tool_call_args_for_ollama(messages)
        payload["messages"] = messages
    else:
        payload["prompt"] = task.get("prompt", "")
    # D-009 Phase 2 — forward tools to Ollama streaming /api/chat.
    if task.get("tools"):
        payload["tools"] = task["tools"]

    logger.info("⚙️  [%s] STREAMING via ollama (%s)...", task_id, model)

    cancelled = False
    pt = ct = 0
    delta_count = 0
    captured_tool_calls: list = []
    captured_thinking_parts: list[str] = []
    # D101 — streaming text-form tool-call normalization (qwen2.5-coder leaks
    # its JSON call into streamed content; Verified 2026-06-16 ollama 0.30.8).
    # When tools were requested and the model has NOT emitted a native call, we
    # may need to suppress raw-JSON content deltas and inject a structured call
    # at stream end. Classifying heuristic: a text-form call's first
    # non-whitespace char is '{' (qwen2.5 JSON) or '<' (qwen3 XML); prose is
    # streamed live (one-token lag). `buffering`: None=undecided, True=hold,
    # False=stream live. Prose detection keeps token streaming intact when no
    # tool call is forming.
    tools_requested = bool(task.get("tools"))
    parse_buffer: list[str] = []
    buffering = None

    async def _post_chunk(chunk_text: str, *, done: bool,
                          prompt_tokens: int = 0, completion_tokens: int = 0,
                          stream_batches: int = 0, stream_final_size: int = 0,
                          tool_calls: list | None = None,
                          reasoning_content: str | None = None):
        nonlocal cancelled
        body = {
            "chunk": chunk_text, "done": done,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "stream_batches": stream_batches, "stream_final_size": stream_final_size,
        }
        if tool_calls:
            body["tool_calls"] = tool_calls
        if reasoning_content:
            body["reasoning_content"] = reasoning_content
        resp = await client.post(hub_stream_url, headers=auth, json=body)
        if resp.status_code == 410:
            cancelled = True
            raise _StreamCancelled()
        if resp.status_code != 200:
            logger.warning("⚠️  [%s] Hub returned %s during streaming", task_id, resp.status_code)

    async def flush_callback(*, chunk, done, **meta):
        # D068: forward batcher telemetry to the hub's done frame so the
        # dashboard task viewer can show batches/final_size without log scraping.
        await _post_chunk(
            chunk, done=done,
            prompt_tokens=meta.get("prompt_tokens", 0),
            completion_tokens=meta.get("completion_tokens", 0),
            stream_batches=meta.get("stream_batches", 0),
            stream_final_size=meta.get("stream_final_size", 0),
            tool_calls=meta.get("tool_calls"),
            reasoning_content=meta.get("reasoning_content"),
        )

    cfg = resolve_batcher_config()
    batcher = StreamBatcher(flush_callback, **cfg)
    batcher.start_timer()

    try:
        async with client.stream("POST", endpoint, json=payload, timeout=600.0) as resp:
            if resp.status_code != 200:
                err_text = f"Ollama stream open failed: {resp.status_code}"
                logger.error("❌ [%s] %s", task_id, err_text)
                await _post_chunk(err_text, done=False)
                await _post_chunk("", done=True)
                return

            async for line in resp.aiter_lines():
                if cancelled:
                    return
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                if data.get("done"):
                    pt = data.get("prompt_eval_count", 0) or 0
                    ct = data.get("eval_count", 0) or 0
                    # D-009/D-010 Phase 2 — tool_calls + thinking may also
                    # land on the done frame (Phase 0 verified: streaming tool
                    # call arrives in a single non-done frame, but defensively
                    # check both — some Ollama versions may differ).
                    if messages:
                        done_msg = data.get("message") or {}
                        if done_msg.get("tool_calls") and not captured_tool_calls:
                            captured_tool_calls.extend(done_msg["tool_calls"])
                        if done_msg.get("thinking"):
                            captured_thinking_parts.append(done_msg["thinking"])
                    break

                if messages:
                    msg_obj = data.get("message") or {}
                    token = msg_obj.get("content", "")
                    # D-009 Phase 2 — capture tool_calls on whichever frame
                    # Ollama emits them (Phase 0 verified: single frame, content="").
                    if msg_obj.get("tool_calls") and not captured_tool_calls:
                        captured_tool_calls.extend(msg_obj["tool_calls"])
                    # D-010 Phase 2/3 — gpt-oss harmony `thinking` may arrive
                    # incrementally; accumulate parts and concat at flush.
                    if msg_obj.get("thinking"):
                        captured_thinking_parts.append(msg_obj["thinking"])
                else:
                    token = data.get("response", "")
                if not token:
                    continue
                delta_count += 1
                # D101 — hold possible text-form tool-call content for end-of-
                # stream classification; stream prose live. Native structured
                # calls (captured_tool_calls set) carry empty content, so this
                # never engages for the qwen3-coder/devstral path.
                if tools_requested and not captured_tool_calls:
                    parse_buffer.append(token)
                    if buffering is None:
                        head = "".join(parse_buffer).lstrip()
                        if head:
                            buffering = head[0] in "{<"
                    if buffering is False:
                        # decided prose — release the held prefix and live-stream
                        await batcher.add("".join(parse_buffer))
                        parse_buffer.clear()
                    continue
                await batcher.add(token)

        # D101 — stream ended: classify any held content. If it parses as a
        # text-form tool call, inject the structured call (content suppressed,
        # same as the native qwen3-coder streaming path); otherwise it was prose
        # we held back — deliver it. extract_tool_calls is pure + fail-open.
        if parse_buffer and not captured_tool_calls:
            buffered = "".join(parse_buffer)
            parsed = extract_tool_calls(buffered)
            if parsed:
                captured_tool_calls.extend(parsed)
                logger.info("🔧 [%s] D101: parsed %d text-form tool_call(s) from streamed content", task_id, len(parsed))
            else:
                await batcher.add(buffered)
        parse_buffer = []

        # Ollama always emits usage on done; if it somehow didn't, fall through
        # to delta_count estimation for symmetry with vLLM/MLX.
        if pt == 0 and ct == 0 and delta_count > 0:
            ct = delta_count
        # D068: capture batcher telemetry just before the final flush.
        # `stats['flushes']` is the count BEFORE this flush; +1 makes it the
        # post-flush total. `current_size` is the about-to-be-flushed buffer.
        _stream_batches_total = batcher.stats['flushes'] + 1
        _stream_final_size = batcher.current_size
        flush_meta: dict = dict(
            prompt_tokens=pt, completion_tokens=ct,
            stream_batches=_stream_batches_total,
            stream_final_size=_stream_final_size,
        )
        if captured_tool_calls:
            flush_meta["tool_calls"] = captured_tool_calls
        if captured_thinking_parts:
            flush_meta["reasoning_content"] = "".join(captured_thinking_parts)
        await batcher.flush(done=True, **flush_meta)
        logger.info(
            "✅ [%s] Ollama stream complete. P:%s C:%s batches=%s final_size=%s",
            task_id, pt, ct, batcher.stats['flushes'], batcher.current_size,
        )

    except _StreamCancelled:
        logger.info("🛑 [%s] Stream cancelled by hub", task_id)
    except StreamBatcherAborted as e:
        logger.error("❌ [%s] Batcher aborted: %s", task_id, e)
        try:
            await _post_chunk(f"\n[Stream aborted: {e}]", done=False)
            await _post_chunk("", done=True, prompt_tokens=pt, completion_tokens=delta_count or ct)
        except Exception:
            pass
    except (httpx.ReadTimeout, httpx.HTTPError, asyncio.TimeoutError) as e:
        logger.error("❌ [%s] Ollama stream interrupted: %s", task_id, e)
        try:
            await _post_chunk(f"\n[Stream interrupted: {e}]", done=False)
            await _post_chunk("", done=True, prompt_tokens=pt, completion_tokens=delta_count or ct)
        except Exception:
            pass
    except Exception as e:
        logger.error("❌ [%s] Unexpected error in Ollama stream: %s", task_id, e)
        try:
            await _post_chunk(f"\n[Stream error: {e}]", done=False)
            await _post_chunk("", done=True, prompt_tokens=pt, completion_tokens=delta_count or ct)
        except Exception:
            pass
    finally:
        await batcher.close()


async def _run_streaming_vllm(client: httpx.AsyncClient, state: AppState, task: dict):
    """Stream vLLM inference token-by-token, batching POSTs to the hub (D040+D041).

    Differences from the Ollama path:
    - vLLM speaks OpenAI-shaped SSE on /v1/chat/completions, not JSONL on
      /api/chat. Parser uses `_iter_sse_events()` over `aiter_bytes()`.
    - State machine for the trailing frames: vLLM emits `finish_reason`,
      then a `usage` chunk (only when `stream_options.include_usage=true`),
      then the `[DONE]` sentinel. Done frame is held until `[DONE]`.
    - Tokens are funneled through `StreamBatcher` so fast vLLM (50-200 tok/s)
      does not flood hub `/stream` with 1 POST per token.
    - Streaming path NEVER calls `/complete` (CF-6). Mid-stream errors emit a
      final chunk POST with the error message and a done sentinel; the D018
      bridge stays inactive on this path.
    - `num_ctx` per-request override is documented as ignored (D015 limitation,
      vLLM context fixed at server startup); a one-shot warning is logged.
    """
    task_id = task["task_id"]
    model = task.get("model", "")
    messages = task.get("messages") or [{"role": "user", "content": task.get("prompt", "")}]
    hub_stream_url = f"{HUB_URL}/tasks/{state.node_id}/{task_id}/stream"
    auth = {"Authorization": f"Bearer {state.node_token}"}

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if task.get("max_tokens"):
        payload["max_tokens"] = task["max_tokens"]
    headers = _vllm_headers()

    if task.get("num_ctx"):
        logger.warning(
            "⚠️  [%s] num_ctx=%s ignored on vLLM — context fixed at server startup (D015)",
            task_id, task.get("num_ctx"),
        )

    logger.info("⚙️  [%s] STREAMING via vllm (%s)...", task_id, model)

    cancelled = False
    pt = ct = 0
    delta_count = 0
    finish_reason = None

    async def _post_chunk(chunk_text: str, *, done: bool, prompt_tokens: int = 0,
                          completion_tokens: int = 0,
                          stream_batches: int = 0, stream_final_size: int = 0):
        nonlocal cancelled
        body = {
            "chunk": chunk_text, "done": done,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "stream_batches": stream_batches, "stream_final_size": stream_final_size,
        }
        resp = await client.post(hub_stream_url, headers=auth, json=body)
        if resp.status_code == 410:
            cancelled = True
            raise _StreamCancelled()
        if resp.status_code != 200:
            logger.warning("⚠️  [%s] Hub returned %s during streaming", task_id, resp.status_code)

    async def flush_callback(*, chunk, done, **meta):
        # D068: forward batcher telemetry to the hub's done frame so the
        # dashboard task viewer can show batches/final_size without log scraping.
        await _post_chunk(
            chunk, done=done,
            prompt_tokens=meta.get("prompt_tokens", 0),
            completion_tokens=meta.get("completion_tokens", 0),
            stream_batches=meta.get("stream_batches", 0),
            stream_final_size=meta.get("stream_final_size", 0),
        )

    cfg = resolve_batcher_config()
    batcher = StreamBatcher(flush_callback, **cfg)
    batcher.start_timer()

    try:
        async with client.stream(
            "POST", f"{VLLM_HOST}/v1/chat/completions",
            json=payload, headers=headers, timeout=600.0,
        ) as resp:
            if resp.status_code != 200:
                err_bytes = await resp.aread()
                err_text = f"vLLM stream open failed: {resp.status_code} - {err_bytes.decode(errors='replace')[:200]}"
                logger.error("❌ [%s] %s", task_id, err_text)
                await _post_chunk(err_text, done=False)
                await _post_chunk("", done=True)
                return

            async for raw in _iter_sse_events(resp.aiter_bytes()):
                if cancelled:
                    return
                if raw == "[DONE]":
                    break
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if "error" in evt:
                    err_msg = json.dumps(evt["error"])
                    logger.error("❌ [%s] vLLM error frame: %s", task_id, err_msg)
                    await batcher.flush(done=False)
                    await _post_chunk(f"\n[vLLM error: {err_msg}]", done=False)
                    break
                if evt.get("usage"):
                    u = evt["usage"]
                    pt = u.get("prompt_tokens", 0) or 0
                    ct = u.get("completion_tokens", 0) or 0
                choices = evt.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    delta_count += 1
                    await batcher.add(content)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

        # Stream consumed cleanly. HP-1: estimate tokens_c from delta count
        # if vLLM did not emit a usage chunk (older vLLM, or stream_options
        # silently ignored). D066: log capability transitions one-shot, not
        # per-task.
        global _vllm_usage_chunk_supported
        usage_seen_this_stream = (pt > 0 or ct > 0)
        if not usage_seen_this_stream and delta_count > 0:
            if _vllm_usage_chunk_supported is None or _vllm_usage_chunk_supported is True:
                logger.warning(
                    "⚠️  vLLM at %s did not emit usage chunk on stream — "
                    "token accounting now uses delta-count estimation for all "
                    "future streams against this backend. Likely vLLM <0.4.1 "
                    "or a proxy stripping stream_options. Set VLLM_STREAMING_ENABLED=false "
                    "to revert to blocking + D018 bridge if exact token counts are required. (D066)",
                    VLLM_HOST,
                )
                _vllm_usage_chunk_supported = False
            ct = delta_count
        elif usage_seen_this_stream and _vllm_usage_chunk_supported is False:
            # Recovery — operator upgraded vLLM mid-session, or proxy fixed.
            logger.info("✓ vLLM at %s emitted usage chunk — capability restored (D066)", VLLM_HOST)
            _vllm_usage_chunk_supported = True
        elif usage_seen_this_stream and _vllm_usage_chunk_supported is None:
            _vllm_usage_chunk_supported = True
        # D068: capture batcher telemetry just before the final flush.
        # `stats['flushes']` is the count BEFORE this flush; +1 makes it the
        # post-flush total. `current_size` is the about-to-be-flushed buffer.
        _stream_batches_total = batcher.stats['flushes'] + 1
        _stream_final_size = batcher.current_size
        await batcher.flush(done=True, prompt_tokens=pt, completion_tokens=ct,
                            stream_batches=_stream_batches_total,
                            stream_final_size=_stream_final_size)
        logger.info(
            "✅ [%s] vLLM stream complete. P:%s C:%s finish=%s batches=%s final_size=%s",
            task_id, pt, ct, finish_reason,
            batcher.stats['flushes'], batcher.current_size,
        )

    except _StreamCancelled:
        logger.info("🛑 [%s] Stream cancelled by hub", task_id)
    except StreamBatcherAborted as e:
        logger.error("❌ [%s] Batcher aborted: %s", task_id, e)
        try:
            await _post_chunk(f"\n[Stream aborted: {e}]", done=False)
            await _post_chunk("", done=True, prompt_tokens=pt, completion_tokens=delta_count or ct)
        except Exception:
            pass
    except (httpx.ReadTimeout, httpx.HTTPError, asyncio.TimeoutError) as e:
        logger.error("❌ [%s] vLLM stream interrupted: %s", task_id, e)
        try:
            await _post_chunk(f"\n[Stream interrupted: {e}]", done=False)
            await _post_chunk("", done=True, prompt_tokens=pt, completion_tokens=delta_count or ct)
        except Exception:
            pass
    except Exception as e:
        logger.error("❌ [%s] Unexpected error in vLLM stream: %s", task_id, e)
        try:
            await _post_chunk(f"\n[Stream error: {e}]", done=False)
            await _post_chunk("", done=True, prompt_tokens=pt, completion_tokens=delta_count or ct)
        except Exception:
            pass
    finally:
        await batcher.close()


async def _run_streaming_mlx(client: httpx.AsyncClient, state: AppState, task: dict):
    """Stream MLX inference token-by-token, batching POSTs to the hub (D059).

    MLX servers in scope: `osaurus` (Apache 2.0 Swift implementation, primary
    target) and `mlx-lm.server` (Python reference). Both speak OpenAI-shaped
    SSE on `/v1/chat/completions`.

    Differences from the vLLM path (`_run_streaming_vllm`):
    - **No auth header.** Local MLX servers do not gate `/v1/chat/completions`;
      `_vllm_headers()` is not used here. If a future deployment wants bearer
      auth, lift the env-var pattern from `VLLM_API_KEY` then.
    - **No `usage` chunk emitted by osaurus** even with
      `stream_options.include_usage=true` (verified against osaurus 2026-05-28).
      Token accounting always falls through to the HP-1 estimation path
      (`completion_tokens = delta_count`, `prompt_tokens = 0`). Kept
      `stream_options` in the payload for forward compat with `mlx-lm.server`
      should it gain usage emission.
    - **`<think>...</think>` reasoning blocks** (qwen3-thinking models on
      osaurus) appear in the content stream as ordinary deltas. Pass-through.
      No agent-side stripping — the hub and clients decide whether to render
      or filter.
    - **`num_ctx` per-request override is ignored** for the same reason as
      vLLM (D015): MLX context is fixed at server startup; per-request
      `num_ctx` is not honored by the MLX backends in scope. One-shot warn
      per task when the client supplies a value.

    Same invariants preserved from D040:
    - CF-1 (`_iter_sse_events()` over `aiter_bytes()` for multi-line events).
    - CF-4 (`finish_reason` → optional `usage` → `[DONE]` tail state machine).
    - CF-5 (D045 piggyback contract — final batcher flush carries the last
      chunk content + `done=True` in a single POST; never an empty done frame
      that drops trailing tokens).
    - CF-6 (streaming path NEVER calls `/complete`; mid-stream errors emit a
      final `/stream` POST with the error text + done sentinel).
    - HP-1 (estimate `completion_tokens` from delta count when usage absent).
    """
    task_id = task["task_id"]
    model = task.get("model", "")
    messages = task.get("messages") or [{"role": "user", "content": task.get("prompt", "")}]
    hub_stream_url = f"{HUB_URL}/tasks/{state.node_id}/{task_id}/stream"
    auth = {"Authorization": f"Bearer {state.node_token}"}

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if task.get("max_tokens"):
        payload["max_tokens"] = task["max_tokens"]

    if task.get("num_ctx"):
        logger.warning(
            "⚠️  [%s] num_ctx=%s ignored on MLX — context fixed at server startup (parallels D015)",
            task_id, task.get("num_ctx"),
        )

    logger.info("⚙️  [%s] STREAMING via mlx (%s)...", task_id, model)

    cancelled = False
    pt = ct = 0
    delta_count = 0
    finish_reason = None

    async def _post_chunk(chunk_text: str, *, done: bool, prompt_tokens: int = 0,
                          completion_tokens: int = 0,
                          stream_batches: int = 0, stream_final_size: int = 0):
        nonlocal cancelled
        body = {
            "chunk": chunk_text, "done": done,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "stream_batches": stream_batches, "stream_final_size": stream_final_size,
        }
        resp = await client.post(hub_stream_url, headers=auth, json=body)
        if resp.status_code == 410:
            cancelled = True
            raise _StreamCancelled()
        if resp.status_code != 200:
            logger.warning("⚠️  [%s] Hub returned %s during streaming", task_id, resp.status_code)

    async def flush_callback(*, chunk, done, **meta):
        # D068: forward batcher telemetry to the hub's done frame so the
        # dashboard task viewer can show batches/final_size without log scraping.
        await _post_chunk(
            chunk, done=done,
            prompt_tokens=meta.get("prompt_tokens", 0),
            completion_tokens=meta.get("completion_tokens", 0),
            stream_batches=meta.get("stream_batches", 0),
            stream_final_size=meta.get("stream_final_size", 0),
        )

    cfg = resolve_batcher_config()
    batcher = StreamBatcher(flush_callback, **cfg)
    batcher.start_timer()

    try:
        async with client.stream(
            "POST", f"{MLX_HOST}/v1/chat/completions",
            json=payload, timeout=600.0,
        ) as resp:
            if resp.status_code != 200:
                err_bytes = await resp.aread()
                err_text = f"MLX stream open failed: {resp.status_code} - {err_bytes.decode(errors='replace')[:200]}"
                logger.error("❌ [%s] %s", task_id, err_text)
                await _post_chunk(err_text, done=False)
                await _post_chunk("", done=True)
                return

            async for raw in _iter_sse_events(resp.aiter_bytes()):
                if cancelled:
                    return
                if raw == "[DONE]":
                    break
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if "error" in evt:
                    err_msg = json.dumps(evt["error"])
                    logger.error("❌ [%s] MLX error frame: %s", task_id, err_msg)
                    await batcher.flush(done=False)
                    await _post_chunk(f"\n[MLX error: {err_msg}]", done=False)
                    break
                if evt.get("usage"):
                    u = evt["usage"]
                    pt = u.get("prompt_tokens", 0) or 0
                    ct = u.get("completion_tokens", 0) or 0
                choices = evt.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    delta_count += 1
                    await batcher.add(content)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

        # HP-1: osaurus emits no usage chunk → always estimate. If a future
        # MLX backend (mlx-lm.server with usage support) populates `pt`/`ct`,
        # this branch is skipped.
        if pt == 0 and ct == 0 and delta_count > 0:
            logger.info(
                "ℹ️  [%s] MLX did not emit usage chunk — estimating tokens_c=%s from delta count",
                task_id, delta_count,
            )
            ct = delta_count
        # D068: capture batcher telemetry just before the final flush.
        # `stats['flushes']` is the count BEFORE this flush; +1 makes it the
        # post-flush total. `current_size` is the about-to-be-flushed buffer.
        _stream_batches_total = batcher.stats['flushes'] + 1
        _stream_final_size = batcher.current_size
        await batcher.flush(done=True, prompt_tokens=pt, completion_tokens=ct,
                            stream_batches=_stream_batches_total,
                            stream_final_size=_stream_final_size)
        logger.info(
            "✅ [%s] MLX stream complete. P:%s C:%s finish=%s batches=%s final_size=%s",
            task_id, pt, ct, finish_reason,
            batcher.stats['flushes'], batcher.current_size,
        )

    except _StreamCancelled:
        logger.info("🛑 [%s] Stream cancelled by hub", task_id)
    except StreamBatcherAborted as e:
        logger.error("❌ [%s] Batcher aborted: %s", task_id, e)
        try:
            await _post_chunk(f"\n[Stream aborted: {e}]", done=False)
            await _post_chunk("", done=True, prompt_tokens=pt, completion_tokens=delta_count or ct)
        except Exception:
            pass
    except (httpx.ReadTimeout, httpx.HTTPError, asyncio.TimeoutError) as e:
        logger.error("❌ [%s] MLX stream interrupted: %s", task_id, e)
        try:
            await _post_chunk(f"\n[Stream interrupted: {e}]", done=False)
            await _post_chunk("", done=True, prompt_tokens=pt, completion_tokens=delta_count or ct)
        except Exception:
            pass
    except Exception as e:
        logger.error("❌ [%s] Unexpected error in MLX stream: %s", task_id, e)
        try:
            await _post_chunk(f"\n[Stream error: {e}]", done=False)
            await _post_chunk("", done=True, prompt_tokens=pt, completion_tokens=delta_count or ct)
        except Exception:
            pass
    finally:
        await batcher.close()


async def _run_embedding_ollama(client: httpx.AsyncClient, state: AppState, task: dict):
    """Run an Ollama embedding task. Tries the batch-capable `/api/embed`
    endpoint first (Ollama 0.2+); falls back to per-input `/api/embeddings`
    when the batch endpoint is missing (404)."""
    task_id = task["task_id"]
    model = task.get("model", "nomic-embed-text")
    payload = task.get("payload", {}) or {}
    inputs = payload.get("input") or []
    if not isinstance(inputs, list):
        inputs = [str(inputs)]

    auth = {"Authorization": f"Bearer {state.node_token}"}
    submit_url = f"{HUB_URL}/tasks/{state.node_id}/complete/{task_id}"

    logger.info("⚙️  [%s] EMBED via ollama (%s, batch=%s)...", task_id, model, len(inputs))
    try:
        # Preferred path: batch /api/embed.
        embeddings: list[list[float]] = []
        prompt_tokens = 0
        resp = await client.post(
            "http://localhost:11434/api/embed",
            json={"model": model, "input": inputs},
            timeout=120.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get("embeddings") or []
            prompt_tokens = data.get("prompt_eval_count", 0) or 0
        elif resp.status_code == 404:
            # Older Ollama: per-input /api/embeddings.
            for s in inputs:
                r = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": model, "prompt": s},
                    timeout=120.0,
                )
                if r.status_code != 200:
                    raise RuntimeError(f"/api/embeddings returned {r.status_code}: {r.text}")
                rd = r.json()
                embeddings.append(rd.get("embedding") or [])
                prompt_tokens += rd.get("prompt_eval_count", 0) or 0
        else:
            raise RuntimeError(f"/api/embed returned {resp.status_code}: {resp.text}")

        if len(embeddings) != len(inputs) or any(not isinstance(v, list) for v in embeddings):
            raise RuntimeError(f"backend returned malformed embeddings for {len(inputs)} inputs")

        await client.post(
            submit_url,
            json={
                "embeddings": embeddings,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
            },
            headers=auth,
        )
        logger.info(
            "✅ [%s] Embedding done. Tokens P:%s dim=%s",
            task_id, prompt_tokens, len(embeddings[0]) if embeddings else 0,
        )
    except httpx.ReadTimeout:
        await client.post(
            submit_url,
            json={"output": "Failed: ollama embedding timed out", "error": True},
            headers=auth,
        )
    except Exception as e:
        await client.post(
            submit_url,
            json={"output": f"Failed to compute embedding via ollama: {e}", "error": True},
            headers=auth,
        )


async def _run_image_mflux(client: httpx.AsyncClient, state: AppState, task: dict):
    """D064: execute one image-gen task via the mflux in-process driver.

    Submits the list of base64 PNGs via the standard /complete endpoint — no
    streaming for image gen v1 (single result batch). The driver runs the
    inference in an executor thread so the agent poll loop and heartbeat
    stay responsive."""
    task_id = task["task_id"]
    model = task.get("model", "")
    payload = task.get("payload") or {}
    auth = {"Authorization": f"Bearer {state.node_token}"}
    submit_url = f"{HUB_URL}/tasks/{state.node_id}/complete/{task_id}"

    logger.info("⚙️  [%s] IMAGE via mflux (%s, n=%s, size=%s, steps=%s)...",
                task_id, model, payload.get("n", 1), payload.get("size"),
                payload.get("steps"))

    try:
        from lib.agent import image_driver_mflux as _img_drv
        images_b64 = await _img_drv.run_image_task(payload, model)
        await client.post(
            submit_url,
            json={"output": images_b64, "prompt_tokens": 0,
                  "completion_tokens": len(images_b64)},
            headers=auth,
        )
        logger.info("✅ [%s] IMAGE done. count=%s", task_id, len(images_b64))
    except Exception as e:
        logger.exception("❌ [%s] IMAGE failed", task_id)
        await client.post(
            submit_url,
            json={"output": f"Image generation failed: {e!r}", "error": True},
            headers=auth,
        )


async def _run_single_task(client: httpx.AsyncClient, state: AppState, task: dict):
    """Execute one inference task against the appropriate backend and submit the result to the hub."""
    task_id = task['task_id']
    model_to_use = task.get("model", "llama3")
    messages = task.get("messages", [])
    kind = task.get("kind", "chat")

    # Embedding tasks dispatch to a dedicated path — they do not share the
    # chat-completions code (different request shape, different result type,
    # no streaming; embedding-on-vLLM/MLX deferred per D028).
    if kind == "embedding":
        await _run_embedding_ollama(client, state, task)
        return

    # D064: image generation tasks dispatch to the mflux in-process driver.
    # Driver returns list[str] of base64 PNGs; agent submits via /complete
    # (same path as blocking inference — no streaming for image gen v1).
    if kind == "image":
        await _run_image_mflux(client, state, task)
        return

    # Determine which local backend has this model (chat path).
    if model_to_use in state.vllm_models:
        backend = "vllm"
        base_url = VLLM_HOST
    elif model_to_use in state.mlx_models:
        backend = "mlx"
        base_url = MLX_HOST
    else:
        backend = "ollama"
        base_url = None

    logger.info("📥 Task %s: model=%r backend=%r", task_id, model_to_use, backend)

    # Streaming dispatch — Ollama always, vLLM behind VLLM_STREAMING_ENABLED
    # (D040, default ON via D044), MLX behind MLX_STREAMING_ENABLED (D059,
    # default OFF until LAB-003 graduates). Disabled paths fall back to
    # blocking + D018 bridge.
    if task.get("stream"):
        if backend == "ollama":
            await _run_streaming_ollama(client, state, task)
            return
        elif backend == "vllm" and VLLM_STREAMING_ENABLED:
            await _run_streaming_vllm(client, state, task)
            return
        elif backend == "mlx" and MLX_STREAMING_ENABLED:
            await _run_streaming_mlx(client, state, task)
            return
        else:
            if backend == "vllm":
                reason = f"vllm streaming disabled (VLLM_STREAMING_ENABLED={VLLM_STREAMING_ENABLED})"
            elif backend == "mlx":
                reason = f"mlx streaming disabled (MLX_STREAMING_ENABLED={MLX_STREAMING_ENABLED})"
            else:
                reason = f"{backend} streaming not implemented"
            logger.warning(
                "⚠️  [%s] Streaming requested but %s — falling back to blocking inference",
                task_id, reason,
            )

    output_text = ""
    p_tokens = c_tokens = 0
    is_error = False
    tool_calls: list = []
    reasoning_content = ""

    try:
        if backend in ("vllm", "mlx"):
            # OpenAI-compatible path — both vLLM and MLX use /v1/chat/completions
            if not messages:
                messages = [{"role": "user", "content": task.get("prompt", "")}]
            logger.info("⚙️  [%s] CHAT via %s (%s)...", task_id, backend, model_to_use)
            req_body = {
                "model": model_to_use,
                "messages": messages,
                "stream": False,
            }
            if task.get("max_tokens"):
                req_body["max_tokens"] = task["max_tokens"]
            # D099 — forward tools to the OpenAI-compatible backend. Both vLLM
            # and MLX (osaurus) honor /v1/chat/completions `tools`. Previously
            # dropped here, so tool-using models silently made 0 calls when they
            # landed on a non-Ollama node (latent until a tool model is served
            # off Ollama). tool_choice stays hub-enforced (filtered into `tools`
            # for named-fn/none + post-validated for required/named-fn), matching
            # the Ollama path's contract — so it is intentionally not forwarded.
            if task.get("tools"):
                req_body["tools"] = task["tools"]
            resp = await client.post(
                f"{base_url}/v1/chat/completions",
                json=req_body,
                headers=_vllm_headers() if backend == "vllm" else {},
            )
            if resp.status_code == 200:
                data = resp.json()
                resp_msg = data["choices"][0]["message"]
                output_text = resp_msg.get("content") or ""
                # D099 — surface tool_calls + reasoning_content from the
                # backend so the hub can shape them (same fields the Ollama
                # branch populates). vLLM emits OpenAI-shape tool_calls;
                # reasoning_content present on reasoning models (DeepSeek-style).
                tool_calls = resp_msg.get("tool_calls") or []
                # D101 follow-up — vLLM-served qwen2.5-coder-7b leaks its tool
                # call as TEXT in content (json/<json>/<response> wrappers) with
                # tool_calls=[] and finish_reason=stop, because the prod vLLM is
                # an older build run WITHOUT a working tool-call parser (hermes
                # does not parse the Coder variant's output — vLLM #29192). The
                # ollama path (D101) and the vLLM *streaming* path already
                # normalize; this non-stream branch was the lone hole. Same
                # guard + parser: only fires when tools were requested, NO native
                # structured call present, content non-empty — so native vLLM
                # tool_calls pass through untouched and the fix is idempotent if
                # the server is ever upgraded to parse natively. Baseline 3/3
                # text-leak verified live 2026-06-17 against mesh prod.
                if should_normalize(tool_calls, output_text, bool(task.get("tools"))):
                    parsed = extract_tool_calls(output_text)
                    if parsed:
                        tool_calls = parsed
                        output_text = ""  # mirror proxy _normalize (content=None)
                        logger.info("🔧 [%s] D101: parsed %d text-form tool_call(s) from %s content", task_id, len(parsed), backend)
                reasoning_content = resp_msg.get("reasoning_content") or ""
                usage = data.get("usage", {})
                p_tokens = usage.get("prompt_tokens", 0)
                c_tokens = usage.get("completion_tokens", 0)
                logger.info("✅ [%s] Done. Tokens P:%s C:%s tool_calls=%d", task_id, p_tokens, c_tokens, len(tool_calls))
            else:
                output_text = f"Error from {backend}: {resp.status_code} - {resp.text}"
                is_error = True
                logger.error("❌ [%s] %s", task_id, output_text)
        elif messages:
            logger.info("⚙️  [%s] CHAT via ollama (%s)...", task_id, model_to_use)
            # D-009 Phase 2 — Ollama 0.30.7 requires assistant-turn
            # tool_calls[].function.arguments as a DICT, not a JSON-string.
            # OpenAI wire shape uses string. Un-coerce here before forwarding
            # so multi-turn tool flows round-trip cleanly. Verified by Phase 0
            # follow-up probe: string args produce empty assistant content; dict
            # args produce the expected natural-language answer.
            messages = _normalize_tool_call_args_for_ollama(messages)
            ollama_body = {
                "model": model_to_use,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": task.get("num_ctx") or OLLAMA_NUM_CTX},
            }
            # D-009 Phase 2 — forward tools to Ollama /api/chat. Ollama returns
            # plain message.content (no tool_calls field) when the model is not
            # tool-trained; hub treats absent tool_calls as "no call emitted"
            # without an allowlist gate.
            if task.get("tools"):
                ollama_body["tools"] = task["tools"]
            ollama_resp = await client.post("http://localhost:11434/api/chat", json=ollama_body)
            if ollama_resp.status_code == 200:
                o_data = ollama_resp.json()
                msg = o_data.get("message") or {}
                # Defensive: backend may return null content; coerce to ""
                # so the hub's stored result is a real string (D025 sibling).
                output_text = msg.get("content") or ""
                tool_calls = msg.get("tool_calls") or []
                # D101 — qwen-family text-form tool calls. ollama leaves
                # qwen2.5-coder calls as TEXT in content (JSON object) with
                # tool_calls=[]; strict clients then no-op. Parse + inject the
                # structured call so they execute. Guards (R4): only when tools
                # were requested, NO native structured call present, content
                # non-empty — protects the native path (qwen3-coder/devstral
                # emit structured tool_calls; Verified 2026-06-16 ollama 0.30.8)
                # and is idempotent. Parser is fail-open (never raises).
                if should_normalize(tool_calls, output_text, bool(task.get("tools"))):
                    parsed = extract_tool_calls(output_text)
                    if parsed:
                        tool_calls = parsed
                        output_text = ""  # mirror proxy _normalize (content=None)
                        logger.info("🔧 [%s] D101: parsed %d text-form tool_call(s) from content", task_id, len(parsed))
                # D-010 Phase 3 — gpt-oss harmony `analysis` channel arrives
                # in message.thinking on Ollama 0.30.7 (Phase 0 verified).
                reasoning_content = msg.get("thinking") or ""
                p_tokens = o_data.get("prompt_eval_count", 0) or 0
                c_tokens = o_data.get("eval_count", 0) or 0
                logger.info("✅ [%s] Done. Tokens P:%s C:%s tool_calls=%d", task_id, p_tokens, c_tokens, len(tool_calls))
            else:
                output_text = f"Error from ollama chat: {ollama_resp.status_code} - {ollama_resp.text}"
                is_error = True
                logger.error("❌ [%s] %s", task_id, output_text)
        else:
            logger.info("⚙️  [%s] RAW via ollama (%s)...", task_id, model_to_use)
            ollama_resp = await client.post("http://localhost:11434/api/generate", json={
                "model": model_to_use,
                "prompt": task.get("prompt", ""),
                "stream": False,
                "options": {"num_ctx": task.get("num_ctx") or OLLAMA_NUM_CTX},
            })
            if ollama_resp.status_code == 200:
                o_data = ollama_resp.json()
                output_text = o_data.get("response") or ""
                p_tokens = o_data.get("prompt_eval_count", 0) or 0
                c_tokens = o_data.get("eval_count", 0) or 0
                logger.info("✅ [%s] Done. Tokens P:%s C:%s", task_id, p_tokens, c_tokens)
            else:
                output_text = f"Error from ollama generate: {ollama_resp.status_code} - {ollama_resp.text}"
                is_error = True
                logger.error("❌ [%s] %s", task_id, output_text)

    except httpx.ReadTimeout:
        output_text = f"Failed: {backend} inference timed out after 600 seconds."
        is_error = True
        logger.error("❌ [%s] %s", task_id, output_text)
    except Exception as e:
        output_text = f"Failed to connect to {backend}: {str(e)}"
        is_error = True
        logger.error("❌ [%s] %s", task_id, output_text)

    result_payload = {"output": output_text, "prompt_tokens": p_tokens, "completion_tokens": c_tokens}
    if tool_calls:
        result_payload["tool_calls"] = tool_calls
    if reasoning_content:
        result_payload["reasoning_content"] = reasoning_content
    if is_error:
        result_payload["error"] = True
    submit_resp = await client.post(
        f"{HUB_URL}/tasks/{state.node_id}/complete/{task_id}",
        json=result_payload,
        headers={"Authorization": f"Bearer {state.node_token}"},
    )
    if submit_resp.status_code == 200:
        logger.info("⤴️  [%s] Result submitted.", task_id)

async def task_polling_loop(state: AppState):
    timeout = httpx.Timeout(600.0, connect=5.0)

    while True:
        if not state.is_connected or not state.node_id:
            await asyncio.sleep(5)
            continue

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(
                    f"{HUB_URL}/tasks/{state.node_id}/pending",
                    headers={"Authorization": f"Bearer {state.node_token}"},
                )
                if resp.status_code == 200:
                    pending = resp.json()
                    if pending:
                        # Take up to parallel_slots tasks and run them concurrently
                        batch = pending[:state.parallel_slots]
                        await asyncio.gather(
                            *[_run_single_task(client, state, t) for t in batch],
                            return_exceptions=True,
                        )
                        continue  # re-poll immediately after completing a batch
        except Exception:
            pass  # heartbeat loop handles disconnect detection

        # Uniform jitter on the idle-poll cadence — same rationale as heartbeat.
        await asyncio.sleep(5 + random.uniform(0, 1.0))

async def connection_manager(state: AppState):
    """Monitors connection state and re-registers when disconnected.

    Before attempting registration, verifies that at least one inference
    backend is serving models.  If no models are found (e.g. Ollama has
    not started yet), retries with exponential backoff up to
    ``_MAX_BACKEND_RETRIES`` times before exiting.  The process supervisor
    (launchd KeepAlive, systemd Restart=always) is expected to respawn it.
    """
    _MAX_BACKEND_RETRIES = 10
    backend_attempt = 0

    while True:
        if not state.is_connected:
            # Gate: require at least one backend with models before registering.
            resources = gather_resources()
            all_models = (
                resources["ollama_models"]
                + resources["vllm_models"]
                + resources["mlx_models"]
            )

            if not all_models:
                backend_attempt += 1
                if backend_attempt > _MAX_BACKEND_RETRIES:
                    logger.error(
                        "No inference backends found after %s attempts. "
                        "Exiting — process supervisor will respawn.",
                        _MAX_BACKEND_RETRIES,
                    )
                    sys.exit(1)
                delay = min(5 * (2 ** (backend_attempt - 1)), 60)
                logger.warning(
                    "No models found on any backend (attempt %s/%s). Retrying in %ss...",
                    backend_attempt, _MAX_BACKEND_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                continue

            backend_attempt = 0  # reset on success for future reconnect cycles

            logger.info("🔄 Attempting to connect/reconnect to Hub...")
            success = await register_with_hub(state)
            if not success:
                logger.warning("⏳ Hub unreachable. Retrying in 60 seconds...")
                await asyncio.sleep(60)
            else:
                logger.info("🚀 Node ready. Parallel slots: %s", state.parallel_slots)
        await asyncio.sleep(5)

async def main():
    state = AppState()

    await asyncio.gather(
        connection_manager(state),
        heartbeat_loop(state),
        task_polling_loop(state)
    )

if __name__ == "__main__":
    # D080: split log streams so launchd's StandardOutPath gets normal
    # operational logs (INFO/DEBUG) and StandardErrorPath gets only WARNING+.
    # Default logging.basicConfig sends everything to stderr, leaving the
    # .log file empty and trapping operators who tail it expecting output.
    _log_level = os.getenv("LLMESH_LOG_LEVEL", "INFO").upper()
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    _stdout_h = logging.StreamHandler(sys.stdout)
    _stdout_h.setLevel(logging.DEBUG)
    _stdout_h.addFilter(lambda r: r.levelno < logging.WARNING)
    _stdout_h.setFormatter(_fmt)

    _stderr_h = logging.StreamHandler(sys.stderr)
    _stderr_h.setLevel(logging.WARNING)
    _stderr_h.setFormatter(_fmt)

    _root = logging.getLogger()
    _root.setLevel(_log_level)
    _root.handlers.clear()
    _root.addHandler(_stdout_h)
    _root.addHandler(_stderr_h)

    asyncio.run(main())
