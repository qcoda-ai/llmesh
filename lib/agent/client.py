import hashlib
import json
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

# Line-buffer stdout/stderr so the agent's `print()` diagnostics appear
# immediately in journalctl/docker logs/etc. without requiring the operator
# to remember `PYTHONUNBUFFERED=1`. Python defaults to block buffering when
# stdout is not a TTY (e.g. when launched by systemd, docker, supervisord),
# which silently swallows minutes of registration/heartbeat/vLLM-detection
# output and makes diagnosis impossible. See decisions.md D017.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

load_dotenv()

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
MLX_HOST = os.getenv("MLX_HOST")              # None unless explicitly set


def _vllm_headers() -> dict:
    """Bearer-auth header for vLLM-path requests, or empty if no key set."""
    return {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}

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
        print(
            f"WARNING: could not persist node salt to {salt_path} ({exc}). "
            f"Falling back to an in-memory salt for this process; the node "
            f"fingerprint will change on restart. Set LLMESH_STATE_DIR to a "
            f"writable directory to fix."
        )
        return secrets.token_hex(16)


def compute_node_fingerprint() -> str:
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
            print(f"vLLM /v1/models returned {resp.status_code}")
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
        print(f"vLLM model listing failed: {e}")
        return [], None


def get_vllm_models() -> list[str]:
    models, _ = _query_vllm_models()
    if models:
        print(f"vLLM models found: {models}")
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
            print(f"MLX models found: {models}")
            return models
        print(f"MLX /v1/models returned {resp.status_code}")
    except Exception as e:
        print(f"MLX model listing failed: {e}")
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
            print(f"vLLM context window: {vllm_max_context} (source: {src})")
        else:
            print(
                "vLLM context window: unknown — `/v1/models` did not expose "
                "`max_model_len` and `VLLM_MAX_CONTEXT` is unset. The hub will "
                f"see this node as a {OLLAMA_NUM_CTX}-token node. Set "
                "`VLLM_MAX_CONTEXT` to fix routing decisions."
            )
    if ollama_embed:
        print(f"Embedding models detected: {ollama_embed}")
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
        "parallel_slots": slots,
        "streaming_capable": True,
        "context_size": context_size,
        "model_context": model_context,
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
    print(f"Gathered local resources: {resources}")
    print(f"Parallel inference slots: {state.parallel_slots}")

    payload = {
        "api_key": API_KEY,
        "resources": resources,
        "node_fingerprint": compute_node_fingerprint(),
    }

    try:
        async with httpx.AsyncClient() as client:
            print(f"Registering with Hub at {HUB_URL}/register...")
            response = await client.post(f"{HUB_URL}/register", json=payload)
            response.raise_for_status()
            node_data = response.json()
            state.node_id = node_data['node_id']
            state.node_token = node_data['node_token']
            state.is_connected = True
            state.consecutive_errors = 0
            print(f"✅ Successfully registered! Node ID: {state.node_id}")
            return True
    except Exception as e:
        print(f"❌ Failed to register with Hub: {e}")
        return False

async def heartbeat_loop(state: AppState):
    psutil.cpu_percent(interval=None)
    last_latency_ms = 0.0
    prev_ollama = None
    prev_vllm = None
    prev_mlx = None

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

            # Re-register when any backend transitions back to available — refreshes model lists
            if prev_ollama is not None:
                if (ollama_up and not prev_ollama) or (vllm_up and not prev_vllm) or (mlx_up and not prev_mlx):
                    print("🔄 Backend came back online — refreshing registration...")
                    await register_with_hub(state)

            prev_ollama = ollama_up
            prev_vllm = vllm_up
            prev_mlx = mlx_up

            payload = {
                "ollama_available": ollama_up,
                "vllm_available": vllm_up,
                "mlx_available": mlx_up,
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
                print(f"⚠️ Heartbeat failed: {e}")

        if state.consecutive_errors >= 3:
            print("🛑 Lost connection to Hub (failed 3 heartbeats). Halting operations to reconnect...")
            state.is_connected = False
            state.node_id = None
            state.node_token = None

async def _run_streaming_ollama(client: httpx.AsyncClient, state: AppState, task: dict):
    """Stream Ollama inference token-by-token, posting each chunk to the hub."""
    task_id = task["task_id"]
    model = task.get("model", "llama3")
    messages = task.get("messages", [])
    hub_stream_url = f"{HUB_URL}/tasks/{state.node_id}/{task_id}/stream"
    auth = {"Authorization": f"Bearer {state.node_token}"}

    endpoint = "http://localhost:11434/api/chat" if messages else "http://localhost:11434/api/generate"
    req_num_ctx = task.get("num_ctx") or OLLAMA_NUM_CTX
    payload = {"model": model, "stream": True, "options": {"num_ctx": req_num_ctx}}
    if messages:
        payload["messages"] = messages
    else:
        payload["prompt"] = task.get("prompt", "")

    print(f"⚙️  [{task_id}] STREAMING via ollama ({model})...")
    try:
        async with client.stream("POST", endpoint, json=payload, timeout=600.0) as resp:
            if resp.status_code != 200:
                error_text = f"Error from ollama streaming: {resp.status_code}"
                print(f"❌ [{task_id}] {error_text}")
                await client.post(
                    f"{HUB_URL}/tasks/{state.node_id}/complete/{task_id}",
                    json={"output": error_text, "error": True}, headers=auth,
                )
                return

            p_tokens = c_tokens = 0
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                if data.get("done"):
                    p_tokens = data.get("prompt_eval_count", 0)
                    c_tokens = data.get("eval_count", 0)
                    chunk_resp = await client.post(
                        hub_stream_url, headers=auth,
                        json={"chunk": "", "done": True,
                              "prompt_tokens": p_tokens, "completion_tokens": c_tokens},
                    )
                    if chunk_resp.status_code == 410:
                        print(f"🛑 [{task_id}] Stream cancelled by hub")
                    elif chunk_resp.status_code != 200:
                        print(f"❌ [{task_id}] Error posting final chunk: {chunk_resp.status_code}")
                    else:
                        print(f"✅ [{task_id}] Stream complete. Tokens P:{p_tokens} C:{c_tokens}")
                    return

                token = data.get("message", {}).get("content", "") if messages else data.get("response", "")
                if not token:
                    continue

                chunk_resp = await client.post(
                    hub_stream_url, headers=auth,
                    json={"chunk": token, "done": False},
                )
                if chunk_resp.status_code == 410:
                    print(f"🛑 [{task_id}] Stream cancelled by hub — aborting")
                    return
                elif chunk_resp.status_code != 200:
                    # Print first error only to avoid spamming the console
                    print(f"⚠️  [{task_id}] Hub returned {chunk_resp.status_code} during streaming")

    except httpx.ReadTimeout:
        await client.post(
            f"{HUB_URL}/tasks/{state.node_id}/complete/{task_id}",
            json={"output": f"Failed: ollama streaming timed out after 600 seconds.", "error": True}, headers=auth,
        )
    except Exception as e:
        await client.post(
            f"{HUB_URL}/tasks/{state.node_id}/complete/{task_id}",
            json={"output": f"Failed to connect to ollama: {e}", "error": True}, headers=auth,
        )


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

    print(f"⚙️  [{task_id}] EMBED via ollama ({model}, batch={len(inputs)})...")
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
        print(f"✅ [{task_id}] Embedding done. Tokens P:{prompt_tokens} dim={len(embeddings[0]) if embeddings else 0}")
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


async def _run_single_task(client: httpx.AsyncClient, state: AppState, task: dict):
    """Execute one inference task against the appropriate backend and submit the result to the hub."""
    task_id = task['task_id']
    model_to_use = task.get("model", "llama3")
    messages = task.get("messages", [])
    kind = task.get("kind", "chat")

    # Embedding tasks dispatch to a dedicated path — they do not share the
    # chat-completions code (different request shape, different result type,
    # no streaming, vLLM/MLX deferred per D028).
    if kind == "embedding":
        await _run_embedding_ollama(client, state, task)
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

    print(f"📥 Task {task_id}: model='{model_to_use}' backend='{backend}'")

    # Streaming dispatch — Ollama only; vLLM and mlx fall back to blocking (beta)
    if task.get("stream"):
        if backend == "ollama":
            await _run_streaming_ollama(client, state, task)
            return
        else:
            print(f"⚠️  [{task_id}] Streaming requested but {backend} is beta — falling back to blocking inference")

    output_text = ""
    p_tokens = c_tokens = 0
    is_error = False

    try:
        if backend in ("vllm", "mlx"):
            # OpenAI-compatible path — both vLLM and MLX use /v1/chat/completions
            if not messages:
                messages = [{"role": "user", "content": task.get("prompt", "")}]
            print(f"⚙️  [{task_id}] CHAT via {backend} ({model_to_use})...")
            resp = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model_to_use,
                    "messages": messages,
                    "stream": False,
                },
                headers=_vllm_headers() if backend == "vllm" else {},
            )
            if resp.status_code == 200:
                data = resp.json()
                output_text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                p_tokens = usage.get("prompt_tokens", 0)
                c_tokens = usage.get("completion_tokens", 0)
                print(f"✅ [{task_id}] Done. Tokens P:{p_tokens} C:{c_tokens}")
            else:
                output_text = f"Error from {backend}: {resp.status_code} - {resp.text}"
                is_error = True
                print(f"❌ [{task_id}] {output_text}")
        elif messages:
            print(f"⚙️  [{task_id}] CHAT via ollama ({model_to_use})...")
            ollama_resp = await client.post("http://localhost:11434/api/chat", json={
                "model": model_to_use,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": task.get("num_ctx") or OLLAMA_NUM_CTX},
            })
            if ollama_resp.status_code == 200:
                o_data = ollama_resp.json()
                # Defensive: backend may return null content; coerce to ""
                # so the hub's stored result is a real string (D025 sibling).
                output_text = (o_data.get("message", {}) or {}).get("content") or ""
                p_tokens = o_data.get("prompt_eval_count", 0) or 0
                c_tokens = o_data.get("eval_count", 0) or 0
                print(f"✅ [{task_id}] Done. Tokens P:{p_tokens} C:{c_tokens}")
            else:
                output_text = f"Error from ollama chat: {ollama_resp.status_code} - {ollama_resp.text}"
                is_error = True
                print(f"❌ [{task_id}] {output_text}")
        else:
            print(f"⚙️  [{task_id}] RAW via ollama ({model_to_use})...")
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
                print(f"✅ [{task_id}] Done. Tokens P:{p_tokens} C:{c_tokens}")
            else:
                output_text = f"Error from ollama generate: {ollama_resp.status_code} - {ollama_resp.text}"
                is_error = True
                print(f"❌ [{task_id}] {output_text}")

    except httpx.ReadTimeout:
        output_text = f"Failed: {backend} inference timed out after 600 seconds."
        is_error = True
        print(f"❌ [{task_id}] {output_text}")
    except Exception as e:
        output_text = f"Failed to connect to {backend}: {str(e)}"
        is_error = True
        print(f"❌ [{task_id}] {output_text}")

    result_payload = {"output": output_text, "prompt_tokens": p_tokens, "completion_tokens": c_tokens}
    if is_error:
        result_payload["error"] = True
    submit_resp = await client.post(
        f"{HUB_URL}/tasks/{state.node_id}/complete/{task_id}",
        json=result_payload,
        headers={"Authorization": f"Bearer {state.node_token}"},
    )
    if submit_resp.status_code == 200:
        print(f"⤴️  [{task_id}] Result submitted.")

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
                    print(
                        f"No inference backends found after {_MAX_BACKEND_RETRIES} "
                        "attempts. Exiting — process supervisor will respawn."
                    )
                    sys.exit(1)
                delay = min(5 * (2 ** (backend_attempt - 1)), 60)
                print(
                    f"No models found on any backend "
                    f"(attempt {backend_attempt}/{_MAX_BACKEND_RETRIES}). "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
                continue

            backend_attempt = 0  # reset on success for future reconnect cycles

            print("🔄 Attempting to connect/reconnect to Hub...")
            success = await register_with_hub(state)
            if not success:
                print("⏳ Hub unreachable. Retrying in 60 seconds...")
                await asyncio.sleep(60)
            else:
                print(f"🚀 Node ready. Parallel slots: {state.parallel_slots}")
        await asyncio.sleep(5)

async def main():
    state = AppState()

    await asyncio.gather(
        connection_manager(state),
        heartbeat_loop(state),
        task_polling_loop(state)
    )

if __name__ == "__main__":
    asyncio.run(main())
