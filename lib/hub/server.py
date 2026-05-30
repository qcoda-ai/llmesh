import logging
import time
import uuid
import os
import secrets
from collections import OrderedDict
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Request, Form, Response, Cookie, Header, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.templating import Jinja2Templates
from . import config as _config  # noqa: F401 — must import first to populate env vars before sessions/metrics read them
from .models import (
    RegistrationRequest, Node, HeartbeatRequest, RegistrationResponse,
    EmbeddingsRequest, TaskKind,
    ImageGenerationRequest, ImageGenerationResponse,
)
import asyncio
import json
from . import storage
from . import tasks
from . import task_store
from . import node_store
from . import metrics
from .metrics import upsert_node_registry, prune_old_metrics
from .sessions import get_session_store, SESSION_MAX_TURNS, SESSION_MEMORY_MODE
from . import compressor

logger = logging.getLogger("llmesh.hub")


# ---------------------------------------------------------------------------
# Dashboard session tokens
# ---------------------------------------------------------------------------
# In-memory mapping of opaque session token → owner_id. The browser cookie
# carries the token (never the owner_id directly), so leaking an owner_id via
# any API response does not allow cookie forgery. Tokens are created at
# /login and destroyed at /logout. Store is in-memory: sessions do not
# survive hub restart — users must re-authenticate, matching the behavior
# of node tokens (see D004).
SESSION_COOKIE_NAME = "llmesh_session"
_session_tokens: dict[str, str] = {}  # token → owner_id


def _resolve_session(token: str | None) -> str | None:
    """Return the owner_id bound to this session token, or None if invalid."""
    if not token:
        return None
    return _session_tokens.get(token)


# ---------------------------------------------------------------------------
# CSRF (D055) — double-submit cookie pattern
# ---------------------------------------------------------------------------
# Browser-driven POST forms (`/login`, `/dashboard/request_inference`) require
# a per-form synchronizer token that matches an httponly cookie set on the
# preceding GET. Defense-in-depth alongside the existing `samesite="Strict"`
# session cookie. Bearer-token API endpoints (`/v1/*`, `/register`, etc) do
# not need CSRF — they do not authenticate via cookies and are not reachable
# from a cross-origin HTML form with valid credentials.
CSRF_COOKIE_NAME = "llmesh_csrf"


def _require_csrf(form_token: str | None, cookie_token: str | None) -> None:
    """Reject the request if the submitted form token does not match the
    cookie token. Timing-safe compare. Called at the top of every POST
    handler that processes a browser form."""
    if (
        not form_token
        or not cookie_token
        or not secrets.compare_digest(form_token, cookie_token)
    ):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
RATE_LIMIT_INFERENCE = os.getenv("RATE_LIMIT_INFERENCE", "60/minute")
# D064: image-gen caps. Hub-side validation before routing.
MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "4"))
MAX_IMAGE_PROMPT_BYTES = int(os.getenv("MAX_IMAGE_PROMPT_BYTES", "8192"))  # ~2k tokens
IMAGE_TASK_TIMEOUT_S = float(os.getenv("IMAGE_TASK_TIMEOUT_S", "300.0"))
RATE_LIMIT_LIST      = os.getenv("RATE_LIMIT_LIST",      "120/minute")
RATE_LIMIT_REGISTER  = os.getenv("RATE_LIMIT_REGISTER",  "20/minute")
RATE_LIMIT_LOGIN     = os.getenv("RATE_LIMIT_LOGIN",     "10/minute")
RATE_LIMIT_HEARTBEAT = os.getenv("RATE_LIMIT_HEARTBEAT", "30/minute")

# Per-owner TTL cache for GET /v1/models. The endpoint scans the node registry
# on every call; under rapid back-to-back client traffic (e.g. qc_eval harness)
# this keeps it off the hot path. Set MODELS_CACHE_TTL=0 to disable. See D027.
# D056: bounded LRU. Without a cap the dict grows one entry per owner_id ever
# queried; a long-running hub with many owners would leak the entries
# indefinitely. Mirror the `_dashboard_cache` pattern from metrics.py.
MODELS_CACHE_TTL = float(os.getenv("MODELS_CACHE_TTL", "10.0"))
MODELS_CACHE_MAX = int(os.getenv("MODELS_CACHE_MAX", "64"))
_models_cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()

# Payload bounds (D032; default raised to 256 KB by D049). Reject oversize
# requests at the hub edge so a single pathological client cannot exhaust the
# event loop's memory budget. Static cap = DoS defence; dynamic per-model
# capacity is surfaced separately via GET /v1/limits.
MAX_INPUT_BYTES = int(os.getenv("MAX_INPUT_BYTES", "262144"))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "200"))
MAX_BATCH_EMBEDDINGS = int(os.getenv("MAX_BATCH_EMBEDDINGS", "128"))

# Bounded stream queue (D033). Drop-oldest on overflow.
STREAM_QUEUE_MAX = int(os.getenv("STREAM_QUEUE_MAX", "256"))

# Default embedding model used when /v1/embeddings is called without one.
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")


def _rate_key(request: Request) -> str:
    """Rate-limit key: API key > x-api-key > node_id path param > client IP."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth.removeprefix("Bearer ")
    x_key = request.headers.get("x-api-key", "")
    if x_key:
        return x_key
    node_id = request.path_params.get("node_id")
    if node_id:
        return node_id
    return get_remote_address(request)


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": {
            "message": f"Rate limit exceeded: {exc.detail}",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }},
        headers={"Retry-After": "60"},
    )


class PayloadTooLarge(Exception):
    """D049: structured 413 raised by payload validators. A dedicated exception
    lets the handler emit `{"error": {...}}` at the top level, distinct from
    FastAPI's default `{"detail": ...}` envelope."""

    def __init__(self, field: str, limit: int, actual: int):
        self.field = field
        self.limit = limit
        self.actual = actual


async def _payload_too_large_handler(request: Request, exc: PayloadTooLarge) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"error": {
            "type": "payload_too_large",
            "field": exc.field,
            "limit_bytes": exc.limit,
            "actual_bytes": exc.actual,
        }},
    )


app = FastAPI(title="LLMesh Hub")

# Setup Jinja2 templates
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "views", "templates"))

_version_path = os.path.join(BASE_DIR, "..", "VERSION")
with open(_version_path) as _vf:
    APP_VERSION = _vf.read().strip()

class InferenceRequest(BaseModel):
    owner_id: str # Added for internal routing
    prompt: Optional[str] = None # Made optional for Anthropic API
    messages: Optional[List[Dict[str, str]]] = None # Added for Anthropic API
    model: str = "llama3"
    num_ctx: Optional[int] = None
    max_tokens: Optional[int] = None

class TaskResponse(BaseModel):
    task_id: str
    node_assigned: str

# Anthropic API Models
class AnthropicMessage(BaseModel):
    role: str
    content: str

class AnthropicRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = 1024 # Default for Anthropic API
    stream: bool = False # Not supported yet
    num_ctx: Optional[int] = None

# OpenAI API Models
class OpenAIMessage(BaseModel):
    role: str
    content: str | list  # Accept plain string OR array of content blocks

class OpenAIRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = None
    stream: bool = False
    num_ctx: Optional[int] = None

limiter = Limiter(
    key_func=_rate_key,
    storage_uri=os.getenv("RATE_LIMIT_STORAGE_URL", "memory://"),
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_exception_handler(PayloadTooLarge, _payload_too_large_handler)

STREAM_CHUNK_TIMEOUT = float(os.getenv("STREAM_CHUNK_TIMEOUT", "300.0"))

# Routing scoring weights (D054 closes D036 §1). The legacy pure-RAM sort
# overloaded the highest-RAM node even while lower-RAM peers sat idle.
# Score = ram_gb - queue_depth * QUEUE_PENALTY - cpu_load * CPU_PENALTY.
# Defaults sized so one queued task costs ~one tier of RAM (8 GB) and a
# fully saturated CPU costs ~10 GB. Both env-tunable.
ROUTING_QUEUE_PENALTY = float(os.getenv("ROUTING_QUEUE_PENALTY", "8.0"))
ROUTING_CPU_PENALTY   = float(os.getenv("ROUTING_CPU_PENALTY",   "0.1"))

DEFAULT_CONTEXT_WINDOW = os.getenv("DEFAULT_CONTEXT_WINDOW")
if DEFAULT_CONTEXT_WINDOW:
    DEFAULT_CONTEXT_WINDOW = int(DEFAULT_CONTEXT_WINDOW)
else:
    DEFAULT_CONTEXT_WINDOW = None


class StreamChunk(BaseModel):
    chunk: str = ""
    done: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # D068: batcher telemetry, populated only on the done frame. Defaults
    # preserve back-compat with agents that haven't been upgraded yet.
    stream_batches: int = 0       # total /stream POSTs this task issued
    stream_final_size: int = 0    # batcher's final per-flush size at end of stream


def _is_error_result(result: dict) -> bool:
    """D034: structured error flag is the only signal. Substring fallback
    against `output` was removed — it coupled retry logic to specific English
    error strings."""
    return bool(result.get("error"))


def _node_has_model(n, model: str) -> bool:
    return (
        model in getattr(n.resources, "ollama_models", []) or
        model in getattr(n.resources, "embedding_models", []) or
        model in getattr(n.resources, "vllm_models", []) or
        model in getattr(n.resources, "mlx_models", [])
    )


def _node_has_embedding_model(n, model: str) -> bool:
    """D028: embedding routing is gated on `embedding_models` only. A chat-only
    model in `ollama_models` does not qualify, even if Ollama would accept the
    request shape — the response would be useless."""
    return model in getattr(n.resources, "embedding_models", [])


def _node_has_image_model(n, model: str) -> bool:
    """D064: image routing is gated on `image_models` only AND the node must
    advertise `image_available=True`. VRAM filtering is layered on top in
    `_select_image_node` because the registry-side `min_vram_gb` for the model
    is consulted at routing time, not membership time."""
    return (
        getattr(n.resources, "image_available", False)
        and model in getattr(n.resources, "image_models", [])
    )


def _node_is_capable(n) -> bool:
    return (
        n.resources.ollama_available or
        getattr(n.resources, "vllm_available", False) or
        getattr(n.resources, "mlx_available", False) or
        getattr(n.resources, "image_available", False)
    )


def _node_model_context(n, model: str) -> int:
    """D030: per-model context lookup with fallback to the legacy node-level
    scalar `context_size`."""
    model_ctx = getattr(n.resources, "model_context", {}) or {}
    if model in model_ctx and isinstance(model_ctx[model], int) and model_ctx[model] > 0:
        return model_ctx[model]
    return getattr(n.resources, "context_size", 8192)


def _payload_too_large(field: str, limit: int, actual: int) -> PayloadTooLarge:
    return PayloadTooLarge(field=field, limit=limit, actual=actual)


def _validate_chat_payload(messages: list[dict]) -> None:
    """D032/D049: reject oversize chat payloads before they hit the routing pipeline."""
    if len(messages) > MAX_MESSAGES:
        raise _payload_too_large(field="messages", limit=MAX_MESSAGES, actual=len(messages))
    for idx, m in enumerate(messages):
        content = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(content, list):
            # Flattened later; size-bound the assembled text.
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        if isinstance(content, str):
            n = len(content.encode("utf-8"))
            if n > MAX_INPUT_BYTES:
                raise _payload_too_large(
                    field=f"messages[{idx}].content",
                    limit=MAX_INPUT_BYTES,
                    actual=n,
                )


def _validate_embeddings_payload(inputs: list[str]) -> None:
    """D032/D049: reject oversize embedding batches and per-input strings."""
    if not inputs:
        raise HTTPException(status_code=400, detail="`input` must be a non-empty string or non-empty list of strings")
    if len(inputs) > MAX_BATCH_EMBEDDINGS:
        raise _payload_too_large(field="input", limit=MAX_BATCH_EMBEDDINGS, actual=len(inputs))
    for idx, s in enumerate(inputs):
        if not isinstance(s, str):
            raise HTTPException(status_code=400, detail="every `input` element must be a string")
        if not s:
            raise HTTPException(status_code=400, detail="`input` strings must be non-empty")
        n = len(s.encode("utf-8"))
        if n > MAX_INPUT_BYTES:
            raise _payload_too_large(
                field=f"input[{idx}]",
                limit=MAX_INPUT_BYTES,
                actual=n,
            )


def _capability_predicate_for_kind(kind: tasks.TaskKind):
    """Pick the model-membership predicate for a task kind. Embedding tasks
    must land on a node that explicitly classifies the model as embedding-
    capable; chat tasks accept any backend that knows the model id."""
    if kind is tasks.TaskKind.EMBEDDING:
        return _node_has_embedding_model
    return _node_has_model


def _score_node(node) -> float:
    """Weighted routing score (D054, closes D036 §1).

    Higher is better. Penalises queue depth heavily and CPU load gently so
    the highest-RAM node does not capture every request while peers sit
    idle. Computed from in-memory state only — no DB read, no heartbeat
    round-trip. Tunable via ROUTING_QUEUE_PENALTY / ROUTING_CPU_PENALTY.
    """
    queue_depth = sum(
        1 for t in tasks.get_tasks_for_node(node.node_id)
        if t.status in ("pending", "claimed")
    )
    cpu_load = getattr(node, "cpu_load", 0.0) or 0.0
    return (
        node.resources.ram_gb
        - queue_depth * ROUTING_QUEUE_PENALTY
        - cpu_load    * ROUTING_CPU_PENALTY
    )


async def _try_requeue(task: tasks.Task) -> bool:
    """Find a capable node not yet attempted and requeue the task. Returns True on success."""
    all_nodes = storage.get_all_nodes()
    current_time = time.time()
    has_model = _capability_predicate_for_kind(task.kind)
    candidates = [
        n for n in all_nodes
        if n.owner_id == task.owner_id
        and n.node_id not in task.attempted_nodes
        and _node_is_capable(n)
        and (current_time - n.last_seen < 30)
        and has_model(n, task.model)
    ]
    if not candidates:
        return False
    candidates.sort(key=_score_node, reverse=True)
    selected = candidates[0]
    task.attempted_nodes.add(selected.node_id)
    await tasks.requeue_task(task, selected.node_id)
    logger.info("Requeued task %s → node %s (retries_left=%s)", task.task_id, selected.node_id, task.retries_left)
    return True


async def _recover_tasks_from_dead_node(dead_node_id: str) -> None:
    """Re-queue or fail any pending/claimed tasks stranded on a pruned node."""
    for task in tasks.get_tasks_for_node(dead_node_id):
        if task.status not in ("pending", "claimed"):
            continue
        task.attempted_nodes.add(dead_node_id)
        if task.retries_left > 0:
            task.retries_left -= 1
            if await _try_requeue(task):
                logger.info("Recovered task %s from dead node %s", task.task_id, dead_node_id)
                continue
        await tasks.fail_task(task.task_id, f"Node {dead_node_id} went offline and no alternate node is available")
        logger.warning("Failed task %s: no capable node after %s went offline", task.task_id, dead_node_id)


@app.on_event("startup")
async def startup():
    # Download and load the compression model before accepting requests.
    # Skipped automatically when SESSION_MEMORY_MODE=cutoff or if packages are missing.
    await compressor.ensure_ready()

    # Start the single module-level metrics flush task (D022).
    await metrics.start_background()

    # Node-registry durability recovery (D026 / D058). Must run BEFORE task
    # recovery so the rehydrated `_node_tasks` mapping references nodes that
    # actually exist in `_nodes`. Rows with last_seen >90s are pruned in-place
    # at load time so a hub restart after a long downtime does not surface
    # dead nodes. Restored nodes carry hash-only credentials; verify falls
    # back to hash compare until the agent re-registers with fresh plaintext.
    try:
        nodes_restored = await storage.load_persisted_nodes(max_age_sec=90.0)
        if nodes_restored:
            logger.info("Node registry restored: %d node(s) carried over from prior process", nodes_restored)
    except Exception as exc:
        logger.warning("Node registry recovery failed: %s — continuing with empty registry", exc)

    # Task durability recovery (D003 / D053). Rebuilds the in-memory task
    # index from rows persisted by the previous hub process. Claimed rows
    # reset to pending so the next eligible node picks them up; mid-flight
    # streaming sessions remain unrecoverable (D003 §Constraints).
    try:
        restored, reset = await tasks.load_persisted()
        if restored or reset:
            logger.info("Task store restored: %d pending, %d claimed-reset-to-pending", restored, reset)
    except Exception as exc:
        logger.warning("Task store recovery failed: %s — continuing with empty queue", exc)

    async def cleanup_loop():
        while True:
            await asyncio.sleep(30)
            pruned = storage.prune_inactive_nodes(max_age_sec=90)
            if pruned:
                logger.info("Pruned %d inactive node(s): %s", len(pruned), pruned)
                for dead_node_id in pruned:
                    await _recover_tasks_from_dead_node(dead_node_id)
                    # D035: drop the per-node queue residue AFTER recovery so
                    # any pending/claimed tasks were given a chance to migrate.
                    tasks.drop_node_queue(dead_node_id)

            evicted = await get_session_store().evict_expired()
            if evicted:
                logger.info("Evicted %d expired session(s)", evicted)

            deleted_e, deleted_s = await prune_old_metrics()
            if deleted_e or deleted_s:
                logger.info("Pruned metrics: %d inference events, %d node snapshots", deleted_e, deleted_s)

            pruned_tasks = await tasks.prune_old_tasks(tasks.TASK_TTL_SECONDS)
            if pruned_tasks:
                logger.info("Pruned %d completed/failed task(s) older than %ss", pruned_tasks, tasks.TASK_TTL_SECONDS)

            # Record node snapshots every 30 seconds
            owner_counts = {}
            for n in storage.get_all_nodes():
                owner_counts[n.owner_id] = owner_counts.get(n.owner_id, 0) + 1
            for owner, count in owner_counts.items():
                metrics.log_node_snapshot(owner, count)

    app.state.cleanup_task = asyncio.create_task(cleanup_loop())


@app.on_event("shutdown")
async def shutdown():
    # Drain any pending metrics events and release the shared DB handle
    # so the WAL is checkpointed cleanly on graceful SIGTERM (D022).
    await metrics.stop_background()
    await metrics.close_db()


async def _summarize_messages(messages: list[dict], owner_id: str) -> "compressor.SummarizeResult | None":
    return await compressor.summarize(messages)


async def _compress_session(session_id: str, owner_id: str, messages: list[dict], model: str):
    store = get_session_store()
    max_msgs = SESSION_MAX_TURNS * 2
    mode = SESSION_MEMORY_MODE

    result = None
    if mode == "aggressive":
        keep = messages[-2:]
        old  = messages[:-2]
        result = await _summarize_messages(old, owner_id)
        if result:
            compressed = [{"role": "system", "content": f"Conversation summary: {result.text}"}] + keep
        else:
            compressed = messages[-max_msgs:]
    elif mode == "balanced":
        half = len(messages) // 2
        old  = messages[:half]
        keep = messages[half:]
        result = await _summarize_messages(old, owner_id)
        if result:
            compressed = [{"role": "system", "content": f"Conversation summary: {result.text}"}] + keep
        else:
            compressed = messages[-max_msgs:]
    else:  # cutoff
        compressed = messages[-max_msgs:]

    await store.save_messages(session_id, owner_id, compressed)

    # Log the compression call as an inference event so the Stats tab can track it
    if result is not None:
        metrics.log_inference_event(
            user_id=owner_id,
            node_id="hub:compressor",
            model=os.getenv("COMPRESS_MODEL_FILE", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            status="success",
            duration_ms=result.duration_ms,
            tokens_prompt=result.prompt_tokens,
            tokens_completion=result.completion_tokens,
            is_compression=True,
        )

def _sse_frame(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sse_event(event_name: str, data: dict) -> str:
    """Render a named SSE event (Anthropic-style) — `event:` + `data:` lines.

    Per Anthropic Messages streaming spec (D061): each event has both a named
    `event:` line and a JSON `data:` payload. OpenAI streams use `_sse_frame`
    (data-only). Anthropic SDKs read the event name to dispatch on type.
    """
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"


def _estimate_input_tokens(messages: list) -> int:
    """Char-divide estimate of input tokens for Anthropic message_start usage.

    Anthropic emits `message_start.message.usage.input_tokens` BEFORE any
    generation. The hub does not yet have the agent-reported `prompt_tokens`
    at that point (the agent hasn't claimed the task yet). Estimate via
    `max(1, total_chars // 4)` — a long-standing rough heuristic for English
    text against tokenizer behavior. The cumulative `output_tokens` in
    `message_delta.usage` carries the agent-reported real number at end of
    stream. Clients that need precise input accounting should use the
    non-streaming `/v1/messages` response which carries exact `usage`.
    """
    total = 0
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for block in c:
                t = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
                if isinstance(t, str):
                    total += len(t)
    return max(1, total // 4)


async def _real_sse_generator_anthropic(
    task: tasks.Task,
    request: Request,
    message_id: str,
    model: str,
    input_tokens_estimate: int,
):
    """Async generator that drains a task's stream_queue and yields Anthropic
    Messages SSE events (D061).

    Event sequence per https://platform.claude.com/docs/en/api/messages-streaming
    (verified 2026-05-28):

        event: message_start          → message envelope w/ usage.input_tokens
        event: content_block_start    → index 0, type:"text", text:""
        event: content_block_delta×N  → delta.type:"text_delta", delta.text:"..."
        event: content_block_stop     → index 0
        event: message_delta          → delta.stop_reason:"end_turn",
                                        usage.output_tokens:N (cumulative)
        event: message_stop

    Errors emit `event: error` with `{"type":"error","error":{"type":...,"message":...}}`
    per spec. Per-event `ping` keepalives are valid per spec but omitted here
    (the hub already pushes deltas at a high enough rate that long quiet
    gaps are not expected; STREAM_CHUNK_TIMEOUT bounds the upper end).

    Reuses task.stream_queue + STREAM_CHUNK_TIMEOUT semantics from the
    OpenAI generator so cancellation, timeout, and node-disconnect paths
    stay structurally identical.
    """
    # 1. message_start — emits the message envelope with input_tokens estimate.
    #    output_tokens starts at 0; the cumulative count lands in message_delta.
    yield _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens_estimate,
                "output_tokens": 0,
            },
        },
    })

    # 2. content_block_start at index 0 — text block.
    yield _sse_event("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    accumulated: list[str] = []
    aborted = False
    error_payload: dict | None = None
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(task.stream_queue.get(), timeout=STREAM_CHUNK_TIMEOUT)
            except asyncio.TimeoutError:
                task.stream_cancelled = True
                logger.warning(
                    "[SSE-anthropic] Task %s aborted: stream timeout (no chunk from node for %ss)",
                    task.task_id, STREAM_CHUNK_TIMEOUT,
                )
                error_payload = {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "Stream timeout — node may be offline",
                    },
                }
                aborted = True
                break

            if chunk is None:  # sentinel — streaming complete
                break

            # D084: capture hub-side TTFT on the first content chunk. See
            # `_real_sse_generator` for the recovered-task guard rationale.
            if getattr(task, "ttft_ms", None) is None:
                _created_at = getattr(task, "created_at", None)
                if _created_at is not None:
                    ttft = (time.time() - _created_at) * 1000.0
                    if ttft <= STREAM_CHUNK_TIMEOUT * 1000.0:
                        task.ttft_ms = ttft

            accumulated.append(chunk)
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": chunk},
            })
    except asyncio.CancelledError:
        task.stream_cancelled = True
        logger.warning("[SSE-anthropic] Task %s aborted: consumer disconnected", task.task_id)
        raise

    if aborted and error_payload is not None:
        yield _sse_event("error", error_payload)
        return

    # 3. content_block_stop — close the text block.
    yield _sse_event("content_block_stop", {
        "type": "content_block_stop",
        "index": 0,
    })

    # 4. message_delta — top-level stop_reason + cumulative output_tokens.
    #    Per docs Warning callout: "The token counts shown in the `usage` field
    #    of the `message_delta` event are *cumulative*."
    output_tokens = task.completion_tokens or 0
    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })

    # 5. message_stop — terminator.
    yield _sse_event("message_stop", {"type": "message_stop"})


async def _real_sse_generator(
    task: tasks.Task,
    request: Request,
    task_id_str: str,
    created: int,
    model: str,
    session_id: str,
    owner_id: str,
    stored_history: list,
    incoming_messages: list,
):
    """Async generator that drains a task's stream_queue and yields SSE frames."""
    def _chunk(delta: dict, finish_reason=None):
        return _sse_frame({
            "id": task_id_str, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        })

    # Role chunk first (OpenAI streaming spec)
    yield _chunk({"role": "assistant"})

    accumulated: list[str] = []
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(task.stream_queue.get(), timeout=STREAM_CHUNK_TIMEOUT)
            except asyncio.TimeoutError:
                task.stream_cancelled = True
                logger.warning("[SSE] Task %s aborted: stream timeout (no chunk from node for %ss)", task.task_id, STREAM_CHUNK_TIMEOUT)
                yield _sse_frame({"error": {"message": "Stream timeout — node may be offline", "type": "stream_error"}})
                return

            if chunk is None:  # sentinel — streaming complete
                break

            # D084: capture hub-side TTFT on the first content chunk. Skip if
            # the gap exceeds STREAM_CHUNK_TIMEOUT — that signature means a
            # persisted-task recovery (D053), not a fresh first-token wait.
            if getattr(task, "ttft_ms", None) is None:
                _created_at = getattr(task, "created_at", None)
                if _created_at is not None:
                    ttft = (time.time() - _created_at) * 1000.0
                    if ttft <= STREAM_CHUNK_TIMEOUT * 1000.0:
                        task.ttft_ms = ttft

            accumulated.append(chunk)
            yield _chunk({"content": chunk})
    except asyncio.CancelledError:
        task.stream_cancelled = True
        logger.warning("[SSE] Task %s aborted: consumer disconnected (connection closed)", task.task_id)
        raise
    finally:
        # Final cleanup for the task if not already resolved
        if not task.stream_cancelled:
            # If we reached here normally, it means the stream is done.
            # We don't mark it cancelled immediately to allow for final done chunk POST.
            pass

    # Final chunk: finish_reason + usage
    full_response = "".join(accumulated)
    yield _sse_frame({
        "id": task_id_str, "object": "chat.completion.chunk",
        "created": created, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": task.prompt_tokens,
            "completion_tokens": task.completion_tokens,
            "total_tokens": task.prompt_tokens + task.completion_tokens,
        },
    })
    yield "data: [DONE]\n\n"

    # Save session after stream completes (mirrors non-streaming path in _process_chat_completion)
    store = get_session_store()
    updated_history = (stored_history or []) + incoming_messages + [{"role": "assistant", "content": full_response}]
    await store.save_messages(session_id, owner_id, updated_history)
    if len(updated_history) >= SESSION_MAX_TURNS * 2:
        asyncio.create_task(_compress_session(session_id, owner_id, updated_history, model))


async def _require_node_token(node_id: str, authorization: str | None = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing node token")
    token = authorization.removeprefix("Bearer ")
    if not storage.verify_node_token(node_id, token):
        raise HTTPException(status_code=403, detail="Invalid node token")

@app.get("/health")
async def health_check():
    # Intentionally minimal — no version string. Unauthenticated version
    # enumeration aids CVE targeting; operators who need the version can
    # read it from an authenticated endpoint or the VERSION file at build.
    return {"status": "ok"}


@app.post("/register", response_model=RegistrationResponse)
@limiter.limit(RATE_LIMIT_REGISTER)
async def register_node(request: Request, reg: RegistrationRequest):
    owner_id = storage.authenticate_owner(reg.api_key)
    if not owner_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )

    node_id = reg.node_fingerprint if reg.node_fingerprint else str(uuid.uuid4())
    existing = storage.get_node(node_id)
    # Re-registration of an existing node: keep the same plaintext token if
    # the previous process is still holding it; otherwise issue fresh
    # plaintext. Hash always recomputed so the persistence layer stays in
    # sync. See D004 (token preservation across re-register) + D058 (hash
    # persistence).
    node_token = existing.node_token if (existing and existing.node_token) else secrets.token_hex(32)
    node_token_hash = storage._hash_token(node_token)
    node = Node(
        node_id=node_id,
        owner_id=owner_id,
        resources=reg.resources,
        last_seen=time.time(),
        node_token=node_token,
        node_token_hash=node_token_hash,
        fingerprint=reg.node_fingerprint or node_id,
    )

    storage.store_node(node)
    asyncio.create_task(upsert_node_registry(
        node_id=node_id,
        owner_id=owner_id,
        os_name=reg.resources.os_name,
        cpu_cores=reg.resources.cpu_cores,
        ram_gb=reg.resources.ram_gb,
        context_size=getattr(reg.resources, "context_size", None),
    ))
    logger.info("Registered node %s for owner %s", node.node_id, owner_id)
    return RegistrationResponse(node_id=node_id, node_token=node_token)

@app.post("/heartbeat/{node_id}")
@limiter.limit(RATE_LIMIT_HEARTBEAT)
async def heartbeat(request: Request, node_id: str, req: HeartbeatRequest, _: None = Depends(_require_node_token)):
    node = storage.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node.last_seen = time.time()
    node.resources.ollama_available = req.ollama_available
    node.resources.vllm_available = req.vllm_available
    node.resources.mlx_available = req.mlx_available
    # D064: image_available toggles per heartbeat (mflux import + model dir
    # state may change between heartbeats — operator installing/removing
    # models). image_models / vram_gb stay sticky from registration; full
    # re-advertise happens on re-register.
    node.resources.image_available = getattr(req, "image_available", False)
    setattr(node, "cpu_load", req.cpu_load)
    setattr(node, "latency_ms", req.latency_ms)
    storage.store_node(node)
    return {"status": "ok"}

@app.get("/nodes")
@limiter.limit(RATE_LIMIT_LIST)
async def list_nodes(request: Request, authorization: str | None = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    api_key = authorization.removeprefix("Bearer ")
    owner_id = storage.authenticate_owner(api_key)
    if not owner_id:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return [n for n in storage.get_all_nodes() if n.owner_id == owner_id]

def _select_node(owner_id: str, model: str, has_model_predicate) -> "Node | None":
    """Find the best currently-online node for an owner+model pair, using the
    given capability predicate. Sorted by `_score_node` desc — weighted by
    queue depth and CPU load so the highest-RAM node does not capture every
    request while peers sit idle (D054 closes D036 §1)."""
    all_nodes = storage.get_all_nodes()
    current_time = time.time()
    capable = [
        n for n in all_nodes
        if n.owner_id == owner_id
        and _node_is_capable(n)
        and (current_time - n.last_seen < 30)
        and has_model_predicate(n, model)
    ]
    if not capable:
        return None
    capable.sort(key=_score_node, reverse=True)
    return capable[0]


async def _route_inference(req: InferenceRequest, stream: bool = False) -> TaskResponse:
    """Internal routing — owner_id must already be validated before calling."""
    selected_node = _select_node(req.owner_id, req.model, _node_has_model)
    if selected_node is None:
        raise HTTPException(status_code=503, detail=f"No capable nodes currently online with model {req.model}")

    task_id = str(uuid.uuid4())

    # Per-model context window (D030): clamp request num_ctx so a client cannot
    # ask for more than the model can serve. Falls back to node scalar.
    model_ctx_cap = _node_model_context(selected_node, req.model)
    requested = req.num_ctx if req.num_ctx is not None else DEFAULT_CONTEXT_WINDOW
    if requested is None:
        num_ctx = None
    else:
        num_ctx = min(int(requested), int(model_ctx_cap))

    task = tasks.Task(
        task_id=task_id,
        kind=tasks.TaskKind.CHAT,
        payload={
            "messages": req.messages,
            "prompt": req.prompt,
            "num_ctx": num_ctx,
            "max_tokens": req.max_tokens,
        },
        model=req.model,
        owner_id=req.owner_id,
    )
    task.attempted_nodes.add(selected_node.node_id)

    if stream and getattr(selected_node.resources, "streaming_capable", False):
        task.stream = True
        task.stream_queue = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)

    await tasks.queue_task_for_node(selected_node.node_id, task)

    logger.info("Routed task %s to node %s using model %s stream=%s", task_id, selected_node.node_id, req.model, task.stream)

    return TaskResponse(task_id=task_id, node_assigned=selected_node.node_id)


async def _select_image_node(owner_id: str, model: str) -> "Node | None":
    """D064 image routing — capability + min_vram filter, then D054 score.

    Layered on top of `_select_node` because the VRAM check pulls from the
    hub-side image registry (`min_vram_gb`) which the generic predicate path
    doesn't have. Returns None if no node qualifies (caller emits 503)."""
    from . import image_registry as _img_reg
    reg_entry = _img_reg.get(model)
    min_vram = reg_entry.min_vram_gb if reg_entry else 0

    all_nodes = storage.get_all_nodes()
    current_time = time.time()
    capable = [
        n for n in all_nodes
        if n.owner_id == owner_id
        and _node_is_capable(n)
        and (current_time - n.last_seen < 30)
        and _node_has_image_model(n, model)
        and getattr(n.resources, "vram_gb", 0.0) >= min_vram
    ]
    if not capable:
        return None
    capable.sort(key=_score_node, reverse=True)
    return capable[0]


async def _route_image(req, owner_id: str) -> TaskResponse:
    """D064: select an image-capable node + min_vram and queue an image task."""
    from . import image_registry as _img_reg

    reg_entry = _img_reg.get(req.model)
    if reg_entry is None:
        # Unknown model id — let routing also try in case operator advertised
        # an `unverified: true` model. Fall through to capability lookup.
        pass

    selected_node = await _select_image_node(owner_id, req.model)
    if selected_node is None:
        # 404 per D064 §"No model fallback / substitution" — return the list
        # of models the owner CAN reach so client can present an actionable
        # choice. Same shape as embeddings 503 but structured.
        all_for_owner = [
            n for n in storage.get_all_nodes()
            if n.owner_id == owner_id and getattr(n.resources, "image_available", False)
        ]
        available_models = sorted({m for n in all_for_owner for m in getattr(n.resources, "image_models", [])})
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "type": "model_not_available",
                    "model": req.model,
                    "available": available_models,
                    "message": (
                        f"No nodes online with image model {req.model!r}. "
                        f"Install with `llmesh-agent install-image-model {req.model}` on an Apple Silicon node "
                        f"and ensure the node has sufficient VRAM."
                    ),
                }
            },
        )

    # Map operator-facing size token → concrete WxH per model family.
    concrete_size = _img_reg.resolve_size(req.model, req.size)
    steps = _img_reg.quality_to_steps(req.quality)

    task_id = str(uuid.uuid4())
    task = tasks.Task(
        task_id=task_id,
        kind=tasks.TaskKind.IMAGE,
        payload={
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "size": concrete_size,
            "n": req.n,
            "seed": req.seed,
            "quality": req.quality,
            "steps": steps,
        },
        model=req.model,
        owner_id=owner_id,
    )
    task.attempted_nodes.add(selected_node.node_id)
    await tasks.queue_task_for_node(selected_node.node_id, task)

    logger.info(
        "Routed image task %s to node %s model=%s size=%s n=%s quality=%s",
        task_id, selected_node.node_id, req.model, concrete_size, req.n, req.quality,
    )
    return TaskResponse(task_id=task_id, node_assigned=selected_node.node_id)


async def _route_embedding(req: EmbeddingsRequest, owner_id: str, inputs: list[str]) -> TaskResponse:
    """D028 routing: select an embedding-capable node and queue an embedding
    task. Inputs must already be normalized (str → [str]) and bounds-checked."""
    selected_node = _select_node(owner_id, req.model, _node_has_embedding_model)
    if selected_node is None:
        hint = ""
        if req.model == DEFAULT_EMBEDDING_MODEL:
            hint = f" — pull it on a node: `ollama pull {req.model}`"
        raise HTTPException(
            status_code=503,
            detail=f"No nodes online with embedding model {req.model}{hint}",
        )

    task_id = str(uuid.uuid4())
    task = tasks.Task(
        task_id=task_id,
        kind=tasks.TaskKind.EMBEDDING,
        payload={"input": inputs},
        model=req.model,
        owner_id=owner_id,
    )
    task.attempted_nodes.add(selected_node.node_id)
    await tasks.queue_task_for_node(selected_node.node_id, task)

    logger.info("Routed embedding task %s to node %s using model %s batch=%d",
                task_id, selected_node.node_id, req.model, len(inputs))
    return TaskResponse(task_id=task_id, node_assigned=selected_node.node_id)


@app.post("/request_inference", response_model=TaskResponse)
@limiter.limit(RATE_LIMIT_INFERENCE)
async def request_inference(request: Request, req: InferenceRequest, authorization: str | None = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    api_key = authorization.removeprefix("Bearer ")
    owner_id = storage.authenticate_owner(api_key)
    if not owner_id:
        raise HTTPException(status_code=403, detail="Invalid API key")
    req.owner_id = owner_id  # override any client-supplied value
    return await _route_inference(req)


@app.get("/tasks/{node_id}/pending")
@limiter.limit(RATE_LIMIT_HEARTBEAT)
async def get_pending_tasks_for_node(request: Request, node_id: str, _: None = Depends(_require_node_token)):
    pending = await tasks.get_pending_tasks(node_id)
    out = []
    for t in pending:
        # Legacy fields (prompt/messages/num_ctx) are kept for chat tasks so
        # older agents that only know the chat shape keep working. New `kind`
        # and `payload` fields drive embedding dispatch (D028).
        entry = {
            "task_id": t.task_id,
            "kind": t.kind.value,
            "payload": t.payload,
            "model": t.model,
            "stream": t.stream,
        }
        if t.kind is tasks.TaskKind.CHAT:
            entry["prompt"] = t.prompt
            entry["messages"] = t.messages
            entry["num_ctx"] = t.num_ctx
            entry["max_tokens"] = t.max_tokens
        out.append(entry)
    return out


@app.post("/tasks/{node_id}/{task_id}/stream")
# NOTE: No @limiter.limit() here — chunk POSTs are per-token (~100–1000/min per active task).
# Applying a rate limit would break streaming mid-response. Revisit as a per-task budget
# if multi-tenant abuse becomes a concern in a future release.
async def stream_task_chunk(
    node_id: str, task_id: str,
    body: StreamChunk,
    _: None = Depends(_require_node_token),
):
    task = tasks.get_task_status(node_id, task_id)
    if not task or not task.stream_queue:
        raise HTTPException(status_code=404, detail="Task not found or not a streaming task")

    if task.stream_cancelled:
        raise HTTPException(status_code=410, detail="Stream cancelled by client")

    def _put_or_drop_oldest(item):
        # D033: bounded queue with drop-oldest. A slow SSE consumer must not
        # block the producing node — and the node should not OOM the hub.
        try:
            task.stream_queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                _ = task.stream_queue.get_nowait()
                logger.warning(
                    "[stream] task=%s queue full (max=%d), dropped oldest chunk",
                    task.task_id, STREAM_QUEUE_MAX,
                )
            except asyncio.QueueEmpty:
                pass
            task.stream_queue.put_nowait(item)

    if body.done:
        # D045: per CF-5 contract from D040+D041, agents may piggyback the
        # final batch onto the done frame as a single POST `{chunk: "tail",
        # done: true}`. Deliver the chunk content BEFORE the close sentinel
        # so SSE consumers receive every token. Empty `chunk` on done is the
        # common case (Ollama path, batcher with empty buffer at done time)
        # and is safe to skip.
        if body.chunk:
            _put_or_drop_oldest(body.chunk)
        task.prompt_tokens = body.prompt_tokens
        task.completion_tokens = body.completion_tokens
        # D068: stash batcher telemetry on the task. Surfaced in metrics +
        # dashboard task viewer. Zero from older agents that don't send it.
        task.stream_batches = body.stream_batches
        task.stream_final_size = body.stream_final_size
        task.status = "completed"
        _put_or_drop_oldest(None)  # sentinel — SSE generator closes on None
        task.done_event.set()
        # Persist terminal status + token accounting for the streaming path
        # (D053). result_json stays NULL — streamed tokens are not reassembled
        # server-side; SSE consumers received them in real time. Best-effort:
        # store failures log + continue, never roll back in-memory state.
        await task_store.get_task_store().save_result(
            task_id, None, body.prompt_tokens, body.completion_tokens, status="completed",
        )
        # D084: emit the inference event from the streaming-done path. Prior
        # to D084 the `/complete` endpoint was the only metric emit site, so
        # streamed chat tasks left NO row in `inference_events` — silently
        # dropping D068 batcher telemetry and (the D084 ask) TTFT. Build the
        # synthetic `result` dict that `_emit_inference_event` expects from
        # the fields the agent reported on the done frame.
        duration_ms = (time.time() - task.start_time) * 1000.0
        _emit_inference_event(
            task, node_id, duration_ms,
            {"prompt_tokens": body.prompt_tokens,
             "completion_tokens": body.completion_tokens},
            status="success",
        )
    else:
        _put_or_drop_oldest(body.chunk)

    return JSONResponse(
        content={"status": "ok"},
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        }
    )


@app.post("/tasks/{node_id}/complete/{task_id}")
@limiter.limit(RATE_LIMIT_HEARTBEAT)
async def submit_task_result(request: Request, node_id: str, task_id: str, result: Dict[str, Any], _: None = Depends(_require_node_token)):
    task = tasks.get_task_status(node_id, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found for this node")

    # Guard against late submissions from dead nodes that were already re-queued or failed
    if task.status in ("completed", "failed"):
        return {"status": "already_resolved"}

    # Embedding result lives in `embeddings` (list[list[float]]) — chat result
    # in `output` (str). Pick the right field per task kind so the stored
    # `task.result` keeps its native type (D029).
    if task.kind is tasks.TaskKind.EMBEDDING:
        result_value: Any = result.get("embeddings")
    else:
        result_value = result.get("output", "")

    task = await tasks.record_task_result(
        task_id,
        result_value,
        prompt_tokens=result.get("prompt_tokens", 0),
        completion_tokens=result.get("completion_tokens", 0),
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    duration_ms = (time.time() - task.start_time) * 1000.0
    is_error = _is_error_result(result)

    if is_error and task.retries_left > 0:
        task.retries_left -= 1
        task.attempted_nodes.add(node_id)
        if await _try_requeue(task):
            _emit_inference_event(task, node_id, duration_ms, result, status="fail")
            return {"status": "requeued"}
        # No alternate node — fall through to terminal failure

    if is_error:
        # On error the agent reports a string under `output`; keep that for the
        # human-readable failure message regardless of kind.
        await tasks.fail_task(task.task_id, result.get("output") or "task failed")
        _emit_inference_event(task, node_id, duration_ms, result, status="fail")
        _bridge_blocking_completion_to_stream_consumer(task, error=True)
    else:
        task.status = "completed"
        task.done_event.set()
        # Persist terminal status (record_task_result already saved result+tokens).
        await task_store.get_task_store().mark_status(task.task_id, "completed")
        _bridge_blocking_completion_to_stream_consumer(task, error=False)
        _emit_inference_event(task, node_id, duration_ms, result, status="success")

    return {"status": "ok"}


def _bridge_blocking_completion_to_stream_consumer(task: tasks.Task, *, error: bool) -> None:
    """D018: bridge a blocking-completion or blocking-failure into the streaming
    consumer queue, if any.

    Beta backends (vLLM, MLX) fall back to blocking inference when streaming is
    requested and submit results via `/complete` instead of `/stream`. Any SSE
    consumer waiting on `stream_queue.get()` — dashboard task viewer at
    `/dashboard/task/<id>/sse` or `/v1/chat/completions` with `stream: true` —
    will hang on the queue until `STREAM_CHUNK_TIMEOUT` and then emit a "node
    may be offline" error, even though the result is sitting in `task.result`
    fully complete.

    On success, push the full result as a single chunk followed by the close
    sentinel; the consumer's accumulator joins the (single-element) list and
    emits the same `done`+`result` frame it would for a real streamed task.

    On failure, push only the close sentinel; the consumer's error path reads
    `task.status` and `task.result` directly to construct the error frame.

    No-op when the task has no `stream_queue` (the regular blocking-API path).
    """
    if task.stream_queue is None:
        return
    if not error:
        task.stream_queue.put_nowait(task.result)
    task.stream_queue.put_nowait(None)


def _emit_inference_event(task: tasks.Task, node_id: str, duration_ms: float,
                          result: dict, status: str):
    metrics.log_inference_event(
        user_id=task.owner_id,
        node_id=node_id,
        model=task.model,
        status=status,
        duration_ms=duration_ms,
        tokens_prompt=result.get("prompt_tokens", 0),
        tokens_completion=result.get("completion_tokens", 0),
        kind=task.kind.value,
        # D068: batcher telemetry passes through from the streaming done frame.
        # Zero on non-streamed tasks; populated on streamed ones.
        stream_batches=getattr(task, "stream_batches", 0),
        stream_final_size=getattr(task, "stream_final_size", 0),
        # D084: hub-side time-to-first-token, set by the SSE generators on the
        # first content chunk. None on non-streamed tasks.
        ttft_ms=getattr(task, "ttft_ms", None),
    )


@app.get("/tasks/status/{node_id}/{task_id}")
async def check_task_status(
    node_id: str,
    task_id: str,
    authorization: str | None = Header(None),
    llmesh_session: str | None = Cookie(None),
):
    owner_id = None
    if authorization and authorization.startswith("Bearer "):
        owner_id = storage.authenticate_owner(authorization.removeprefix("Bearer "))
    elif llmesh_session:
        owner_id = _resolve_session(llmesh_session)
    if not owner_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    task = tasks.get_task_status(node_id, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.owner_id != owner_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return {
        "status": task.status,
        "result": task.result,
        "prompt_tokens": task.prompt_tokens,
        "completion_tokens": task.completion_tokens
    }

@app.get("/api/nodes")
@limiter.limit(RATE_LIMIT_LIST)
async def get_nodes_for_owner(
    request: Request,
    authorization: str | None = Header(None),
    llmesh_session: str | None = Cookie(None)
):
    owner_id = None
    if authorization:
        api_key = authorization.removeprefix("Bearer ")
        owner_id = storage.authenticate_owner(api_key)
    if not owner_id and llmesh_session:
        owner_id = _resolve_session(llmesh_session)

    if not owner_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    all_nodes = storage.get_all_nodes()
    owner_nodes = [n for n in all_nodes if n.owner_id == owner_id]
    current_time = time.time()
    
    # Format the data for the frontend so JS has less to do
    result = []
    for n in owner_nodes:
        result.append({
            # Send full node_id so dashboard renders operator-supplied
            # labels (D048) at full length. Prior `[:8] + "..."` truncation
            # made auto-fingerprints (`node_<16hex>`) marginally readable
            # but defeats LLMESH_NODE_ID. CSS handles any overflow.
            "node_id": n.node_id,
            "node_id_full": n.node_id,
            "cpu_cores": n.resources.cpu_cores,
            "ram_gb": n.resources.ram_gb,
            "os_name": n.resources.os_name,
            "cpu_load": getattr(n, "cpu_load", 0.0),
            "latency_ms": getattr(n, "latency_ms", 0.0),
            "context_size": getattr(n.resources, "context_size", 8192),
            "ollama_available": n.resources.ollama_available,
            "ollama_models": getattr(n.resources, "ollama_models", []),
            "embedding_models": getattr(n.resources, "embedding_models", []),
            "vllm_available": getattr(n.resources, "vllm_available", False),
            "vllm_models": getattr(n.resources, "vllm_models", []),
            "mlx_available": getattr(n.resources, "mlx_available", False),
            "mlx_models": getattr(n.resources, "mlx_models", []),
            "image_available": getattr(n.resources, "image_available", False),
            "image_models": getattr(n.resources, "image_models", []),
            "agent_version": getattr(n.resources, "agent_version", "0.1x"),
            "hub_version": APP_VERSION,
            "model_context": getattr(n.resources, "model_context", {}),
            "parallel_slots": getattr(n.resources, "parallel_slots", 1),
            "last_seen_sec": round(current_time - n.last_seen, 1)
        })
    return result

@app.get("/api/metrics")
@limiter.limit(RATE_LIMIT_LIST)
async def get_metrics(request: Request, llmesh_session: str | None = Cookie(None)):
    owner_id = _resolve_session(llmesh_session)
    if not owner_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return await metrics.get_dashboard_stats(owner_id)


@app.get("/api/recent")
@limiter.limit(RATE_LIMIT_LIST)
async def get_recent(request: Request, llmesh_session: str | None = Cookie(None)):
    """Return the 10 most recent rows from each SQLite table for the dashboard Recent tab."""
    owner_id = _resolve_session(llmesh_session)
    if not owner_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    import aiosqlite, os as _os
    db_path = _os.getenv("SESSION_DB", "./sessions.db")
    if db_path == ":memory:":
        return {"inference_events": [], "node_snapshots": [], "sessions": []}

    try:
        async with aiosqlite.connect(db_path) as conn:
            conn.row_factory = aiosqlite.Row

            async def _query(sql, params=()):
                try:
                    async with conn.execute(sql, params) as cur:
                        rows = await cur.fetchall()
                        return [dict(r) for r in rows]
                except Exception:
                    return []

            inference_events = await _query(
                "SELECT id, strftime('%Y-%m-%d %H:%M:%S', timestamp, 'unixepoch', 'localtime') AS ts, "
                "user_id, node_id, model, status, ROUND(duration_ms) AS duration_ms, "
                "tokens_prompt, tokens_completion "
                "FROM inference_events WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10",
                (owner_id,)
            )
            node_snapshots = await _query(
                "SELECT id, strftime('%Y-%m-%d %H:%M:%S', timestamp, 'unixepoch', 'localtime') AS ts, "
                "user_id, active_nodes "
                "FROM node_snapshots WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10",
                (owner_id,)
            )
            sessions = await _query(
                "SELECT session_id, owner_id, "
                "json_array_length(messages) AS message_count, "
                "strftime('%Y-%m-%d %H:%M:%S', created_at, 'unixepoch', 'localtime') AS created_at, "
                "strftime('%Y-%m-%d %H:%M:%S', last_active, 'unixepoch', 'localtime') AS last_active "
                "FROM sessions WHERE owner_id = ? ORDER BY last_active DESC LIMIT 10",
                (owner_id,)
            )

        return {
            "inference_events": inference_events,
            "node_snapshots": node_snapshots,
            "sessions": sessions,
        }
    except Exception as exc:
        return {"inference_events": [], "node_snapshots": [], "sessions": [], "error": str(exc)}

# --- Completions API Routes ---

@app.get("/v1/models")
@limiter.limit(RATE_LIMIT_LIST)
async def list_models(request: Request, authorization: str | None = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")

    api_key = authorization.replace("Bearer ", "")
    owner_id = storage.authenticate_owner(api_key)
    if not owner_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

    now_mono = time.monotonic()
    if MODELS_CACHE_TTL > 0:
        cached = _models_cache.get(owner_id)
        if cached and cached[0] > now_mono:
            _models_cache.move_to_end(owner_id)
            return cached[1]

    all_nodes = storage.get_all_nodes()

    # Aggregate per model:
    #   max ctx (D031 dedup-by-id, max wins)
    #   capability set ("chat" | "embed") drawn from which list it came from
    # Capabilities are additive — a model registered as both ollama_models and
    # embedding_models on different nodes ends up as ["chat","embed"].
    model_ctx: dict[str, int] = {}
    model_caps: dict[str, set[str]] = {}
    now = int(time.time())

    def _record(model_id: str, capability: str, ctx: int):
        if ctx and ctx > 0:
            model_ctx[model_id] = max(model_ctx.get(model_id, 0), ctx)
        else:
            model_ctx.setdefault(model_id, 0)
        model_caps.setdefault(model_id, set()).add(capability)

    for n in all_nodes:
        if n.owner_id != owner_id:
            continue
        node_default_ctx = getattr(n.resources, "context_size", 8192)
        per_model = getattr(n.resources, "model_context", {}) or {}

        for m in getattr(n.resources, "ollama_models", []):
            _record(m, "chat", per_model.get(m, node_default_ctx))
        for m in getattr(n.resources, "vllm_models", []):
            _record(m, "chat", per_model.get(m, node_default_ctx))
        for m in getattr(n.resources, "mlx_models", []):
            _record(m, "chat", per_model.get(m, node_default_ctx))
        for m in getattr(n.resources, "embedding_models", []):
            _record(m, "embed", per_model.get(m, node_default_ctx))

    model_list = [
        {
            "id": m,
            "object": "model",
            "created": now,
            "owned_by": "llmesh-node",
            "context_length": model_ctx.get(m, 0) or 8192,
            "capabilities": sorted(model_caps[m]),
        }
        for m in sorted(model_caps.keys())
    ]

    response = {"object": "list", "data": model_list}
    if MODELS_CACHE_TTL > 0:
        _models_cache[owner_id] = (now_mono + MODELS_CACHE_TTL, response)
        _models_cache.move_to_end(owner_id)
        while len(_models_cache) > MODELS_CACHE_MAX:
            _models_cache.popitem(last=False)
    return response


# Approximate chars-per-token for the byte-budget estimate surfaced on
# /v1/limits. 4 is the OpenAI/Anthropic rule-of-thumb for English text. Used
# only for the informational `context_bytes_estimate` — actual validation
# is byte-exact against MAX_INPUT_BYTES.
_CHARS_PER_TOKEN_ESTIMATE = 4


@app.get("/v1/limits")
@limiter.limit(RATE_LIMIT_LIST)
async def list_limits(request: Request, authorization: str | None = Header(None)):
    """D049: discoverable payload bounds.

    Static block: global DoS-defence caps enforced at request edge.
    `models` block: per-model context capacity derived from connected-node
    `model_context`/`context_size`, scoped to the calling owner. Empty when
    no capable nodes are online. Informational only — clients use it to
    pre-clamp before sending; validation is still byte-exact against
    `max_input_bytes`."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    api_key = authorization.replace("Bearer ", "")
    owner_id = storage.authenticate_owner(api_key)
    if not owner_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

    per_model_ctx: dict[str, int] = {}
    for n in storage.get_all_nodes():
        if n.owner_id != owner_id:
            continue
        node_default_ctx = getattr(n.resources, "context_size", 8192)
        per_model = getattr(n.resources, "model_context", {}) or {}
        for attr in ("ollama_models", "vllm_models", "mlx_models", "embedding_models"):
            for m in getattr(n.resources, attr, []) or []:
                ctx = per_model.get(m, node_default_ctx) or 0
                if ctx > 0:
                    per_model_ctx[m] = max(per_model_ctx.get(m, 0), ctx)

    models_block = {
        m: {
            "context_tokens": ctx,
            "context_bytes_estimate": min(MAX_INPUT_BYTES, ctx * _CHARS_PER_TOKEN_ESTIMATE),
        }
        for m, ctx in per_model_ctx.items()
    }

    return {
        "max_input_bytes": MAX_INPUT_BYTES,
        "max_messages": MAX_MESSAGES,
        "max_batch_embeddings": MAX_BATCH_EMBEDDINGS,
        "stream_queue_max": STREAM_QUEUE_MAX,
        "models": models_block,
    }


async def _process_chat_completion(
    req_model: str,
    messages: list[dict],
    api_key: str,
    session_id: str | None = None,
    want_stream: bool = False,
    num_ctx: int | None = None,
    max_tokens: int | None = None,
) -> tuple:
    owner_id = storage.authenticate_owner(api_key)
    if not owner_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

    _validate_chat_payload(messages)

    store = get_session_store()

    if session_id is None:
        session_id = str(uuid.uuid4())

    stored_history = await store.get_messages(session_id, owner_id)
    full_messages = (stored_history or []) + messages

    inf_req = InferenceRequest(
        owner_id=owner_id,
        model=req_model,
        messages=full_messages,
        num_ctx=num_ctx,
        max_tokens=max_tokens,
    )

    try:
        task_response = await _route_inference(inf_req, stream=want_stream)
        node_id = task_response.node_assigned
        task_id = task_response.task_id
    except HTTPException as e:
        return None, JSONResponse(status_code=503, content={"error": {"message": e.detail, "type": "server_error"}}), session_id, owner_id, stored_history, messages

    queued_task = tasks.get_task_status(node_id, task_id)
    if queued_task is None:
        return None, JSONResponse(status_code=500, content={"error": {"message": "Task not found after queuing", "type": "server_error"}}), session_id, owner_id, stored_history, messages
    queued_task.session_id = session_id

    # Streaming path — return immediately; SSE generator handles wait + session save
    if queued_task.stream_queue is not None:
        return queued_task, None, session_id, owner_id, stored_history, messages

    # Non-streaming path — wait for completion
    try:
        await asyncio.wait_for(queued_task.done_event.wait(), timeout=600.0)
    except asyncio.TimeoutError:
        return None, JSONResponse(status_code=504, content={"error": {"message": "LLMesh task timed out", "type": "timeout"}}), session_id, owner_id, stored_history, messages

    if queued_task.status == "completed":
        updated_history = (stored_history or []) + messages + [{"role": "assistant", "content": queued_task.result}]
        await store.save_messages(session_id, owner_id, updated_history)
        if len(updated_history) >= SESSION_MAX_TURNS * 2:
            asyncio.create_task(_compress_session(session_id, owner_id, updated_history, req_model))
        return queued_task, None, session_id, owner_id, stored_history, messages

    return None, JSONResponse(status_code=500, content={"error": {"message": queued_task.result, "type": "server_error"}}), session_id, owner_id, stored_history, messages

@app.post("/v1/chat/completions")
@limiter.limit(RATE_LIMIT_INFERENCE)
async def openai_chat_completions(
    request: Request,
    req: OpenAIRequest,
    response: Response,
    authorization: str | None = Header(None),
    x_session_id: str | None = Header(None),
    x_qcoda_execution_id: str | None = Header(None),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")

    # QCoda LAB-006 cross-stack correlation hook. QCoda injects this header
    # via lib/agents/base.py::_call_llm (QCoda D-084 / nodemesh-side log
    # surface for joint qcoda + llmesh + ollama lab rig). Logged at INFO so
    # `grep <exec_id> logs/*.log` aligns QCoda's per-agent traces with the
    # hub-side dispatch line. None when caller is not QCoda.
    if x_qcoda_execution_id:
        logger.info("QCoda execution_id=%s model=%s stream=%s", x_qcoda_execution_id, req.model, req.stream)

    api_key = authorization.replace("Bearer ", "")

    def extract_content(content) -> str:
        """Flatten content blocks (e.g. [{type: text, text: ...}]) to a plain string."""
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return content or ""

    messages = [{"role": m.role, "content": extract_content(m.content)} for m in req.messages]

    task, err_resp, session_id, owner_id, stored_history, incoming_messages = await _process_chat_completion(
        req.model, messages, api_key, x_session_id, want_stream=req.stream,
        num_ctx=req.num_ctx, max_tokens=req.max_tokens
    )
    if err_resp:
        return err_resp

    retry_count = task.initial_retries - task.retries_left
    task_id = f"chatcmpl-{task.task_id}"
    created = int(time.time())

    if req.stream:
        sse_headers = {
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id, 
            "X-Retry-Count": str(retry_count)
        }
        if task.stream_queue is not None:
            # Real streaming — node is streaming_capable, tokens arrive via chunk endpoint
            return StreamingResponse(
                _real_sse_generator(task, request, task_id, created, req.model,
                                    session_id, owner_id, stored_history, incoming_messages),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        # Fallback: non-capable node — wait for completion then fake-stream the full result
        try:
            await asyncio.wait_for(task.done_event.wait(), timeout=600.0)
        except asyncio.TimeoutError:
            return JSONResponse(status_code=504, content={"error": {"message": "LLMesh task timed out", "type": "timeout"}})
        if task.status != "completed":
            return JSONResponse(status_code=500, content={"error": {"message": task.result, "type": "server_error"}})

        async def _fake_sse():
            yield f"data: {json.dumps({'id': task_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': task_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': task.result}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': task_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_fake_sse(), media_type="text/event-stream", headers=sse_headers)

    response.headers["X-Session-ID"] = session_id
    response.headers["X-Retry-Count"] = str(retry_count)
    return {
        "id": task_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": task.result,
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": task.prompt_tokens,
            "completion_tokens": task.completion_tokens,
            "total_tokens": task.prompt_tokens + task.completion_tokens
        }
    }

@app.post("/v1/messages")
@limiter.limit(RATE_LIMIT_INFERENCE)
async def anthropic_messages(
    request: Request,
    req: AnthropicRequest,
    response: Response,
    x_api_key: str | None = Header(None),
    x_session_id: str | None = Header(None)
):
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing x-api-key header")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    task, err_resp, session_id, _owner_id, _stored_history, _incoming = await _process_chat_completion(
        req.model, messages, x_api_key, x_session_id,
        want_stream=req.stream, num_ctx=req.num_ctx, max_tokens=req.max_tokens,
    )
    if err_resp:
        return err_resp

    # Anthropic streaming branch (D061). Reuses the OpenAI streaming pipeline
    # (task.stream_queue or fake-SSE fallback) but emits anthropic named events
    # instead of the OpenAI chat.completion.chunk shape.
    if req.stream:
        message_id = f"msg_{task.task_id}"
        sse_headers = {
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id,
            "X-Retry-Count": str(task.initial_retries - task.retries_left),
        }
        input_estimate = _estimate_input_tokens(messages)
        if task.stream_queue is not None:
            return StreamingResponse(
                _real_sse_generator_anthropic(task, request, message_id, req.model, input_estimate),
                media_type="text/event-stream",
                headers=sse_headers,
            )

        # Fallback: non-streaming-capable node — wait for completion then
        # emit the full text as a single content_block_delta. Preserves the
        # event-sequence contract so SDK accumulators still parse cleanly.
        try:
            await asyncio.wait_for(task.done_event.wait(), timeout=600.0)
        except asyncio.TimeoutError:
            return JSONResponse(status_code=504, content={"error": {"message": "LLMesh task timed out", "type": "timeout"}})
        if task.status != "completed":
            return JSONResponse(status_code=500, content={"error": {"message": task.result, "type": "server_error"}})

        async def _fake_anthropic_sse():
            yield _sse_event("message_start", {
                "type": "message_start",
                "message": {
                    "id": message_id, "type": "message", "role": "assistant",
                    "content": [], "model": req.model,
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": input_estimate, "output_tokens": 0},
                },
            })
            yield _sse_event("content_block_start", {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""},
            })
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": task.result or ""},
            })
            yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield _sse_event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": task.completion_tokens or 0},
            })
            yield _sse_event("message_stop", {"type": "message_stop"})

        return StreamingResponse(_fake_anthropic_sse(), media_type="text/event-stream", headers=sse_headers)

    response.headers["X-Session-ID"] = session_id
    response.headers["X-Retry-Count"] = str(task.initial_retries - task.retries_left)

    return {
        "id": f"msg_{task.task_id}",
        "type": "message",
        "role": "assistant",
        "model": req.model,
        "content": [
            {
                "type": "text",
                "text": task.result
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": task.prompt_tokens,
            "output_tokens": task.completion_tokens
        }
    }

@app.post("/v1/embeddings")
@limiter.limit(RATE_LIMIT_INFERENCE)
async def embeddings(
    request: Request,
    req: EmbeddingsRequest,
    response: Response,
    authorization: str | None = Header(None),
):
    """OpenAI-compatible embeddings endpoint (D028).

    Routes to an embedding-capable Ollama node selected via the same node
    pool as chat. Stateless — no session storage. Returns a fully assembled
    OpenAI shape (object/data/model/usage)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    api_key = authorization.replace("Bearer ", "")
    owner_id = storage.authenticate_owner(api_key)
    if not owner_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

    # Normalize input → list[str] regardless of whether the client sent a
    # single string or an array. Validate bounds before routing so we reject
    # oversize requests without consuming a node slot (D032).
    inputs = [req.input] if isinstance(req.input, str) else list(req.input)
    _validate_embeddings_payload(inputs)

    task_response = await _route_embedding(req, owner_id, inputs)
    queued_task = tasks.get_task_status(task_response.node_assigned, task_response.task_id)
    if queued_task is None:
        raise HTTPException(status_code=500, detail="Task not found after queuing")

    try:
        await asyncio.wait_for(queued_task.done_event.wait(), timeout=600.0)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": {"message": "LLMesh embedding task timed out", "type": "timeout"}})

    if queued_task.status != "completed":
        return JSONResponse(status_code=500, content={"error": {"message": queued_task.result or "embedding task failed", "type": "server_error"}})

    vectors = queued_task.result or []
    if not isinstance(vectors, list) or len(vectors) != len(inputs):
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"node returned {len(vectors) if isinstance(vectors, list) else 'invalid'} vectors for {len(inputs)} inputs", "type": "server_error"}},
        )

    data = [
        {"object": "embedding", "index": i, "embedding": vec}
        for i, vec in enumerate(vectors)
    ]
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": queued_task.prompt_tokens,
            "total_tokens": queued_task.prompt_tokens,
        },
    }


@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
@limiter.limit(RATE_LIMIT_INFERENCE)
async def images_generations(
    request: Request,
    req: ImageGenerationRequest,
    response: Response,
    authorization: str | None = Header(None),
):
    """OpenAI-compatible image generation endpoint (D064).

    Routes to an `image`-capable node with sufficient VRAM for the requested
    model (per hub-side `image_registry.min_vram_gb`). Stateless — no session
    storage. Returns OpenAI-shaped `{"created":N, "data":[{"b64_json":"..."}, ...]}`.

    Limitations (v1 — see D064):
      * `response_format` must be `"b64_json"` (URL mode deferred to v2).
      * Backend is mflux on Apple Silicon only.
      * No model substitution: missing model → 404 with structured error.
      * No content filter; operator policy per docs/image_gen.md.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    api_key = authorization.replace("Bearer ", "")
    owner_id = storage.authenticate_owner(api_key)
    if not owner_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

    # Bounds validation before consuming a node slot (D049 pattern).
    if req.n < 1 or req.n > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "param": "n",
                    "message": f"`n` must be between 1 and {MAX_IMAGES_PER_REQUEST} (got {req.n})",
                }
            },
        )
    prompt_bytes = len((req.prompt or "").encode("utf-8"))
    if prompt_bytes > MAX_IMAGE_PROMPT_BYTES:
        raise _payload_too_large(field="prompt", limit=MAX_IMAGE_PROMPT_BYTES, actual=prompt_bytes)

    task_response = await _route_image(req, owner_id)
    queued_task = tasks.get_task_status(task_response.node_assigned, task_response.task_id)
    if queued_task is None:
        raise HTTPException(status_code=500, detail="Task not found after queuing")

    try:
        await asyncio.wait_for(queued_task.done_event.wait(), timeout=IMAGE_TASK_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail={
            "error": {"type": "timeout", "message": f"Image generation timed out after {IMAGE_TASK_TIMEOUT_S:.0f}s"}
        })

    if queued_task.status != "completed":
        raise HTTPException(status_code=502, detail={
            "error": {"type": "backend_error", "message": str(queued_task.result)}
        })

    # `task.result` is a list[str] of base64 PNGs from the agent driver.
    images: list[str] = queued_task.result or []
    if not isinstance(images, list):
        raise HTTPException(status_code=500, detail="Image task returned non-list result")

    # D064 metrics hook (image_event). Tokens are zero on this path; counts +
    # wall-time + resolution + steps are the units that matter.
    metrics.log_image_event(
        owner_id=owner_id,
        node_id=task_response.node_assigned,
        model=req.model,
        status="success",
        duration_ms=int((time.time() - queued_task.start_time) * 1000),
        image_count=len(images),
        size=queued_task.payload.get("size", ""),
        steps=int(queued_task.payload.get("steps", 0) or 0),
        quality_tier=queued_task.payload.get("quality", "draft"),
    )

    return {
        "created": int(time.time()),
        "data": [{"b64_json": b} for b in images],
    }


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str, authorization: str | None = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    api_key = authorization.replace("Bearer ", "")
    owner_id = storage.authenticate_owner(api_key)
    if not owner_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    await get_session_store().delete_session(session_id, owner_id)
    return {"status": "ok"}


# --- UI Routes ---

@app.get("/", response_class=RedirectResponse)
async def root_redirect():
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/login", response_class=HTMLResponse)
async def login_view(request: Request):
    existing = request.cookies.get(CSRF_COOKIE_NAME)
    csrf = existing or secrets.token_urlsafe(32)
    response = templates.TemplateResponse(
        request, "login.html", {"error": None, "csrf_token": csrf}
    )
    if not existing:
        response.set_cookie(
            key=CSRF_COOKIE_NAME, value=csrf,
            httponly=True, secure=True, samesite="Strict",
        )
    return response

@app.post("/login", response_class=HTMLResponse)
@limiter.limit(RATE_LIMIT_LOGIN)
async def login_submit(
    request: Request,
    api_key: str = Form(...),
    csrf_token: str = Form(...),
    llmesh_csrf: str | None = Cookie(None),
):
    _require_csrf(csrf_token, llmesh_csrf)
    owner_id = storage.authenticate_owner(api_key)
    if not owner_id:
        # Re-render login with the same csrf token so the user can retry.
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "Invalid API Key", "csrf_token": csrf_token},
        )

    # Mint an opaque session token bound to the owner. Never expose the
    # owner_id in the cookie — a leaked owner_id must not grant session access.
    token = secrets.token_urlsafe(32)
    _session_tokens[token] = owner_id

    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=True,
        samesite="Strict",
    )
    return response

@app.get("/logout", response_class=RedirectResponse)
async def logout(llmesh_session: str | None = Cookie(None)):
    if llmesh_session:
        _session_tokens.pop(llmesh_session, None)
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_view(request: Request, llmesh_session: str | None = Cookie(None)):
    owner_id = _resolve_session(llmesh_session)
    if not owner_id:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    # Filter nodes for this owner
    all_nodes = storage.get_all_nodes()
    owner_nodes = [n for n in all_nodes if n.owner_id == owner_id]
    
    # Collect all available models across the owner's online nodes and their counts
    current_time = time.time()
    model_counts = {}
    for n in owner_nodes:
        if _node_is_capable(n) and (current_time - n.last_seen < 30):
            # D028: chat dropdown excludes embedding-only models (they cannot
            # answer /v1/chat/completions). Per-node Embed row in node card
            # surfaces embedding models separately.
            chat_models = (
                getattr(n.resources, "ollama_models", []) +
                getattr(n.resources, "vllm_models", []) +
                getattr(n.resources, "mlx_models", [])
            )
            for model in chat_models:
                model_counts[model] = model_counts.get(model, 0) + 1

    # Sort by count descending, then alphabetically by name
    available_models = [
        {"name": name, "count": count}
        for name, count in sorted(model_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    # D064: deduped union of image_models advertised by this owner's nodes.
    # Empty list → the dashboard image-gen card hides itself (template guard).
    image_models_set: set[str] = set()
    for n in owner_nodes:
        if getattr(n.resources, "image_available", False):
            for m in getattr(n.resources, "image_models", []):
                image_models_set.add(m)
    image_models_list = sorted(image_models_set)

    # Calculate base_url from request
    base_url = str(request.base_url).rstrip('/')

    existing_csrf = request.cookies.get(CSRF_COOKIE_NAME)
    csrf = existing_csrf or secrets.token_urlsafe(32)

    response = templates.TemplateResponse(
        request, "dashboard.html",
        {
            "nodes": owner_nodes,
            "owner_id": owner_id,
            "current_time": time.time(),
            "available_models": available_models,
            "image_models": image_models_list,
            "base_url": base_url,
            "version": APP_VERSION,
            "csrf_token": csrf,
        }
    )
    if not existing_csrf:
        response.set_cookie(
            key=CSRF_COOKIE_NAME, value=csrf,
            httponly=True, secure=True, samesite="Strict",
        )
    return response

@app.post("/dashboard/request_inference")
async def dashboard_submit_inference(
    request: Request,
    prompt: str = Form(...),
    model: str = Form(...),
    csrf_token: str = Form(...),
    llmesh_session: str | None = Cookie(None),
    llmesh_csrf: str | None = Cookie(None),
):
    _require_csrf(csrf_token, llmesh_csrf)
    owner_id = _resolve_session(llmesh_session)
    if not owner_id:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    # Reuse the JSON logic by building the Pydantic model
    try:
        inf_req = InferenceRequest(owner_id=owner_id, prompt=prompt, model=model)
        result = await _route_inference(inf_req, stream=True)
        
        # Redirect the user to the polling page for this specific task
        return RedirectResponse(
            url=f"/dashboard/task/{result.node_assigned}/{result.task_id}", 
            status_code=status.HTTP_303_SEE_OTHER
        )
    except HTTPException as e:
        # If no nodes available, show a simple error
        return HTMLResponse(f"<h3>Error: {e.detail}</h3><a href='/dashboard'>Back</a>", status_code=e.status_code)

@app.post("/dashboard/request_image")
async def dashboard_submit_image(
    request: Request,
    prompt: str = Form(...),
    model: str = Form(...),
    size: str = Form("square"),
    quality: str = Form("draft"),
    csrf_token: str = Form(...),
    llmesh_session: str | None = Cookie(None),
    llmesh_csrf: str | None = Cookie(None),
):
    """D082: dashboard image-gen submit. Queues the task and redirects to a
    status view that polls the result endpoint — never blocks the browser.
    Prior D064-era behaviour blocked the request thread for the full inference
    duration (up to IMAGE_TASK_TIMEOUT_S), so the page hung with no progress
    indicator. Now the same pattern as chat: submit → redirect → poll → render.
    """
    _require_csrf(csrf_token, llmesh_csrf)
    owner_id = _resolve_session(llmesh_session)
    if not owner_id:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    try:
        req = ImageGenerationRequest(
            model=model, prompt=prompt, size=size, quality=quality, n=1,
        )
    except Exception as exc:
        return HTMLResponse(
            f"<h3>Invalid request: {exc}</h3><a href='/dashboard'>Back</a>",
            status_code=400,
        )

    try:
        task_response = await _route_image(req, owner_id)
    except HTTPException as e:
        msg = e.detail if isinstance(e.detail, str) else json.dumps(e.detail)
        return HTMLResponse(
            f"<h3>Error: {msg}</h3><a href='/dashboard'>Back</a>",
            status_code=e.status_code,
        )

    return RedirectResponse(
        url=f"/dashboard/image/{task_response.node_assigned}/{task_response.task_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.get("/dashboard/image/{node_id}/{task_id}", response_class=HTMLResponse)
async def dashboard_image_status_view(
    request: Request, node_id: str, task_id: str,
    llmesh_session: str | None = Cookie(None),
):
    """D082: image-gen status view. Renders prompt + model + live elapsed
    timer + spinner; JS polls /dashboard/image/<node>/<task>/result every
    1s until it returns the b64 PNG (or an error)."""
    owner_id = _resolve_session(llmesh_session)
    if not owner_id:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    task = tasks.get_task_status(node_id, task_id)
    if not task:
        return HTMLResponse("Task not found", status_code=404)
    if task.owner_id != owner_id:
        return HTMLResponse("Access denied", status_code=403)
    return templates.TemplateResponse(
        request, "image_status.html",
        {
            "task_id": task_id, "node_id": node_id,
            "model": task.model,
            "prompt": (task.payload or {}).get("prompt", ""),
            "size": (task.payload or {}).get("size", ""),
            "quality": (task.payload or {}).get("quality", "draft"),
            "timeout_s": int(IMAGE_TASK_TIMEOUT_S),
        },
    )


@app.get("/dashboard/image/{node_id}/{task_id}/result")
async def dashboard_image_result(
    node_id: str, task_id: str,
    llmesh_session: str | None = Cookie(None),
):
    """D082: poll target for the image status view. Returns one of:
      {"status": "pending", "elapsed_ms": N}
      {"status": "done", "images": ["<b64>"], "elapsed_ms": N}
      {"status": "failed", "error": "...", "elapsed_ms": N}
    Logs the image event exactly once on the first poll that observes
    completion (idempotent via a flag on the task object)."""
    owner_id = _resolve_session(llmesh_session)
    if not owner_id:
        return JSONResponse({"status": "failed", "error": "unauthorized"}, status_code=401)
    task = tasks.get_task_status(node_id, task_id)
    if not task:
        return JSONResponse({"status": "failed", "error": "task not found"}, status_code=404)
    if task.owner_id != owner_id:
        return JSONResponse({"status": "failed", "error": "access denied"}, status_code=403)

    elapsed_ms = int((time.time() - task.start_time) * 1000)

    if task.status not in ("completed", "failed"):
        return JSONResponse({"status": "pending", "elapsed_ms": elapsed_ms})

    # One-shot metrics log on first observation of terminal state.
    if not getattr(task, "_image_event_logged", False):
        try:
            metrics.log_image_event(
                owner_id=owner_id,
                node_id=node_id,
                model=task.model,
                status="success" if task.status == "completed" else "failed",
                duration_ms=elapsed_ms,
                image_count=len(task.result) if isinstance(task.result, list) else 0,
                size=(task.payload or {}).get("size", ""),
                steps=int((task.payload or {}).get("steps", 0) or 0),
                quality_tier=(task.payload or {}).get("quality", "draft"),
            )
        except Exception as exc:
            logger.warning("image event log failed for task %s: %r", task_id, exc)
        task._image_event_logged = True

    if task.status == "failed":
        err = task.result if isinstance(task.result, str) else "unknown error"
        return JSONResponse({"status": "failed", "error": err, "elapsed_ms": elapsed_ms})

    images = task.result if isinstance(task.result, list) else []
    return JSONResponse({"status": "done", "images": images, "elapsed_ms": elapsed_ms})


@app.get("/dashboard/task/{node_id}/{task_id}", response_class=HTMLResponse)
async def dashboard_task_status_view(request: Request, node_id: str, task_id: str, llmesh_session: str | None = Cookie(None)):
    owner_id = _resolve_session(llmesh_session)
    if not owner_id:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    task = tasks.get_task_status(node_id, task_id)
    if not task:
        return HTMLResponse("Task not found", status_code=404)
    if task.owner_id != owner_id:
        return HTMLResponse("Access denied", status_code=403)

    return templates.TemplateResponse(
        request, "task_status.html",
        {"task_id": task_id, "node_id": node_id, "task": task, "owner_id": owner_id}
    )


@app.get("/dashboard/task/{node_id}/{task_id}/sse")
async def dashboard_task_sse(
    request: Request,
    node_id: str,
    task_id: str,
    llmesh_session: str | None = Cookie(None),
):
    """SSE endpoint for the dashboard task status page — streams inference tokens in real-time."""
    owner_id = _resolve_session(llmesh_session)
    if not owner_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    task = tasks.get_task_status(node_id, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.owner_id != owner_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Base headers for all SSE responses
    sse_headers = {
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    # Already completed — send result immediately
    if task.status == "completed":
        async def _completed():
            yield f"data: {json.dumps({'done': True, 'result': task.result, 'prompt_tokens': task.prompt_tokens, 'completion_tokens': task.completion_tokens})}\n\n"
        return StreamingResponse(_completed(), media_type="text/event-stream", headers=sse_headers)

    if task.status == "failed":
        async def _failed():
            yield f"data: {json.dumps({'error': task.result or 'Task failed'})}\n\n"
        return StreamingResponse(_failed(), media_type="text/event-stream", headers=sse_headers)

    # Non-streaming task — wait for done_event and send result as a single burst
    if task.stream_queue is None:
        async def _poll_sse():
            try:
                await asyncio.wait_for(task.done_event.wait(), timeout=600.0)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'error': 'Task timed out'})}\n\n"
                return
            if task.status == "completed":
                yield f"data: {json.dumps({'done': True, 'result': task.result, 'prompt_tokens': task.prompt_tokens, 'completion_tokens': task.completion_tokens})}\n\n"
            else:
                yield f"data: {json.dumps({'error': task.result or 'Task failed'})}\n\n"
        return StreamingResponse(_poll_sse(), media_type="text/event-stream", headers=sse_headers)

    # Streaming task — drain stream_queue token by token
    async def _stream_sse():
        # Initial status frame so the UI knows the stream is active
        yield f"data: {json.dumps({'status': 'thinking'})}\n\n"
        accumulated = []
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(task.stream_queue.get(), timeout=STREAM_CHUNK_TIMEOUT)
                except asyncio.TimeoutError:
                    task.stream_cancelled = True
                    logger.warning("[Dashboard SSE] Task %s aborted: stream timeout (no chunk from node for %ss)", task.task_id, STREAM_CHUNK_TIMEOUT)
                    yield f"data: {json.dumps({'error': 'Stream timeout — node may be offline'})}\n\n"
                    return
                if chunk is None:  # sentinel — stream complete
                    break
                accumulated.append(chunk)
                yield f"data: {json.dumps({'token': chunk})}\n\n"
        except asyncio.CancelledError:
            task.stream_cancelled = True
            logger.warning("[Dashboard SSE] Task %s aborted: consumer disconnected (connection closed)", task.task_id)
            raise
        finally:
            # Only set cancelled if we didn't finish normally
            if chunk is not None:
                task.stream_cancelled = True

        full_text = "".join(accumulated)
        yield f"data: {json.dumps({'done': True, 'result': full_text, 'prompt_tokens': task.prompt_tokens, 'completion_tokens': task.completion_tokens})}\n\n"

    return StreamingResponse(_stream_sse(), media_type="text/event-stream", headers=sse_headers)


