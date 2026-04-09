import logging
import time
import uuid
import os
import secrets
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Request, Form, Response, Cookie, Header, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.templating import Jinja2Templates
from . import config as _config  # noqa: F401 — must import first to populate env vars before sessions/metrics read them
from .models import RegistrationRequest, Node, HeartbeatRequest, RegistrationResponse
import asyncio
import json
from . import storage
from . import tasks
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
# Rate limiting
# ---------------------------------------------------------------------------
RATE_LIMIT_INFERENCE = os.getenv("RATE_LIMIT_INFERENCE", "60/minute")
RATE_LIMIT_LIST      = os.getenv("RATE_LIMIT_LIST",      "120/minute")
RATE_LIMIT_REGISTER  = os.getenv("RATE_LIMIT_REGISTER",  "20/minute")
RATE_LIMIT_LOGIN     = os.getenv("RATE_LIMIT_LOGIN",     "10/minute")
RATE_LIMIT_HEARTBEAT = os.getenv("RATE_LIMIT_HEARTBEAT", "30/minute")


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

STREAM_CHUNK_TIMEOUT = float(os.getenv("STREAM_CHUNK_TIMEOUT", "300.0"))
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


def _is_error_result(result: dict) -> bool:
    if result.get("error"):
        return True
    output = result.get("output", "")
    return output.startswith("Failed") or output.startswith("Error from")


def _node_has_model(n, model: str) -> bool:
    return (
        model in getattr(n.resources, "ollama_models", []) or
        model in getattr(n.resources, "vllm_models", []) or
        model in getattr(n.resources, "mlx_models", [])
    )


def _node_is_capable(n) -> bool:
    return (
        n.resources.ollama_available or
        getattr(n.resources, "vllm_available", False) or
        getattr(n.resources, "mlx_available", False)
    )


def _try_requeue(task: tasks.Task) -> bool:
    """Find a capable node not yet attempted and requeue the task. Returns True on success."""
    all_nodes = storage.get_all_nodes()
    current_time = time.time()
    candidates = [
        n for n in all_nodes
        if n.owner_id == task.owner_id
        and n.node_id not in task.attempted_nodes
        and _node_is_capable(n)
        and (current_time - n.last_seen < 30)
        and _node_has_model(n, task.model)
    ]
    if not candidates:
        return False
    candidates.sort(key=lambda n: n.resources.ram_gb, reverse=True)
    selected = candidates[0]
    task.attempted_nodes.add(selected.node_id)
    tasks.requeue_task(task, selected.node_id)
    logger.info("Requeued task %s → node %s (retries_left=%s)", task.task_id, selected.node_id, task.retries_left)
    return True


def _recover_tasks_from_dead_node(dead_node_id: str) -> None:
    """Re-queue or fail any pending/claimed tasks stranded on a pruned node."""
    for task in tasks.get_tasks_for_node(dead_node_id):
        if task.status not in ("pending", "claimed"):
            continue
        task.attempted_nodes.add(dead_node_id)
        if task.retries_left > 0:
            task.retries_left -= 1
            if _try_requeue(task):
                logger.info("Recovered task %s from dead node %s", task.task_id, dead_node_id)
                continue
        tasks.fail_task(task.task_id, f"Node {dead_node_id} went offline and no alternate node is available")
        logger.warning("Failed task %s: no capable node after %s went offline", task.task_id, dead_node_id)


@app.on_event("startup")
async def startup():
    # Download and load the compression model before accepting requests.
    # Skipped automatically when SESSION_MEMORY_MODE=cutoff or if packages are missing.
    await compressor.ensure_ready()

    # Start the single module-level metrics flush task (D022).
    await metrics.start_background()

    async def cleanup_loop():
        while True:
            await asyncio.sleep(30)
            pruned = storage.prune_inactive_nodes(max_age_sec=90)
            if pruned:
                logger.info("Pruned %d inactive node(s): %s", len(pruned), pruned)
                for dead_node_id in pruned:
                    _recover_tasks_from_dead_node(dead_node_id)

            evicted = await get_session_store().evict_expired()
            if evicted:
                logger.info("Evicted %d expired session(s)", evicted)

            deleted_e, deleted_s = await prune_old_metrics()
            if deleted_e or deleted_s:
                logger.info("Pruned metrics: %d inference events, %d node snapshots", deleted_e, deleted_s)

            pruned_tasks = tasks.prune_old_tasks(tasks.TASK_TTL_SECONDS)
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
    node_token = existing.node_token if (existing and existing.node_token) else secrets.token_hex(32)
    node = Node(
        node_id=node_id,
        owner_id=owner_id,
        resources=reg.resources,
        last_seen=time.time(),
        node_token=node_token,
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

def _route_inference(req: InferenceRequest, stream: bool = False) -> TaskResponse:
    """Internal routing — owner_id must already be validated before calling."""
    all_nodes = storage.get_all_nodes()
    current_time = time.time()

    capable_nodes = [
        n for n in all_nodes
        if n.owner_id == req.owner_id
        and _node_is_capable(n)
        and (current_time - n.last_seen < 30)
        and _node_has_model(n, req.model)
    ]

    if not capable_nodes:
        raise HTTPException(status_code=503, detail=f"No capable nodes currently online with model {req.model}")

    capable_nodes.sort(key=lambda n: n.resources.ram_gb, reverse=True)
    selected_node = capable_nodes[0]

    task_id = str(uuid.uuid4())
    # Choose final num_ctx: request-specific > hub default
    num_ctx = req.num_ctx if req.num_ctx is not None else DEFAULT_CONTEXT_WINDOW

    task = tasks.Task(task_id=task_id, prompt=req.prompt, messages=req.messages,
                      model=req.model, owner_id=req.owner_id, num_ctx=num_ctx)
    task.attempted_nodes.add(selected_node.node_id)

    if stream and getattr(selected_node.resources, "streaming_capable", False):
        task.stream = True
        task.stream_queue = asyncio.Queue()

    tasks.queue_task_for_node(selected_node.node_id, task)

    logger.info("Routed task %s to node %s using model %s stream=%s", task_id, selected_node.node_id, req.model, task.stream)

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
    return _route_inference(req)


@app.get("/tasks/{node_id}/pending")
@limiter.limit(RATE_LIMIT_HEARTBEAT)
async def get_pending_tasks_for_node(request: Request, node_id: str, _: None = Depends(_require_node_token)):
    pending = tasks.get_pending_tasks(node_id)
    return [{"task_id": t.task_id, "prompt": t.prompt, "messages": t.messages,
             "model": t.model, "stream": t.stream, "num_ctx": t.num_ctx} for t in pending]


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

    if body.done:
        task.prompt_tokens = body.prompt_tokens
        task.completion_tokens = body.completion_tokens
        task.status = "completed"
        task.stream_queue.put_nowait(None)  # sentinel — SSE generator closes on None
        task.done_event.set()
    else:
        task.stream_queue.put_nowait(body.chunk)

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

    task = tasks.record_task_result(
        task_id,
        result.get("output", ""),
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
        if _try_requeue(task):
            _emit_inference_event(task, node_id, duration_ms, result, status="fail")
            return {"status": "requeued"}
        # No alternate node — fall through to terminal failure

    if is_error:
        tasks.fail_task(task.task_id, task.result)
        _emit_inference_event(task, node_id, duration_ms, result, status="fail")
        _bridge_blocking_completion_to_stream_consumer(task, error=True)
    else:
        task.status = "completed"
        task.done_event.set()
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
            "node_id": n.node_id[:8] + "...",
            "node_id_full": n.node_id,
            "cpu_cores": n.resources.cpu_cores,
            "ram_gb": n.resources.ram_gb,
            "os_name": n.resources.os_name,
            "cpu_load": getattr(n, "cpu_load", 0.0),
            "latency_ms": getattr(n, "latency_ms", 0.0),
            "context_size": getattr(n.resources, "context_size", 8192),
            "ollama_available": n.resources.ollama_available,
            "ollama_models": getattr(n.resources, "ollama_models", []),
            "vllm_available": getattr(n.resources, "vllm_available", False),
            "vllm_models": getattr(n.resources, "vllm_models", []),
            "mlx_available": getattr(n.resources, "mlx_available", False),
            "mlx_models": getattr(n.resources, "mlx_models", []),
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

    all_nodes = storage.get_all_nodes()
    
    # Aggregate models and their maximum context capacity across all nodes
    model_info = {} # model_id -> max_ctx
    now = int(time.time())
    
    for n in all_nodes:
        if n.owner_id != owner_id:
            continue
            
        ctx = getattr(n.resources, "context_size", 8192)
        node_models = (
            getattr(n.resources, "ollama_models", []) +
            getattr(n.resources, "vllm_models", []) +
            getattr(n.resources, "mlx_models", [])
        )
        for m in node_models:
            model_info[m] = max(model_info.get(m, 0), ctx)

    model_list = [
        {
            "id": m,
            "object": "model",
            "created": now,
            "owned_by": "llmesh-node",
            "context_length": ctx_len
        }
        for m, ctx_len in model_info.items()
    ]

    return {"object": "list", "data": model_list}

async def _process_chat_completion(
    req_model: str,
    messages: list[dict],
    api_key: str,
    session_id: str | None = None,
    want_stream: bool = False,
    num_ctx: int | None = None,
) -> tuple:
    owner_id = storage.authenticate_owner(api_key)
    if not owner_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

    store = get_session_store()

    if session_id is None:
        session_id = str(uuid.uuid4())

    stored_history = await store.get_messages(session_id, owner_id)
    full_messages = (stored_history or []) + messages

    inf_req = InferenceRequest(
        owner_id=owner_id,
        model=req_model,
        messages=full_messages,
        num_ctx=num_ctx
    )

    try:
        task_response = _route_inference(inf_req, stream=want_stream)
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
    x_session_id: str | None = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")

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
        req.model, messages, api_key, x_session_id, want_stream=req.stream, num_ctx=req.num_ctx
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

    if req.stream:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Streaming is not supported on /v1/messages. Use /v1/chat/completions with stream=true.",
                    "type": "invalid_request_error",
                }
            },
        )

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    task, err_resp, session_id, _owner_id, _stored_history, _incoming = await _process_chat_completion(
        req.model, messages, x_api_key, x_session_id, num_ctx=req.num_ctx
    )
    if err_resp:
        return err_resp

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
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login", response_class=HTMLResponse)
@limiter.limit(RATE_LIMIT_LOGIN)
async def login_submit(request: Request, api_key: str = Form(...)):
    owner_id = storage.authenticate_owner(api_key)
    if not owner_id:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid API Key"})

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
            all_node_models = (
                getattr(n.resources, "ollama_models", []) +
                getattr(n.resources, "vllm_models", []) +
                getattr(n.resources, "mlx_models", [])
            )
            for model in all_node_models:
                model_counts[model] = model_counts.get(model, 0) + 1
                
    # Sort by count descending, then alphabetically by name
    available_models = [
        {"name": name, "count": count}
        for name, count in sorted(model_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    
    # Calculate base_url from request
    base_url = str(request.base_url).rstrip('/')
    
    return templates.TemplateResponse(
        "dashboard.html", 
        {
            "request": request, 
            "nodes": owner_nodes, 
            "owner_id": owner_id, 
            "current_time": time.time(), 
            "available_models": available_models,
            "base_url": base_url,
            "version": APP_VERSION
        }
    )

@app.post("/dashboard/request_inference")
async def dashboard_submit_inference(
    request: Request,
    prompt: str = Form(...),
    model: str = Form(...),
    llmesh_session: str | None = Cookie(None)
):
    owner_id = _resolve_session(llmesh_session)
    if not owner_id:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    # Reuse the JSON logic by building the Pydantic model
    try:
        inf_req = InferenceRequest(owner_id=owner_id, prompt=prompt, model=model)
        result = _route_inference(inf_req, stream=True)
        
        # Redirect the user to the polling page for this specific task
        return RedirectResponse(
            url=f"/dashboard/task/{result.node_assigned}/{result.task_id}", 
            status_code=status.HTTP_303_SEE_OTHER
        )
    except HTTPException as e:
        # If no nodes available, show a simple error
        return HTMLResponse(f"<h3>Error: {e.detail}</h3><a href='/dashboard'>Back</a>", status_code=e.status_code)

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
        "task_status.html",
        {"request": request, "task_id": task_id, "node_id": node_id, "task": task, "owner_id": owner_id}
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


