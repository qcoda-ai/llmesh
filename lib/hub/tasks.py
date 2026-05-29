import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from .models import TaskKind
from . import task_store as _task_store_mod

logger = logging.getLogger("llmesh.hub.tasks")

TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", "3600"))


class Task:
    """Generic task envelope. `kind` selects payload/result shape:
      - CHAT:      payload = {"messages": list[dict], "prompt": str|None, "num_ctx": int|None}
                   result  = str (assistant text)
      - EMBEDDING: payload = {"input": list[str]}      # always normalized to list
                   result  = list[list[float]]         # one vector per input, in submission order
      - IMAGE:     payload = {"prompt": str, "negative_prompt": str|None,
                              "size": "WxH", "n": int, "seed": int|None,
                              "quality": "draft"|"quality"}
                   result  = list[str]                # base64-encoded PNG, one per image (D064)
    """

    def __init__(
        self,
        task_id: str,
        kind: TaskKind = TaskKind.CHAT,
        payload: dict | None = None,
        model: str = "llama3",
        owner_id: str = "",
        retries_left: int = 2,
        # Backwards-compat shim: callers may still pass prompt/messages/num_ctx
        # for chat tasks. Promoted into `payload` if `payload` is not given.
        prompt: str | None = None,
        messages: list[dict] | None = None,
        num_ctx: int | None = None,
    ):
        self.task_id = task_id
        self.kind = TaskKind(kind) if not isinstance(kind, TaskKind) else kind
        if payload is None:
            if self.kind is TaskKind.CHAT:
                payload = {
                    "messages": messages or [],
                    "prompt": prompt,
                    "num_ctx": num_ctx,
                }
            elif self.kind is TaskKind.IMAGE:
                # Image-gen callers pass payload= explicitly; this branch is a
                # safety net so tests / shims that construct a bare image Task
                # without payload still produce a valid envelope.
                payload = {
                    "prompt": "", "negative_prompt": None,
                    "size": "1024x1024", "n": 1, "seed": None,
                    "quality": "draft",
                }
            else:
                payload = {"input": []}
        self.payload: dict = payload
        self.model = model
        self.owner_id = owner_id
        self.status = "pending"
        self.result: Any = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # D068: batcher telemetry — populated by /stream done frame for
        # streamed tasks. Zero on non-streamed tasks. Surfaced in metrics +
        # dashboard task viewer so operators can see batcher convergence
        # without grepping agent logs.
        self.stream_batches: int = 0
        self.stream_final_size: int = 0
        self.start_time = time.time()
        self.created_at: float = time.time()
        self.session_id: str | None = None
        self.done_event = asyncio.Event()
        self.retries_left: int = retries_left
        self.initial_retries: int = retries_left
        self.attempted_nodes: set[str] = set()
        # Streaming fields (chat only)
        self.stream: bool = False
        self.stream_queue: Optional[asyncio.Queue] = None
        self.stream_cancelled: bool = False

    # --- Compat shims so existing chat callers keep working unchanged ---
    @property
    def messages(self) -> list[dict]:
        return self.payload.get("messages", []) if self.kind is TaskKind.CHAT else []

    @property
    def prompt(self) -> str | None:
        return self.payload.get("prompt") if self.kind is TaskKind.CHAT else None

    @property
    def num_ctx(self) -> int | None:
        return self.payload.get("num_ctx") if self.kind is TaskKind.CHAT else None

    @property
    def max_tokens(self) -> int | None:
        return self.payload.get("max_tokens") if self.kind is TaskKind.CHAT else None


# Per-node task lists (for pending task polling by nodes)
_node_tasks: Dict[str, List[Task]] = {}
# Flat index for O(1) status lookup by task_id
_task_index: Dict[str, Task] = {}


async def queue_task_for_node(node_id: str, task: Task):
    if node_id not in _node_tasks:
        _node_tasks[node_id] = []
    _node_tasks[node_id].append(task)
    _task_index[task.task_id] = task
    await _task_store_mod.get_task_store().save_task(task, node_id, status="pending")


async def get_pending_tasks(node_id: str) -> List[Task]:
    pending = [t for t in _node_tasks.get(node_id, []) if t.status == "pending"]
    store = _task_store_mod.get_task_store()
    for p in pending:
        p.status = "claimed"
        await store.mark_status(p.task_id, "claimed", node_id=node_id)
    return pending


async def record_task_result(
    task_id: str,
    result: Any,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> "Task | None":
    """Update result fields and return the Task. Does NOT signal done_event."""
    t = _task_index.get(task_id)
    if t is None:
        return None
    t.result = result
    t.prompt_tokens = prompt_tokens
    t.completion_tokens = completion_tokens
    await _task_store_mod.get_task_store().save_result(
        task_id, result, prompt_tokens, completion_tokens,
    )
    return t


async def complete_task(node_id: str, task_id: str, result: Any,
                        prompt_tokens: int = 0, completion_tokens: int = 0):
    """Legacy helper — records result and immediately signals done_event as success."""
    t = await record_task_result(task_id, result, prompt_tokens, completion_tokens)
    if t is None:
        return False
    t.status = "completed"
    t.done_event.set()
    await _task_store_mod.get_task_store().mark_status(task_id, "completed")
    return True


async def fail_task(task_id: str, reason: str) -> bool:
    """Mark task as failed and signal done_event so waiters can unblock."""
    t = _task_index.get(task_id)
    if t is None:
        return False
    t.status = "failed"
    t.result = reason
    t.done_event.set()
    await _task_store_mod.get_task_store().save_result(
        task_id, reason, t.prompt_tokens, t.completion_tokens, status="failed",
    )
    return True


async def requeue_task(task: Task, new_node_id: str) -> None:
    """Move task to a new node's queue without touching done_event."""
    task.status = "pending"
    task.start_time = time.time()  # reset timer for the new attempt
    if new_node_id not in _node_tasks:
        _node_tasks[new_node_id] = []
    _node_tasks[new_node_id].append(task)
    await _task_store_mod.get_task_store().save_task(task, new_node_id, status="pending")


def get_tasks_for_node(node_id: str) -> List[Task]:
    """Return all tasks queued for a node regardless of status."""
    return list(_node_tasks.get(node_id, []))


def get_task_status(node_id: str, task_id: str) -> "Task | None":
    return _task_index.get(task_id)


def drop_node_queue(node_id: str) -> None:
    """Remove the per-node task list entry (D035 — fixes the empty-list residue
    leak that prune_inactive_nodes used to leave behind for every disconnected
    node)."""
    _node_tasks.pop(node_id, None)


async def prune_old_tasks(ttl_seconds: int) -> int:
    """Remove completed/failed tasks older than ttl_seconds. Returns count removed."""
    now = time.time()
    to_remove = [
        task_id for task_id, task in _task_index.items()
        if task.status in ("completed", "failed")
        and (now - task.created_at) > ttl_seconds
    ]
    for task_id in to_remove:
        task = _task_index.pop(task_id)
        for node_task_list in _node_tasks.values():
            try:
                node_task_list.remove(task)
            except ValueError:
                pass
    await _task_store_mod.get_task_store().evict_expired(ttl_seconds)
    return len(to_remove)


def _rehydrate_task(row: dict) -> Task:
    """Reconstruct a Task object from a load_persisted row. asyncio primitives
    (`done_event`, `stream_queue`) get fresh instances; stream-related runtime
    state is intentionally lost — D003 documents mid-flight streams as
    unrecoverable across a restart."""
    t = Task(
        task_id=row["task_id"],
        kind=row["kind"],
        payload=row["payload"],
        model=row["model"],
        owner_id=row["owner_id"],
        retries_left=row["retries_left"],
    )
    t.status = row["status"]
    t.result = row["result"]
    t.prompt_tokens = row["prompt_tokens"]
    t.completion_tokens = row["completion_tokens"]
    t.initial_retries = row["initial_retries"]
    t.attempted_nodes = set(row["attempted_nodes"])
    t.session_id = row["session_id"]
    t.stream = row["stream"]
    t.created_at = row["created_at"]
    return t


async def load_persisted(store=None) -> tuple[int, int]:
    """Restore pending and claimed tasks from the persistence store into the
    in-memory dicts. Claimed rows reset to pending (status update + DB write)
    so the next eligible node picks them up. Returns (pending_restored,
    claimed_reset). Safe to call before any tasks have been queued; idempotent
    when the store has no rows.
    """
    if store is None:
        store = _task_store_mod.get_task_store()
    rows = await store.load_persisted()
    pending_restored = 0
    claimed_reset = 0
    for row in rows:
        was_claimed = row["status"] == "claimed"
        task = _rehydrate_task(row)
        if was_claimed:
            task.status = "pending"
            claimed_reset += 1
            await store.mark_status(task.task_id, "pending", node_id=row["node_id"])
        else:
            pending_restored += 1
        node_id = row["node_id"]
        _task_index[task.task_id] = task
        if node_id:
            _node_tasks.setdefault(node_id, []).append(task)
    return pending_restored, claimed_reset
