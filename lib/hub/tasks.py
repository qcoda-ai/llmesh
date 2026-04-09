import asyncio
import os
import time
from typing import Dict, List, Optional

TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", "3600"))

class Task:
    def __init__(self, task_id: str, prompt: str = None, messages: list[dict] = None,
                 model: str = "llama3", owner_id: str = "", retries_left: int = 2,
                 num_ctx: int = None):
        self.task_id = task_id
        self.prompt = prompt
        self.messages = messages or []
        self.model = model
        self.owner_id = owner_id
        self.num_ctx = num_ctx
        self.status = "pending"
        self.result = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.start_time = time.time()
        self.created_at: float = time.time()
        self.session_id: str | None = None
        self.done_event = asyncio.Event()
        self.retries_left: int = retries_left
        self.initial_retries: int = retries_left
        self.attempted_nodes: set[str] = set()
        # Streaming fields
        self.stream: bool = False
        self.stream_queue: Optional[asyncio.Queue] = None
        self.stream_cancelled: bool = False

# Per-node task lists (for pending task polling by nodes)
_node_tasks: Dict[str, List[Task]] = {}
# Flat index for O(1) status lookup by task_id
_task_index: Dict[str, Task] = {}

def queue_task_for_node(node_id: str, task: Task):
    if node_id not in _node_tasks:
        _node_tasks[node_id] = []
    _node_tasks[node_id].append(task)
    _task_index[task.task_id] = task

def get_pending_tasks(node_id: str) -> List[Task]:
    pending = [t for t in _node_tasks.get(node_id, []) if t.status == "pending"]
    for p in pending:
        p.status = "claimed"
    return pending

def record_task_result(task_id: str, result: str,
                       prompt_tokens: int = 0, completion_tokens: int = 0) -> "Task | None":
    """Update result fields and return the Task. Does NOT signal done_event."""
    t = _task_index.get(task_id)
    if t is None:
        return None
    t.result = result
    t.prompt_tokens = prompt_tokens
    t.completion_tokens = completion_tokens
    return t

def complete_task(node_id: str, task_id: str, result: str,
                  prompt_tokens: int = 0, completion_tokens: int = 0):
    """Legacy helper — records result and immediately signals done_event as success."""
    t = record_task_result(task_id, result, prompt_tokens, completion_tokens)
    if t is None:
        return False
    t.status = "completed"
    t.done_event.set()
    return True

def fail_task(task_id: str, reason: str) -> bool:
    """Mark task as failed and signal done_event so waiters can unblock."""
    t = _task_index.get(task_id)
    if t is None:
        return False
    t.status = "failed"
    t.result = reason
    t.done_event.set()
    return True

def requeue_task(task: Task, new_node_id: str) -> None:
    """Move task to a new node's queue without touching done_event."""
    task.status = "pending"
    task.start_time = time.time()  # reset timer for the new attempt
    if new_node_id not in _node_tasks:
        _node_tasks[new_node_id] = []
    _node_tasks[new_node_id].append(task)

def get_tasks_for_node(node_id: str) -> List[Task]:
    """Return all tasks queued for a node regardless of status."""
    return list(_node_tasks.get(node_id, []))

def get_task_status(node_id: str, task_id: str) -> "Task | None":
    return _task_index.get(task_id)


def prune_old_tasks(ttl_seconds: int) -> int:
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
    return len(to_remove)
