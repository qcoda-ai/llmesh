"""
Verify every mutator in lib/hub/tasks routes through the persistence store
(D053). Uses a recording fake store; no aiosqlite needed. Companion to
test_task_store.py which exercises the SQLite backend directly.
"""
import asyncio

import pytest

from lib.hub import tasks as tasks_mod
from lib.hub import task_store as task_store_mod
from lib.hub.models import TaskKind


class RecordingStore:
    """Captures every store call so assertions can inspect the routing
    without spinning up a real DB. Methods match SQLiteTaskStore /
    MemoryTaskStore signatures."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def save_task(self, task, node_id, status="pending"):
        self.calls.append(("save_task", task.task_id, node_id, status))

    async def mark_status(self, task_id, status, node_id=None):
        self.calls.append(("mark_status", task_id, status, node_id))

    async def save_result(self, task_id, result, prompt_tokens=0,
                          completion_tokens=0, status=None):
        self.calls.append(("save_result", task_id, result, prompt_tokens,
                           completion_tokens, status))

    async def delete_task(self, task_id):
        self.calls.append(("delete_task", task_id))

    async def evict_expired(self, ttl_seconds):
        self.calls.append(("evict_expired", ttl_seconds))
        return 0

    async def load_persisted(self):
        return []

    async def close(self):
        pass


@pytest.fixture(autouse=True)
def isolated_store_and_state():
    """Replace the module singleton with a fresh recording store for each
    test, and clear in-memory dicts so tests don't leak state."""
    prev = task_store_mod._task_store
    store = RecordingStore()
    task_store_mod.set_task_store(store)
    tasks_mod._task_index.clear()
    tasks_mod._node_tasks.clear()
    yield store
    task_store_mod.set_task_store(prev)
    tasks_mod._task_index.clear()
    tasks_mod._node_tasks.clear()


def _run(coro):
    return asyncio.run(coro)


def _new_task(task_id="t1"):
    return tasks_mod.Task(
        task_id=task_id,
        kind=TaskKind.CHAT,
        owner_id="alice",
        messages=[{"role": "user", "content": "hi"}],
    )


def test_queue_task_for_node_writes_to_store(isolated_store_and_state):
    store = isolated_store_and_state
    task = _new_task("q1")
    _run(tasks_mod.queue_task_for_node("node-A", task))
    assert ("save_task", "q1", "node-A", "pending") in store.calls
    assert tasks_mod._task_index["q1"] is task
    assert task in tasks_mod._node_tasks["node-A"]


def test_get_pending_tasks_marks_claimed_in_store(isolated_store_and_state):
    store = isolated_store_and_state
    task = _new_task("p1")
    _run(tasks_mod.queue_task_for_node("node-A", task))
    store.calls.clear()
    claimed = _run(tasks_mod.get_pending_tasks("node-A"))
    assert len(claimed) == 1
    assert claimed[0].status == "claimed"
    assert ("mark_status", "p1", "claimed", "node-A") in store.calls


def test_record_task_result_persists_result(isolated_store_and_state):
    store = isolated_store_and_state
    task = _new_task("r1")
    _run(tasks_mod.queue_task_for_node("node-A", task))
    store.calls.clear()
    out = _run(tasks_mod.record_task_result("r1", "answer", 5, 10))
    assert out is task
    assert ("save_result", "r1", "answer", 5, 10, None) in store.calls


def test_complete_task_marks_completed(isolated_store_and_state):
    store = isolated_store_and_state
    task = _new_task("c1")
    _run(tasks_mod.queue_task_for_node("node-A", task))
    store.calls.clear()
    ok = _run(tasks_mod.complete_task("node-A", "c1", "done", 1, 2))
    assert ok is True
    assert task.status == "completed"
    assert task.done_event.is_set()
    assert ("save_result", "c1", "done", 1, 2, None) in store.calls
    assert ("mark_status", "c1", "completed", None) in store.calls


def test_fail_task_persists_failure(isolated_store_and_state):
    store = isolated_store_and_state
    task = _new_task("f1")
    _run(tasks_mod.queue_task_for_node("node-A", task))
    store.calls.clear()
    ok = _run(tasks_mod.fail_task("f1", "node offline"))
    assert ok is True
    assert task.status == "failed"
    assert ("save_result", "f1", "node offline", 0, 0, "failed") in store.calls


def test_requeue_task_writes_new_node(isolated_store_and_state):
    store = isolated_store_and_state
    task = _new_task("rq1")
    _run(tasks_mod.queue_task_for_node("node-A", task))
    task.status = "claimed"
    store.calls.clear()
    _run(tasks_mod.requeue_task(task, "node-B"))
    assert task.status == "pending"
    assert task in tasks_mod._node_tasks["node-B"]
    assert ("save_task", "rq1", "node-B", "pending") in store.calls


def test_prune_old_tasks_calls_evict(isolated_store_and_state):
    store = isolated_store_and_state
    task = _new_task("old1")
    _run(tasks_mod.queue_task_for_node("node-A", task))
    task.status = "completed"
    task.created_at = 0.0  # very old
    store.calls.clear()
    removed = _run(tasks_mod.prune_old_tasks(ttl_seconds=10))
    assert removed == 1
    assert ("evict_expired", 10) in store.calls
    assert "old1" not in tasks_mod._task_index


def test_missing_task_id_returns_none_or_false(isolated_store_and_state):
    """record_task_result / fail_task / complete_task on unknown IDs should
    return None/False without touching the store (no row to update)."""
    store = isolated_store_and_state
    out = _run(tasks_mod.record_task_result("nope", "x"))
    assert out is None
    failed = _run(tasks_mod.fail_task("nope", "x"))
    assert failed is False
    completed = _run(tasks_mod.complete_task("n", "nope", "x"))
    assert completed is False
    # No store calls should have fired for the missing-task paths
    assert not any(c[0] == "save_result" for c in store.calls)
    assert not any(c[0] == "mark_status" for c in store.calls)


def test_load_persisted_resets_claimed_to_pending(isolated_store_and_state):
    """A store returning a claimed row should drive load_persisted to reset
    it to pending in memory AND issue a mark_status call back to the store."""
    store = isolated_store_and_state

    # Replace the recorded load_persisted with a canned row
    async def fake_load():
        return [{
            "task_id": "recover-1",
            "node_id": "node-A",
            "owner_id": "alice",
            "kind": "chat",
            "model": "llama3",
            "status": "claimed",
            "payload": {"messages": [], "prompt": None, "num_ctx": None},
            "result": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "retries_left": 2,
            "initial_retries": 2,
            "attempted_nodes": set(),
            "session_id": None,
            "stream": False,
            "created_at": 0.0,
            "updated_at": 0.0,
        }]
    store.load_persisted = fake_load

    restored, reset = _run(tasks_mod.load_persisted(store))
    assert restored == 0
    assert reset == 1
    assert tasks_mod._task_index["recover-1"].status == "pending"
    assert ("mark_status", "recover-1", "pending", "node-A") in store.calls


def test_load_persisted_keeps_pending_status(isolated_store_and_state):
    store = isolated_store_and_state

    async def fake_load():
        return [{
            "task_id": "pend-1",
            "node_id": "node-B",
            "owner_id": "alice",
            "kind": "chat",
            "model": "llama3",
            "status": "pending",
            "payload": {"messages": [], "prompt": None, "num_ctx": None},
            "result": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "retries_left": 2,
            "initial_retries": 2,
            "attempted_nodes": set(),
            "session_id": None,
            "stream": False,
            "created_at": 0.0,
            "updated_at": 0.0,
        }]
    store.load_persisted = fake_load

    restored, reset = _run(tasks_mod.load_persisted(store))
    assert restored == 1
    assert reset == 0
    assert tasks_mod._task_index["pend-1"].status == "pending"
    assert tasks_mod._task_index["pend-1"] in tasks_mod._node_tasks["node-B"]


def test_rehydrated_task_gets_fresh_async_primitives(isolated_store_and_state):
    """asyncio.Queue / asyncio.Event are never persisted (D003). After
    load_persisted the in-memory Task must have a fresh done_event and
    stream_queue=None, regardless of what the previous process held."""
    store = isolated_store_and_state

    async def fake_load():
        return [{
            "task_id": "fresh-1",
            "node_id": "node-A",
            "owner_id": "alice",
            "kind": "chat",
            "model": "llama3",
            "status": "pending",
            "payload": {"messages": [], "prompt": None, "num_ctx": None},
            "result": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "retries_left": 2,
            "initial_retries": 2,
            "attempted_nodes": set(),
            "session_id": None,
            "stream": True,
            "created_at": 0.0,
            "updated_at": 0.0,
        }]
    store.load_persisted = fake_load

    _run(tasks_mod.load_persisted(store))
    task = tasks_mod._task_index["fresh-1"]
    assert task.done_event is not None
    assert task.done_event.is_set() is False
    assert task.stream_queue is None  # D003: stream_queue never persists
    assert task.stream is True         # but the flag does
