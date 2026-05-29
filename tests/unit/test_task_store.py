"""
Unit tests for lib/hub/task_store.SQLiteTaskStore (D053 implementation of
D003). Exercises the persistence interface against an on-disk SQLite DB so
the WAL pragma and schema DDL are actually applied.
"""
import asyncio
import json
import os
import tempfile

import pytest

from lib.hub import tasks as tasks_mod
from lib.hub import task_store
from lib.hub.models import TaskKind

# SQLiteTaskStore requires aiosqlite. The hub itself falls back to
# MemoryTaskStore when aiosqlite is missing — guard the SQLite-specific
# round-trip tests with skipif so the MemoryTaskStore test still runs.
try:
    import aiosqlite  # noqa: F401
    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

needs_aiosqlite = pytest.mark.skipif(
    not _HAS_AIOSQLITE,
    reason="aiosqlite not installed; SQLiteTaskStore unreachable in this env",
)


def _new_task(task_id="t1", kind=TaskKind.CHAT, owner="alice"):
    """Build a Task without going through queue_task_for_node (which would
    write to the module-level store). Direct construction lets the test
    drive the store explicitly."""
    return tasks_mod.Task(
        task_id=task_id,
        kind=kind,
        owner_id=owner,
        messages=[{"role": "user", "content": "hello"}] if kind is TaskKind.CHAT else None,
    )


@pytest.fixture
def store():
    """Fresh SQLiteTaskStore against an isolated tempfile DB. aiosqlite's
    `:memory:` URI does not share state across connections, so a real file
    is the simplest way to verify durability across save → load round trips
    within a single test."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="llmesh-test-")
    os.close(fd)
    s = task_store.SQLiteTaskStore(path)
    yield s
    asyncio.run(s.close())
    for ext in ("", "-wal", "-shm"):
        try:
            os.unlink(path + ext)
        except FileNotFoundError:
            pass


def _run(coro):
    return asyncio.run(coro)


@needs_aiosqlite
def test_save_task_then_load_persisted_returns_pending(store):
    task = _new_task("save-1")
    _run(store.save_task(task, node_id="node-A", status="pending"))
    rows = _run(store.load_persisted())
    assert len(rows) == 1
    row = rows[0]
    assert row["task_id"] == "save-1"
    assert row["node_id"] == "node-A"
    assert row["status"] == "pending"
    assert row["owner_id"] == "alice"
    assert row["kind"] == "chat"
    assert row["payload"]["messages"][0]["content"] == "hello"


@needs_aiosqlite
def test_mark_status_round_trip(store):
    task = _new_task("mark-1")
    _run(store.save_task(task, node_id="node-A", status="pending"))
    _run(store.mark_status("mark-1", "claimed"))
    rows = _run(store.load_persisted())
    assert rows[0]["status"] == "claimed"


@needs_aiosqlite
def test_save_result_populates_fields(store):
    task = _new_task("res-1")
    _run(store.save_task(task, node_id="node-A", status="pending"))
    _run(store.save_result("res-1", "the answer", prompt_tokens=12, completion_tokens=34))
    # save_result without status leaves row in pending — still returned
    rows = _run(store.load_persisted())
    assert rows[0]["result"] == "the answer"
    assert rows[0]["prompt_tokens"] == 12
    assert rows[0]["completion_tokens"] == 34


@needs_aiosqlite
def test_save_result_with_status_terminal_removes_from_load_persisted(store):
    task = _new_task("done-1")
    _run(store.save_task(task, node_id="node-A", status="pending"))
    _run(store.save_result("done-1", "ok", 1, 1, status="completed"))
    rows = _run(store.load_persisted())
    assert rows == []  # load_persisted excludes terminal rows


@needs_aiosqlite
def test_completed_and_failed_excluded_from_load_persisted(store):
    t1 = _new_task("p1")
    t2 = _new_task("p2")
    t3 = _new_task("p3")
    _run(store.save_task(t1, node_id="n", status="pending"))
    _run(store.save_task(t2, node_id="n", status="completed"))
    _run(store.save_task(t3, node_id="n", status="failed"))
    rows = _run(store.load_persisted())
    assert {r["task_id"] for r in rows} == {"p1"}


@needs_aiosqlite
def test_evict_expired_removes_terminal_only(store):
    import time
    t_pending = _new_task("p")
    t_done = _new_task("d")
    t_pending.created_at = 0.0
    t_done.created_at = 0.0
    _run(store.save_task(t_pending, node_id="n", status="pending"))
    _run(store.save_task(t_done, node_id="n", status="completed"))
    # Both rows were just written so updated_at = now. ttl=0 means anything
    # older than `now - 0` = now → strict `< now` excludes the just-written
    # rows. Sleep briefly so the cutoff passes the write timestamp.
    time.sleep(0.01)
    removed = _run(store.evict_expired(ttl_seconds=0))
    assert removed == 1  # only the completed row
    rows = _run(store.load_persisted())
    assert {r["task_id"] for r in rows} == {"p"}


@needs_aiosqlite
def test_embedding_payload_round_trip(store):
    task = tasks_mod.Task(
        task_id="emb-1",
        kind=TaskKind.EMBEDDING,
        payload={"input": ["alpha", "beta"]},
        owner_id="alice",
        model="nomic-embed-text",
    )
    _run(store.save_task(task, node_id="n", status="pending"))
    rows = _run(store.load_persisted())
    assert rows[0]["kind"] == "embedding"
    assert rows[0]["payload"]["input"] == ["alpha", "beta"]


@needs_aiosqlite
def test_attempted_nodes_preserved_as_set(store):
    task = _new_task("att-1")
    task.attempted_nodes = {"node-A", "node-B"}
    _run(store.save_task(task, node_id="node-C", status="pending"))
    rows = _run(store.load_persisted())
    assert rows[0]["attempted_nodes"] == {"node-A", "node-B"}


@needs_aiosqlite
def test_stream_flag_persists(store):
    task = _new_task("stream-1")
    task.stream = True
    _run(store.save_task(task, node_id="n", status="pending"))
    rows = _run(store.load_persisted())
    assert rows[0]["stream"] is True


@needs_aiosqlite
def test_save_task_is_upsert(store):
    """Same task_id written twice updates in place, no UNIQUE-violation."""
    task = _new_task("up-1")
    _run(store.save_task(task, node_id="n1", status="pending"))
    _run(store.save_task(task, node_id="n2", status="claimed"))
    rows = _run(store.load_persisted())
    assert len(rows) == 1
    assert rows[0]["node_id"] == "n2"
    assert rows[0]["status"] == "claimed"


def test_memory_store_is_noop():
    """MemoryTaskStore satisfies the interface but persists nothing."""
    s = task_store.MemoryTaskStore()
    task = _new_task("mem-1")
    _run(s.save_task(task, node_id="n", status="pending"))
    _run(s.mark_status("mem-1", "claimed"))
    _run(s.save_result("mem-1", "x", 1, 1))
    rows = _run(s.load_persisted())
    assert rows == []
    removed = _run(s.evict_expired(0))
    assert removed == 0
