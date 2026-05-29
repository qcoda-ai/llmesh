"""
Unit tests for lib/hub/node_store.SQLiteNodeStore (D058 implementation of
D026 node-slice). Exercises CRUD + last_seen-based prune-on-load against an
on-disk SQLite DB so the WAL pragma and schema DDL actually apply.
"""
import asyncio
import os
import tempfile
import time

import pytest

from lib.hub import node_store
from lib.hub.models import Node, ResourceCaps

try:
    import aiosqlite  # noqa: F401
    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

needs_aiosqlite = pytest.mark.skipif(
    not _HAS_AIOSQLITE,
    reason="aiosqlite not installed; SQLiteNodeStore unreachable in this env",
)


def _make_node(node_id="node-A", owner="alice", token_hash="abc123",
               last_seen: float | None = None, fingerprint="fp-A"):
    return Node(
        node_id=node_id,
        owner_id=owner,
        resources=ResourceCaps(
            cpu_cores=4,
            ram_gb=16.0,
            os_name="linux",
            ollama_available=True,
        ),
        last_seen=last_seen if last_seen is not None else time.time(),
        node_token="plaintext-not-persisted",
        node_token_hash=token_hash,
        fingerprint=fingerprint,
    )


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db", prefix="llmesh-nodes-")
    os.close(fd)
    s = node_store.SQLiteNodeStore(path)
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
def test_save_node_then_load_returns_row(store):
    node = _make_node("node-A", token_hash="hashA")
    _run(store.save_node(node, fingerprint="fp-A"))
    rows = _run(store.load_persisted(max_age_sec=90.0))
    assert len(rows) == 1
    row = rows[0]
    assert row["node_id"] == "node-A"
    assert row["owner_id"] == "alice"
    assert row["node_token_hash"] == "hashA"
    assert row["fingerprint"] == "fp-A"
    assert row["resources"]["cpu_cores"] == 4
    assert row["resources"]["ram_gb"] == 16.0


@needs_aiosqlite
def test_load_persisted_drops_stale_rows(store):
    """last_seen older than max_age_sec must be removed by load_persisted —
    matches steady-state prune behaviour so a hub restart after a long
    downtime does not surface dead nodes."""
    fresh = _make_node("node-fresh", last_seen=time.time())
    stale = _make_node("node-stale", last_seen=time.time() - 300.0)
    _run(store.save_node(fresh, fingerprint="fp-fresh"))
    _run(store.save_node(stale, fingerprint="fp-stale"))

    rows = _run(store.load_persisted(max_age_sec=90.0))
    ids = {r["node_id"] for r in rows}
    assert ids == {"node-fresh"}

    # Stale row was deleted in-place: second load still excludes it.
    rows_again = _run(store.load_persisted(max_age_sec=90.0))
    assert {r["node_id"] for r in rows_again} == {"node-fresh"}


@needs_aiosqlite
def test_save_node_upserts_on_conflict(store):
    """Re-registering the same fingerprint must overwrite, not error."""
    original = _make_node("node-A", token_hash="hashOLD")
    _run(store.save_node(original, fingerprint="fp-A"))

    refreshed = _make_node("node-A", token_hash="hashNEW")
    _run(store.save_node(refreshed, fingerprint="fp-A"))

    rows = _run(store.load_persisted())
    assert len(rows) == 1
    assert rows[0]["node_token_hash"] == "hashNEW"


@needs_aiosqlite
def test_update_heartbeat_changes_last_seen(store):
    node = _make_node("node-A", last_seen=1000.0)
    _run(store.save_node(node, fingerprint="fp-A"))
    _run(store.update_heartbeat("node-A", last_seen=time.time(),
                                 cpu_load=0.42, latency_ms=12.0))
    rows = _run(store.load_persisted(max_age_sec=90.0))
    assert len(rows) == 1
    assert rows[0]["cpu_load"] == 0.42
    assert rows[0]["latency_ms"] == 12.0


@needs_aiosqlite
def test_delete_node_removes_row(store):
    node = _make_node("node-A")
    _run(store.save_node(node, fingerprint="fp-A"))
    _run(store.delete_node("node-A"))
    rows = _run(store.load_persisted())
    assert rows == []


def test_memory_store_is_noop():
    """MemoryNodeStore satisfies the interface without touching disk."""
    mem = node_store.MemoryNodeStore()
    node = _make_node("node-A")
    _run(mem.save_node(node, fingerprint="fp-A"))
    _run(mem.update_heartbeat("node-A", last_seen=time.time(), cpu_load=0.1, latency_ms=1.0))
    _run(mem.delete_node("node-A"))
    rows = _run(mem.load_persisted())
    assert rows == []
