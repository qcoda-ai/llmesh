"""
Unit tests for lib/hub/storage.load_persisted_nodes (D058 startup recovery).

Verifies that:
  * Persisted rows hydrate back into the in-memory `_nodes` dict.
  * The restored Node has empty plaintext + populated hash + populated fingerprint.
  * verify_node_token (hash path) accepts the same plaintext the agent
    originally received at /register time.
  * A node that re-registers after restore overwrites the hash-only entry
    with fresh plaintext, switching the verify path back to plaintext compare.
"""
import asyncio
import os
import tempfile
import time

import pytest

from lib.hub import node_store, storage
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


def _make_node(node_id="node-A", token="plaintext-1", fingerprint="fp-A"):
    return Node(
        node_id=node_id,
        owner_id="alice",
        resources=ResourceCaps(
            cpu_cores=4, ram_gb=16.0, os_name="linux", ollama_available=True,
        ),
        last_seen=time.time(),
        node_token=token,
        node_token_hash=storage._hash_token(token),
        fingerprint=fingerprint,
    )


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def sqlite_store():
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


@pytest.fixture(autouse=True)
def isolate_registry():
    storage._nodes.clear()
    yield
    storage._nodes.clear()


@needs_aiosqlite
def test_load_persisted_nodes_hydrates_memory(sqlite_store):
    token = "the-real-token"
    node = _make_node(token=token)
    _run(sqlite_store.save_node(node, fingerprint="fp-A"))

    node_store.set_node_store(sqlite_store)
    try:
        restored = _run(storage.load_persisted_nodes(max_age_sec=90.0))
    finally:
        node_store.reset_node_store()

    assert restored == 1
    assert "node-A" in storage._nodes

    n = storage._nodes["node-A"]
    assert n.node_token == ""  # plaintext NOT restored
    assert n.node_token_hash == storage._hash_token(token)
    assert n.fingerprint == "fp-A"
    assert n.owner_id == "alice"
    assert n.resources.cpu_cores == 4


@needs_aiosqlite
def test_restored_node_verifies_via_hash_path(sqlite_store):
    """The running agent still presents the plaintext it got at /register —
    verify_node_token must accept it via the hash fallback."""
    token = "the-real-token"
    node = _make_node(token=token)
    _run(sqlite_store.save_node(node, fingerprint="fp-A"))

    node_store.set_node_store(sqlite_store)
    try:
        _run(storage.load_persisted_nodes(max_age_sec=90.0))
    finally:
        node_store.reset_node_store()

    assert storage.verify_node_token("node-A", token) is True
    assert storage.verify_node_token("node-A", "wrong") is False


@needs_aiosqlite
def test_re_register_after_restore_switches_to_plaintext_path(sqlite_store):
    """After a restored node re-registers (or any code overwrites the in-memory
    Node with one that has plaintext), verify falls back to the plaintext
    path. Hash path only matters until plaintext is repopulated."""
    original_token = "old-token"
    node = _make_node(token=original_token)
    _run(sqlite_store.save_node(node, fingerprint="fp-A"))

    node_store.set_node_store(sqlite_store)
    try:
        _run(storage.load_persisted_nodes(max_age_sec=90.0))
    finally:
        node_store.reset_node_store()

    # Simulate re-register: in-memory Node now carries fresh plaintext.
    storage._nodes["node-A"] = _make_node(token="brand-new")
    assert storage.verify_node_token("node-A", "brand-new") is True
    # Old hash no longer matches the new plaintext-only verify path.
    assert storage.verify_node_token("node-A", original_token) is False


@needs_aiosqlite
def test_load_persisted_skips_stale_rows(sqlite_store):
    fresh = _make_node("node-fresh")
    stale = _make_node("node-stale")
    stale.last_seen = time.time() - 1000.0
    _run(sqlite_store.save_node(fresh, fingerprint="fp-fresh"))
    _run(sqlite_store.save_node(stale, fingerprint="fp-stale"))

    node_store.set_node_store(sqlite_store)
    try:
        restored = _run(storage.load_persisted_nodes(max_age_sec=90.0))
    finally:
        node_store.reset_node_store()

    assert restored == 1
    assert "node-fresh" in storage._nodes
    assert "node-stale" not in storage._nodes


@needs_aiosqlite
def test_load_persisted_with_corrupt_resources_skips_row(sqlite_store):
    """A row whose resources_json fails to deserialize must not crash startup;
    the row is logged and skipped."""
    import json
    await_ = asyncio.run
    await_(sqlite_store._ensure_init())
    now = time.time()
    await_(sqlite_store._conn.execute(
        """INSERT INTO nodes (node_id, owner_id, node_token_hash, fingerprint,
                              resources_json, last_seen, cpu_load, latency_ms, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("bad-node", "alice", "h", "fp",
         json.dumps({"missing": "fields"}),  # ResourceCaps requires cpu_cores/ram_gb/etc
         now, 0.0, 0.0, now),
    ))
    await_(sqlite_store._conn.commit())

    node_store.set_node_store(sqlite_store)
    try:
        restored = _run(storage.load_persisted_nodes(max_age_sec=90.0))
    finally:
        node_store.reset_node_store()

    assert restored == 0
    assert "bad-node" not in storage._nodes
