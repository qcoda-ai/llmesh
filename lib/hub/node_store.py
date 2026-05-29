"""
Node-registry persistence layer (D026 node-slice implementation, see decisions.md::D058).

Mirrors the SQLiteTaskStore pattern in `lib/hub/task_store.py`. The hub's
in-memory `storage._nodes` dict remains authoritative at runtime — every
mutation writes through to this store as an additive, best-effort durability
hook. Write failures are logged and swallowed; in-memory state is never
rolled back. On startup the hub reads back rows whose `last_seen` is fresher
than the standard 90s prune cutoff so already-connected agents continue
operating under their existing node_id without re-registering.

Persisted attributes are JSON-serializable only. The node_token plaintext is
**never** persisted — only its sha256 hex digest. A running agent's in-memory
token validates against the restored hash after a hub restart (see
storage.verify_node_token dual path).

Schema is the nodes-table slice of feature_hub_state_durability.md §4.2.
The companion `tasks` table belongs to D053 (already shipped) and lives in
task_store.py.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger("llmesh.hub.node_store")


_SQLITE_DDL = [
    """CREATE TABLE IF NOT EXISTS nodes (
        node_id          TEXT PRIMARY KEY,
        owner_id         TEXT NOT NULL,
        node_token_hash  TEXT NOT NULL,
        fingerprint      TEXT NOT NULL,
        resources_json   TEXT NOT NULL,
        last_seen        REAL NOT NULL,
        cpu_load         REAL NOT NULL DEFAULT 0.0,
        latency_ms       REAL NOT NULL DEFAULT 0.0,
        created_at       REAL NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_nodes_owner       ON nodes(owner_id)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_fingerprint ON nodes(fingerprint)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_last_seen   ON nodes(last_seen)",
]


def _node_row(node, fingerprint: str, now: float) -> tuple:
    return (
        node.node_id,
        node.owner_id,
        node.node_token_hash,
        fingerprint,
        json.dumps(node.resources.model_dump() if hasattr(node.resources, "model_dump") else node.resources.dict()),
        node.last_seen,
        node.cpu_load,
        node.latency_ms,
        now,
    )


class MemoryNodeStore:
    """No-op store used when aiosqlite is missing or DB is `:memory:` for tests.

    Methods mirror SQLiteNodeStore's surface so callers do not branch on backend.
    All writes are silently dropped; `load_persisted` returns an empty list.
    """

    async def save_node(self, node, fingerprint: str) -> None:
        return

    async def update_heartbeat(self, node_id: str, last_seen: float,
                                cpu_load: float, latency_ms: float,
                                resources_json: str | None = None) -> None:
        return

    async def delete_node(self, node_id: str) -> None:
        return

    async def load_persisted(self, max_age_sec: float = 90.0) -> list[dict]:
        return []

    async def close(self) -> None:
        return


class SQLiteNodeStore:
    """aiosqlite-backed node durability layer. Lazy init opens the DB on
    first call and applies `CREATE TABLE IF NOT EXISTS`; WAL mode for
    concurrent readers (mirrors task_store + sessions).
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            import aiosqlite
            self._conn = await aiosqlite.connect(self._db_path)
            await self._conn.execute("PRAGMA journal_mode=WAL")
            for stmt in _SQLITE_DDL:
                await self._conn.execute(stmt)
            await self._conn.commit()
            self._initialized = True

    async def save_node(self, node, fingerprint: str) -> None:
        try:
            await self._ensure_init()
            now = time.time()
            row = _node_row(node, fingerprint, now)
            await self._conn.execute(
                """
                INSERT INTO nodes (
                    node_id, owner_id, node_token_hash, fingerprint,
                    resources_json, last_seen, cpu_load, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    owner_id        = excluded.owner_id,
                    node_token_hash = excluded.node_token_hash,
                    fingerprint     = excluded.fingerprint,
                    resources_json  = excluded.resources_json,
                    last_seen       = excluded.last_seen,
                    cpu_load        = excluded.cpu_load,
                    latency_ms      = excluded.latency_ms
                """,
                row,
            )
            await self._conn.commit()
        except Exception as exc:
            logger.warning("save_node failed for %s: %s", node.node_id, exc)

    async def update_heartbeat(self, node_id: str, last_seen: float,
                                cpu_load: float, latency_ms: float,
                                resources_json: str | None = None) -> None:
        try:
            await self._ensure_init()
            if resources_json is not None:
                await self._conn.execute(
                    """UPDATE nodes
                       SET last_seen = ?, cpu_load = ?, latency_ms = ?, resources_json = ?
                       WHERE node_id = ?""",
                    (last_seen, cpu_load, latency_ms, resources_json, node_id),
                )
            else:
                await self._conn.execute(
                    """UPDATE nodes
                       SET last_seen = ?, cpu_load = ?, latency_ms = ?
                       WHERE node_id = ?""",
                    (last_seen, cpu_load, latency_ms, node_id),
                )
            await self._conn.commit()
        except Exception as exc:
            logger.warning("update_heartbeat failed for %s: %s", node_id, exc)

    async def delete_node(self, node_id: str) -> None:
        try:
            await self._ensure_init()
            await self._conn.execute("DELETE FROM nodes WHERE node_id = ?", (node_id,))
            await self._conn.commit()
        except Exception as exc:
            logger.warning("delete_node failed for %s: %s", node_id, exc)

    async def load_persisted(self, max_age_sec: float = 90.0) -> list[dict]:
        """Return all rows whose last_seen is fresher than max_age_sec.
        Stale rows are deleted in-place so the table self-prunes on startup
        without a separate cleanup pass. Caller reconstructs Node objects
        and inserts them into the in-memory `_nodes` dict.
        """
        try:
            await self._ensure_init()
            cutoff = time.time() - max_age_sec
            await self._conn.execute("DELETE FROM nodes WHERE last_seen < ?", (cutoff,))
            await self._conn.commit()
            async with self._conn.execute(
                """SELECT node_id, owner_id, node_token_hash, fingerprint,
                          resources_json, last_seen, cpu_load, latency_ms, created_at
                   FROM nodes"""
            ) as cursor:
                rows = await cursor.fetchall()
            return [
                {
                    "node_id": r[0],
                    "owner_id": r[1],
                    "node_token_hash": r[2],
                    "fingerprint": r[3],
                    "resources": json.loads(r[4]),
                    "last_seen": r[5],
                    "cpu_load": r[6],
                    "latency_ms": r[7],
                    "created_at": r[8],
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning("load_persisted failed: %s", exc)
            return []

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._initialized = False


_node_store: Any = None


def get_node_store():
    """Module-level singleton accessor. Backend chosen from env:
      - NODE_DB or TASK_DB or SESSION_DB == ':memory:'  → MemoryNodeStore
      - aiosqlite import fails                          → MemoryNodeStore (warn)
      - otherwise                                       → SQLiteNodeStore at the resolved path
    """
    global _node_store
    if _node_store is not None:
        return _node_store

    db_path = (
        os.getenv("NODE_DB")
        or os.getenv("TASK_DB")
        or os.getenv("SESSION_DB")
        or "./nodes.db"
    )
    if db_path == ":memory:":
        _node_store = MemoryNodeStore()
        return _node_store

    try:
        import aiosqlite  # noqa: F401
        _node_store = SQLiteNodeStore(db_path)
    except ImportError:
        logger.warning("aiosqlite not available, falling back to in-memory node store")
        _node_store = MemoryNodeStore()
    return _node_store


def set_node_store(store) -> None:
    """Test injection hook. Replace the module-level singleton."""
    global _node_store
    _node_store = store


def reset_node_store() -> None:
    """Clear the singleton so the next get_node_store() rebuilds from env."""
    global _node_store
    _node_store = None
