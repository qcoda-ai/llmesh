"""
Task-queue persistence layer (D003 implementation, see decisions.md::D053).

Mirrors the SQLiteSessionStore pattern in `lib/hub/sessions.py`. The hub's
in-memory `_task_index` / `_node_tasks` dicts (lib/hub/tasks.py) remain
authoritative at runtime — every mutation writes through to this store as
an additive, best-effort durability hook. Write failures are logged and
swallowed; in-memory state is never rolled back. On startup the hub reads
back pending and claimed rows, resetting claimed → pending so any task the
previous process forgot mid-flight gets re-routed.

Persisted attributes are JSON-serializable only. asyncio primitives on Task
(`done_event`, `stream_queue`) are never persisted — D003 explicitly accepts
that mid-flight streaming sessions cannot survive a hub restart.

Schema is the tasks-table slice of feature_hub_state_durability.md §4.2.
The companion `nodes` table from that spec belongs to the D026 follow-up
and is out of scope here.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

from .models import TaskKind

logger = logging.getLogger("llmesh.hub.task_store")


_SQLITE_DDL = [
    """CREATE TABLE IF NOT EXISTS tasks (
        task_id              TEXT PRIMARY KEY,
        node_id              TEXT,
        owner_id             TEXT NOT NULL,
        kind                 TEXT NOT NULL,
        model                TEXT NOT NULL,
        status               TEXT NOT NULL,
        payload_json         TEXT NOT NULL,
        result_json          TEXT,
        prompt_tokens        INTEGER NOT NULL DEFAULT 0,
        completion_tokens    INTEGER NOT NULL DEFAULT 0,
        retries_left         INTEGER NOT NULL DEFAULT 0,
        initial_retries      INTEGER NOT NULL DEFAULT 0,
        attempted_nodes_json TEXT NOT NULL DEFAULT '[]',
        session_id           TEXT,
        stream_flag          INTEGER NOT NULL DEFAULT 0,
        created_at           REAL NOT NULL,
        updated_at           REAL NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_tasks_node_status ON tasks(node_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_owner       ON tasks(owner_id)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_updated_at  ON tasks(updated_at)",
]


def _task_row(task, node_id: str | None, status: str, now: float) -> tuple:
    return (
        task.task_id,
        node_id,
        task.owner_id,
        task.kind.value if isinstance(task.kind, TaskKind) else str(task.kind),
        task.model,
        status,
        json.dumps(task.payload),
        json.dumps(task.result) if task.result is not None else None,
        task.prompt_tokens,
        task.completion_tokens,
        task.retries_left,
        task.initial_retries,
        json.dumps(sorted(task.attempted_nodes)),
        task.session_id,
        1 if task.stream else 0,
        task.created_at,
        now,
    )


class MemoryTaskStore:
    """No-op store used when aiosqlite is missing or DB is `:memory:` for tests.

    Methods mirror SQLiteTaskStore's surface so callers do not branch on backend.
    All writes are silently dropped; `load_persisted` returns an empty list.
    """

    async def save_task(self, task, node_id: str | None, status: str = "pending") -> None:
        return

    async def mark_status(self, task_id: str, status: str, node_id: str | None = None) -> None:
        return

    async def save_result(
        self,
        task_id: str,
        result: Any,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        status: str | None = None,
    ) -> None:
        return

    async def delete_task(self, task_id: str) -> None:
        return

    async def evict_expired(self, ttl_seconds: int) -> int:
        return 0

    async def load_persisted(self) -> list[dict]:
        return []

    async def close(self) -> None:
        return


class SQLiteTaskStore:
    """aiosqlite-backed task durability layer. Lazy init opens the DB on
    first call and applies `CREATE TABLE IF NOT EXISTS`; WAL mode for
    concurrent readers (mirrors the hub's sessions store).
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

    async def save_task(self, task, node_id: str | None, status: str = "pending") -> None:
        try:
            await self._ensure_init()
            now = time.time()
            row = _task_row(task, node_id, status, now)
            await self._conn.execute(
                """
                INSERT INTO tasks (
                    task_id, node_id, owner_id, kind, model, status,
                    payload_json, result_json, prompt_tokens, completion_tokens,
                    retries_left, initial_retries, attempted_nodes_json,
                    session_id, stream_flag, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    node_id              = excluded.node_id,
                    status               = excluded.status,
                    payload_json         = excluded.payload_json,
                    result_json          = excluded.result_json,
                    prompt_tokens        = excluded.prompt_tokens,
                    completion_tokens    = excluded.completion_tokens,
                    retries_left         = excluded.retries_left,
                    attempted_nodes_json = excluded.attempted_nodes_json,
                    session_id           = excluded.session_id,
                    stream_flag          = excluded.stream_flag,
                    updated_at           = excluded.updated_at
                """,
                row,
            )
            await self._conn.commit()
        except Exception as exc:
            logger.warning("save_task failed for %s: %s", task.task_id, exc)

    async def mark_status(self, task_id: str, status: str, node_id: str | None = None) -> None:
        try:
            await self._ensure_init()
            now = time.time()
            if node_id is not None:
                await self._conn.execute(
                    "UPDATE tasks SET status = ?, node_id = ?, updated_at = ? WHERE task_id = ?",
                    (status, node_id, now, task_id),
                )
            else:
                await self._conn.execute(
                    "UPDATE tasks SET status = ?, updated_at = ? WHERE task_id = ?",
                    (status, now, task_id),
                )
            await self._conn.commit()
        except Exception as exc:
            logger.warning("mark_status failed for %s: %s", task_id, exc)

    async def save_result(
        self,
        task_id: str,
        result: Any,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        status: str | None = None,
    ) -> None:
        try:
            await self._ensure_init()
            now = time.time()
            result_json = json.dumps(result) if result is not None else None
            if status is not None:
                await self._conn.execute(
                    """
                    UPDATE tasks SET
                        result_json = ?,
                        prompt_tokens = ?,
                        completion_tokens = ?,
                        status = ?,
                        updated_at = ?
                    WHERE task_id = ?
                    """,
                    (result_json, prompt_tokens, completion_tokens, status, now, task_id),
                )
            else:
                await self._conn.execute(
                    """
                    UPDATE tasks SET
                        result_json = ?,
                        prompt_tokens = ?,
                        completion_tokens = ?,
                        updated_at = ?
                    WHERE task_id = ?
                    """,
                    (result_json, prompt_tokens, completion_tokens, now, task_id),
                )
            await self._conn.commit()
        except Exception as exc:
            logger.warning("save_result failed for %s: %s", task_id, exc)

    async def delete_task(self, task_id: str) -> None:
        try:
            await self._ensure_init()
            await self._conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            await self._conn.commit()
        except Exception as exc:
            logger.warning("delete_task failed for %s: %s", task_id, exc)

    async def evict_expired(self, ttl_seconds: int) -> int:
        try:
            await self._ensure_init()
            cutoff = time.time() - ttl_seconds
            cursor = await self._conn.execute(
                "DELETE FROM tasks WHERE status IN ('completed','failed') AND updated_at < ?",
                (cutoff,),
            )
            await self._conn.commit()
            return cursor.rowcount
        except Exception as exc:
            logger.warning("evict_expired failed: %s", exc)
            return 0

    async def load_persisted(self) -> list[dict]:
        """Return all rows with status pending or claimed. The caller is
        responsible for reconstructing Task objects and resetting claimed
        rows to pending (both in memory and via mark_status)."""
        await self._ensure_init()
        async with self._conn.execute(
            """
            SELECT task_id, node_id, owner_id, kind, model, status,
                   payload_json, result_json, prompt_tokens, completion_tokens,
                   retries_left, initial_retries, attempted_nodes_json,
                   session_id, stream_flag, created_at, updated_at
            FROM tasks WHERE status IN ('pending','claimed')
            """
        ) as cursor:
            rows = await cursor.fetchall()
        out = []
        for r in rows:
            out.append({
                "task_id": r[0],
                "node_id": r[1],
                "owner_id": r[2],
                "kind": r[3],
                "model": r[4],
                "status": r[5],
                "payload": json.loads(r[6]),
                "result": json.loads(r[7]) if r[7] else None,
                "prompt_tokens": r[8],
                "completion_tokens": r[9],
                "retries_left": r[10],
                "initial_retries": r[11],
                "attempted_nodes": set(json.loads(r[12])),
                "session_id": r[13],
                "stream": bool(r[14]),
                "created_at": r[15],
                "updated_at": r[16],
            })
        return out

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._initialized = False


_task_store = None


def get_task_store():
    """Module-level singleton accessor. Backend chosen from env:
      - TASK_DB or SESSION_DB == ':memory:'  → MemoryTaskStore
      - aiosqlite import fails                → MemoryTaskStore (warn)
      - otherwise                             → SQLiteTaskStore at TASK_DB
    """
    global _task_store
    if _task_store is not None:
        return _task_store

    db_path = os.getenv("TASK_DB") or os.getenv("SESSION_DB") or "./tasks.db"
    if db_path == ":memory:":
        _task_store = MemoryTaskStore()
        return _task_store

    try:
        import aiosqlite  # noqa: F401
        _task_store = SQLiteTaskStore(db_path)
    except ImportError:
        logger.warning("aiosqlite not available, falling back to in-memory task store")
        _task_store = MemoryTaskStore()
    return _task_store


def set_task_store(store) -> None:
    """Test injection hook. Replace the module-level singleton with an
    explicit instance (real SQLite at a tmp path, MemoryTaskStore, or a
    recording mock)."""
    global _task_store
    _task_store = store


def reset_task_store() -> None:
    """Clear the singleton so the next get_task_store() rebuilds from env.
    Test helper."""
    global _task_store
    _task_store = None
