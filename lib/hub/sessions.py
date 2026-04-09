import os
import json
import time
import asyncio

SESSION_DB = os.getenv("SESSION_DB", "./sessions.db")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "7200"))
SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "20"))
SESSION_MEMORY_MODE = os.getenv("SESSION_MEMORY_MODE", "aggressive")

_SQLITE_DDL = [
    """CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        messages TEXT NOT NULL,
        created_at REAL NOT NULL,
        last_active REAL NOT NULL,
        PRIMARY KEY (session_id, owner_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active)",
]

_PG_DDL = [
    """CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        messages TEXT NOT NULL,
        created_at DOUBLE PRECISION NOT NULL,
        last_active DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (session_id, owner_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active)",
]


class MemorySessionStore:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def _key(self, session_id: str, owner_id: str) -> str:
        return f"{owner_id}:{session_id}"

    async def get_messages(self, session_id: str, owner_id: str) -> list[dict] | None:
        entry = self._store.get(self._key(session_id, owner_id))
        if entry is None:
            return None
        entry["last_active"] = time.time()
        return entry["messages"]

    async def save_messages(self, session_id: str, owner_id: str, messages: list[dict]) -> None:
        now = time.time()
        key = self._key(session_id, owner_id)
        if key not in self._store:
            self._store[key] = {"created_at": now}
        self._store[key]["messages"] = messages
        self._store[key]["last_active"] = now

    async def evict_expired(self) -> int:
        cutoff = time.time() - SESSION_TTL_SECONDS
        to_delete = [k for k, v in self._store.items() if v.get("last_active", 0) < cutoff]
        for k in to_delete:
            del self._store[k]
        return len(to_delete)

    async def delete_session(self, session_id: str, owner_id: str) -> None:
        self._store.pop(self._key(session_id, owner_id), None)


class SQLiteSessionStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_init(self):
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

    async def get_messages(self, session_id: str, owner_id: str) -> list[dict] | None:
        await self._ensure_init()
        async with self._init_lock:
            async with self._conn.execute(
                "SELECT messages FROM sessions WHERE session_id = ? AND owner_id = ?",
                (session_id, owner_id)
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None
            await self._conn.execute(
                "UPDATE sessions SET last_active = ? WHERE session_id = ? AND owner_id = ?",
                (time.time(), session_id, owner_id)
            )
            await self._conn.commit()
        return json.loads(row[0])

    async def save_messages(self, session_id: str, owner_id: str, messages: list[dict]) -> None:
        await self._ensure_init()
        now = time.time()
        await self._conn.execute("""
            INSERT INTO sessions (session_id, owner_id, messages, created_at, last_active)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id, owner_id) DO UPDATE SET
                messages = excluded.messages,
                last_active = excluded.last_active
        """, (session_id, owner_id, json.dumps(messages), now, now))
        await self._conn.commit()

    async def evict_expired(self) -> int:
        await self._ensure_init()
        cutoff = time.time() - SESSION_TTL_SECONDS
        cursor = await self._conn.execute(
            "DELETE FROM sessions WHERE last_active < ?", (cutoff,)
        )
        await self._conn.commit()
        return cursor.rowcount

    async def delete_session(self, session_id: str, owner_id: str) -> None:
        await self._ensure_init()
        await self._conn.execute(
            "DELETE FROM sessions WHERE session_id = ? AND owner_id = ?",
            (session_id, owner_id)
        )
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._initialized = False


class PostgresSessionStore:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_init(self):
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            import asyncpg
            self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
            async with self._pool.acquire() as conn:
                for stmt in _PG_DDL:
                    await conn.execute(stmt)
            self._initialized = True

    async def get_messages(self, session_id: str, owner_id: str) -> list[dict] | None:
        await self._ensure_init()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT messages FROM sessions WHERE session_id = $1 AND owner_id = $2",
                session_id, owner_id
            )
            if row is None:
                return None
            await conn.execute(
                "UPDATE sessions SET last_active = $1 WHERE session_id = $2 AND owner_id = $3",
                time.time(), session_id, owner_id
            )
        return json.loads(row["messages"])

    async def save_messages(self, session_id: str, owner_id: str, messages: list[dict]) -> None:
        await self._ensure_init()
        now = time.time()
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO sessions (session_id, owner_id, messages, created_at, last_active)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (session_id, owner_id) DO UPDATE SET
                    messages = EXCLUDED.messages,
                    last_active = EXCLUDED.last_active
            """, session_id, owner_id, json.dumps(messages), now, now)

    async def evict_expired(self) -> int:
        await self._ensure_init()
        cutoff = time.time() - SESSION_TTL_SECONDS
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM sessions WHERE last_active < $1", cutoff
            )
        # asyncpg returns "DELETE N" as a string
        return int(result.split()[-1])

    async def delete_session(self, session_id: str, owner_id: str) -> None:
        await self._ensure_init()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM sessions WHERE session_id = $1 AND owner_id = $2",
                session_id, owner_id
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False


_session_store = None


def get_session_store():
    global _session_store
    if _session_store is not None:
        return _session_store

    backend = os.getenv("SESSION_BACKEND", "sqlite")

    if backend == "postgres":
        dsn = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        if not dsn:
            raise RuntimeError(
                "SESSION_BACKEND=postgres requires DATABASE_URL or POSTGRES_URL to be set. "
                "See docs/postgres.md."
            )
        _session_store = PostgresSessionStore(dsn)
        return _session_store

    db_path = SESSION_DB
    if db_path == ":memory:":
        _session_store = MemorySessionStore()
        return _session_store

    try:
        import aiosqlite  # noqa: F401
        _session_store = SQLiteSessionStore(db_path)
    except ImportError:
        print("Warning: aiosqlite not available, falling back to in-memory session store")
        _session_store = MemorySessionStore()
    return _session_store
