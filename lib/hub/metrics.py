import logging
import os
import time
import asyncio
from collections import OrderedDict

logger = logging.getLogger("llmesh.hub.metrics")

# Metrics share the same SQLite file as sessions (SESSION_DB).
# A module-level connection is opened once at first use and closed on hub
# shutdown via close_db(). All metrics reads/writes share this single handle.
_db_conn = None
_db_init_lock = asyncio.Lock()

_DDL = [
    """CREATE TABLE IF NOT EXISTS inference_events (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp         REAL    NOT NULL,
        user_id           TEXT    NOT NULL,
        node_id           TEXT    NOT NULL,
        model             TEXT    NOT NULL,
        status            TEXT    NOT NULL,
        duration_ms       REAL,
        tokens_prompt     INTEGER,
        tokens_completion INTEGER,
        is_compression    INTEGER DEFAULT 0,
        kind              TEXT DEFAULT 'chat'
    )""",
    "CREATE INDEX IF NOT EXISTS idx_inf_user_ts ON inference_events(user_id, timestamp)",
    """CREATE TABLE IF NOT EXISTS node_snapshots (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp    REAL    NOT NULL,
        user_id      TEXT    NOT NULL,
        active_nodes INTEGER
    )""",
    "CREATE INDEX IF NOT EXISTS idx_snap_user_ts ON node_snapshots(user_id, timestamp)",
    """CREATE TABLE IF NOT EXISTS node_registry (
        node_id    TEXT PRIMARY KEY,
        owner_id   TEXT NOT NULL,
        hostname   TEXT,
        os_name    TEXT,
        cpu_cores  INTEGER,
        ram_gb     REAL,
        context_size INTEGER,
        first_seen REAL NOT NULL,
        last_seen  REAL NOT NULL
    )""",
]

# Migration statements — run best-effort on existing DBs.
_MIGRATIONS = [
    "ALTER TABLE inference_events ADD COLUMN is_compression INTEGER DEFAULT 0",
    "ALTER TABLE node_registry ADD COLUMN context_size INTEGER",
    "ALTER TABLE inference_events ADD COLUMN kind TEXT DEFAULT 'chat'",
]


async def _get_db():
    global _db_conn
    if _db_conn is not None:
        return _db_conn
    async with _db_init_lock:
        if _db_conn is not None:
            return _db_conn
        import aiosqlite
        db_path = os.getenv("SESSION_DB", "./sessions.db")
        _db_conn = await aiosqlite.connect(db_path)
        await _db_conn.execute("PRAGMA journal_mode=WAL")
        for stmt in _DDL:
            await _db_conn.execute(stmt)
        # Migrate existing DBs — ignore errors for already-applied migrations
        for migration in _MIGRATIONS:
            try:
                await _db_conn.execute(migration)
            except Exception:
                pass
        await _db_conn.commit()
    return _db_conn


async def close_db() -> None:
    """Close the shared metrics DB connection. Call from hub shutdown to
    ensure WAL is checkpointed cleanly and the FD is released."""
    global _db_conn
    if _db_conn is None:
        return
    try:
        await _db_conn.commit()
        await _db_conn.close()
    except Exception:
        logger.exception("metrics: error while closing DB connection")
    finally:
        _db_conn = None


async def prune_old_metrics() -> tuple[int, int]:
    """Delete old rows from inference_events and node_snapshots. Returns
    (deleted_events, deleted_snapshots). Both deletes run under a single
    transaction — if the second delete fails, the first is rolled back so
    the two retention windows stay consistent."""
    retention_events = int(os.getenv("METRICS_RETENTION_DAYS", "30"))
    retention_snaps  = int(os.getenv("SNAPSHOT_RETENTION_DAYS", "7"))
    cutoff_events = time.time() - (retention_events * 86400)
    cutoff_snaps  = time.time() - (retention_snaps  * 86400)
    db = await _get_db()
    try:
        cur_e = await db.execute("DELETE FROM inference_events WHERE timestamp < ?", (cutoff_events,))
        cur_s = await db.execute("DELETE FROM node_snapshots  WHERE timestamp < ?", (cutoff_snaps,))
        await db.commit()
        return cur_e.rowcount or 0, cur_s.rowcount or 0
    except Exception:
        await db.rollback()
        logger.exception("metrics: prune_old_metrics failed, rolled back")
        return 0, 0


async def upsert_node_registry(node_id: str, owner_id: str, os_name: str | None,
                               cpu_cores: int | None, ram_gb: float | None,
                               context_size: int | None = None,
                               hostname: str | None = None):
    now = time.time()
    db = await _get_db()
    await db.execute("""
        INSERT INTO node_registry (node_id, owner_id, hostname, os_name, cpu_cores, ram_gb, context_size, first_seen, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
            last_seen = excluded.last_seen,
            hostname  = excluded.hostname,
            os_name   = excluded.os_name,
            cpu_cores = excluded.cpu_cores,
            ram_gb    = excluded.ram_gb,
            context_size = excluded.context_size
    """, (node_id, owner_id, hostname, os_name, cpu_cores, ram_gb, context_size, now, now))
    await db.commit()


async def get_node_ratings(owner_id: str) -> dict:
    db = await _get_db()
    ratings = {}
    async with db.execute("""
        SELECT node_id,
               COUNT(*) AS total_tasks,
               ROUND(AVG(CASE WHEN status='success' THEN 1.0 ELSE 0.0 END)*100, 1) AS success_pct,
               ROUND(AVG(duration_ms), 0) AS avg_latency_ms,
               ROUND(AVG(tokens_prompt + tokens_completion), 0) AS avg_tokens
        FROM inference_events
        WHERE user_id = ? AND is_compression = 0
        GROUP BY node_id
    """, (owner_id,)) as cur:
        async for row in cur:
            node_id, total, pct, latency, tokens = row
            ratings[node_id] = {
                "total_tasks": total,
                "success_pct": pct,
                "avg_latency_ms": int(latency) if latency is not None else None,
                "avg_tokens": int(tokens) if tokens is not None else None,
            }
    return ratings


# ---------------------------------------------------------------------------
# Global event buffering (D022)
# ---------------------------------------------------------------------------
# One shared buffer for all owners; one background flush task for the whole
# hub. Prior design (one MetricsLogger instance per owner_id, each spawning
# its own _periodic_flush asyncio task) leaked a permanent task per unique
# owner, had no exception handler in the flush loop, and used fire-and-forget
# asyncio.create_task calls that can be GC'd mid-execution on Python 3.11+.
# See decisions.md D022.

_FLUSH_INTERVAL_SEC = 5.0
_DASHBOARD_CACHE_TTL = 15.0
_DASHBOARD_CACHE_MAX = 64

_event_buffer: list[dict] = []
_flush_task: asyncio.Task | None = None
_dashboard_cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()


async def start_background() -> None:
    """Start the single module-level flush task. Called from hub startup."""
    global _flush_task
    if _flush_task is not None and not _flush_task.done():
        return
    _flush_task = asyncio.create_task(_flush_loop())


async def stop_background() -> None:
    """Cancel the flush task and drain any remaining buffered events.
    Called from hub shutdown so SIGTERM does not lose the last ~5s of events."""
    global _flush_task
    if _flush_task is not None:
        _flush_task.cancel()
        try:
            await _flush_task
        except (asyncio.CancelledError, Exception):
            pass
        _flush_task = None
    await _flush_buffer()


async def _flush_loop() -> None:
    """Run until cancelled. Each iteration is wrapped in try/except so a
    single failing flush does not kill the loop — prior design let an
    aiosqlite error silently stop all metrics writes for an owner forever."""
    while True:
        try:
            await asyncio.sleep(_FLUSH_INTERVAL_SEC)
            await _flush_buffer()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("metrics: flush loop iteration failed, continuing")


async def _flush_buffer() -> None:
    """Drain the global event buffer into the DB in one batch.
    Safe without a lock: log_* writers are sync (no await between list.append
    and return), and this is the only coroutine that reads+clears the buffer."""
    if not _event_buffer:
        return
    events = _event_buffer[:]
    _event_buffer.clear()

    db = await _get_db()

    inf_rows = [
        (e["timestamp"], e["user_id"], e["node_id"], e["model"], e["status"],
         e.get("duration_ms"), e.get("tokens_prompt", 0), e.get("tokens_completion", 0),
         1 if e.get("is_compression") else 0,
         e.get("kind", "chat"))
        for e in events if e.get("event") == "inference"
    ]
    snap_rows = [
        (e["timestamp"], e["user_id"], e.get("active_nodes", 0))
        for e in events if e.get("event") == "node_snapshot"
    ]

    try:
        if inf_rows:
            await db.executemany(
                "INSERT INTO inference_events "
                "(timestamp, user_id, node_id, model, status, duration_ms, "
                " tokens_prompt, tokens_completion, is_compression, kind) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                inf_rows
            )
        if snap_rows:
            await db.executemany(
                "INSERT INTO node_snapshots (timestamp, user_id, active_nodes) VALUES (?,?,?)",
                snap_rows
            )
        if inf_rows or snap_rows:
            await db.commit()
    except Exception:
        logger.exception("metrics: batch insert failed, %d events dropped", len(events))


def log_inference_event(user_id: str, node_id: str, model: str, status: str,
                        duration_ms: float, tokens_prompt: int, tokens_completion: int,
                        is_compression: bool = False, kind: str = "chat") -> None:
    """Queue an inference event for the next batch flush. Sync-by-design:
    one list.append, no await, no coroutine spawning. Safe to call from any
    context; the flush task picks it up on the next tick.

    `kind` distinguishes chat vs embedding traffic for analytics. Stored in
    the buffer as a tag — the underlying schema is unchanged for now (D028
    follow-up may add a column once we have a migration cadence)."""
    _event_buffer.append({
        "event": "inference",
        "timestamp": time.time(),
        "user_id": user_id,
        "node_id": node_id,
        "model": model,
        "status": status,
        "duration_ms": duration_ms,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "is_compression": is_compression,
        "kind": kind,
    })


def log_node_snapshot(user_id: str, active_nodes: int) -> None:
    """Queue a node-count snapshot for the next batch flush. See
    log_inference_event for the sync-by-design rationale."""
    _event_buffer.append({
        "event": "node_snapshot",
        "timestamp": time.time(),
        "user_id": user_id,
        "active_nodes": active_nodes,
    })


def _short_node(node_id: str) -> str:
    return node_id[:13] if len(node_id) > 13 else node_id


def _pivot(rows: list, missing=None) -> dict:
    """Pivot (bucket, node_id, value) rows into {labels, datasets:[{node, data}]}."""
    labels_order: dict = {}
    nodes_order: dict = {}
    raw: dict = {}
    for bucket, node_id, value in rows:
        labels_order[bucket] = None
        nodes_order[node_id] = None
        raw[(bucket, node_id)] = value
    labels = list(labels_order)
    nodes = list(nodes_order)
    datasets = [
        {"node": _short_node(nid),
         "data": [raw.get((b, nid), missing) for b in labels]}
        for nid in nodes
    ]
    return {"labels": labels, "datasets": datasets}


async def _get_node_perf_charts(owner_id: str, cutoff: float, node_ratings: dict) -> dict:
    db = await _get_db()

    throughput_rows = []
    async with db.execute("""
        SELECT strftime('%m-%d %H:%M', CAST(timestamp/300 AS INTEGER)*300, 'unixepoch') AS bucket,
               node_id, COUNT(*) AS tasks
        FROM inference_events
        WHERE user_id = ? AND timestamp >= ? AND is_compression = 0
        GROUP BY bucket, node_id ORDER BY bucket ASC
    """, (owner_id, cutoff)) as cur:
        async for row in cur:
            throughput_rows.append(row)

    latency_rows = []
    async with db.execute("""
        SELECT strftime('%m-%d %H:%M', CAST(timestamp/300 AS INTEGER)*300, 'unixepoch') AS bucket,
               node_id, ROUND(AVG(duration_ms), 0) AS avg_ms
        FROM inference_events
        WHERE user_id = ? AND timestamp >= ? AND is_compression = 0 AND duration_ms IS NOT NULL
        GROUP BY bucket, node_id ORDER BY bucket ASC
    """, (owner_id, cutoff)) as cur:
        async for row in cur:
            latency_rows.append(row)

    success_rows = []
    async with db.execute("""
        SELECT strftime('%m-%d %H:%M', CAST(timestamp/300 AS INTEGER)*300, 'unixepoch') AS bucket,
               node_id,
               ROUND(AVG(CASE WHEN status='success' THEN 100.0 ELSE 0.0 END), 1) AS success_pct
        FROM inference_events
        WHERE user_id = ? AND timestamp >= ? AND is_compression = 0
        GROUP BY bucket, node_id ORDER BY bucket ASC
    """, (owner_id, cutoff)) as cur:
        async for row in cur:
            success_rows.append(row)

    # Distribution derived from node_ratings (already computed, no extra query)
    distribution = {
        "labels": [_short_node(nid) for nid in node_ratings],
        "data":   [node_ratings[nid]["total_tasks"] for nid in node_ratings],
    }

    return {
        "throughput":   _pivot(throughput_rows, missing=None),
        "latency":      _pivot(latency_rows, missing=None),
        "success_rate": _pivot(success_rows, missing=None),
        "distribution": distribution,
    }


async def get_dashboard_stats(owner_id: str, days: int = 7) -> dict:
    """Aggregate dashboard stats for an owner. Results are cached per-owner
    for 15s to absorb dashboard polling. Cache is bounded to
    _DASHBOARD_CACHE_MAX owners, LRU-evicted."""
    now = time.time()
    cached = _dashboard_cache.get(owner_id)
    if cached is not None and (now - cached[0]) < _DASHBOARD_CACHE_TTL:
        _dashboard_cache.move_to_end(owner_id)
        return cached[1]

    # Flush any pending writes so the stats reflect the latest events
    await _flush_buffer()

    cutoff = now - (days * 24 * 3600)
    db = await _get_db()

    # --- inference timeline: 5-minute buckets (user calls only) ---
    timeline = []
    async with db.execute("""
        SELECT
            strftime('%m-%d %H:%M', CAST(timestamp/300 AS INTEGER)*300, 'unixepoch') AS bucket,
            COUNT(*)                                                    AS total_calls,
            COALESCE(SUM(tokens_prompt + tokens_completion), 0)         AS total_tokens,
            COALESCE(SUM(tokens_prompt), 0)                             AS prompt_tokens,
            COALESCE(SUM(tokens_completion), 0)                         AS completion_tokens,
            COALESCE(SUM(duration_ms), 0.0)                             AS total_duration,
            SUM(CASE WHEN status='success' THEN 1 ELSE 0 END)           AS success,
            SUM(CASE WHEN status='fail'    THEN 1 ELSE 0 END)           AS fail
        FROM inference_events
        WHERE user_id = ? AND timestamp >= ? AND is_compression = 0
        GROUP BY bucket ORDER BY bucket DESC LIMIT 60
    """, (owner_id, cutoff)) as cur:
        async for row in cur:
            bucket, calls, tokens, prompt_tk, comp_tk, dur, success, fail = row
            timeline.append({
                "date": bucket,
                "calls": calls or 0,
                "tokens": tokens or 0,
                "prompt_tokens": prompt_tk or 0,
                "completion_tokens": comp_tk or 0,
                "speed_per_token": (dur / tokens) if tokens else 0,
                "success": success or 0,
                "fail": fail or 0,
            })
    timeline.reverse()

    # --- model breakdown (user calls only) ---
    models = {}
    async with db.execute(
        "SELECT model, COUNT(*) FROM inference_events "
        "WHERE user_id=? AND timestamp>=? AND is_compression=0 GROUP BY model",
        (owner_id, cutoff)
    ) as cur:
        async for row in cur:
            models[row[0]] = row[1]

    # --- overall success / fail (user calls only) ---
    success_rate = {"success": 0, "fail": 0}
    async with db.execute("""
        SELECT
            SUM(CASE WHEN status='success' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status='fail'    THEN 1 ELSE 0 END)
        FROM inference_events WHERE user_id=? AND timestamp>=? AND is_compression=0
    """, (owner_id, cutoff)) as cur:
        row = await cur.fetchone()
        if row:
            success_rate = {"success": row[0] or 0, "fail": row[1] or 0}

    # --- node snapshot timeline: 5-minute buckets ---
    node_timeline = []
    async with db.execute("""
        SELECT
            strftime('%m-%d %H:%M', CAST(timestamp/300 AS INTEGER)*300, 'unixepoch') AS bucket,
            MAX(active_nodes) AS nodes
        FROM node_snapshots
        WHERE user_id = ? AND timestamp >= ?
        GROUP BY bucket ORDER BY bucket DESC LIMIT 60
    """, (owner_id, cutoff)) as cur:
        async for row in cur:
            node_timeline.append({"time": row[0], "nodes": row[1]})
    node_timeline.reverse()

    # --- compression timeline: 5-minute buckets ---
    compression_timeline = []
    async with db.execute("""
        SELECT
            strftime('%m-%d %H:%M', CAST(timestamp/300 AS INTEGER)*300, 'unixepoch') AS bucket,
            COUNT(*)                                          AS compress_calls,
            COALESCE(SUM(tokens_prompt), 0)                  AS tokens_prompt,
            COALESCE(SUM(tokens_completion), 0)              AS tokens_completion,
            COALESCE(AVG(duration_ms), 0)                    AS avg_duration_ms
        FROM inference_events
        WHERE user_id = ? AND timestamp >= ? AND is_compression = 1
        GROUP BY bucket ORDER BY bucket DESC LIMIT 60
    """, (owner_id, cutoff)) as cur:
        async for row in cur:
            bucket, calls, prompt_tk, comp_tk, avg_dur = row
            compression_timeline.append({
                "date": bucket,
                "compress_calls": calls or 0,
                "tokens_prompt": prompt_tk or 0,
                "tokens_completion": comp_tk or 0,
                "avg_duration_ms": round(avg_dur or 0),
            })
    compression_timeline.reverse()

    # --- active sessions count (from sessions table) ---
    sessions_timeline = []
    async with db.execute("""
        SELECT
            strftime('%m-%d %H:%M', CAST(last_active/300 AS INTEGER)*300, 'unixepoch') AS bucket,
            COUNT(*) AS session_count,
            ROUND(AVG(json_array_length(messages)), 1) AS avg_messages
        FROM sessions
        WHERE owner_id = ? AND last_active >= ?
        GROUP BY bucket ORDER BY bucket DESC LIMIT 60
    """, (owner_id, cutoff)) as cur:
        async for row in cur:
            sessions_timeline.append({
                "date": row[0],
                "session_count": row[1] or 0,
                "avg_messages": row[2] or 0,
            })
    sessions_timeline.reverse()

    node_ratings = await get_node_ratings(owner_id)
    node_perf = await _get_node_perf_charts(owner_id, cutoff, node_ratings)

    result = {
        "timeline": timeline,
        "models": models,
        "success_rate": success_rate,
        "active_nodes_timeline": node_timeline,
        "compression_timeline": compression_timeline,
        "sessions_timeline": sessions_timeline,
        "node_ratings": node_ratings,
        "node_perf": node_perf,
    }

    # LRU insert with cap
    _dashboard_cache[owner_id] = (time.time(), result)
    _dashboard_cache.move_to_end(owner_id)
    while len(_dashboard_cache) > _DASHBOARD_CACHE_MAX:
        _dashboard_cache.popitem(last=False)
    return result
