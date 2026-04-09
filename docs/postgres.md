# PostgreSQL Backend

By default the hub stores everything — session history and inference metrics — in
a single local SQLite file (`sessions.db`). PostgreSQL is an optional drop-in
for deployments that need shared state across multiple hub instances, long-term
queryable analytics, or production-grade durability.

No code changes are required on nodes. Nodes are unaware of the storage backend.

---

## What lives in the database

| Table | Content | Backend |
|---|---|---|
| `sessions` | Per-owner conversation history (JSON), TTL-managed | SQLite or Postgres |
| `inference_events` | One row per inference call: model, tokens, latency, status | SQLite only (Postgres planned) |
| `node_snapshots` | Active node count per owner, sampled every 30 s | SQLite only (Postgres planned) |

All three tables share the same file/database. There is one config point, one
connection, and zero extra infrastructure for the default SQLite path.

---

## Architecture diagrams

### SQLite (default — zero config)

```
  ┌──────────────┐         ┌──────────────────────────────────────────┐
  │    Client    │         │              LLMesh Hub                  │
  │              │──req───▶│                                          │
  │  X-Session-ID│         │  1. load session history                 │
  │  (optional)  │         │  2. prepend to messages                  │
  └──────────────┘         │  3. route task to node                   │
                           │  4. await done_event                     │
  ┌──────────────┐         │  5. save updated history                 │
  │     Node     │◀─task──▶│  6. log inference event                  │
  │  (Ollama)    │         │                                          │
  └──────────────┘         └───────────────┬──────────────────────────┘
                                           │ read/write
                                           ▼
                                  ┌─────────────────┐
                                  │   sessions.db   │
                                  │  (local file,   │
                                  │   WAL mode)     │
                                  │                 │
                                  │  sessions       │
                                  │  inference_evts │
                                  │  node_snapshots │
                                  └─────────────────┘
```

Single process, single file. Survives restarts. Lost if the disk is lost.

---

### PostgreSQL (optional — multi-instance)

```
  ┌──────────────┐    ┌─────────────────┐
  │    Client    │    │  Load Balancer  │
  │              │───▶│  (nginx, etc.)  │
  └──────────────┘    └────────┬────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
  ┌─────────────────────┐       ┌─────────────────────┐
  │   Hub  instance 1   │       │   Hub  instance 2   │
  │                     │       │                     │
  │  in-process:        │       │  in-process:        │
  │  · task queue       │       │  · task queue       │
  │  · node registry    │       │  · node registry    │
  │  · metrics buffer   │       │  · metrics buffer   │
  └──────────┬──────────┘       └──────────┬──────────┘
             │                             │
             │      SESSION_BACKEND        │
             │      = postgres             │
             └──────────┬──────────────────┘
                        │ asyncpg pool
                        ▼
             ┌─────────────────────┐
             │      PostgreSQL     │
             │                    │
             │  sessions          │  ← shared across all hub instances
             └─────────────────────┘

  Note: inference_events and node_snapshots are still written to
  a local sessions.db on each hub instance (SQLite). Postgres
  metrics support is a planned future addition.
```

Any hub instance can serve any client session — the `X-Session-ID` header is
the only sticky state, and it resolves to a row in the shared `sessions` table.

---

## Prerequisites

- PostgreSQL 13 or later
- `asyncpg==0.29.0` (already in `pyproject.toml` base dependencies)

---

## Quick start

### 1. Create a database and user

```sql
CREATE DATABASE llmesh;
CREATE USER llmesh_hub WITH PASSWORD 'your-password-here';
GRANT ALL PRIVILEGES ON DATABASE llmesh TO llmesh_hub;

\c llmesh
GRANT ALL ON SCHEMA public TO llmesh_hub;
```

### 2. Configure the hub

In `server_config.json`:

```json
{
    "session": {
        "backend": "postgres"
    }
}
```

Set the connection URL as an environment variable (keep credentials out of the
config file):

```bash
export DATABASE_URL="postgresql://llmesh_hub:your-password-here@localhost:5432/llmesh"
```

Or use environment variables alone (takes precedence over `server_config.json`):

```bash
export SESSION_BACKEND=postgres
export DATABASE_URL="postgresql://llmesh_hub:your-password-here@localhost:5432/llmesh"
```

### 3. Start the hub

```bash
uvicorn lib.hub.server:app --port 8000
```

The `sessions` table is created automatically on the first inference request.
No migration step is required.

---

## Configuration reference

Environment variables always override `server_config.json`.

| `server_config.json` key | Environment variable | Default | Description |
|---|---|---|---|
| `session.backend` | `SESSION_BACKEND` | `sqlite` | `sqlite` or `postgres` |
| `session.db` | `SESSION_DB` | `./sessions.db` | SQLite path (SQLite only) |
| — | `DATABASE_URL` | _(required for postgres)_ | Postgres DSN |
| — | `POSTGRES_URL` | — | Alias for `DATABASE_URL` |
| `session.ttl_seconds` | `SESSION_TTL_SECONDS` | `7200` | Seconds before inactive sessions are evicted |
| `session.max_turns` | `SESSION_MAX_TURNS` | `20` | Turn count that triggers history compression |
| `session.memory_mode` | `SESSION_MEMORY_MODE` | `aggressive` | `aggressive`, `balanced`, or `cutoff` |
| `session.compress_model` | `SESSION_COMPRESS_MODEL` | _(request model)_ | Model used for summarisation |

---

## Schema

Both backends use the same logical schema. Created automatically on first use.

### `sessions`

```sql
CREATE TABLE sessions (
    session_id   TEXT             NOT NULL,
    owner_id     TEXT             NOT NULL,
    messages     TEXT             NOT NULL,  -- JSON array of {role, content} objects
    created_at   DOUBLE PRECISION NOT NULL,  -- Unix timestamp
    last_active  DOUBLE PRECISION NOT NULL,  -- Unix timestamp, updated each turn
    PRIMARY KEY (session_id, owner_id)
);
CREATE INDEX idx_sessions_owner       ON sessions(owner_id);
CREATE INDEX idx_sessions_last_active ON sessions(last_active);
```

### `inference_events` (SQLite only)

```sql
CREATE TABLE inference_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         REAL    NOT NULL,   -- Unix timestamp
    user_id           TEXT    NOT NULL,
    node_id           TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    status            TEXT    NOT NULL,   -- 'success' or 'fail'
    duration_ms       REAL,
    tokens_prompt     INTEGER,
    tokens_completion INTEGER
);
CREATE INDEX idx_inf_user_ts ON inference_events(user_id, timestamp);
```

### `node_snapshots` (SQLite only)

```sql
CREATE TABLE node_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    REAL    NOT NULL,
    user_id      TEXT    NOT NULL,
    active_nodes INTEGER
);
CREATE INDEX idx_snap_user_ts ON node_snapshots(user_id, timestamp);
```

---

## Useful queries (Postgres sessions table)

```sql
-- Active sessions in the last hour
SELECT owner_id, COUNT(*) AS sessions
FROM sessions
WHERE last_active > extract(epoch FROM now()) - 3600
GROUP BY owner_id ORDER BY sessions DESC;

-- Average conversation length per owner
SELECT owner_id,
       ROUND(AVG(json_array_length(messages::json)), 1) AS avg_turns
FROM sessions GROUP BY owner_id;

-- Longest active sessions
SELECT session_id, owner_id,
       json_array_length(messages::json)     AS turns,
       to_timestamp(last_active)             AS last_active
FROM sessions ORDER BY turns DESC LIMIT 20;
```

## Useful queries (SQLite inference_events)

```sql
-- Calls per model in the last 24h
SELECT model, COUNT(*) AS calls
FROM inference_events
WHERE timestamp > unixepoch() - 86400
GROUP BY model ORDER BY calls DESC;

-- Hourly call volume
SELECT strftime('%Y-%m-%d %H:00', timestamp, 'unixepoch') AS hour,
       COUNT(*) AS calls,
       SUM(tokens_prompt + tokens_completion) AS tokens
FROM inference_events
GROUP BY hour ORDER BY hour DESC LIMIT 48;

-- p95 latency per model
SELECT model,
       COUNT(*)                             AS calls,
       ROUND(AVG(duration_ms), 0)          AS avg_ms,
       MAX(duration_ms)                    AS max_ms
FROM inference_events
GROUP BY model;
```

---

## Connection pooling

`PostgresSessionStore` uses `asyncpg.create_pool` with `min_size=2, max_size=10`.
For high-concurrency deployments, run PgBouncer in transaction-pooling mode in
front of Postgres and point `DATABASE_URL` at PgBouncer.

---

## Migrating from SQLite to Postgres

Sessions do not need to be migrated — they are short-lived by design.

1. Set `SESSION_BACKEND=postgres` and `DATABASE_URL`, restart the hub
2. New sessions go to Postgres immediately
3. Old SQLite sessions expire within `SESSION_TTL_SECONDS` (default 2 h)
4. Delete `sessions.db` after the TTL window closes

Inference metrics remain in `sessions.db` on disk. They are no longer written
after the switch. Historical data is still readable via SQLite directly.

---

## Multi-instance constraint

Task routing (`_node_tasks`, `_task_index`, `done_event`) is in-process memory.
Running two hub instances behind a load balancer shares session history (via
Postgres) but not task state — a client must maintain a persistent connection
to the same hub instance for the duration of a single inference call. Session
continuity across calls works with any instance.
