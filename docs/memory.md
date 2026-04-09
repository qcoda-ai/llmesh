# Session Memory

LLMesh Hub maintains per-session conversation history at the hub level. This means full message context is assembled centrally and injected into each inference task — regardless of which node handles the request. Clients do not need to resend the full history on every turn.

---

## How It Works

1. Client sends a request to `/v1/chat/completions` or `/v1/messages`.
2. Hub looks up any existing history for the provided `X-Session-ID`.
3. Stored history is prepended to the incoming messages before routing to a node.
4. After the node responds, the hub appends the user message and assistant reply to the stored history.
5. The session ID (new or existing) is returned in the `X-Session-ID` response header.

---

## Using X-Session-ID

**First request** — omit the header; the hub generates a new session ID:

```
POST /v1/chat/completions
Authorization: Bearer <your-api-key>

{ "model": "llama3", "messages": [{"role": "user", "content": "Hello"}] }
```

Response includes:
```
X-Session-ID: a3f2c1d0-...
```

**Subsequent requests** — pass the returned ID to continue the conversation:

```
POST /v1/chat/completions
Authorization: Bearer <your-api-key>
X-Session-ID: a3f2c1d0-...

{ "model": "llama3", "messages": [{"role": "user", "content": "What did I just say?"}] }
```

The hub injects the prior history so the node sees the full context.

---

## Default Behavior (Zero Config)

- Sessions and inference metrics are stored in `./sessions.db` (SQLite, created automatically on first use).
- Sessions expire after 2 hours of inactivity.
- Compression triggers at 20 turns (40 messages stored).
- Default compression mode: `aggressive` — summarizes history into a single system message, keeping only the last 2 turns verbatim.

---

## Configuration

All settings are optional. They can be placed in `server_config.json` or set as environment variables (env vars take precedence).

### Session settings

| `server_config.json` key | Environment variable | Default | Description |
|---|---|---|---|
| `session.backend` | `SESSION_BACKEND` | `sqlite` | `sqlite` or `postgres` |
| `session.db` | `SESSION_DB` | `./sessions.db` | SQLite file path, or `:memory:` |
| `session.ttl_seconds` | `SESSION_TTL_SECONDS` | `7200` | Evict sessions inactive longer than this (seconds) |
| `session.max_turns` | `SESSION_MAX_TURNS` | `20` | Turn count that triggers history compression |
| `session.memory_mode` | `SESSION_MEMORY_MODE` | `aggressive` | Compression strategy: `aggressive`, `balanced`, or `cutoff` |

### Compression model settings

Compression runs entirely in-process — no Ollama or external server required. The hub downloads a small GGUF model from HuggingFace on first startup (~300 MB, cached locally). The server does not accept requests until the model is ready.

| `server_config.json` key | Environment variable | Default | Description |
|---|---|---|---|
| `compress.model_repo` | `COMPRESS_MODEL_REPO` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | HuggingFace repo ID for the compression model |
| `compress.model_file` | `COMPRESS_MODEL_FILE` | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | GGUF filename within that repo |
| `compress.context_size` | `COMPRESS_MODEL_CTX` | `4096` | Model context window (tokens) |
| `compress.n_threads` | `COMPRESS_N_THREADS` | _(CPU count)_ | CPU threads used for inference |

The default model (`Qwen2.5-0.5B-Instruct`, Q4_K_M quantisation) runs on CPU only and requires ~500 MB of RAM. It is Apache 2.0 licensed. Set `SESSION_MEMORY_MODE=cutoff` to disable compression entirely — the model will not be downloaded.

**Use in-memory store** (no file, resets on hub restart):
```
SESSION_DB=:memory:
```

---

## Compression Modes

### `aggressive` (default)
Triggers at `SESSION_MAX_TURNS`. Summarizes all prior history into a single dense system message and retains only the last 2 turns verbatim. Minimizes stored token footprint for long-running sessions.

### `balanced`
Triggers at `SESSION_MAX_TURNS`. Compresses the older half of the conversation into a summary; keeps the newer half verbatim. Moderate token footprint.

### `cutoff`
No LLM summarization. Drops the oldest messages when the cap is exceeded. Zero compute cost; suitable for simple Q&A where verbatim continuity past the window is not needed.

Compression runs as a background task after the response is returned — zero added latency to the caller. Summarization is handled in-process by the hub; no agent nodes or external servers are involved. If the compression model fails to load, the mode falls back to `cutoff`.

---

## Clearing a Session

Send a `DELETE` request with the session ID:

```
DELETE /v1/sessions/<session-id>
Authorization: Bearer <your-api-key>
```

Response:
```json
{ "status": "ok" }
```

---

## Context Window Integration

Session history interacts directly with the hub's context window orchestration: the hub decides each request's effective `num_ctx` and truncates or compresses history to fit.

1. **Window Size**: The Hub determines the target `num_ctx` for each task (Request > Hub Default > Agent Default). 
2. **History Truncation**: If the assembled history + the new user message exceeds the target `num_ctx`, Ollama will silently truncate the oldest messages in the history.
3. **Prevention**: Using `aggressive` or `balanced` compression modes helps keep the history well within the context window, preventing unexpected truncation and maintaining conversation quality.

---

## Storage backends

| Backend | How to enable | Notes |
|---|---|---|
| SQLite (default) | Nothing — zero config | Single file `sessions.db`, WAL mode |
| PostgreSQL | `SESSION_BACKEND=postgres` + `DATABASE_URL` | Shared across hub instances, see `docs/postgres.md` |

Inference metrics (`inference_events`, `node_snapshots`) always use SQLite for now.
Postgres metrics support is planned.
