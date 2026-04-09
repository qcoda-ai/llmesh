# Node Security Model

LLMesh uses a two-phase authentication model to secure node-hub communication. The API key authenticates the owner at registration. A per-node token — issued at registration and scoped to that node only — authenticates all subsequent operations.

---

## Two-Phase Auth

### Phase 1 — Registration (API key)

The agent registers with the hub by presenting its owner API key. The hub validates the key, creates the node record, generates a `node_token`, and returns it in the registration response.

```
POST /register
{ "api_key": "...", "resources": {...}, "node_fingerprint": "..." }

→ { "node_id": "abc-123", "node_token": "nt_a7f3c9d2e1b84f0a..." }
```

The API key is **not used again** after registration. It is never stored in the node record or returned in any subsequent response.

### Phase 2 — Operations (node token)

All agent-facing endpoints require the node token as a Bearer token in the `Authorization` header:

```
Authorization: Bearer <node_token>
```

Covered endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /heartbeat/{node_id}` | Node liveness updates |
| `GET /tasks/{node_id}/pending` | Task polling |
| `POST /tasks/{node_id}/complete/{task_id}` | Result submission |

The hub verifies:
1. The `Authorization` header is present and uses the `Bearer` scheme
2. The token maps to a known node
3. The token's node matches the `node_id` in the path

A missing or invalid token returns `401`. A token that doesn't match the path node_id returns `403`.

---

## Node Listing (`GET /nodes`)

The `/nodes` endpoint requires an owner API key and returns only nodes belonging to that owner:

```
GET /nodes
Authorization: Bearer <api_key>

→ [ { "node_id": "...", "owner_id": "...", "resources": {...} }, ... ]
```

Cross-tenant node data is never returned. The `node_token` field is excluded from all list responses.

---

## Token Properties

| Property | Value |
|---|---|
| Format | 64-character hex string (`secrets.token_hex(32)`) |
| Scope | Single node only — cannot be used for other node_ids |
| Comparison | Constant-time (`secrets.compare_digest`) — timing-safe |
| Persistence | In-memory only — not written to disk or database |
| Lifetime | Valid until hub restart or node re-registration |
| Rotation | Automatically replaced on re-registration |

---

## Token Lifecycle

```
Agent starts
    ↓
POST /register  (API key)
    ↓
Hub issues node_token
    ↓
Agent stores node_token in memory
    ↓
All operations use node_token
    ↓
Hub restarts  ──→  node_token lost  ──→  agent re-registers  ──→  new node_token issued
Backend comes back online  ──→  agent re-registers  ──→  new node_token issued
3 failed heartbeats  ──→  agent clears node_id + node_token  ──→  re-registration triggered
```

---

## What This Protects Against

- **Task theft** — an attacker who knows a `node_id` cannot poll `/tasks/{node_id}/pending` without the token
- **Result poisoning** — a rogue caller cannot submit fabricated results to `/tasks/{node_id}/complete/{task_id}`
- **Fake heartbeats** — an attacker cannot keep a dead node alive in the routing pool
- **Cross-tenant node listing** — `/nodes` only returns the calling owner's nodes

## Known Limitations

- **Tokens are not persisted.** A hub restart invalidates all tokens. Agents detect this via heartbeat failures and automatically re-register, but there is a brief gap during which in-flight tasks will be lost.
- **Tokens are not rotated proactively.** A compromised token remains valid until the node re-registers. For environments where token compromise is a concern, place the hub behind a TLS-terminating reverse proxy (see [nginx_deployment.md](nginx_deployment.md)) so tokens are never transmitted in plaintext.
- **Hub restart = node re-registration required.** This is inherent to the in-memory node registry. Nodes will reconnect automatically; plan for ~5–15 seconds of unavailability during hub restarts.

---

## Deployment Recommendation

Run the hub behind an HTTPS reverse proxy in any non-local environment. Tokens and API keys transmitted over plain HTTP are visible to anyone on the network path. See [nginx_deployment.md](nginx_deployment.md) for a reference configuration.
