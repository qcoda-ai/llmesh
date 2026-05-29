# Known Limitations

LLMesh ships an opinionated v0/v1 surface. The gaps below are documented choices, not bugs. Each links to the authoritative reference so you can verify the trade-off before adopting.

For the canonical roadmap status (shipped / planned / deferred), see [`PROJECT.md`](../.qcoda/PROJECT.md). For per-feature design notes, see the files under `docs/` and `.qcoda/features/`.

---

## 1. OpenAI API compatibility (`/v1/chat/completions`)

| Parameter | Status | Notes |
|---|---|---|
| `stream: true` | **Supported** | Ollama always; vLLM default-ON per D044; MLX default-ON per D060. See [`docs/streaming.md`](streaming.md). |
| `tools` / `tool_choice` | Not supported | Function-calling out of scope for v1. Roadmap. |
| `response_format` (JSON mode) | Not supported | Roadmap. |
| `temperature`, `top_p`, `presence_penalty`, `frequency_penalty` | Accepted, not forwarded | No backend gets these knobs in v1. |
| `logprobs` | Not supported | No probability output. |
| `n > 1` | Not supported | One completion per request. |
| Vision / multimodal content blocks | Not supported | Text input only. |

Ref: [`.qcoda/api.md` §"NOT Implemented"](../.qcoda/api.md).

---

## 2. Anthropic API compatibility (`/v1/messages`)

| Feature | Status | Notes |
|---|---|---|
| Basic messages | Supported | Anthropic schema in/out. |
| `stream: true` | **Supported** per D061 | Canonical event sequence (`message_start` → `content_block_delta` → ...). Compatible with the official `anthropic` Python/TS/Go/Java/Ruby/PHP SDKs via `base_url=<hub>`. |
| `tools` / `tool_choice` | Not supported | Roadmap. |
| Vision / multimodal content blocks | Not supported | Roadmap. |
| Top-level `system` field | Not supported | Only the `messages` array is read. |
| Thinking blocks, ping events | Not emitted | Roadmap. |
| `input_tokens` in `message_start.usage` | Estimated | char/4 rule of thumb; agent-reported `prompt_tokens` not available at first-event emission time. Cumulative `output_tokens` in `message_delta.usage` is exact. |

Ref: [`.qcoda/api.md` §"Anthropic Compatibility"](../.qcoda/api.md), [`decisions.md::D061`](../.qcoda/decisions.md).

---

## 3. Embeddings (`/v1/embeddings`)

| Aspect | Status | Notes |
|---|---|---|
| OpenAI shape | Supported | Single + batch input, `index`-ordered response. |
| Ollama backend | Supported | `nomic-embed-text` default. |
| vLLM backend | Not supported | Deferred per D028 / D036. |
| MLX backend | Not supported | Deferred per D028 / D036. |
| Max input bytes | 256 KB default | `MAX_INPUT_BYTES`, env-tunable. Raised from 32 KB per D049. |
| Max batch count | 128 default | `MAX_BATCH_EMBEDDINGS`. |
| Streaming | N/A | Embeddings are stateless. |

Ref: [`.qcoda/api.md` §"Embeddings"](../.qcoda/api.md).

---

## 4. Image generation (`/v1/images/generations`)

| Aspect | Status | Notes |
|---|---|---|
| Platform | **Apple Silicon Macs only** | mflux is MLX-backed. No Intel Mac, Linux, Windows. |
| Model family | **FLUX-only** | mflux 0.17.5 does not support SDXL or SD 1.5. Reintroducing them requires a different driver (Diffusers, ComfyUI, A1111) — deferred to v2. |
| Models in registry | `flux-schnell`, `flux-dev` | ~32 GB diffusers-layout repo per model. |
| `response_format` | Only `"b64_json"` | URL mode deferred to v2. |
| `n` (images per request) | Capped at 4 | `MAX_IMAGES_PER_REQUEST`. |
| `prompt` length | Capped at 8 KB | `MAX_IMAGE_PROMPT_BYTES`. |
| Model substitution on miss | None | Missing model → 404 with structured `model_not_available`. |
| Content filter | **None shipped** | Operator policy applies. Do not expose to shared devices without a classifier in front. |
| ControlNet / LoRA / img2img / inpainting / upscaling | Not supported | Roadmap. |
| Progress streaming (SSE preview frames) | Not supported | Roadmap. |
| Server-side image history | None | Result returned, then discarded. |
| Auto-download of weights | **Never** | Operator runs `python scripts/install_image_model.py install <id>` explicitly. |

Ref: [`docs/image_gen.md`](image_gen.md), [`decisions.md::D064`](../.qcoda/decisions.md), [`decisions.md::D071`](../.qcoda/decisions.md).

---

## 5. Streaming

| Backend | Streaming default | Notes |
|---|---|---|
| Ollama | Always on | Native streaming. |
| vLLM | Default ON per D044 | Set `VLLM_STREAMING_ENABLED=false` for blocking + D018 bridge (single SSE frame on completion). |
| MLX | Default ON per D060 | osaurus and mlx-lm.server compatible. Set `MLX_STREAMING_ENABLED=false` for blocking. |
| Anthropic `/v1/messages` | Supported per D061 | Canonical Anthropic SSE event sequence. |
| Mid-flight streaming SSE recovery on hub restart | **Not supported** | Task queue itself persists per D053, but a stream in progress is lost. Client must reconnect. |
| Dashboard task view | Supported | SSE token-by-token render. |

Ref: [`docs/streaming.md`](streaming.md), [`.qcoda/features/feature_streaming.md`](../.qcoda/features/feature_streaming.md).

---

## 6. Auth, multi-tenancy, sessions

| Aspect | Status | Notes |
|---|---|---|
| Self-service registration / signup | **None** | Hub operator provisions owners via `server_config.json`. |
| Sample-key startup guard | Enforced per D013 | Hub refuses to start with `change-me-key-*` placeholders. |
| Node token auth | Two-phase per D004 | API key at registration, node token for operations. |
| Dashboard session cookie | Opaque server-side token | `httponly`, `secure`, `samesite=Strict`. |
| CSRF protection on dashboard | Enforced per D055 | Double-submit cookie pattern on `POST /login` + `POST /dashboard/request_inference` + `POST /dashboard/request_image`. `GET /logout` remains GET-driven (worst case: log victim out). |
| Cross-owner data isolation | Enforced | Nodes, sessions, tasks, metrics all scoped by `owner_id`. |
| Owner SSO / OIDC | Not supported | Roadmap. |

Ref: [`.qcoda/authentication.md`](../.qcoda/authentication.md), [`.qcoda/security_privacy.md`](../.qcoda/security_privacy.md).

---

## 7. Storage, durability, recovery

| Component | Backing store | Restart behaviour |
|---|---|---|
| Node registry | In-memory | Lost on restart; nodes re-register on next heartbeat (~5s). Node-slice durability deferred per D026. |
| Task queue | In-memory authoritative + SQLite write-through (D053) | Pending + claimed tasks survive restart; mid-flight streaming SSE is unrecoverable. |
| Sessions | SQLite (default) or PostgreSQL | Survive restart. |
| Session encryption at rest | **None** per D070 | Delegated to FileVault on macOS / EBS-encryption on cloud. Revisit on HIPAA/GDPR/SOC2 ask. **Do not sync `sessions.db` to iCloud/Drive/Dropbox without wrapping it first** (FileVault doesn't protect cloud copies). |
| Hub clustering / HA | Not supported | Architectural change, requires shared state layer. Roadmap. |
| Config hot-reload | Not supported | Hub restart required for `server_config.json` changes. |

Ref: [`.qcoda/architecture.md`](../.qcoda/architecture.md), [`decisions.md::D026`](../.qcoda/decisions.md), [`decisions.md::D053`](../.qcoda/decisions.md), [`decisions.md::D070`](../.qcoda/decisions.md).

---

## 8. Routing

| Aspect | Status | Notes |
|---|---|---|
| Algorithm | Weighted score per D054 | `ram_gb − queue_depth*ROUTING_QUEUE_PENALTY − cpu_load*ROUTING_CPU_PENALTY`. Penalties env-tunable. |
| Latency-aware tiebreaker | Deferred | D054 §"latency_ms" — not implemented. |
| Smart routing / hybrid local-cloud | **Deliberately out of scope** | App's decision, not LLMesh's. We surface token visibility; you build the policy. |
| Image-gen VRAM filter | Enforced | Per-model `min_vram_gb` filter runs before D054 weighted score. |
| Automatic retry on node failure | Enforced | Failed node excluded from candidate set on next attempt. |

Ref: [`.qcoda/nodes.md`](../.qcoda/nodes.md), [`decisions.md::D054`](../.qcoda/decisions.md).

---

## 9. Polling, latency, real-time

| Aspect | Status | Notes |
|---|---|---|
| Agent poll cadence | 5s | Pull-based per D007. Minimum ~5s task pickup latency. |
| WebSocket / SSE long-poll agent model | **Not supported** | High effort, non-blocking for v0/v1. Roadmap. |
| Dashboard auto-refresh | 5s polling | No SSE for status cards. |

Ref: [`.qcoda/architecture.md`](../.qcoda/architecture.md), [`decisions.md::D007`](../.qcoda/decisions.md).

---

## 10. Observability, audit

| Aspect | Status | Notes |
|---|---|---|
| Per-event token metrics | Enforced | Per-owner rollups, configurable retention. |
| Dashboard analytics | 9-chart panel | 7-day window. |
| Structured API key usage / failed-auth audit log | **None** | Forensic investigation falls back on raw server logs. Known gap. |
| Distributed tracing / OpenTelemetry | Not supported | Roadmap. |
| Per-request log correlation IDs | Not exposed | Internal `task_id` only. |

Ref: [`.qcoda/security_privacy.md` §"Missing audit log"](../.qcoda/security_privacy.md).

---

## 11. Deployment

| Aspect | Status | Notes |
|---|---|---|
| Docker + docker-compose | Supported | `Dockerfile` + `docker-compose.yml` ship. |
| nginx fronting | Supported | See `docs/nginx_deployment.md`. |
| macOS launchd plist | Supported | See `docs/macos_launchd.md`. |
| Kubernetes manifests | **Not shipped** | Out of scope for v0/v1. |
| Helm chart | Not shipped | Roadmap. |

Ref: [`.qcoda/deployment.md`](../.qcoda/deployment.md).

---

## 12. UX / accessibility

| Aspect | Status | Notes |
|---|---|---|
| Native mobile clients | **None** | Out of scope. Use any OpenAI/Anthropic-compatible client. |
| WCAG AA audit | Not run | Dashboard is admin-only; revisit before wider distribution. |
| Marketing landing page | Out of LLMesh scope | LLMesh is the open-source broker; the landing page is separate (`llmesh.net`). |

Ref: [`.qcoda/PROJECT.md` §"Out of scope"](../.qcoda/PROJECT.md), [`.qcoda/users_and_stories.md`](../.qcoda/users_and_stories.md).

---

## Found something not on this list?

The decision ledger ([`.qcoda/decisions.md`](../.qcoda/decisions.md)) and discussion log ([`.qcoda/discussions.md`](../.qcoda/discussions.md)) are append-only and tend to land before this page is updated. If you spot a documented gap that should be promoted here, open an issue against the upstream repo — patches welcome.
