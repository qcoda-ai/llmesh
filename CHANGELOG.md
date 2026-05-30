# Changelog

All notable changes to LLMesh. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Per-change provenance lives in [`.qcoda/decisions.md`](.qcoda/decisions.md) — every line below resolves to one or more `Dxxx` entries that record the design rationale, alternatives, and operator impact.

---

## [Unreleased]

Targeted for the next patch on top of v0.20.1. Tag pending.

---

## [0.20.1] — 2026-05-30 — CI/CD + TTFT + image-gen BETA polish

Patch on the v0.2 marketing bundle (v0.20.0). Headline lift is the CI/CD pipeline (D085–D091) — first push-to-deploy on `mk4.schwabe.net` landed 2026-05-30. TTFT dashboard chart (D084) and image-gen BETA tightening (D083) are the user-facing additions; everything else is operator-infra hygiene. See `decisions.md::D092` for the release-bundle decision.

### Changed
- **Agent (`meshclient.service`) consolidated onto hub's codebase + auto-restart on deploy** (D091). New `deploy/meshclient.service` template (User=llmesh, WorkingDirectory=/opt/llmesh/app, EnvironmentFile=/opt/llmesh/app/.env.agent, same hardening as hub). New `deploy/.env.agent.example` documenting `LLMESH_API_KEY`, `HUB_URL`, `LLMESH_NODE_ID`, vLLM + Ollama + MLX backend blocks. `deploy/deploy.sh` gains a conditional agent-restart hook after the hub `/health` gate — hub-only hosts skip via `systemctl list-unit-files`. New `docs/cicd_setup_circleci.md` §1.13 (one-time install + sudoers extension). Closes the drift where mk4's agent ran code from before D068 / D078 / D079 without per-deploy updates.
- **Hub binds `:8003` in production deploy to dodge vLLM port collision** (D088). vLLM defaults to `:8000` and frequently shares a host with LLMesh in SOHO + production-mesh setups. `deploy/llmesh.service`, `deploy/llmesh-nginx.conf`, `deploy/deploy.sh`, `.circleci/config.yml`, and the production-context bits of `docs/cicd_setup_circleci.md` are all swept to `:8003`. Local dev install (uvicorn directly from a checkout) keeps `:8000` — operator muscle memory and no vLLM clash on dev boxes.
- **Relocate app to `/opt/llmesh/app`** (D087). `/home/llmesh/app` was blocked by SELinux Enforcing on RHEL-family (Rocky Linux 9 was the canary — first `systemctl start llmesh` failed with `EnvironmentFile: Permission denied`; systemd policy refuses to read `/home/*`). `/opt` is the default-allowed location for non-distro-package services. The `llmesh` user account stays at `/home/llmesh` (home dir, `~/.ssh`, `~/deploy.sh` symlink). Only the app + venv + state files move. `docs/cicd_setup_circleci.md` §1.2 rewritten with the new clone path + `chown` step; §1.5 updated to recommend `cp + restorecon` on SELinux-Enforcing systems (with `ln -s` documented as the Debian/Ubuntu alternative — both shapes still handled by `deploy.sh`'s drift check). Includes one-time server-side migration commands for existing partial installs.
- **systemd unit refactored to gunicorn-supervised uvicorn worker + nginx config template + Let's Encrypt step** (D086). `deploy/llmesh.service` now launches `gunicorn lib.hub.server:app -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8000 --timeout 600 --graceful-timeout 30 --forwarded-allow-ips '*'`. **`-w 1` is pinned** — hub holds in-memory authoritative state (D058 node registry, D053 task queue, D027/D056 models cache, D005 compression model) that breaks under multi-worker. New `deploy/llmesh-nginx.conf` template adapted from `../qcoda` (rate limit, scanner-trap, security headers, SSE-tuned `proxy_buffering off` + `proxy_read_timeout 600s`, certbot placeholders). New `docs/cicd_setup_circleci.md` §1.10 (nginx install) + §1.11 (Let's Encrypt) + §1.12 (loopback/firewall hardening). `gunicorn==23.0.0` added to runtime dependencies. `deploy/deploy.sh` now verifies the gunicorn binary post-install + worker count post-restart.

### Added
- **CI/CD via CircleCI** (D085). `.circleci/config.yml`, `deploy/deploy.sh`, `deploy/llmesh.service`, `docs/cicd_setup_circleci.md`. Single environment on `your-deploy-host.example.com` under systemd. Branch `main` auto-deploys after three gates (`pytest tests/unit/`, `scripts/validate_decisions.py`, pinned `gitleaks`). PRs run the test gate only. Pattern mirrors `../qcoda` + `../talential-python-backend`. Comparison + Bitbucket Pipelines alternative preserved in `.qcoda/strategy/cicd_circleci_vs_bitbucket_pipelines.md`.
- **Hub-side time-to-first-token (TTFT) metric + dashboard chart** (D084). New `inference_events.ttft_ms` column (NULL-safe additive migration). Captured at the first non-sentinel chunk in both the OpenAI and Anthropic SSE generators. Surfaced on the dashboard Stats tab as a p50/p95-per-model bar chart over the last 24h, sorted ascending by p50, models with fewer than 20 samples omitted. Hub-side definition (includes routing + agent dispatch + backend cold-start) — operator responsiveness signal, not a pure backend benchmark. Lets operators measure how responsive their mesh feels per model. Stdlib-only (`statistics.quantiles`); no numpy / pandas. 8 unit tests.
- **Image-gen non-blocking submit + status view + result poll** (D082). `POST /dashboard/request_image` redirects to a per-task status page that polls `/dashboard/image/{node}/{task}/result` every 1s. Live elapsed-time counter; inline base64 PNG render + download link on success; structured failure panel that reads from the agent's `.err` stream.
- **Image-gen `test` quality tier** (D081). 1-step FLUX-schnell smoke path; renders in ~5 s on M1 Ultra for fast operator verification.
- **Agent log split** (D080). stdout = INFO/DEBUG, stderr = WARNING+. Fixes the "empty `.log` so the operator can't see why something failed" trap.

### Changed
- **Image generation flagged BETA + tightened system requirements** (D083). 64 GB UMA Apple Silicon minimum, 128 GB recommended for production. **Do not co-run mflux with other large MLX/LLM processes** (Ollama with a big model loaded, mlx-lm.server, osaurus, vLLM) — co-resident large RSS has caused a kernel panic + hard reboot on M1 Ultra 64 GB (2026-05-29 23:12, panic log analyzed: `watchdog timeout` triggered by VM compressor saturation, not direct OOM). Dashboard `Image` tab carries a BETA badge + sysreq alert; `docs/image_gen.md` opens with the stability advisory. See [`docs/known_limitations.md`](docs/known_limitations.md) §1a.
- **Image error path** — `logger.exception` + `repr(e)` on the wire so the dashboard failure panel + agent log together carry the full traceback context.

### Fixed
- **Relative-import trap in agent inference path** (D079). `from .` removed from every module reachable from `python lib/agent/client.py` (launchd executes script form, not `python -m`). Same bug class as D078.
- **Image capability probe silent fail** (D078). Promoted from `logger.debug` to `WARNING` + switched to absolute imports — operators now see the cause in the agent error log instead of a missing capability with no signal.
- **Per-node `agent_version` surfaced on hub + image-transition re-register trigger** (D076). Image gen install/uninstall now re-registers the node so the dashboard reflects capability changes within one heartbeat.
- **Dashboard image-gen capability surface** (D075). Capability was plumbed in D064 routing but never serialised to `/api/nodes` or rendered. Image card on the dashboard now matches the underlying routing state.

### Internal
- `agent_version` default tuned to `"0.1x"` with neutral badge palette (D077) — stops flagging the working baseline as a warning state.

---

## [0.20.0] — 2026-05-29 — "v0.2" bundle release

The headline release for LLMesh v0.2. Bundles every shipped feature from `D040` through `D073` under a single version tag. Internal `0.11.0 → 0.20.0` jump is intentional and matches the marketing-`v0.2` → `0.20.0` mapping; the hold on the version bump (D063) gated specifically on image-gen graduating LAB-005.

### Headline features

- **Adaptive chunked SSE streaming (`StreamBatcher`)** (D041 + D067 + D068). Three flush triggers (size, time-since-flush, target client PPS) + TPS-driven sliding window. Converges to ~8× token aggregation at MLX rates without regressing time-to-first-token. Cuts hub `/stream` syscall pressure under fast clusters by ~80%. **Unified across all three backends** (Ollama, vLLM, MLX) — no per-backend streaming divergence (D067 closed a 6-week gap). Per-batch telemetry surfaced agent → hub → dashboard (D068). Operator escape hatch: `STREAM_BATCH_FIXED=N` pins batch size and disables adaptive auto-tune. Tunables: `STREAM_BATCH_INITIAL` / `_TIME_MS` / `_TARGET_PPS` / `_MIN` / `_MAX` / `_MAX_BUFFER`.
- **MLX real per-token streaming (default ON)** (D059 + D060). New `_run_streaming_mlx()` in the agent mirrors the vLLM streaming helper minus the bearer header. **Osaurus** is the verified primary MLX backend (Apache 2.0 Swift); `mlx-lm.server` also works. Wire shape verified end-to-end on M1 Ultra. Runs on top of `StreamBatcher` (above). Set `MLX_STREAMING_ENABLED=false` to revert to the D018 bridge path. LAB-003 graduated CONCLUDED (6/6 + 3 manual items, 3 consecutive passes).
- **Image generation v1 (BETA, see D083 in Unreleased)** (D064 + D071 + D073). OpenAI-compatible `POST /v1/images/generations` routes to image-capable nodes. **mflux in-process** on Apple Silicon Macs; FLUX-schnell and FLUX-dev. Operator-explicit model install (`scripts/install_image_model.py`) — never auto-downloads weights. Install layout is diffusers multi-file via per-file curl with resume (D071, resolves D-005). Ship-nothing safety, `image_event` metrics. LAB-005 graduated CONCLUDED (3 consecutive 6/6 on M1 Ultra). Dashboard `Image` tab + per-node capability badge. See [`docs/image_gen.md`](docs/image_gen.md) — **read the BETA + sysreq advisory before enabling**.
- **Anthropic Messages SSE streaming on `/v1/messages`** (D061). Hub-side reframing only (zero agent changes) — reuses the OpenAI `stream_queue`. Canonical 7-event sequence (`message_start` → `content_block_start` → `content_block_delta`×N → `content_block_stop` → `message_delta` → `message_stop`). `input_tokens` estimated from char/4 (agent prompt_tokens not available pre-stream); cumulative `output_tokens` in `message_delta` is exact. Verified inline against the official `anthropic` Python SDK 0.39.0 in LAB-004 (CONCLUDED).
- **vLLM real per-token streaming (default ON)** (D040 + D044 + D045 + D047). `_run_streaming_vllm()` in the agent. Runs on top of `StreamBatcher`. Set `VLLM_STREAMING_ENABLED=false` to revert. LAB-002 graduated CONCLUDED. Forwards client `max_tokens` (D044). Hub `/stream` done frame delivers chunk content before close sentinel (D045, real production bug).
- **Hub state durability** (D053 task queue + D058 node registry). Pending + claimed tasks recover across restart (claimed → reset to pending). Node-slice persisted to SQLite; hash-only token storage; dual-verify path. Closes D026 node-slice durability. Best-effort write-through; in-memory authoritative.
- **Weighted node routing** (D054). `ram_gb − queue_depth*ROUTING_QUEUE_PENALTY − cpu_load*ROUTING_CPU_PENALTY` (defaults 8.0 / 0.1). Drops the pure-RAM tiebreaker for a richer load-aware score. Set `ROUTING_QUEUE_PENALTY=0` to revert.
- **`/v1/limits` endpoint + `MAX_INPUT_BYTES` raised 32 KB → 256 KB** (D049). Structured `payload_too_large` error type. Per-model context block returned for client-side pre-clamping.
- **CSRF protection on dashboard + login POST forms** (D055).
- **Per-owner `/v1/models` LRU cache** (D027 + D056). `MODELS_CACHE_MAX=64` bound.

### Changed

- **Ledger Law pre-commit gate flipped warn-only → BLOCKING** (D065). `OPEN` entries in `.qcoda/decisions.md` now block commits, as does the doc-coverage gate for `lib/hub/`, `lib/agent/`, and `lib/views/templates/` changes.
- **Operator-supplied `LLMESH_NODE_ID`** (D048). Replaces auto-fingerprint with operator label. Regex `^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$`.
- **`AGENTS.md` is publish-friendly** (D050). Internal-only governance lives in `.claude/CLAUDE.md`. Public mirror at `github.com/qcoda-ai/llmesh` rsynced from private Bitbucket via `publish.sh`.
- **vLLM `stream_options.include_usage` capability detection** (D066). One-shot logging; behaviour-based, not version-string parsing.

### Decisions resolved

- D063 closes (hold version bump until image-gen ships).
- D026 node-slice closes (durability shipped via D058).
- Image-gen exploratory thread D-003 resolves → D064.
- LAB-005 install-layout blocker D-005 resolves → D071.

### Known limitations

See [`docs/known_limitations.md`](docs/known_limitations.md). v1 does not ship tool/function calling, JSON mode, logprobs, `n > 1`, vision/multimodal input, ComfyUI/A1111/Draw Things image backends, or hub clustering/HA.

---

## [0.11.x] and earlier — historical

Pre-v0.2 work shipped under `0.11.x` patch versions without per-release tags. The decision ledger is the authoritative source of pre-v0.2 history:

- **Node token auth + rate limiting + SSE streaming for Ollama** (D004, D008, D009, D010, D016, D017, D019, D023, D024).
- **vLLM bearer auth, context auto-detect, streaming bridge for non-streaming backends, structured error-flag signal** (D014, D015, D018, D025, D034, D044, D045).
- **`/v1/embeddings` (Ollama)** (D028 + D029 + D030).
- **`/v1/models` capabilities field** (D031). Per-owner TTL cache (D027).
- **Session memory: in-process compression on the hub** (D005). SQLite default, Postgres production (D002).
- **Task queue: DB-backed with in-memory authoritative** (D003).
- **Two-phase node authentication** (D004). API key at registration; node token for operations.
- **Routing algorithm: RAM-first MVP baseline** (D001). Superseded by weighted routing in v0.20.0 (D054).
- **Ledger enforcement infrastructure** (D042 + D043). Pre-commit hook + validator (warn-only initially; flipped BLOCKING in v0.20.0 per D065).
- **Labs pattern formalisation** (D046). LAB-002 (vLLM streaming) graduated CONCLUDED (D047).
- **Sample-key startup guard** (D013). Hub refuses to start with the shipped sample keys.
- **No rebrand; ship as LLMesh; launched mid-May 2026** (D057). Public site `https://llmesh.net`. Closes pre-launch `todo.md` phases 0 + 1.
- **Decline session-history encryption at rest** (D070). Documented decline; deployment model makes it redundant.

For granular per-decision provenance, see [`.qcoda/decisions.md`](.qcoda/decisions.md) (~83 committed decisions through v0.20.0).
