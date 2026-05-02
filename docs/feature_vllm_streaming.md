# Feature Plan — vLLM real per-token streaming + adaptive token batching

**Status:** PROPOSED — pending approval of OPEN decisions D040 + D041 in `.qcoda/decisions.md`
**Owner:** Andrew
**Last updated:** 2026-04-29
**Effort:** ~4-5 days realistic (was 2.5 — brutal re-eval added P0 fixes and 25 risk items)

---

## Goal

1. Replace blocking-via-D018 vLLM inference with real per-token streaming on `/v1/chat/completions stream:true`.
2. Add a generic agent-side adaptive token batcher (`StreamBatcher`) so fast vLLM (50-200 tok/s) does not flood hub `/stream` with 1 POST per token.
3. Operator escape hatch: `STREAM_BATCH_FIXED=N` env var pins batch size, disables auto-tune.

## Non-goals

- Per-request `num_ctx` for vLLM (D015 limitation, server-fixed).
- Hub-side batch consumption / aggregation. Hub stays unchanged.
- MLX streaming (separate decision later).
- Cancellation propagation hub → agent (existing limitation, document only).
- Hot-reload of `STREAM_BATCH_*` env vars (restart required, document).
- Refactor of `_run_streaming_ollama` to share `StreamBatcher` (separate PR + decision after vLLM stabilizes).

---

## Architecture

### Agent dispatcher
`lib/agent/client.py` lines 704-710 currently:
```
if task.get("stream"):
    if backend == "ollama":
        await _run_streaming_ollama(...)
    else:
        # vLLM/MLX falls back to blocking
```
Change: route `backend == "vllm"` to new `_run_streaming_vllm()`. Blocking path untouched for `stream:false` requests.

### `_run_streaming_vllm()` — new
Lives in `lib/agent/client.py`. Uses:
- `httpx.AsyncClient.stream("POST", "/v1/chat/completions", ...)` with `stream: True`, `stream_options: {"include_usage": true}`, headers from `_vllm_headers()` (D014).
- SSE event parser (see Critical fixes below).
- `StreamBatcher` instance fed by parsed deltas.
- State machine: `finish_reason` flips flag → wait for `usage` chunk → `[DONE]` triggers final flush with done frame.

### `StreamBatcher` — new module
`lib/agent/streaming_batcher.py`. Single-coroutine model: stream loop calls `await batcher.add(token)`, batcher checks both size and time on every add. No background timer task = no race. **EXCEPT** when stream stalls (vLLM mid-burst gap), no token = no flush. Mitigation: timer task with `asyncio.Lock` covers the stall case. Both belt and braces.

Constructor:
```
StreamBatcher(
    flush_callback,          # async fn(chunk: str, *, done: bool, **meta)
    initial_size=10,
    time_cap_ms=100,
    target_pps=10,
    min_size=1,
    max_size=100,
    fixed_size=None,         # if set, disable auto-tune
    max_buffer=1000,         # hard cap, abort task if exceeded
)
```

Auto-tune (when `fixed_size is None`):
- Sliding window of `(timestamp, count)` for last 5s.
- TPS metric: `tokens_in_window / (last_add - first_add)` — inter-token rate, NOT flush-to-flush. Defends against slow-hub death spiral.
- New size: `clamp(round(tps / target_pps), min_size, max_size)`.
- Smoothing: `size = round(0.7 * old + 0.3 * new)`.
- Recompute on each flush.

Fixed mode:
- `fixed_size=20` → batch always 20.
- Time cap (`STREAM_BATCH_TIME_MS`) still honored (latency floor).
- `_retune()` short-circuits.
- Window still tracked for metrics surface.

### Wire format — preserve hub contract
Each batch POST to `/tasks/{node_id}/{task_id}/stream`:
```json
{"chunk": "concatenated tokens", "done": false}
```
Final POST piggybacks done + usage:
```json
{"chunk": "last batch tokens", "done": true, "tokens_p": 31, "tokens_c": 47, "batch_size_final": 12}
```
Single POST when possible (last batch combined with done) — protects against last-batch-fails-then-no-done bug.

Hub side: zero change. `_real_sse_generator` consumes as today.

---

## Critical fixes (P0 — must be in implementation, not optional)

### CF-1 — SSE event boundary parsing
- `httpx.aiter_lines()` returns ONE LINE AT A TIME, not events. Single SSE event spans multiple lines + terminator `\n\n`.
- `data:` value can theoretically span lines (folded). vLLM doesn't fold in practice but parser must not assume.
- **Implementation**: switch to `aiter_bytes()` + manual SSE parser, OR use `httpx-sse` lib. Buffer until `\n\n`, then parse event block: extract `data:` lines, concatenate.
- **Tests**: simulated transport that returns truncated reads (`data: {"x":` then `1}\n\n`) → parser reassembles correctly.
- Skip lines starting with `:` (heartbeat comments — LiteLLM proxy emits these).

### CF-2 — Race between batcher timer and stream loop
- Background timer flushes from coroutine A, stream loop calls `add()` from coroutine B → torn writes to `self.buffer`.
- **Implementation**: `asyncio.Lock` around `flush()` and `add()`. Timer task uses lock. Stream loop also checks elapsed time on each add (eliminates dependency on timer for non-stall case).
- Timer task: `asyncio.wait_for(condition, timeout=time_cap_ms)` instead of polling — wakes on token-arrives event OR timeout, no busy poll.
- **Tests**: hammer with concurrent add + simulated timer fires → no lost tokens, no double flush.

### CF-3 — Serial POST ordering
- Adaptive batcher = variable POST cadence. Out-of-order delivery → consumer sees tokens reordered.
- **Implementation**: `await POST` per batch, never `asyncio.create_task(POST)`. Strictly serial. Document explicitly in `_run_streaming_vllm` docstring.
- **Tests**: inject artificial delay on alternate POSTs → consumer order preserved.

### CF-4 — finish_reason vs [DONE] vs usage state machine
- vLLM order: `data: {choices: [{delta:{}, finish_reason: "stop"}]}` → `data: {usage: {...}}` → `data: [DONE]`.
- Naive flush-on-finish-reason → `usage` arrives after, lost.
- **Implementation**: state machine
  - SAW_FINISH_REASON: capture finish, do not flush
  - SAW_USAGE: capture tokens_p, tokens_c
  - SAW_DONE: trigger final batch flush + done frame with captured usage
- **Tests**: parser sees finish → usage → [DONE] in correct order, done frame populated.

### CF-5 — Done frame must always fire
- Last batch POST fails → done frame skipped → SSE consumer hangs until `STREAM_CHUNK_TIMEOUT`.
- **Implementation**: combine last batch + done into single POST when possible (piggyback). If last batch must be separate (oversize), wrap in try/except and ALWAYS attempt done frame after.
- **Tests**: last batch POST raises → done frame still emitted with `error: "last batch dropped"` flag.

### CF-6 — D018 bridge double-fire prevention
- D018 pushes result to `stream_queue` on `/complete` finalize.
- Real streaming path emits chunks AND a final done sentinel via `/stream`. If agent ALSO calls `/complete` (current path does), bridge fires after stream sentinel → consumer sees double delivery.
- **Implementation**: streaming path NEVER calls `/complete`. Mid-stream errors emit error frame via `/stream` with `done:true, error:...`. Hub's `submit_task_result` is reached only via blocking path.
- Bridge unchanged (still active for MLX, blocking fallback). Add defensive check: bridge no-ops if `stream_queue` already received sentinel.
- **Tests**: streaming vLLM task → `/complete` never called. Verify via mock hub call counter.

---

## High-priority risks (P1 — address before flag flip)

### HP-1 — Token accounting on partial failure
- Mid-stream disconnect, no `usage` chunk. `tokens_c=null` is silently wrong.
- **Implementation**: agent counts `delta.content` chunks as fallback (`usage_source: "estimated"` flag in done meta). `tokens_p=0` with explicit flag, hub displays `~` prefix.
- **Tests**: simulated disconnect after 50 deltas → done frame has tokens_c=50, source=estimated.

### HP-2 — STREAM_CHUNK_TIMEOUT during vLLM cold start
- First token can take 2-5s while model loads. If hub timeout is absolute (not per-chunk), stream dies before first token.
- **Implementation**: read `lib/hub/server.py` `STREAM_CHUNK_TIMEOUT` semantics. If absolute → emit keepalive frame (empty chunk, special meta) every 1s during gap. If per-chunk reset → no action needed, document.
- **Tests**: simulated 5s pre-token delay → consumer not timed out.

### HP-3 — TPS metric robustness
- TPS using flush-to-flush time → slow hub POST inflates time → batch shrinks → more POSTs → death spiral.
- **Implementation**: window stores `add()` timestamps only. TPS = `count / (last_add_ts - first_add_ts)`. POST latency excluded.
- **Tests**: simulated 200ms-per-POST hub → batch grows correctly based on real vLLM speed.

### HP-4 — Drop per-backend cache
- Cache `_last_batch_size_per_backend` is wrong for multi-model backends (7B vs 70B speed differs 10x).
- **Implementation**: drop cache. Each task starts at `STREAM_BATCH_INITIAL`. 5s adaptation is fast enough.
- **Tests**: two tasks back-to-back with different speeds → both adapt independently from initial.

### HP-5 — Memory cap on stuck batcher
- Hub unreachable + retries → buffer grows unbounded.
- **Implementation**: `STREAM_BATCH_MAX_BUFFER=1000` hard cap. Beyond → log error, abort task, close vLLM stream.
- **Tests**: simulated hub-unreachable → buffer stops at cap, task aborts cleanly.

---

## Medium / hygiene (P2)

- **Logging discipline**: NOT per token, NOT per batch. Per task: start, batch-size changes >2x only, end with summary.
- **`httpx.AsyncClient` reuse**: shared per agent, not per task. Connection pool stays bounded.
- **`[DONE]` literal**: NOT JSON. Explicit string check before `json.loads`.
- **Heartbeat comment skip**: `:` lines ignored.
- **Mid-stream error frame**: vLLM may emit `data: {"error": {...}}` (OOM, model crash). Parser checks `error` key, fails task gracefully via stream done frame.
- **`num_ctx` ignored warning**: log once per task when `num_ctx` set + value < server max.
- **Cancellation hint**: agent reads hub POST response. If hub returns 410/cancelled → tear down vLLM stream, drop batcher.
- **Done meta `batch_size_final`**: include adaptive end-state batch size for ops visibility.

---

## Configuration — env vars (agent)

| Var | Default | Purpose |
|-----|---------|---------|
| `VLLM_STREAMING_ENABLED` | `false` | Master gate. Default off until manual real-vLLM verification clean. |
| `STREAM_BATCH_INITIAL` | `10` | Starting batch size for adaptive mode. |
| `STREAM_BATCH_TIME_MS` | `100` | Latency cap. Forces flush even if size not reached. |
| `STREAM_BATCH_TARGET_PPS` | `10` | Target POST/s rate. Adaptive math aims here. |
| `STREAM_BATCH_MIN` | `1` | Adaptive floor. |
| `STREAM_BATCH_MAX` | `100` | Adaptive ceiling. |
| `STREAM_BATCH_MAX_BUFFER` | `1000` | Hard cap before abort. |
| `STREAM_BATCH_FIXED` | (unset) | If set to N>=1, disables auto-tune. Operator escape hatch. |

Resolution priority: `STREAM_BATCH_FIXED` > `STREAM_BATCH_INITIAL` > defaults. `STREAM_BATCH_FIXED=0` or invalid → fallback to adaptive + warning.

---

## Test plan (~40 tests total)

### Unit — `tests/test_streaming_batcher.py` (new, ~20 tests)
1. Size trigger — 10 tokens → 1 flush.
2. Time trigger — 5 tokens + 100ms wait → flush.
3. Done trigger — flush partial + done frame.
4. Auto-tune — 100 tok/s for 5s → batch grows toward 10.
5. Auto-tune — 5 tok/s → batch shrinks to 1.
6. Smoothing — sudden 10x speed change does not overshoot.
7. Empty stream — done with no tokens → 1 POST (done only).
8. Window pruning — old entries (>5s) excluded.
9. Time-cap timer fires when no new tokens arrive.
10. Hub 5xx mid-stream → 1 retry, then error frame, no infinite loop.
11. Concurrent batchers per task → state isolated.
12. Race: concurrent add + timer fires → no lost tokens, no double flush (CF-2).
13. `STREAM_BATCH_FIXED=20` → batch never deviates across speed changes.
14. `STREAM_BATCH_FIXED=1` → 1 token per POST.
15. `STREAM_BATCH_FIXED=20` + tokens stop → time cap fires partial flush.
16. `STREAM_BATCH_FIXED=0` → fallback to adaptive + warning logged.
17. `STREAM_BATCH_FIXED=invalid` → fallback to adaptive + warning logged.
18. Slow-hub TPS measurement — simulated 200ms POST → TPS reflects vLLM speed (HP-3).
19. Memory cap — buffer stops at `STREAM_BATCH_MAX_BUFFER`, task aborts (HP-5).
20. Last-batch-fails → done frame still fires (CF-5).

### Unit — `tests/test_vllm_streaming.py` (new, ~15 tests)
1. SSE parser — well-formed `data: {...}` frames → expected chunk POSTs.
2. SSE parser — `[DONE]` sentinel triggers done frame.
3. SSE parser — empty `delta.content` skipped.
4. SSE parser — role-only first chunk skipped.
5. SSE parser — heartbeat `:` comment lines skipped.
6. SSE parser — multi-read event reassembly (CF-1).
7. SSE parser — `[DONE]` literal not JSON-parsed.
8. State machine — finish_reason → usage → [DONE] order respected (CF-4).
9. Token accounting — usage chunk → tokens_p, tokens_c on done frame.
10. Token accounting — missing usage → estimated counts + flag (HP-1).
11. Mid-stream disconnect → error frame via `/stream`, no `/complete` call (CF-6).
12. HTTP 429 / 503 on stream open → error path, no aiter invocation.
13. Mid-stream `error` JSON frame → graceful task fail with done sentinel.
14. Bearer auth header present on stream POST (D014).
15. `num_ctx` set + smaller than server max → warning logged once.

### Regression — extend existing
- `tests/test_streaming_bridge.py` — vLLM streaming path → bridge does NOT double-fire (CF-6 negative test).
- `tests/test_streaming_bridge.py` — bridge still no-ops when `stream_queue` is None.
- `tests/test_vllm_auth.py` — extend: streaming path uses same headers as blocking.
- `tests/test_vllm_context.py` — unchanged.

### Integration — `tests/test_vllm_streaming_integration.py` (new)
- Mock vLLM server via httpx mock_transport, real SSE captures replayed.
- 100 tokens @ 200 tok/s → expect ~10 POSTs (not 100).
- 10 tokens @ 5 tok/s → expect ~10 POSTs (each token, time-capped).
- Serial POST ordering preserved under injected delay (CF-3).
- Unicode-heavy stream (CJK, emoji) → bytes-equal output to non-batched reference.
- Concurrent 4 streaming tasks → no chunk interleaving across tasks.

### Manual real-vLLM gate (pre-flag-flip)
- Local vLLM with small model, real `/v1/chat/completions stream:true`.
- Dashboard task viewer fills token-by-token.
- Verify usage shows correct counts.
- Kill vLLM mid-stream → hub shows error within 1s.
- Concurrent 4 streaming tasks → no hub crash, no chunk reorder.
- Operator-style deployment per D015/D018 lessons (real hardware surfaces real bugs).

---

## Sequencing

1. Plan + decisions OPEN — this doc + D040, D041 entries written, status OPEN.
2. `StreamBatcher` class + 20 unit tests — TDD, standalone, no integration.
3. SSE event parser + 7 SSE tests — TDD, standalone.
4. `_run_streaming_vllm()` wiring batcher + parser + 8 vLLM tests.
5. Dispatcher branch wired behind `VLLM_STREAMING_ENABLED=false`.
6. Regression tests pass — bridge no-double-fire, blocking path untouched.
7. Mock-server integration test passes.
8. Manual real-vLLM gate — operator verification.
9. Flip flag default → on. Decisions D040 + D041 → COMMITTED.
10. Docs update: `feature_streaming.md`, `nodes.md`, `README.md`, `.env.example`. Remove D028 deferred-vLLM comment at `client.py:686`.
11. Demo gif (per recent commit pattern).

## Rollback

- `VLLM_STREAMING_ENABLED=false` reverts to current blocking + D018 bridge.
- Zero hub-side changes = zero hub rollback.
- `STREAM_BATCH_FIXED=1` = effectively per-token mode (debug parity with current Ollama).
- Single env var on agent restart returns to last known-good.

## Decision dependency

D040 ships D041 inside it (StreamBatcher is foundation, not optional). Adaptive auto-tune is a flag on top. If D040 ships without D041 by accident → vLLM streaming floods hub. Mitigation: implementation order makes batcher a hard dependency of `_run_streaming_vllm()` — no streaming code path that does not go through batcher.

## Top 3 bite risks if any P0/P1 ignored

1. SSE line/event boundary parsing (CF-1) → silent corruption, garbled output.
2. Race between batcher timer and stream loop (CF-2) → lost tokens, double POSTs.
3. finish_reason / [DONE] / usage ordering (CF-4) → tokens_c always missing on done frame.

---

## Open questions (resolve before COMMITTED)

- [ ] `STREAM_CHUNK_TIMEOUT` semantics — absolute or per-chunk? Decide HP-2 keepalive design.
- [ ] vLLM version detection viable, or always assume `include_usage` and treat absence as known case?
- [ ] Ollama refactor onto `StreamBatcher` — separate decision after D040 ships, or bundled?
- [ ] Metrics surface (current batch size, POST/s) — out of scope for D040, follow-up decision?
