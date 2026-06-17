# SPEC — Tool-Call Normalization in nodemesh (graduate the qcoda proxy)

> **Status:** IMPLEMENTED 2026-06-16 as **D101** (v0.21.4). Gates G1–G3 re-probed on ollama 0.30.8 — reality differed from §1/§3 (qwen3-coder emits **native** structured calls, not XML-in-content; qwen2.5-coder leaks in streaming too). See `.qcoda/decisions.md::D101`.
> **Target repo:** `qcoda-nodemesh` (LLMesh) — **NOT** qcoda core.
> **Authored:** 2026-06-16 by aschwabe@gmail.com (via Claude Code, qcoda session).
> **Implement in:** a fresh Claude Code session rooted in `qcoda-nodemesh`, following that repo's `.qcoda/CONVENTIONS.md` + `AGENTS.md`.
> **Sibling of:** nodemesh decision **D098** (hub→agent `tools` serialization). This is the **response-in** sibling of that **request-out** fix.

---

## 1. Problem

ollama leaves **qwen-family** tool calls as **text in `message.content`** instead of populating structured `tool_calls`:
- `qwen2.5-coder` → JSON: `{"name": "...", "arguments": {...}}` in content
- `qwen3-coder` → XML: `<function=name>...<parameter=k>v</parameter></function>` in content
- `devstral` → emits structured `tool_calls` **natively** (no fix needed)

D098 already fixed the upstream half: `tools` now reach ollama so the model *sees* them. But when the model *answers* with a text-form call, nodemesh returns `content=<text>` + `tool_calls=[]`. Any client expecting structured tool_calls (pi, goose, opencode, MCP clients via Claude Desktop / Chatwise / Goose) silently **no-ops** — it never sees a tool call to execute.

**Goal:** parse text-form tool calls out of `content` and inject structured `tool_calls` at the serving layer, so every client gets working tool calls on qwen models — without each client needing its own shim.

A working reference implementation exists as a dev-time proxy in the **qcoda** repo (port 8765, A/B proven: stock pi→qwen2.5 = no-op; pi→proxy→qwen2.5 = executes edit). This spec graduates its **parse rules** into nodemesh's serving path. **Do NOT ship the standalone port-8765 proxy to prod** — it's a dev shim (extra hop, SPOF). Port the *rules*, not the process.

---

## 2. Verified grounding (read-only recon, 2026-06-16)

All citations verified against `qcoda-nodemesh` HEAD this session. Implementer should re-grep to confirm line numbers haven't drifted.

**Architecture:** hub-and-spoke, NOT single-process.
- **Hub** = `lib/hub/server.py` (FastAPI) — accepts client request, queues `Task`, shapes OpenAI/Anthropic response back.
- **Agent** = `lib/agent/client.py` — runs per compute node, polls hub, calls **local** backend (ollama `localhost:11434` / vLLM / MLX), submits result back.
- Seam: HTTP task-queue (`/tasks/{node}/pending` out, `/tasks/{node}/complete/{task}` back).

**Backend dispatch (agent-side):** `lib/agent/client.py:1416-1425` — `vllm_models`→vllm, `mlx_models`→mlx, **else ollama (default/fallback)**. qwen-family serves via **ollama** in practice (every qwen ref in repo is ollama-path; vLLM/MLX emit structured `tool_calls` natively at `client.py:1490-1497`).

**D098 sibling fix:** commit `51c20b6`, `lib/hub/server.py:1226-1230` (`entry["tools"] = t.tools` in `get_pending_tasks_for_node()`). Request-out path. This spec is the response-in counterpart.

**Response-in hook point (THE place to normalize):** `lib/agent/client.py:1528-1540` — ollama non-stream reads `msg.get("content")` (1533) + `msg.get("tool_calls")` (1534). **This is where the qwen text call sits in `content` with `tool_calls=[]`.** Parse + inject slots in between 1534 and 1540. Result submitted at `client.py:1573-1577`.

**No existing text-form parser** anywhere in `lib/` (grep-confirmed). Only structured-shape coercion exists: `_format_tool_calls_for_openai` (`server.py:551-577`) reshapes already-structured dicts to OpenAI wire shape — it does NOT parse text. Your parser is net-new and feeds this existing coercion.

**Streaming caveat:** ollama **always streams** in this codebase (`client.py:1433-1436` → `_run_streaming_ollama` ~718). Streaming capture (`client.py:835-847`) reads structured `tool_calls` off frames. **If qwen emits a text call in streaming mode, it lands in accumulated content → needs a SECOND parse path.** This is unverified — see Gate G2.

**Reference parser to port (qcoda repo):** `labs/agent_eval/auto_code/toolcall_proxy.py`:
- `_extract_tool_calls` (line 40) — XML `<function=…>` regex (36-37) + JSON brace-matcher
- `_json_objects` (line 80) — balanced-brace JSON extraction
- `_mk_call` (line 70) — builds the structured call dict
- `_normalize` (line 108) — sets `content=None` + `finish_reason="tool_calls"`

---

## 3. Empirical-verify GATES (run BEFORE implementing — do not assume)

Per nodemesh epistemic discipline. STOP and report if any gate's reality differs from this spec.

**G1 — Confirm qwen serves via ollama in the target deployment.**
```
# On a node, or against mesh.qcoda.com:
curl -s $VLLM_HOST/health   # is vLLM even up for qwen?
# Check node resources advertise qwen under ollama_models, not vllm_models.
```
If qwen is served via **vLLM** in prod, the correct fix is vLLM's native `--tool-call-parser hermes|qwen3_coder` on the vLLM server (operator config, NOT nodemesh code) — see §7. Only proceed with the code fix if ollama is the qwen backend.

**G2 — Determine whether qwen emits text calls in STREAMING mode.** ollama always streams here. Probe a streamed completion with `tools` against qwen3-coder + qwen2.5-coder and inspect whether the tool call arrives as a structured frame or as accumulated text content:
```
curl -sN localhost:11434/api/chat -d '{"model":"qwen2.5-coder:32b","stream":true,"messages":[...],"tools":[...]}'
```
- If structured frame → streaming path already works, only the non-stream/accumulated path needs the parser.
- If text in content → **a second parse path is required** on the accumulated streamed content (`client.py:835-847` capture + `_run_streaming_ollama`). Scope it in.

**G3 — Confirm the two text formats per model.** Re-probe qwen2.5-coder (expect JSON-in-content) and qwen3-coder (expect XML `<function=>`) to confirm the parser covers the actual emitted shapes on the target ollama version. Quote the raw `content` in the decision entry.

---

## 4. Design decision — hook at the AGENT layer

Normalize at `lib/agent/client.py` response-in (~line 1534), **not** the hub. Rationale:
- Closest to ground truth — raw ollama JSON before the hub collapses anything.
- Symmetric with D098 (which made tools *arrive* at the agent). Tools-in and calls-out live in the same process.
- No new task-queue field needed — the agent submits already-normalized `tool_calls` at `client.py:1573-1577`; the hub's existing `_format_tool_calls_for_openai`/`_format_tool_calls_for_anthropic` shaping then works unchanged for BOTH the OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` client surfaces.

Streaming path (if G2 shows text calls): parse the **accumulated content** in the streaming capture and emit the structured call the same way the non-stream path does. The hub's SSE synthesis (`server.py:860-889 _real_sse_generator`) already turns a structured `tool_calls` into a delta frame — feed it one.

---

## 5. Implementation requirements

**R1 — Port the parser.** New module (suggest `lib/agent/toolcall_parse.py`) with pure functions ported from `toolcall_proxy.py`:
- `extract_tool_calls(content: str) -> list[dict]` — handles BOTH qwen2.5 JSON-in-content AND qwen3-coder XML `<function=>`. Balanced-brace JSON extraction (not naive regex) per the reference `_json_objects`.
- Builds OpenAI-shape call dicts `{id, type:"function", function:{name, arguments:<JSON string>}}` (let the existing `_format_tool_calls_for_openai` do final coercion, or emit final shape directly — pick one, document it).
- Pure + deterministic + unit-testable in isolation (no I/O), mirroring how `_format_tool_calls_for_openai` is tested.

**R2 — Wire at the non-stream hook** (`client.py:~1534`). After reading `msg`:
```
content = msg.get("content")
tool_calls = msg.get("tool_calls") or []
if not tool_calls and content and <tools_were_requested>:
    parsed = extract_tool_calls(content)
    if parsed:
        tool_calls = parsed
        content = None            # mirror proxy _normalize
        finish_reason = "tool_calls"
```

**R3 — Streaming path** (only if G2 confirms text calls in stream): parse accumulated content in `_run_streaming_ollama` capture; emit structured call.

**R4 — Guards (mandatory):**
- **Only parse when `tool_calls` is empty AND tools were requested AND content present.** Never touch a response that already has structured calls (protects devstral native path + idempotency).
- Sits **behind `OPENAI_TOOLS_ENABLED`** kill-switch (`server.py:1827`) — if tools are globally off, no parsing.
- Parse failure = **fail-open to original behavior** (return content as-is, log). Never crash the completion on a malformed parse.
- Do not alter vLLM/MLX paths (`client.py:1490-1497`) — they're already structured.

---

## 6. Tests required (part of "done", not follow-up)

Mirror `tests/unit/test_openai_tools_ollama.py` (pytest + pytest-asyncio; construct objects directly, monkeypatch backend, no live ollama). Required cases:
1. qwen2.5 JSON-in-content → injected structured `tool_calls`, `content=None`, `finish_reason="tool_calls"`.
2. qwen3-coder XML `<function=>` → same.
3. Multiple calls in one content → all extracted.
4. **No-op when `tool_calls` already structured** (devstral path) — response unchanged.
5. **No-op when no `tools` requested** — content untouched.
6. **No-op / fail-open on malformed content** — no crash, original returned.
7. Idempotency — running the parser twice yields the same result.
8. (If R3) streaming accumulated-content parse → structured delta.
9. Regression analog to `test_pending_task_serialization_flattens_tools_to_top_level` — confirm tools reach the parse layer.

---

## 7. Non-goals / explicitly out of scope

- **Do NOT ship the port-8765 standalone proxy** to prod. Port the rules only.
- **vLLM tool-call parsing is the operator's responsibility**, not nodemesh code. If qwen ever serves via vLLM, configure `--enable-auto-tool-choice --tool-call-parser hermes|qwen3_coder` on the vLLM server (tracked in nodemesh `.qcoda/discussions.md:574`, `CHANGELOG.md:63`; vLLM issue #29192 caveat — `[needs verify: empirical curl confirms structured tool_calls before relying on it]`). This spec covers the **ollama** gap only.

> **UPDATE (D102, 2026-06-17) — server-config route rejected on evidence; code fix landed.** `qwen2.5-coder-7b` IS vLLM-served in prod and leaks text-form calls. The #29192 caveat above resolved **against** the server-config route: vLLM **`hermes` does not parse the Qwen2.5-Coder variant** (it emits json/`<tools>`, not hermes `<tool_call>`), and the operator runs an **older vLLM on purpose** (old-hardware support), so latest parsers/docs may not apply. Live mesh probe 2026-06-17: 3/3 non-deterministic text-leak, all parsed by the existing D101 `extract_tool_calls`. Fix = wire the D101 guard+parser into the **vLLM/MLX non-stream branch** (`client.py` ~1538), mirroring the ollama branch and the already-normalizing vLLM *streaming* branch. Idempotent (no-ops if vLLM later parses natively). See `decisions.md::D102`.
- Non-qwen text-call formats (llama3_json, mistral, etc.) — out of scope unless G3 surfaces them on a served model. Note any in the decision entry as follow-up.

---

## 8. Ledger / process requirements (nodemesh conventions)

- **Decision entry FIRST**, then code, then set `COMMITTED` (nodemesh CONVENTIONS Ledger Law).
- **Decision number: next committed D-NNN** — recon found highest = **D100**; this is likely **D101**. **Verify** before writing: `grep -E '^## D[0-9]' .qcoda/decisions.md | head -1` (append-only, newest at top).
- Entry template: `## DNNN — <title>` / Status / Date / Builds on (D098) / Context / Decision (A/B) / Files touched / **Verified:** (cite the G1–G3 probe commands + raw qwen content) / Why this matters / Cross-references.
- **Attribution:** aschwabe@gmail.com (NOT "user"/"operator"; note nodemesh CONVENTIONS example text shows a different email — use aschwabe@gmail.com).
- **Epistemic discipline (inviolable):** every flag/version/path/symbol claim carries inline `Verified: <cmd or URL>`. No "probably/likely/should/by default". Per AGENTS.md line 25: verify ollama field names / SSE schema against official ollama docs — do not rely on memory.
- **Version bump = 3 artifacts** in the same commit: `pyproject.toml:7`, `/VERSION` (no `v` prefix), dashboard footer. Plus D100 tag discipline.
- **Pre-commit is BLOCKING:** `scripts/validate_decisions.py` (fails on OPEN decisions + doc coverage) + `gitleaks protect --staged`. Don't `--no-verify`.
- **Never push unless asked** (AGENTS.md line 21).

---

## 9. Definition of done

1. G1–G3 gates run + results quoted in the decision entry.
2. Parser module + agent-layer wiring (+ streaming path iff G2 requires).
3. All §6 tests green; full suite green.
4. A/B reproduction documented: a real client (pi or an MCP tool) against qwen2.5-coder via nodemesh — no-op before, executes tool call after. Quote it.
5. D-NNN decision entry COMMITTED; affected docs updated (`.qcoda/api.md` tool surface, CHANGELOG).
6. Version bumped (3 artifacts). Not pushed unless operator asks.

---

## 10. Why this is worth doing

The fix is the unlock for every structured-tool-call client on qwen models — pi, goose, opencode, and MCP clients (Claude Desktop, Chatwise, Goose-as-operator-interface). It's the serving-layer half of D-003 (remote MCP) and D-106 (dev-team operator interface): those need tool calls to work on local qwen models, and only a serving-layer fix reaches third-party tools that hit the endpoint directly. aider sidesteps this (prompted SEARCH/REPLACE, model-agnostic) — but everything else is hostage to tool-call format until this lands.
