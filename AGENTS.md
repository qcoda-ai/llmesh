# AGENTS.md — Universal Preamble for AI Coding Tools

> Single entry point every AI coding tool reads first when working in this repo.
> All deep governance lives in **`.qcoda/CONVENTIONS.md`** — read it in full before any work.
> Use a brutal re-evaluation approach for recommendations and analysis of user requests.
> Be direct and concise; avoid overly positive wording.

---

## Mandatory Reading (every session, every tool)

1. **`.qcoda/CONVENTIONS.md`** — Ledger Law, Documentation Gate, Workflow Orchestration, Task Management, Core Principles. Inviolable.
2. **`.qcoda/PROJECT.md`** — current project state and scope.
3. **`.qcoda/decisions.md`** — review for any `OPEN` entries; resolve before new work.
4. **`.qcoda/discussions.md`** — pre-decision strategy items; promote to `decisions.md` when converged.
5. **`.qcoda/lessons.md`** — anti-patterns and corrections from prior sessions; avoid repeating.

A decision that isn't logged in `.qcoda/decisions.md` did not happen. No code change is complete until `.qcoda/` reflects it.

---

## Universal Rules (apply to every tool)

- **Never push to remote unless the user explicitly asks.** No exceptions.
- **Tests are part of done**, not a follow-up.
- **Security-relevant changes are three artifacts in one turn**: code change + `COMMITTED` decision entry + at least one unit test exercising the failure path. No "circle back" to the ledger or tests.
- **Surface architectural alternatives** before silently taking the local-optimum fix when a domain has a known idiomatic answer (packaging, dependency management, auth, config, transport choice). Present 2-3 options ordered by scope.
- **Cross-reference propagation**: after writing a decision entry, grep `.qcoda/` for keywords the decision touches and add one-line `see DXXX` references to every matching doc.
- **Verify integration code with official documentation.** LLMesh integrates Ollama, vLLM, MLX, OpenAI, Anthropic — moving APIs. Do not rely on memory for function signatures or field names.
- **Auto-fix CI without being told.** Red CI is a blocker, not a backlog item.
- **Ledger Law and Documentation Gate are inviolable.** See `.qcoda/CONVENTIONS.md` for the full rules and enforcement layers.

---

## Tool-Specific Notes

### Claude Code
- Use plan mode for any non-trivial task (3+ steps or architectural decisions).
- Use subagents liberally to keep the main context window clean. One task per subagent.
- See also `.claude/CLAUDE.md` (thin pointer to this file).

### Other tools
- Read this file first; load `.qcoda/CONVENTIONS.md` next.
- Tool-specific configs (e.g. `.aider.conf.yml` for aider, `AGENTS.md` for Antigravity) reference this file; do not duplicate rules in tool configs.

---

## Enforcement

Three layers ensure the ledger stays current:

1. **Pre-commit hook** (`scripts/git_hooks/pre-commit`) — Ledger Law check + gitleaks secrets scan. Install via `bash scripts/install_git_hooks.sh`. Currently in **warn-only** mode for the Ledger Law check (first-week rollout per decisions.md D042); gitleaks block is always active.
2. **Post-commit hook** (`scripts/post_commit_hook.sh`) — prints a doc-update checklist when `lib/` or `alembic/` files are committed.
3. **Session obligation** — every AI tool session starts by reading `CONVENTIONS.md` and `PROJECT.md`; OPEN decisions are resolved before new work begins.
