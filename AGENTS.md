# AGENTS.md — Universal Preamble for AI Coding Tools

> Single entry point every AI coding tool reads first when working in this repo.
> Use a brutal re-evaluation approach for recommendations and analysis of user requests.
> Be direct and concise; avoid overly positive wording.

---

## Mandatory Reading (every session, every tool)

1. **`README.md`** — project overview, install, run, configuration.
2. **`docs/`** — feature-level design notes, where present.
3. **Tool-specific config** (e.g. `.claude/CLAUDE.md` for Claude Code, `.aider.conf.yml` for aider) — extends these universal rules with tool-specific or workspace-specific guidance.

A code change is not complete until tests pass and any user-visible behavior is documented.

---

## Universal Rules (apply to every tool)

- **Never push to remote unless the user explicitly asks.** No exceptions.
- **Tests are part of done**, not a follow-up.
- **Security-relevant changes are three artifacts in one turn**: code change + clear commit message naming the threat and operator-facing impact + at least one unit test exercising the failure path. No "circle back" to the tests later.
- **Surface architectural alternatives** before silently taking the local-optimum fix when a domain has a known idiomatic answer (packaging, dependency management, auth, config, transport choice). Present 2-3 options ordered by scope.
- **Verify integration code with official documentation.** This project integrates Ollama, vLLM, MLX, OpenAI, and Anthropic — moving APIs. Do not rely on memory for function signatures, field names, or SSE schemas; check the upstream docs before writing the call.
- **Auto-fix CI without being told.** Red CI is a blocker, not a backlog item.

---

## Tool-Specific Notes

### Claude Code
- Use plan mode for any non-trivial task (3+ steps or architectural decisions).
- Use subagents liberally to keep the main context window clean. One task per subagent.
- See `.claude/CLAUDE.md` for additional configuration.

### aider, Antigravity, and other AI coding tools
- Read this file first; load tool-specific config next.
- Tool-specific configs reference this file and do not duplicate rules.

---

## Enforcement

Two layers:

1. **Pre-commit hook** (`scripts/git_hooks/pre-commit`, installed via `bash scripts/install_git_hooks.sh`) — runs `gitleaks protect --staged --config .gitleaks.toml` for secrets scanning. Always blocking.
2. **Post-commit hook** (`scripts/post_commit_hook.sh`) — prints a documentation checklist whenever `lib/`, `main.py`, or `manage.py` files were committed. Reminder only; never blocks.

**First-time setup for contributors**: `bash scripts/install_git_hooks.sh` after cloning. The script also checks for `gitleaks` and prints install hints if missing (`brew install gitleaks` on macOS). Without gitleaks installed locally, the pre-commit hook will block all commits with a clear install message.
