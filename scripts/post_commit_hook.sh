#!/usr/bin/env bash
# post-commit hook — Documentation checklist reminder.
# Installed via: bash scripts/install_git_hooks.sh
# Source: scripts/post_commit_hook.sh
#
# Prints a doc-update checklist whenever code files are committed.
# Post-commit hooks must never block — always exits 0.

CHANGED=$(git diff --name-only HEAD~1 HEAD 2>/dev/null \
          || git show --name-only --format="" HEAD 2>/dev/null)

CODE_CHANGED=$(echo "$CHANGED" | grep -E "^lib/|^main\.py|^manage\.py" | head -1)
if [ -z "$CODE_CHANGED" ]; then
    exit 0
fi

echo ""
echo "─────────────────────────────────────────────────────"
echo "  LLMesh Documentation Gate — post-commit checklist"
echo "─────────────────────────────────────────────────────"
echo ""
echo "  Code was committed. Verify before ending the session:"
echo ""
echo "  [ ] .qcoda/ docs updated for any new features or changes"
echo "  [ ] .qcoda/decisions.md has a COMMITTED entry for this work"
echo "  [ ] README.md updated if env vars, routes, or commands changed"
echo "  [ ] .qcoda/todo.md reflects current state"
echo ""
echo "  Ledger Law: a decision that isn't in .qcoda/ did not happen."
echo "─────────────────────────────────────────────────────"
echo ""

exit 0
