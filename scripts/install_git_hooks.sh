#!/usr/bin/env bash
# Installs LLMesh git hooks into .git/hooks/.
# Idempotent: re-running re-installs the hooks from versioned source.
#
# Usage:
#   bash scripts/install_git_hooks.sh
#
# Hooks installed:
#   pre-commit  — Ledger Law check (warn-only first-week per D042) + gitleaks secrets scan
#   post-commit — Documentation gate reminder

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="$REPO_ROOT/.git/hooks"

install_hook() {
    local name="$1"
    local source="$2"
    local target="$HOOK_DIR/$name"
    if [ ! -f "$source" ]; then
        echo "  ✗ $name — source missing: $source"
        return 1
    fi
    install -m 0755 "$source" "$target"
    echo "  ✓ $name → $source"
}

mkdir -p "$HOOK_DIR"
install_hook pre-commit  "$REPO_ROOT/scripts/git_hooks/pre-commit"
install_hook post-commit "$REPO_ROOT/scripts/post_commit_hook.sh"

if command -v gitleaks >/dev/null 2>&1; then
    echo "  ✓ gitleaks $(gitleaks version 2>&1 | head -1) detected"
else
    echo ""
    echo "  ⚠  gitleaks NOT installed — pre-commit will block until you run:"
    echo "       brew install gitleaks      (macOS)"
    echo "       https://github.com/gitleaks/gitleaks#installing  (other)"
fi

echo ""
echo "Hooks installed. Source-of-truth lives in scripts/git_hooks/."
