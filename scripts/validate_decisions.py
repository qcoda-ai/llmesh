#!/usr/bin/env python3
"""
validate_decisions.py
---------------------
Enforces the Ledger Law at commit time. Two checks run in sequence:

  1. OPEN-decision check
     Flags commits if any decision in .qcoda/decisions.md still has
     status OPEN. All accepted decisions must be COMMITTED before
     a commit is allowed.

  2. Doc-coverage check
     If any hub/agent/template file is staged, at least one .qcoda/
     file must also be staged. Architecture changes without a matching
     doc update are a Ledger Law violation.

     Trigger patterns (staged file → required companion):
       lib/hub/<*>.py             →  .qcoda/ must have staged changes
       lib/agent/<*>.py           →  .qcoda/ must have staged changes
       lib/views/templates/<*>    →  .qcoda/ must have staged changes

Usage (standalone):
    python scripts/validate_decisions.py

Usage (as git pre-commit hook):
    Installed via: bash scripts/install_git_hooks.sh
    Source:        scripts/git_hooks/pre-commit

Exit codes:
    0 — all checks pass, safe to proceed
    1 — violation found
"""

import subprocess
import sys
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DECISIONS_FILE = REPO_ROOT / ".qcoda" / "decisions.md"

OPEN_PATTERN = re.compile(
    # Tolerates both `**Status**:` (qcoda style) and `**Status:**` (LLMesh style).
    r"^\s*[-*]?\s*\*\*Status[\*:]{2,4}\s*OPEN\s*$",
    re.IGNORECASE | re.MULTILINE,
)

DOC_TRIGGER_PREFIXES = (
    "lib/hub/",
    "lib/agent/",
    "lib/views/templates/",
)


def _staged_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, check=True,
            cwd=str(REPO_ROOT),
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _open_entries(content: str) -> list[str]:
    entries = re.split(r"^## ", content, flags=re.MULTILINE)
    return [
        entry.split("\n")[0].strip()
        for entry in entries
        if OPEN_PATTERN.search(entry)
    ]


def check_open_decisions() -> int:
    if not DECISIONS_FILE.exists():
        print("ERROR: .qcoda/decisions.md not found. Ledger Law requires this file.")
        return 1

    content = DECISIONS_FILE.read_text()
    open_titles = _open_entries(content)

    if not open_titles:
        print("✓ decisions.md — all decisions COMMITTED.")
        return 0

    print(f"\n✗ LEDGER LAW VIOLATION — {len(open_titles)} OPEN decision(s) in .qcoda/decisions.md:\n")
    for title in open_titles:
        print(f"  • {title}")
    print(
        "\nAll accepted decisions must be written into the relevant .qcoda/*.md files "
        "and their status updated to COMMITTED before this commit can proceed.\n"
        "See .qcoda/decisions.md and .qcoda/CONVENTIONS.md for the Ledger Law.\n"
    )
    return 1


def check_doc_coverage() -> int:
    staged = _staged_files()
    if not staged:
        return 0

    triggered_by = [f for f in staged if any(f.startswith(p) for p in DOC_TRIGGER_PREFIXES)]
    if not triggered_by:
        print("✓ doc-coverage  — no hub/agent/template changes staged.")
        return 0

    qcoda_staged = [f for f in staged if f.startswith(".qcoda/")]

    if qcoda_staged:
        print(
            f"✓ doc-coverage  — {len(triggered_by)} hub/agent/template file(s) staged "
            f"with {len(qcoda_staged)} .qcoda/ update(s)."
        )
        return 0

    print("\n✗ DOC COVERAGE VIOLATION — hub/agent/template changes staged without .qcoda/ updates:\n")
    for f in triggered_by:
        print(f"  • {f}")
    print(
        "\nArchitecture changes (hub, agent, templates) require at least one .qcoda/ "
        "document to be updated and staged in the same commit.\n"
        "Update the relevant .qcoda/*.md file(s) and add them with `git add`.\n"
        "See .qcoda/CONVENTIONS.md — Documentation Gate rule.\n"
    )
    return 1


def main() -> int:
    rc1 = check_open_decisions()
    rc2 = check_doc_coverage()
    return max(rc1, rc2)


if __name__ == "__main__":
    sys.exit(main())
