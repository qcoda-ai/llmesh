"""
Regression guard: shipped runtime modules must not call `print()`.

All operator-facing output flows through the `logging` module so it can be
filtered by level, formatted by uvicorn/systemd/docker, and shipped to
log aggregators. Bare `print()` interleaves with structured log output and
draws negative attention from operators in managed environments.

See decisions.md D051.
"""
import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCAN_DIRS = ["lib/hub", "lib/agent"]


def _iter_py_files():
    for sub in SCAN_DIRS:
        for path in (REPO_ROOT / sub).rglob("*.py"):
            yield path


def _print_call_lines(tree: ast.AST) -> list[int]:
    hits: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("path", list(_iter_py_files()), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_print_calls(path: pathlib.Path) -> None:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    hits = _print_call_lines(tree)
    assert not hits, (
        f"{path.relative_to(REPO_ROOT)} contains bare print() at line(s) {hits}. "
        "Use the module logger instead (see decisions.md D051)."
    )
