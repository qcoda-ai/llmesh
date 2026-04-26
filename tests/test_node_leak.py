"""Unit tests for the _node_tasks leak fix (D035).

Before D035, prune_inactive_nodes deleted the node from storage._nodes but
left tasks._node_tasks[node_id] (an empty list) in place forever. Fix is
storage.prune_inactive_nodes returns the stale ids and the cleanup loop
calls tasks.drop_node_queue(...) AFTER recovery moves any pending/claimed
tasks off the dead node.
"""
import time

import pytest

from lib.hub import storage, tasks
from lib.hub.models import Node, ResourceCaps, TaskKind


@pytest.fixture(scope="module", autouse=True)
def hub_and_node():
    """Override the subprocess hub fixture from conftest.py."""
    yield


@pytest.fixture(autouse=True)
def reset_state():
    storage._nodes.clear()
    tasks._node_tasks.clear()
    tasks._task_index.clear()
    yield
    storage._nodes.clear()
    tasks._node_tasks.clear()
    tasks._task_index.clear()


def _register(node_id: str, last_seen: float):
    n = Node(
        node_id=node_id,
        owner_id="o1",
        resources=ResourceCaps(
            cpu_cores=4, ram_gb=16, os_name="Linux",
            ollama_available=True, ollama_models=["m1"],
        ),
        last_seen=last_seen,
        node_token="t",
    )
    storage.store_node(n)


def test_drop_node_queue_clears_residue():
    _register("dead", last_seen=time.time() - 1000)
    t = tasks.Task(task_id="x", kind=TaskKind.CHAT, model="m1", owner_id="o1")
    t.status = "completed"  # not eligible for recovery
    tasks.queue_task_for_node("dead", t)
    assert "dead" in tasks._node_tasks

    stale = storage.prune_inactive_nodes(max_age_sec=1)
    assert stale == ["dead"]
    # Caller must drop the queue after pruning; simulate cleanup_loop's path.
    for nid in stale:
        tasks.drop_node_queue(nid)
    assert "dead" not in tasks._node_tasks, "queue residue not cleaned"


def test_prune_returns_stale_does_not_drop_yet():
    """prune_inactive_nodes itself must not touch tasks._node_tasks — that is
    the caller's job AFTER recovery has had a chance to migrate tasks."""
    _register("dead", last_seen=time.time() - 1000)
    t = tasks.Task(task_id="x", kind=TaskKind.CHAT, model="m1", owner_id="o1")
    tasks.queue_task_for_node("dead", t)

    stale = storage.prune_inactive_nodes(max_age_sec=1)
    assert stale == ["dead"]
    # Queue still present — recovery in server.cleanup_loop reads from it.
    assert "dead" in tasks._node_tasks
    assert t in tasks._node_tasks["dead"]


def test_drop_node_queue_idempotent():
    tasks.drop_node_queue("nonexistent")  # must not raise
    tasks._node_tasks["x"] = []
    tasks.drop_node_queue("x")
    assert "x" not in tasks._node_tasks
