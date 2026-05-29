"""
Unit tests for the weighted routing score in `lib/hub/server.py::_score_node`
(D054, closes D036 §1).
"""
import pytest

from lib.hub import server, tasks as tasks_mod, task_store as task_store_mod
from lib.hub.models import Node, ResourceCaps, TaskKind


def _node(node_id, ram_gb, cpu_load=0.0, owner="alice"):
    res = ResourceCaps(
        cpu_cores=4, ram_gb=ram_gb, os_name="linux",
        ollama_available=True, ollama_models=["llama3"],
    )
    n = Node(node_id=node_id, owner_id=owner, resources=res, last_seen=0.0)
    n.cpu_load = cpu_load
    return n


def _enqueue(node_id, n_pending=0, n_claimed=0, owner="alice"):
    """Push synthetic tasks into the in-memory dicts (no store writes —
    routing reads only from `_node_tasks`)."""
    for i in range(n_pending):
        t = tasks_mod.Task(
            task_id=f"{node_id}-p-{i}", kind=TaskKind.CHAT,
            owner_id=owner, messages=[{"role": "user", "content": "x"}],
        )
        t.status = "pending"
        tasks_mod._task_index[t.task_id] = t
        tasks_mod._node_tasks.setdefault(node_id, []).append(t)
    for i in range(n_claimed):
        t = tasks_mod.Task(
            task_id=f"{node_id}-c-{i}", kind=TaskKind.CHAT,
            owner_id=owner, messages=[{"role": "user", "content": "x"}],
        )
        t.status = "claimed"
        tasks_mod._task_index[t.task_id] = t
        tasks_mod._node_tasks.setdefault(node_id, []).append(t)


@pytest.fixture(autouse=True)
def isolate_tasks_state():
    """Swap the module-level task store for a noop and clear in-memory state
    so test ordering doesn't matter."""
    prev = task_store_mod._task_store
    task_store_mod.set_task_store(task_store_mod.MemoryTaskStore())
    tasks_mod._task_index.clear()
    tasks_mod._node_tasks.clear()
    yield
    task_store_mod.set_task_store(prev)
    tasks_mod._task_index.clear()
    tasks_mod._node_tasks.clear()


def test_equal_ram_idle_beats_busy():
    """Two equal-RAM nodes — the one with no queue wins over one with 2 queued."""
    busy = _node("busy", ram_gb=16.0)
    idle = _node("idle", ram_gb=16.0)
    _enqueue("busy", n_pending=2)
    assert server._score_node(idle) > server._score_node(busy)


def test_equal_ram_low_cpu_beats_high_cpu():
    """Two equal-RAM equal-queue nodes — lower CPU wins."""
    hot = _node("hot", ram_gb=16.0, cpu_load=80.0)
    cool = _node("cool", ram_gb=16.0, cpu_load=5.0)
    assert server._score_node(cool) > server._score_node(hot)


def test_higher_ram_wins_when_nothing_else_differs():
    """No queue, no CPU load — RAM still drives the choice."""
    big = _node("big", ram_gb=32.0)
    small = _node("small", ram_gb=8.0)
    assert server._score_node(big) > server._score_node(small)


def test_queue_penalty_overwhelms_ram_advantage():
    """16 GB + 2 queued ≈ 0 ; 8 GB + 0 queued = 8. Lower-RAM idle wins."""
    big_busy = _node("big_busy", ram_gb=16.0)
    small_idle = _node("small_idle", ram_gb=8.0)
    _enqueue("big_busy", n_pending=2)
    assert server._score_node(small_idle) > server._score_node(big_busy)


def test_claimed_tasks_also_count_as_load():
    """Claimed tasks are in-flight on the node — must penalise too."""
    busy_claimed = _node("busy_claimed", ram_gb=16.0)
    idle = _node("idle", ram_gb=16.0)
    _enqueue("busy_claimed", n_claimed=2)
    assert server._score_node(idle) > server._score_node(busy_claimed)


def test_terminal_tasks_do_not_count_as_load():
    """Completed and failed tasks linger in `_node_tasks` until pruned but
    should NOT degrade the node's routing score."""
    node = _node("n", ram_gb=16.0)
    # Push 3 terminal tasks
    for i in range(3):
        t = tasks_mod.Task(
            task_id=f"done-{i}", kind=TaskKind.CHAT,
            owner_id="alice", messages=[],
        )
        t.status = "completed" if i < 2 else "failed"
        tasks_mod._task_index[t.task_id] = t
        tasks_mod._node_tasks.setdefault("n", []).append(t)
    assert server._score_node(node) == 16.0


def test_env_var_disables_penalty(monkeypatch):
    """ROUTING_QUEUE_PENALTY=0 + ROUTING_CPU_PENALTY=0 reverts to pure-RAM."""
    monkeypatch.setattr(server, "ROUTING_QUEUE_PENALTY", 0.0)
    monkeypatch.setattr(server, "ROUTING_CPU_PENALTY", 0.0)
    big_busy = _node("big_busy", ram_gb=16.0, cpu_load=90.0)
    small_idle = _node("small_idle", ram_gb=8.0)
    _enqueue("big_busy", n_pending=5)
    assert server._score_node(big_busy) > server._score_node(small_idle)


def test_missing_cpu_load_defaults_to_zero():
    """Heartbeat may not have arrived yet; cpu_load absent should not crash."""
    n = _node("fresh", ram_gb=16.0)
    delattr(n, "cpu_load")  # simulate pre-heartbeat state
    assert server._score_node(n) == 16.0
