"""
Unit tests for D018: bridge blocking task completions into streaming consumer
queues.

When the dashboard task viewer or `/v1/chat/completions` with `stream: true`
creates a task with a `stream_queue`, the SSE consumer waits on
`stream_queue.get()` for chunks. Beta backends (vLLM, MLX) fall back to
blocking inference and submit results via `/complete` instead of `/stream`,
which sets `task.done_event` but never touches `stream_queue`. Without the
D018 bridge, the SSE consumer hangs on the queue until `STREAM_CHUNK_TIMEOUT`
and emits a misleading "node may be offline" error — even though the result
is sitting in `task.result` fully complete.

These tests pin the bridging behavior of `_bridge_blocking_completion_to_stream_consumer`
so future refactors cannot silently break the dashboard / streaming-API path
for beta backends.

Pure-function tests — no hub startup, no fixtures, no HTTP.
"""
import asyncio

import pytest

from lib.hub import tasks
from lib.hub.server import _bridge_blocking_completion_to_stream_consumer


def _make_task_with_stream_queue(result: str = "the answer") -> tasks.Task:
    t = tasks.Task(task_id="test-d018", model="test-model", owner_id="owner_test")
    t.stream_queue = asyncio.Queue()
    t.result = result
    return t


def _make_task_without_stream_queue(result: str = "the answer") -> tasks.Task:
    t = tasks.Task(task_id="test-d018", model="test-model", owner_id="owner_test")
    # stream_queue defaults to None
    t.result = result
    return t


# ── Success path ───────────────────────────────────────────────────────


def test_success_pushes_result_then_sentinel():
    """On a successful blocking completion with a stream_queue present, the
    bridge must push exactly two items: the full result followed by the close
    sentinel `None`. The SSE consumer accumulates the (single-element) list
    and emits the same done+result frame it would for a real streamed task."""
    task = _make_task_with_stream_queue("Hello! How can I assist you today?")

    _bridge_blocking_completion_to_stream_consumer(task, error=False)

    assert task.stream_queue.qsize() == 2
    assert task.stream_queue.get_nowait() == "Hello! How can I assist you today?"
    assert task.stream_queue.get_nowait() is None


def test_success_no_op_when_stream_queue_is_none():
    """The blocking-API path (non-streaming clients) creates tasks without a
    stream_queue. The bridge must be a no-op for those — calling `put_nowait`
    on `None` would crash."""
    task = _make_task_without_stream_queue("the answer")

    # Must not raise
    _bridge_blocking_completion_to_stream_consumer(task, error=False)

    assert task.stream_queue is None


# ── Failure path ───────────────────────────────────────────────────────


def test_failure_pushes_only_sentinel():
    """On a blocking failure with a stream_queue present, the bridge must
    push only the close sentinel — not the result. The SSE generator's error
    path reads `task.status` and `task.result` directly to construct the
    error frame, so pushing the result would double-deliver."""
    task = _make_task_with_stream_queue("error: model crashed")

    _bridge_blocking_completion_to_stream_consumer(task, error=True)

    assert task.stream_queue.qsize() == 1
    assert task.stream_queue.get_nowait() is None


def test_failure_no_op_when_stream_queue_is_none():
    """Failure path on a non-streaming task is also a no-op."""
    task = _make_task_without_stream_queue("error: model crashed")

    _bridge_blocking_completion_to_stream_consumer(task, error=True)

    assert task.stream_queue is None


# ── Regression: order matters ──────────────────────────────────────────


def test_success_order_is_result_then_sentinel_not_reversed():
    """The result MUST be pushed before the sentinel. If reversed, the SSE
    generator sees `None` first, breaks out of its drain loop, and emits a
    `done` frame with an empty `result` — silently dropping the model's
    output. This is a load-bearing ordering invariant."""
    task = _make_task_with_stream_queue("the load-bearing answer")

    _bridge_blocking_completion_to_stream_consumer(task, error=False)

    first = task.stream_queue.get_nowait()
    second = task.stream_queue.get_nowait()
    assert first == "the load-bearing answer", "result must be pushed before sentinel"
    assert second is None, "sentinel must come after result"


def test_success_handles_empty_result_string():
    """Edge case: if the model returned an empty string (zero output tokens),
    the bridge must still push it as a chunk. The accumulator emits the empty
    string and the consumer sees a `done` frame with `result: ''`, which is
    correct — empty output is a valid result, not a failure."""
    task = _make_task_with_stream_queue("")

    _bridge_blocking_completion_to_stream_consumer(task, error=False)

    assert task.stream_queue.qsize() == 2
    assert task.stream_queue.get_nowait() == ""
    assert task.stream_queue.get_nowait() is None
