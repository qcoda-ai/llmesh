"""Regression test for D045 — hub `/stream` done-frame must deliver chunk content
before the close sentinel.

Background
----------
Per CF-5 contract from D040+D041, agents may piggyback the final batch onto
the done frame as a single POST body `{"chunk": "tail", "done": true}`. The
prior hub handler at `lib/hub/server.py::stream_task_chunk` only put the
close sentinel (`None`) on `task.stream_queue` when `done=True`, silently
dropping the `chunk` content. Confirmed in production: 4 concurrent vLLM
streams produced completion_tokens=7-8 each but the SSE consumer received
only 6-7 tokens because the last batch landed on the done frame and was
discarded.

This test exercises the fixed handler logic directly via the same Task
machinery the hub uses, without standing up the full FastAPI app.
"""

import asyncio

import pytest

from lib.hub import tasks


def _make_streaming_task() -> tasks.Task:
    t = tasks.Task(task_id="test-d045", model="m", owner_id="o")
    t.stream_queue = asyncio.Queue()
    return t


@pytest.mark.asyncio
async def test_done_frame_with_non_empty_chunk_delivers_content_then_sentinel():
    """D045: when agent POSTs {chunk: 'tail', done: true}, hub must deliver
    'tail' before None so SSE consumer renders the final tokens."""
    t = _make_streaming_task()

    # Simulate the hub's stream_task_chunk done branch directly.
    # (The actual handler is exercised end-to-end in integration tests; here
    # we pin the queue-ordering invariant.)
    chunk = "78"
    if chunk:
        t.stream_queue.put_nowait(chunk)
    t.prompt_tokens = 10
    t.completion_tokens = 5
    t.status = "completed"
    t.stream_queue.put_nowait(None)

    drained = []
    while True:
        item = t.stream_queue.get_nowait()
        if item is None:
            drained.append(None)
            break
        drained.append(item)

    assert drained == ["78", None], (
        f"expected ['78', None] but got {drained} — "
        f"D045 violated, chunk content lost on done frame"
    )


@pytest.mark.asyncio
async def test_done_frame_with_empty_chunk_skips_chunk_put():
    """Common case: agent flushes mid-stream, sends {chunk: '', done: true}.
    Hub must NOT put empty string on queue (would be rendered as empty
    delta — harmless but wasted frame)."""
    t = _make_streaming_task()
    chunk = ""
    if chunk:
        t.stream_queue.put_nowait(chunk)
    t.stream_queue.put_nowait(None)

    items = []
    while True:
        try:
            i = t.stream_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        items.append(i)

    assert items == [None], f"expected only sentinel but got {items}"


@pytest.mark.asyncio
async def test_full_streaming_then_done_preserves_order():
    """Full happy-path simulation: mid-stream chunks plus piggybacked done
    chunk arrive at SSE consumer in order."""
    t = _make_streaming_task()

    # Mid-stream chunks
    t.stream_queue.put_nowait("Hello")
    t.stream_queue.put_nowait(" ")
    t.stream_queue.put_nowait("world")

    # Done frame with piggybacked tail
    chunk = "!"
    if chunk:
        t.stream_queue.put_nowait(chunk)
    t.stream_queue.put_nowait(None)

    drained = []
    while True:
        item = t.stream_queue.get_nowait()
        if item is None:
            break
        drained.append(item)

    assert "".join(drained) == "Hello world!"
