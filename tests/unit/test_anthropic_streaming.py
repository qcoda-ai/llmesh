"""Unit tests for Anthropic Messages SSE streaming (D061).

Covers:
  * _sse_event renders both `event:` name and `data:` JSON lines.
  * _estimate_input_tokens char/4 heuristic with various message shapes.
  * _real_sse_generator_anthropic emits the canonical 7-event sequence:
      message_start → content_block_start → content_block_delta×N →
      content_block_stop → message_delta → message_stop
  * Text deltas land in delta.text per spec.
  * message_delta carries cumulative output_tokens from task.completion_tokens.
  * Empty stream still emits content_block_start + content_block_stop.
  * Stream timeout emits `event: error` with `type:"api_error"` shape, no
    message_delta / message_stop after error.
  * Cancellation (asyncio.CancelledError) sets task.stream_cancelled and
    propagates.

The hub's task.stream_queue is the only upstream contract — same shape as
the OpenAI generator's input. No agent-side changes required.
"""

import asyncio
import json
import re
import types

import pytest

from lib.hub import server as hub_server
from lib.hub.server import (
    _real_sse_generator_anthropic,
    _sse_event,
    _estimate_input_tokens,
)


def _parse_sse(blob: str) -> list[tuple[str, dict]]:
    """Parse a raw SSE byte stream into list of (event_name, parsed_data)."""
    events: list[tuple[str, dict]] = []
    for chunk in blob.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        event_name = None
        data_lines = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if event_name is None or not data_lines:
            continue
        try:
            data = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        events.append((event_name, data))
    return events


def _make_task(task_id="t1", stream_queue=None, completion_tokens=0):
    t = types.SimpleNamespace()
    t.task_id = task_id
    t.stream_queue = stream_queue
    t.stream_cancelled = False
    t.completion_tokens = completion_tokens
    return t


async def _drain(gen) -> str:
    out = []
    async for ev in gen:
        out.append(ev)
    return "".join(out)


# --- _sse_event encoder ------------------------------------------------------

def test_sse_event_renders_both_lines():
    raw = _sse_event("message_start", {"type": "message_start", "x": 1})
    assert raw.startswith("event: message_start\n")
    assert "data: " in raw
    assert raw.endswith("\n\n")


def test_sse_event_data_is_valid_json():
    raw = _sse_event("ping", {"type": "ping"})
    data_line = [l for l in raw.split("\n") if l.startswith("data:")][0]
    assert json.loads(data_line[5:].strip()) == {"type": "ping"}


# --- _estimate_input_tokens --------------------------------------------------

def test_estimate_input_tokens_char_divide():
    msgs = [{"role": "user", "content": "Hello world!"}]  # 12 chars
    assert _estimate_input_tokens(msgs) == 3


def test_estimate_input_tokens_floor_is_1():
    assert _estimate_input_tokens([]) == 1
    assert _estimate_input_tokens([{"role": "user", "content": ""}]) == 1


def test_estimate_input_tokens_handles_list_content_blocks():
    msgs = [{
        "role": "user",
        "content": [{"type": "text", "text": "abcd"}, {"type": "text", "text": "efgh"}],
    }]
    assert _estimate_input_tokens(msgs) == 2  # 8 chars // 4


# --- generator: happy path ----------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_generator_emits_canonical_sequence():
    q: asyncio.Queue = asyncio.Queue()
    await q.put("Hello")
    await q.put(" ")
    await q.put("world")
    await q.put(None)  # done sentinel

    task = _make_task(task_id="abc", stream_queue=q, completion_tokens=3)
    blob = await _drain(_real_sse_generator_anthropic(
        task, request=None, message_id="msg_abc", model="claude-opus-4-7",
        input_tokens_estimate=12,
    ))
    events = _parse_sse(blob)
    names = [n for n, _ in events]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


@pytest.mark.asyncio
async def test_anthropic_generator_message_start_carries_input_tokens():
    q: asyncio.Queue = asyncio.Queue()
    await q.put(None)
    task = _make_task(stream_queue=q)
    blob = await _drain(_real_sse_generator_anthropic(
        task, request=None, message_id="msg_x", model="claude-opus-4-7",
        input_tokens_estimate=42,
    ))
    events = _parse_sse(blob)
    ms = [d for n, d in events if n == "message_start"][0]
    assert ms["message"]["id"] == "msg_x"
    assert ms["message"]["model"] == "claude-opus-4-7"
    assert ms["message"]["role"] == "assistant"
    assert ms["message"]["content"] == []
    assert ms["message"]["usage"]["input_tokens"] == 42
    assert ms["message"]["usage"]["output_tokens"] == 0
    assert ms["message"]["stop_reason"] is None


@pytest.mark.asyncio
async def test_anthropic_generator_content_block_start_is_text_type():
    q: asyncio.Queue = asyncio.Queue()
    await q.put(None)
    blob = await _drain(_real_sse_generator_anthropic(
        _make_task(stream_queue=q), request=None,
        message_id="m", model="m", input_tokens_estimate=1,
    ))
    cbs = [d for n, d in _parse_sse(blob) if n == "content_block_start"][0]
    assert cbs == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }


@pytest.mark.asyncio
async def test_anthropic_generator_deltas_use_text_delta_type():
    q: asyncio.Queue = asyncio.Queue()
    for c in ["Hi", "!"]:
        await q.put(c)
    await q.put(None)
    blob = await _drain(_real_sse_generator_anthropic(
        _make_task(stream_queue=q), request=None,
        message_id="m", model="m", input_tokens_estimate=1,
    ))
    deltas = [d for n, d in _parse_sse(blob) if n == "content_block_delta"]
    texts = [d["delta"]["text"] for d in deltas]
    assert texts == ["Hi", "!"]
    for d in deltas:
        assert d["type"] == "content_block_delta"
        assert d["index"] == 0
        assert d["delta"]["type"] == "text_delta"


@pytest.mark.asyncio
async def test_anthropic_generator_message_delta_has_cumulative_output_tokens():
    """Per docs Warning callout: message_delta.usage.output_tokens is CUMULATIVE."""
    q: asyncio.Queue = asyncio.Queue()
    for c in ["A", "B", "C"]:
        await q.put(c)
    await q.put(None)
    task = _make_task(stream_queue=q, completion_tokens=17)
    blob = await _drain(_real_sse_generator_anthropic(
        task, request=None, message_id="m", model="m", input_tokens_estimate=1,
    ))
    md = [d for n, d in _parse_sse(blob) if n == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "end_turn"
    assert md["delta"]["stop_sequence"] is None
    assert md["usage"]["output_tokens"] == 17


@pytest.mark.asyncio
async def test_anthropic_generator_message_stop_is_terminator():
    q: asyncio.Queue = asyncio.Queue()
    await q.put(None)
    blob = await _drain(_real_sse_generator_anthropic(
        _make_task(stream_queue=q), request=None,
        message_id="m", model="m", input_tokens_estimate=1,
    ))
    events = _parse_sse(blob)
    last_name, last_data = events[-1]
    assert last_name == "message_stop"
    assert last_data == {"type": "message_stop"}


# --- empty stream ------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_generator_empty_stream_still_well_formed():
    """Zero deltas — still emits start/stop block boundaries + message_stop."""
    q: asyncio.Queue = asyncio.Queue()
    await q.put(None)
    blob = await _drain(_real_sse_generator_anthropic(
        _make_task(stream_queue=q), request=None,
        message_id="m", model="m", input_tokens_estimate=1,
    ))
    names = [n for n, _ in _parse_sse(blob)]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


# --- timeout path -------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_generator_timeout_emits_error_event(monkeypatch):
    """STREAM_CHUNK_TIMEOUT exceeded → emit `event: error` then return.
    No message_delta / message_stop after error per spec."""
    monkeypatch.setattr(hub_server, "STREAM_CHUNK_TIMEOUT", 0.05)
    q: asyncio.Queue = asyncio.Queue()  # never gets a chunk
    task = _make_task(stream_queue=q)
    blob = await _drain(_real_sse_generator_anthropic(
        task, request=None, message_id="m", model="m", input_tokens_estimate=1,
    ))
    events = _parse_sse(blob)
    names = [n for n, _ in events]
    assert "error" in names
    assert "message_stop" not in names
    err = [d for n, d in events if n == "error"][0]
    assert err["type"] == "error"
    assert err["error"]["type"] == "api_error"
    assert "timeout" in err["error"]["message"].lower()
    assert task.stream_cancelled is True


# --- cancellation path -------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_generator_cancellation_sets_stream_cancelled():
    """Consumer disconnect → CancelledError → mark task and re-raise."""
    q: asyncio.Queue = asyncio.Queue()
    task = _make_task(stream_queue=q)

    async def _run():
        async for _ in _real_sse_generator_anthropic(
            task, request=None, message_id="m", model="m", input_tokens_estimate=1,
        ):
            pass

    t = asyncio.create_task(_run())
    await asyncio.sleep(0.02)  # let it pass message_start + content_block_start, then block on queue
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert task.stream_cancelled is True


# --- spec-compliance encoding sanity ------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_generator_raw_bytes_have_event_and_data_lines():
    """Both `event: <name>` AND `data: {json}` lines required per SSE spec.
    Anthropic SDK parsers reject events missing either line."""
    q: asyncio.Queue = asyncio.Queue()
    await q.put("hi")
    await q.put(None)
    blob = await _drain(_real_sse_generator_anthropic(
        _make_task(stream_queue=q), request=None,
        message_id="m", model="m", input_tokens_estimate=1,
    ))
    for ev in blob.split("\n\n"):
        ev = ev.strip()
        if not ev:
            continue
        assert re.search(r"^event: \w+$", ev, flags=re.MULTILINE), ev
        assert re.search(r"^data: \{.*\}$", ev, flags=re.MULTILINE), ev
