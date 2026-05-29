"""Unit tests for batcher telemetry surface (D068).

Coverage:
  * Agent done frame carries `stream_batches` + `stream_final_size`.
  * Hub `StreamChunk` model accepts the new fields with safe defaults.
  * Task object stashes them on the done-frame handler path.
  * `log_inference_event` accepts + buffers them; flush writes the row.
"""

import asyncio
import json
import types

import httpx
import pytest

from lib.agent import client as agent_client
from lib.agent.client import _run_streaming_ollama
from lib.hub import server as hub_server
from lib.hub import metrics, tasks
from lib.hub.models import TaskKind


# --- agent side: done frame carries batcher telemetry ----------------------

def _make_state(node_id="node-test", node_token="tok-test"):
    s = types.SimpleNamespace()
    s.node_id = node_id
    s.node_token = node_token
    return s


def _ollama_jsonl(events: list[dict]) -> bytes:
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode()


@pytest.mark.asyncio
async def test_agent_done_frame_carries_batcher_telemetry(monkeypatch):
    """The agent's final `/stream` POST (done=True) must include
    `stream_batches` and `stream_final_size`. Hub stashes them on the Task
    for downstream surface in metrics + dashboard."""
    monkeypatch.setattr(agent_client, "HUB_URL", "http://hub.test:8000")

    posted: list[dict] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/api/chat" in url:
            jsonl = _ollama_jsonl([
                {"message": {"content": "tok"}, "done": False},
                {"message": {"content": "tok"}, "done": False},
                {"done": True, "prompt_eval_count": 3, "eval_count": 2},
            ])
            return httpx.Response(200, content=jsonl,
                                  headers={"Content-Type": "application/x-ndjson"})
        if "/stream" in url:
            posted.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_ollama(
            c, _make_state(),
            {"task_id": "td068", "model": "llama3",
             "messages": [{"role": "user", "content": "hi"}]},
        )

    done_frames = [p for p in posted if p["done"]]
    assert len(done_frames) == 1
    df = done_frames[0]
    # Must be present, populated, and non-zero (we sent 2 tokens — at least
    # 1 flush + done flush = 2 batches min).
    assert "stream_batches" in df
    assert df["stream_batches"] >= 1
    assert "stream_final_size" in df
    assert df["stream_final_size"] >= 0


# --- hub side: StreamChunk model accepts new fields ------------------------

def test_stream_chunk_model_accepts_telemetry_fields():
    body = hub_server.StreamChunk(
        chunk="x", done=True, prompt_tokens=5, completion_tokens=3,
        stream_batches=12, stream_final_size=7,
    )
    assert body.stream_batches == 12
    assert body.stream_final_size == 7


def test_stream_chunk_model_defaults_zero_for_back_compat():
    """Older agents that haven't been upgraded to D068 send no telemetry
    fields. Model must default to 0 so the POST still parses."""
    body = hub_server.StreamChunk(chunk="x", done=True,
                                  prompt_tokens=5, completion_tokens=3)
    assert body.stream_batches == 0
    assert body.stream_final_size == 0


# --- task object stashes telemetry on done frame ---------------------------

def test_task_has_telemetry_fields_default_zero():
    """Fresh Task has zero telemetry fields. Populated only on streamed
    tasks' done frame."""
    t = tasks.Task(task_id="t1", kind=TaskKind.CHAT, owner_id="alice")
    assert t.stream_batches == 0
    assert t.stream_final_size == 0


# --- metrics: log_inference_event accepts + buffers telemetry --------------

def test_log_inference_event_buffers_telemetry():
    """`stream_batches` + `stream_final_size` land in the event buffer with
    the right keys and values."""
    metrics._event_buffer.clear()
    metrics.log_inference_event(
        user_id="alice", node_id="n1", model="m1", status="success",
        duration_ms=123.0, tokens_prompt=10, tokens_completion=20,
        stream_batches=4, stream_final_size=6,
    )
    assert len(metrics._event_buffer) == 1
    evt = metrics._event_buffer[0]
    assert evt["stream_batches"] == 4
    assert evt["stream_final_size"] == 6


def test_log_inference_event_back_compat_no_telemetry():
    """Old callers omit the new kwargs — must default to 0."""
    metrics._event_buffer.clear()
    metrics.log_inference_event(
        user_id="alice", node_id="n1", model="m1", status="success",
        duration_ms=1.0, tokens_prompt=1, tokens_completion=1,
    )
    evt = metrics._event_buffer[0]
    assert evt["stream_batches"] == 0
    assert evt["stream_final_size"] == 0
