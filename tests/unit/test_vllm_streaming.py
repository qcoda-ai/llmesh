"""Unit tests for vLLM real per-token streaming (D040).

Covers:
- _iter_sse_events parser (CF-1): event boundaries, multi-read reassembly,
  heartbeat skip, [DONE] literal not JSON-parsed.
- _run_streaming_vllm: happy path, finish_reason/usage/[DONE] state machine
  (CF-4), missing-usage estimation (HP-1), no-/complete invariant (CF-6),
  bearer auth (D014), num_ctx warning (D015).

Mock vLLM and hub via httpx.MockTransport routed on URL host.
"""

import asyncio
import json
import types
import unittest.mock as mock
from typing import Iterable

import httpx
import pytest

from lib.agent import client as agent_client
from lib.agent.client import _iter_sse_events, _run_streaming_vllm


# --- helpers -----------------------------------------------------------------

def _make_state(node_id="node-test", node_token="tok-test"):
    s = types.SimpleNamespace()
    s.node_id = node_id
    s.node_token = node_token
    return s


async def _bytes_async_iter(chunks: Iterable[bytes]):
    for c in chunks:
        yield c


def _sse(events: list[str], terminator: bool = True) -> bytes:
    """Render OpenAI-style SSE events as bytes. terminator adds [DONE]."""
    out = b""
    for e in events:
        out += f"data: {e}\n\n".encode()
    if terminator:
        out += b"data: [DONE]\n\n"
    return out


def _delta(content=None, role=None, finish_reason=None, usage=None):
    """Build an OpenAI streaming chunk JSON string."""
    payload = {"id": "x", "object": "chat.completion.chunk",
               "created": 0, "model": "m"}
    if usage is not None:
        payload["usage"] = usage
        payload["choices"] = []
    else:
        d = {}
        if role is not None:
            d["role"] = role
        if content is not None:
            d["content"] = content
        payload["choices"] = [
            {"index": 0, "delta": d, "finish_reason": finish_reason}
        ]
    return json.dumps(payload)


# --- 1. SSE parser: well-formed events --------------------------------------

@pytest.mark.asyncio
async def test_iter_sse_events_yields_payloads_for_each_event():
    raw = b"data: {\"a\": 1}\n\ndata: {\"b\": 2}\n\n"
    out = []
    async for evt in _iter_sse_events(_bytes_async_iter([raw])):
        out.append(evt)
    assert out == ['{"a": 1}', '{"b": 2}']


# --- 2. SSE parser: [DONE] terminator ---------------------------------------

@pytest.mark.asyncio
async def test_iter_sse_events_passes_done_through_unparsed():
    raw = b"data: {\"x\": 1}\n\ndata: [DONE]\n\n"
    out = []
    async for evt in _iter_sse_events(_bytes_async_iter([raw])):
        out.append(evt)
    assert out == ['{"x": 1}', "[DONE]"]


# --- 3. SSE parser: empty / no-data events skipped --------------------------

@pytest.mark.asyncio
async def test_iter_sse_events_skips_data_less_events():
    raw = b"event: ping\nid: 42\n\ndata: {\"y\": 9}\n\n"
    out = []
    async for evt in _iter_sse_events(_bytes_async_iter([raw])):
        out.append(evt)
    assert out == ['{"y": 9}']


# --- 4. SSE parser: heartbeat ':' comments skipped --------------------------

@pytest.mark.asyncio
async def test_iter_sse_events_skips_heartbeat_comments():
    raw = b": keepalive\n\ndata: {\"z\": 1}\n\n: another\n\n"
    out = []
    async for evt in _iter_sse_events(_bytes_async_iter([raw])):
        out.append(evt)
    assert out == ['{"z": 1}']


# --- 5. SSE parser: multi-read event reassembly (CF-1) ----------------------

@pytest.mark.asyncio
async def test_iter_sse_events_reassembles_event_split_across_reads():
    # Single event arriving in three pieces, terminator in last piece.
    parts = [b"data: {\"split", b"\": tr", b"ue}\n\n"]
    out = []
    async for evt in _iter_sse_events(_bytes_async_iter(parts)):
        out.append(evt)
    assert out == ['{"split": true}']


# --- 6. SSE parser: leading whitespace stripped from data: lines ------------

@pytest.mark.asyncio
async def test_iter_sse_events_strips_leading_whitespace_after_data_colon():
    raw = b"data:    {\"ws\": 1}\n\n"
    out = []
    async for evt in _iter_sse_events(_bytes_async_iter([raw])):
        out.append(evt)
    assert out == ['{"ws": 1}']


# --- vLLM stream integration: shared mock infrastructure --------------------

class _PostRecorder:
    """Tracks all hub-side POSTs (chunk, done, /complete attempts)."""
    def __init__(self):
        self.stream_posts: list[dict] = []
        self.complete_posts: list[dict] = []
        self.cancel_at_post: int | None = None  # 1-indexed; None = never

    def handler(self, vllm_sse_bytes: bytes):
        """Build a httpx.MockTransport handler closure."""
        async def _handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/v1/chat/completions" in url:
                # vLLM upstream — return SSE stream.
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    content=vllm_sse_bytes,
                )
            if "/stream" in url:
                self.stream_posts.append(json.loads(request.content))
                if self.cancel_at_post is not None and len(self.stream_posts) >= self.cancel_at_post:
                    return httpx.Response(410, json={"detail": "cancelled"})
                return httpx.Response(200, json={"status": "ok"})
            if "/complete" in url:
                self.complete_posts.append(json.loads(request.content))
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(404, text=f"unmocked: {url}")
        return _handler


def _vllm_handler_fail_open(status: int, body: bytes):
    """Stream open returns non-200 — agent should not aiter."""
    posts: dict = {"stream_posts": [], "complete_posts": []}
    async def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/v1/chat/completions" in url:
            return httpx.Response(status, content=body)
        if "/stream" in url:
            posts["stream_posts"].append(json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})
        if "/complete" in url:
            posts["complete_posts"].append(json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)
    return _handler, posts


@pytest.fixture
def patch_hosts(monkeypatch):
    monkeypatch.setattr(agent_client, "VLLM_HOST", "http://vllm.test:8001")
    monkeypatch.setattr(agent_client, "HUB_URL", "http://hub.test:8000")
    monkeypatch.setattr(agent_client, "VLLM_API_KEY", None)
    yield


# --- 7. happy path: deltas + usage + [DONE] ---------------------------------

@pytest.mark.asyncio
async def test_streaming_happy_path_with_usage(patch_hosts):
    sse = _sse([
        _delta(role="assistant"),
        _delta(content="Hello"),
        _delta(content=" world"),
        _delta(content="!"),
        _delta(finish_reason="stop"),
        _delta(usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}),
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_vllm(
            c, _make_state(),
            {"task_id": "t1", "model": "m1", "messages": [{"role": "user", "content": "hi"}]},
        )

    # All chunks delivered, done frame fires, usage on done.
    done_posts = [p for p in rec.stream_posts if p["done"]]
    assert len(done_posts) == 1
    assert done_posts[0]["prompt_tokens"] == 7
    assert done_posts[0]["completion_tokens"] == 3
    # Concatenated tokens across all chunk POSTs (batched or per-token) equal "Hello world!"
    chunks = "".join(p["chunk"] for p in rec.stream_posts)
    assert chunks == "Hello world!"


# --- 8. HP-1: missing usage → estimated tokens_c ----------------------------

@pytest.mark.asyncio
async def test_streaming_missing_usage_estimates_tokens_c(patch_hosts):
    sse = _sse([
        _delta(role="assistant"),
        _delta(content="A"),
        _delta(content="B"),
        _delta(content="C"),
        _delta(finish_reason="stop"),
        # No usage chunk (older vLLM behavior).
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_vllm(
            c, _make_state(),
            {"task_id": "t2", "model": "m1", "messages": []},
        )

    done = [p for p in rec.stream_posts if p["done"]][0]
    # Estimated from delta count == 3.
    assert done["completion_tokens"] == 3
    assert done["prompt_tokens"] == 0


# --- 9. CF-4 state machine: finish_reason → usage → [DONE] ------------------

@pytest.mark.asyncio
async def test_streaming_state_machine_captures_usage_after_finish_reason(patch_hosts):
    """vLLM emits finish_reason FIRST, then usage, then [DONE]. Done frame
    must hold until [DONE] so usage is captured."""
    sse = _sse([
        _delta(content="x"),
        _delta(content="y"),
        _delta(finish_reason="stop"),                                  # finish
        _delta(usage={"prompt_tokens": 99, "completion_tokens": 2}),   # usage AFTER finish
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_vllm(
            c, _make_state(),
            {"task_id": "t3", "model": "m1", "messages": []},
        )
    done = [p for p in rec.stream_posts if p["done"]][0]
    assert done["prompt_tokens"] == 99  # captured AFTER finish_reason — would be 0 if state-machine wrong
    assert done["completion_tokens"] == 2


# --- 10. empty delta.content skipped ----------------------------------------

@pytest.mark.asyncio
async def test_streaming_skips_empty_delta_content(patch_hosts):
    sse = _sse([
        _delta(content="ok"),
        _delta(content=""),       # empty — should be skipped
        _delta(content=None),     # absent — should be skipped
        _delta(finish_reason="stop"),
        _delta(usage={"prompt_tokens": 1, "completion_tokens": 1}),
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_vllm(
            c, _make_state(),
            {"task_id": "t4", "model": "m1", "messages": []},
        )
    chunks = "".join(p["chunk"] for p in rec.stream_posts)
    assert chunks == "ok"


# --- 11. role-only first chunk skipped --------------------------------------

@pytest.mark.asyncio
async def test_streaming_skips_role_only_chunk(patch_hosts):
    sse = _sse([
        _delta(role="assistant"),     # no content — should not POST
        _delta(content="hi"),
        _delta(finish_reason="stop"),
        _delta(usage={"prompt_tokens": 1, "completion_tokens": 1}),
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_vllm(
            c, _make_state(),
            {"task_id": "t5", "model": "m1", "messages": []},
        )
    chunks = "".join(p["chunk"] for p in rec.stream_posts)
    assert chunks == "hi"


# --- 12. HTTP non-200 on stream open → error chunk + done -------------------

@pytest.mark.asyncio
async def test_streaming_non_200_open_emits_error_and_done(patch_hosts):
    handler, posts = _vllm_handler_fail_open(503, b"vLLM not ready")
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_vllm(
            c, _make_state(),
            {"task_id": "t6", "model": "m1", "messages": []},
        )
    # Two POSTs: error chunk then done sentinel.
    assert len(posts["stream_posts"]) == 2
    assert "vLLM stream open failed: 503" in posts["stream_posts"][0]["chunk"]
    assert posts["stream_posts"][1]["done"] is True
    # CF-6 invariant
    assert len(posts["complete_posts"]) == 0


# --- 13. mid-stream error JSON frame → graceful fail ------------------------

@pytest.mark.asyncio
async def test_streaming_mid_stream_error_frame_graceful_fail(patch_hosts):
    sse = (
        f"data: {_delta(content='partial')}\n\n".encode()
        + b'data: {"error": {"message": "oom", "type": "internal"}}\n\n'
        + b"data: [DONE]\n\n"
    )
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_vllm(
            c, _make_state(),
            {"task_id": "t7", "model": "m1", "messages": []},
        )
    # Final POST(s) include error text and a done sentinel.
    chunks_text = "".join(p["chunk"] for p in rec.stream_posts)
    assert "oom" in chunks_text
    assert any(p["done"] for p in rec.stream_posts)
    # CF-6 invariant
    assert len(rec.complete_posts) == 0


# --- 14. CF-6: NO /complete called from streaming path ----------------------

@pytest.mark.asyncio
async def test_streaming_never_calls_complete(patch_hosts):
    sse = _sse([
        _delta(content="ok"),
        _delta(finish_reason="stop"),
        _delta(usage={"prompt_tokens": 1, "completion_tokens": 1}),
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_vllm(
            c, _make_state(),
            {"task_id": "t8", "model": "m1", "messages": []},
        )
    assert len(rec.complete_posts) == 0
    assert any(p["done"] for p in rec.stream_posts)


# --- 15. D014: bearer auth header present on stream POST --------------------

@pytest.mark.asyncio
async def test_streaming_attaches_vllm_bearer_when_set(monkeypatch):
    monkeypatch.setattr(agent_client, "VLLM_HOST", "http://vllm.test:8001")
    monkeypatch.setattr(agent_client, "HUB_URL", "http://hub.test:8000")
    monkeypatch.setattr(agent_client, "VLLM_API_KEY", "sk-secret")

    captured: list[str] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/v1/chat/completions" in url:
            captured.append(request.headers.get("authorization", ""))
            sse = _sse([
                _delta(content="hi"),
                _delta(finish_reason="stop"),
                _delta(usage={"prompt_tokens": 1, "completion_tokens": 1}),
            ])
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=sse)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_vllm(
            c, _make_state(),
            {"task_id": "t9", "model": "m1", "messages": []},
        )
    assert captured == ["Bearer sk-secret"]


# --- 16. D015: num_ctx warning logged ---------------------------------------

@pytest.mark.asyncio
async def test_streaming_warns_when_num_ctx_set(patch_hosts, capsys):
    sse = _sse([
        _delta(content="ok"),
        _delta(finish_reason="stop"),
        _delta(usage={"prompt_tokens": 1, "completion_tokens": 1}),
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_vllm(
            c, _make_state(),
            {"task_id": "t10", "model": "m1", "messages": [], "num_ctx": 4096},
        )
    captured = capsys.readouterr().out
    assert "num_ctx=4096 ignored" in captured
    assert "D015" in captured


# --- 17. dispatcher: VLLM_STREAMING_ENABLED default (D044) ------------------

def test_vllm_streaming_enabled_by_default():
    """Per D044 — flag default ON after operator verification."""
    import os
    if "VLLM_STREAMING_ENABLED" in os.environ:
        # Env override set; check resolved correctly.
        assert agent_client.VLLM_STREAMING_ENABLED == (
            os.environ["VLLM_STREAMING_ENABLED"].lower() in ("true", "1", "yes")
        )
    else:
        # No env override; default must be ON.
        assert agent_client.VLLM_STREAMING_ENABLED is True


# --- 18. D044: max_tokens forwarded from task to vLLM payload ---------------

@pytest.mark.asyncio
async def test_streaming_forwards_max_tokens_when_set(patch_hosts):
    """Pass: when task includes max_tokens, agent forwards it to vLLM payload."""
    captured_payload: dict = {}

    async def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/v1/chat/completions" in url:
            captured_payload.update(json.loads(request.content))
            sse = _sse([
                _delta(content="ok"),
                _delta(finish_reason="stop"),
                _delta(usage={"prompt_tokens": 1, "completion_tokens": 1}),
            ])
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=sse)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_vllm(
            c, _make_state(),
            {"task_id": "tmt1", "model": "m1", "messages": [], "max_tokens": 2000},
        )
    assert captured_payload.get("max_tokens") == 2000


@pytest.mark.asyncio
async def test_streaming_omits_max_tokens_when_unset(patch_hosts):
    """Pass: when task does not specify max_tokens, payload does not include it."""
    captured_payload: dict = {}

    async def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/v1/chat/completions" in url:
            captured_payload.update(json.loads(request.content))
            sse = _sse([
                _delta(content="ok"),
                _delta(finish_reason="stop"),
                _delta(usage={"prompt_tokens": 1, "completion_tokens": 1}),
            ])
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=sse)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_vllm(
            c, _make_state(),
            {"task_id": "tmt2", "model": "m1", "messages": []},
        )
    assert "max_tokens" not in captured_payload
