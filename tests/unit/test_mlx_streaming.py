"""Unit tests for MLX real per-token streaming (D059).

Mirrors test_vllm_streaming.py structure; primary divergences:
  * No bearer header — verifies `_run_streaming_mlx` does NOT attach
    Authorization on the upstream POST.
  * Osaurus (the primary target MLX backend) emits no `usage` chunk even with
    `stream_options.include_usage=true` (verified 2026-05-28). Tests force
    estimation-only path and lock the wire shape.
  * MLX_HOST is the configured upstream.
  * num_ctx warning text references the MLX-context-fixed-at-startup limit
    (parallels D015 for vLLM).
"""

import json
import types
from typing import Iterable

import httpx
import pytest

from lib.agent import client as agent_client
from lib.agent.client import _run_streaming_mlx


def _make_state(node_id="node-test", node_token="tok-test"):
    s = types.SimpleNamespace()
    s.node_id = node_id
    s.node_token = node_token
    return s


def _sse(events: list[str], terminator: bool = True) -> bytes:
    out = b""
    for e in events:
        out += f"data: {e}\n\n".encode()
    if terminator:
        out += b"data: [DONE]\n\n"
    return out


def _delta(content=None, role=None, finish_reason=None, usage=None):
    payload = {"id": "x", "object": "chat.completion.chunk",
               "created": 0, "model": "qwen3-8b-4bit"}
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


class _PostRecorder:
    """Tracks all hub-side POSTs + upstream MLX request inspection."""
    def __init__(self):
        self.stream_posts: list[dict] = []
        self.complete_posts: list[dict] = []
        self.mlx_request_headers: dict | None = None
        self.mlx_request_body: dict | None = None
        self.cancel_at_post: int | None = None

    def handler(self, mlx_sse_bytes: bytes):
        async def _handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/v1/chat/completions" in url:
                self.mlx_request_headers = dict(request.headers)
                try:
                    self.mlx_request_body = json.loads(request.content)
                except Exception:
                    self.mlx_request_body = None
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    content=mlx_sse_bytes,
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


@pytest.fixture
def patch_hosts(monkeypatch):
    monkeypatch.setattr(agent_client, "MLX_HOST", "http://mlx.test:1337")
    monkeypatch.setattr(agent_client, "HUB_URL", "http://hub.test:8000")
    monkeypatch.setattr(agent_client, "VLLM_API_KEY", "should-not-leak-into-mlx")
    yield


# --- 1. happy path (osaurus shape: no usage chunk, finish_reason → [DONE]) --

@pytest.mark.asyncio
async def test_streaming_happy_path_osaurus_shape(patch_hosts):
    """Replays exact wire shape captured from osaurus 2026-05-28: deltas,
    finish_reason stop, [DONE]. No usage chunk — agent must estimate."""
    sse = _sse([
        _delta(role="assistant", content=""),
        _delta(content="Hello"),
        _delta(content=" world"),
        _delta(content="!"),
        _delta(finish_reason="stop"),
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_mlx(
            c, _make_state(),
            {"task_id": "m1", "model": "qwen3-8b-4bit",
             "messages": [{"role": "user", "content": "hi"}]},
        )

    done_posts = [p for p in rec.stream_posts if p["done"]]
    assert len(done_posts) == 1
    # Estimation: 3 content deltas → tokens_c=3, tokens_p=0.
    assert done_posts[0]["completion_tokens"] == 3
    assert done_posts[0]["prompt_tokens"] == 0

    chunks = "".join(p["chunk"] for p in rec.stream_posts)
    assert chunks == "Hello world!"


# --- 2. no auth header attached to upstream MLX --------------------------------

@pytest.mark.asyncio
async def test_streaming_does_not_attach_vllm_bearer_to_mlx(patch_hosts):
    """Regression guard: VLLM_API_KEY must NOT leak into MLX requests.
    osaurus and local mlx-lm.server do not gate /v1/chat/completions."""
    sse = _sse([
        _delta(content="ok"),
        _delta(finish_reason="stop"),
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_mlx(
            c, _make_state(),
            {"task_id": "m2", "model": "qwen3-8b-4bit", "messages": []},
        )
    assert rec.mlx_request_headers is not None
    auth = rec.mlx_request_headers.get("authorization", "")
    assert auth == "" or "should-not-leak-into-mlx" not in auth


# --- 3. payload includes stream + stream_options + max_tokens ------------------

@pytest.mark.asyncio
async def test_streaming_payload_shape(patch_hosts):
    sse = _sse([_delta(content="x"), _delta(finish_reason="stop")])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_mlx(
            c, _make_state(),
            {"task_id": "m3", "model": "qwen3-8b-4bit", "messages": [],
             "max_tokens": 256},
        )
    assert rec.mlx_request_body is not None
    body = rec.mlx_request_body
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 256
    assert body["model"] == "qwen3-8b-4bit"


# --- 4. CF-4 state machine: usage present (mlx-lm.server forward-compat) ----

@pytest.mark.asyncio
async def test_streaming_uses_usage_chunk_when_present(patch_hosts):
    """If a future MLX backend (mlx-lm.server) emits usage, we use it instead
    of estimating. Forward-compat lock."""
    sse = _sse([
        _delta(content="x"),
        _delta(content="y"),
        _delta(finish_reason="stop"),
        _delta(usage={"prompt_tokens": 42, "completion_tokens": 9}),
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_mlx(
            c, _make_state(),
            {"task_id": "m4", "model": "m1", "messages": []},
        )
    done = [p for p in rec.stream_posts if p["done"]][0]
    assert done["prompt_tokens"] == 42
    assert done["completion_tokens"] == 9


# --- 5. CF-6: streaming path NEVER calls /complete --------------------------

@pytest.mark.asyncio
async def test_streaming_never_calls_complete(patch_hosts):
    sse = _sse([
        _delta(content="ok"),
        _delta(finish_reason="stop"),
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_mlx(
            c, _make_state(),
            {"task_id": "m5", "model": "m1", "messages": []},
        )
    assert len(rec.complete_posts) == 0
    assert any(p["done"] for p in rec.stream_posts)


# --- 6. mid-stream error frame → graceful fail ------------------------------

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
        await _run_streaming_mlx(
            c, _make_state(),
            {"task_id": "m6", "model": "m1", "messages": []},
        )
    chunks_text = "".join(p["chunk"] for p in rec.stream_posts)
    assert "oom" in chunks_text
    assert any(p["done"] for p in rec.stream_posts)
    assert len(rec.complete_posts) == 0


# --- 7. HTTP non-200 on stream open → error chunk + done --------------------

@pytest.mark.asyncio
async def test_streaming_non_200_open_emits_error_and_done(patch_hosts):
    posts: dict = {"stream_posts": [], "complete_posts": []}
    async def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/v1/chat/completions" in url:
            return httpx.Response(503, content=b"model not loaded")
        if "/stream" in url:
            posts["stream_posts"].append(json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})
        if "/complete" in url:
            posts["complete_posts"].append(json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_mlx(
            c, _make_state(),
            {"task_id": "m7", "model": "m1", "messages": []},
        )
    assert len(posts["stream_posts"]) == 2
    assert "MLX stream open failed: 503" in posts["stream_posts"][0]["chunk"]
    assert posts["stream_posts"][1]["done"] is True
    assert len(posts["complete_posts"]) == 0


# --- 8. hub 410 → stream cancelled, no further work --------------------------

@pytest.mark.asyncio
async def test_streaming_hub_410_cancels(patch_hosts):
    sse = _sse([
        _delta(content="a"),
        _delta(content="b"),
        _delta(content="c"),
        _delta(content="d"),
        _delta(finish_reason="stop"),
    ])
    rec = _PostRecorder()
    rec.cancel_at_post = 1
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_mlx(
            c, _make_state(),
            {"task_id": "m8", "model": "m1", "messages": []},
        )
    assert len(rec.stream_posts) >= 1
    assert len(rec.complete_posts) == 0


# --- 9. think-tag content passes through unchanged ---------------------------

@pytest.mark.asyncio
async def test_streaming_passes_through_think_tags(patch_hosts):
    """qwen3-thinking on osaurus emits <think>...</think> blocks as ordinary
    deltas. Agent must not strip or transform; downstream renders decide."""
    sse = _sse([
        _delta(content="<think>"),
        _delta(content="reason"),
        _delta(content="</think>"),
        _delta(content="answer"),
        _delta(finish_reason="stop"),
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_mlx(
            c, _make_state(),
            {"task_id": "m9", "model": "qwen3-8b-4bit", "messages": []},
        )
    text = "".join(p["chunk"] for p in rec.stream_posts)
    assert text == "<think>reason</think>answer"


# --- 10. num_ctx supplied → one-shot warning logged --------------------------

@pytest.mark.asyncio
async def test_streaming_warns_when_num_ctx_set(patch_hosts, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="llmesh.agent")
    sse = _sse([_delta(content="ok"), _delta(finish_reason="stop")])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(sse))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_mlx(
            c, _make_state(),
            {"task_id": "m10", "model": "m1", "messages": [], "num_ctx": 8192},
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any("num_ctx=8192 ignored on MLX" in m for m in messages), messages


# --- 11. MLX_STREAMING_ENABLED default ON per D060 ---------------------------

def test_mlx_streaming_enabled_by_default():
    """D059 shipped behind the flag default-OFF; D060 (2026-05-28) flipped the
    default ON after LAB-003 graduated (3 consecutive 6/6 automated passes +
    hub round-trip + STREAM_BATCH_FIXED=1 parity + 410 cancel all observed)."""
    import os
    if "MLX_STREAMING_ENABLED" in os.environ:
        assert agent_client.MLX_STREAMING_ENABLED == (
            os.environ["MLX_STREAMING_ENABLED"].lower() in ("true", "1", "yes")
        )
    else:
        assert agent_client.MLX_STREAMING_ENABLED is True
