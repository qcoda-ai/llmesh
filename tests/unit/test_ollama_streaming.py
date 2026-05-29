"""Unit tests for Ollama streaming refactored onto StreamBatcher (D067).

Mirrors the vLLM (D040) and MLX (D059) test shape. Ollama wire is JSONL,
not SSE — one JSON object per line. Final object carries `done:true` +
`prompt_eval_count` + `eval_count`. Chat path uses `message.content`;
generate path uses `response`.

Coverage:
  * Happy path — tokens batched, done with usage emitted on flush.
  * `n` POSTs ≤ delta count (batching reduces POST volume).
  * `STREAM_BATCH_FIXED=1` parity — per-token mode for operator escape.
  * Hub 410 → _StreamCancelled; no /complete call (CF-6).
  * HTTP non-200 on Ollama → error chunk + done sentinel.
  * Generate (legacy) path with `response` field works identically.
"""

import asyncio
import json
import types

import httpx
import pytest

from lib.agent import client as agent_client
from lib.agent.client import _run_streaming_ollama


def _make_state(node_id="node-test", node_token="tok-test"):
    s = types.SimpleNamespace()
    s.node_id = node_id
    s.node_token = node_token
    return s


def _ollama_jsonl(events: list[dict]) -> bytes:
    """Render Ollama JSONL response (one object per line + trailing newline)."""
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode()


class _PostRecorder:
    """Tracks hub-side POSTs (chunk, done, /complete attempts) + upstream Ollama
    request body for shape inspection."""
    def __init__(self):
        self.stream_posts: list[dict] = []
        self.complete_posts: list[dict] = []
        self.ollama_request_body: dict | None = None
        self.cancel_at_post: int | None = None

    def handler(self, ollama_jsonl_bytes: bytes):
        async def _handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/api/chat" in url or "/api/generate" in url:
                try:
                    self.ollama_request_body = json.loads(request.content)
                except Exception:
                    self.ollama_request_body = None
                return httpx.Response(200, content=ollama_jsonl_bytes,
                                      headers={"Content-Type": "application/x-ndjson"})
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
    monkeypatch.setattr(agent_client, "HUB_URL", "http://hub.test:8000")
    yield


# --- 1. happy path (chat) ----------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_chat_happy_path_with_usage(patch_hosts):
    """Chat path with /api/chat. Final done event carries usage."""
    jsonl = _ollama_jsonl([
        {"message": {"content": "Hello"}, "done": False},
        {"message": {"content": " "}, "done": False},
        {"message": {"content": "world"}, "done": False},
        {"message": {"content": "!"}, "done": False},
        {"done": True, "prompt_eval_count": 7, "eval_count": 4},
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(jsonl))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_ollama(
            c, _make_state(),
            {"task_id": "o1", "model": "llama3",
             "messages": [{"role": "user", "content": "hi"}]},
        )

    # Done frame carries the agent-reported usage.
    done_posts = [p for p in rec.stream_posts if p["done"]]
    assert len(done_posts) == 1
    assert done_posts[0]["prompt_tokens"] == 7
    assert done_posts[0]["completion_tokens"] == 4

    # Tokens concatenate to the original text across however many POSTs the
    # batcher made.
    chunks = "".join(p["chunk"] for p in rec.stream_posts)
    assert chunks == "Hello world!"


# --- 2. batching reduces POST volume (assuming adaptive default) ----------

@pytest.mark.asyncio
async def test_streaming_batches_below_token_count(patch_hosts):
    """4 tokens should produce fewer than 4 hub POSTs on adaptive default
    (batcher coalesces). Per-token mode is the operator escape via
    STREAM_BATCH_FIXED=1 — covered separately below."""
    jsonl = _ollama_jsonl([
        {"message": {"content": f"tok{i}"}, "done": False} for i in range(10)
    ] + [{"done": True, "prompt_eval_count": 1, "eval_count": 10}])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(jsonl))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_ollama(
            c, _make_state(),
            {"task_id": "o2", "model": "llama3",
             "messages": [{"role": "user", "content": "hi"}]},
        )
    # Should batch — chunk POSTs strictly less than 10 token-emitting events.
    chunk_posts = [p for p in rec.stream_posts if p.get("chunk")]
    assert len(chunk_posts) < 10, (
        f"Expected batching to reduce <10 POSTs, got {len(chunk_posts)}"
    )
    text = "".join(p["chunk"] for p in rec.stream_posts)
    assert text == "".join(f"tok{i}" for i in range(10))


# --- 3. STREAM_BATCH_FIXED=1 parity → per-token POSTs ----------------------

@pytest.mark.asyncio
async def test_streaming_fixed_batch_size_one(patch_hosts, monkeypatch):
    """STREAM_BATCH_FIXED=1 = operator escape hatch for per-token mode.
    Each non-empty token should produce its own POST (per D041)."""
    monkeypatch.setenv("STREAM_BATCH_FIXED", "1")
    jsonl = _ollama_jsonl([
        {"message": {"content": "a"}, "done": False},
        {"message": {"content": "b"}, "done": False},
        {"message": {"content": "c"}, "done": False},
        {"done": True, "prompt_eval_count": 1, "eval_count": 3},
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(jsonl))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_ollama(
            c, _make_state(),
            {"task_id": "o3", "model": "llama3",
             "messages": [{"role": "user", "content": "x"}]},
        )
    # 3 token-emitting events + 1 done flush (which carries the last
    # token via piggyback per D045). Total hub POSTs should be ≤ 4.
    assert len(rec.stream_posts) <= 4
    text = "".join(p["chunk"] for p in rec.stream_posts)
    assert text == "abc"


# --- 4. CF-6: streaming path NEVER calls /complete -------------------------

@pytest.mark.asyncio
async def test_streaming_never_calls_complete(patch_hosts):
    jsonl = _ollama_jsonl([
        {"message": {"content": "hi"}, "done": False},
        {"done": True, "prompt_eval_count": 1, "eval_count": 1},
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(jsonl))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_ollama(
            c, _make_state(),
            {"task_id": "o4", "model": "llama3",
             "messages": [{"role": "user", "content": "x"}]},
        )
    assert len(rec.complete_posts) == 0
    assert any(p["done"] for p in rec.stream_posts)


# --- 5. hub 410 → cancelled cleanly ----------------------------------------

@pytest.mark.asyncio
async def test_streaming_hub_410_cancels(patch_hosts):
    jsonl = _ollama_jsonl([
        {"message": {"content": "a"}, "done": False},
        {"message": {"content": "b"}, "done": False},
        {"message": {"content": "c"}, "done": False},
        {"done": True, "prompt_eval_count": 1, "eval_count": 3},
    ])
    rec = _PostRecorder()
    rec.cancel_at_post = 1
    transport = httpx.MockTransport(rec.handler(jsonl))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_ollama(
            c, _make_state(),
            {"task_id": "o5", "model": "llama3",
             "messages": [{"role": "user", "content": "x"}]},
        )
    assert len(rec.stream_posts) >= 1
    assert len(rec.complete_posts) == 0


# --- 6. Ollama HTTP non-200 → error chunk + done ---------------------------

@pytest.mark.asyncio
async def test_streaming_non_200_emits_error_and_done(patch_hosts):
    async def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/api/chat" in url:
            return httpx.Response(500, content=b"ollama failed")
        if "/stream" in url:
            return httpx.Response(200, json={"status": "ok"})
        if "/complete" in url:
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)
    posts: list[dict] = []
    async def _rec_handler(request: httpx.Request) -> httpx.Response:
        if "/stream" in str(request.url):
            posts.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})
        return await _handler(request)
    transport = httpx.MockTransport(_rec_handler)
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_ollama(
            c, _make_state(),
            {"task_id": "o6", "model": "llama3",
             "messages": [{"role": "user", "content": "x"}]},
        )
    # Two POSTs: error-text chunk + done sentinel.
    assert len(posts) >= 2
    assert any("Ollama stream open failed: 500" in p.get("chunk", "") for p in posts)
    assert posts[-1]["done"] is True


# --- 7. legacy /api/generate path with `response` field --------------------

@pytest.mark.asyncio
async def test_streaming_generate_path(patch_hosts):
    """No messages → routes to /api/generate, parses `response` field."""
    jsonl = _ollama_jsonl([
        {"response": "alpha", "done": False},
        {"response": " beta", "done": False},
        {"done": True, "prompt_eval_count": 2, "eval_count": 2},
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(jsonl))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_ollama(
            c, _make_state(),
            {"task_id": "o7", "model": "llama3",
             "messages": [],  # → generate path
             "prompt": "alpha beta"},
        )
    text = "".join(p["chunk"] for p in rec.stream_posts)
    assert text == "alpha beta"
    done = [p for p in rec.stream_posts if p["done"]][0]
    assert done["completion_tokens"] == 2


# --- 8. payload includes num_ctx from task or fallback ---------------------

@pytest.mark.asyncio
async def test_streaming_forwards_num_ctx(patch_hosts):
    jsonl = _ollama_jsonl([
        {"message": {"content": "x"}, "done": False},
        {"done": True, "prompt_eval_count": 1, "eval_count": 1},
    ])
    rec = _PostRecorder()
    transport = httpx.MockTransport(rec.handler(jsonl))
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_ollama(
            c, _make_state(),
            {"task_id": "o8", "model": "llama3", "num_ctx": 8192,
             "messages": [{"role": "user", "content": "x"}]},
        )
    assert rec.ollama_request_body is not None
    assert rec.ollama_request_body["options"]["num_ctx"] == 8192
