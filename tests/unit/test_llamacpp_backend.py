"""Unit tests for the llama.cpp `llama-server` backend (D104).

llama-server is wire-identical to MLX (OpenAI-compat `/v1/chat/completions`,
`/v1/models`), so streaming is exercised by the shared `_run_streaming_openai`
path that `test_mlx_streaming.py` already covers. These tests lock the
llamacpp-specific surface: optional bearer auth, the `/health` liveness probe,
model listing, and that the streaming wrapper targets `LLAMACPP_HOST` and
injects the auth header (the one behavioural difference from MLX).
"""

import json
import types

import httpx
import pytest

from lib.agent import client as agent_client
from lib.agent.client import (
    _llamacpp_headers,
    get_llamacpp_models,
    _run_streaming_llamacpp,
)


def _make_state(node_id="node-test", node_token="tok-test"):
    s = types.SimpleNamespace()
    s.node_id = node_id
    s.node_token = node_token
    return s


# --- auth header ------------------------------------------------------------

def test_headers_empty_when_no_key(monkeypatch):
    monkeypatch.setattr(agent_client, "LLAMACPP_API_KEY", "")
    assert _llamacpp_headers() == {}


def test_headers_bearer_when_key_set(monkeypatch):
    monkeypatch.setattr(agent_client, "LLAMACPP_API_KEY", "secret")
    assert _llamacpp_headers() == {"Authorization": "Bearer secret"}


# --- model listing ----------------------------------------------------------

def test_get_models_empty_when_host_unset(monkeypatch):
    monkeypatch.setattr(agent_client, "LLAMACPP_HOST", None)
    assert get_llamacpp_models() == []


def test_get_models_parses_v1_models(monkeypatch):
    monkeypatch.setattr(agent_client, "LLAMACPP_HOST", "http://llama.test:8080")
    monkeypatch.setattr(agent_client, "LLAMACPP_API_KEY", "")

    def _fake_get(url, timeout=None, headers=None):
        assert url == "http://llama.test:8080/v1/models"
        return httpx.Response(200, json={"data": [{"id": "qwen2.5-coder-7b"}]})

    monkeypatch.setattr(agent_client.httpx, "get", _fake_get)
    assert get_llamacpp_models() == ["qwen2.5-coder-7b"]


# --- streaming wrapper: targets LLAMACPP_HOST + injects auth -----------------

@pytest.mark.asyncio
async def test_streaming_targets_host_and_injects_auth(monkeypatch):
    monkeypatch.setattr(agent_client, "LLAMACPP_HOST", "http://llama.test:8080")
    monkeypatch.setattr(agent_client, "LLAMACPP_API_KEY", "secret")
    monkeypatch.setattr(agent_client, "HUB_URL", "http://hub.test:8000")

    seen = {"upstream_url": None, "upstream_auth": None, "stream_posts": []}

    def _sse() -> bytes:
        def _d(content=None, finish_reason=None):
            return json.dumps({
                "id": "x", "object": "chat.completion.chunk", "created": 0,
                "model": "m", "choices": [
                    {"index": 0, "delta": ({"content": content} if content else {}),
                     "finish_reason": finish_reason}
                ],
            })
        return (
            f"data: {_d(content='hi')}\n\n"
            f"data: {_d(finish_reason='stop')}\n\n"
            "data: [DONE]\n\n"
        ).encode()

    async def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/v1/chat/completions" in url:
            seen["upstream_url"] = url
            seen["upstream_auth"] = request.headers.get("authorization")
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, content=_sse(),
            )
        if "/stream" in url:
            seen["stream_posts"].append(json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, text=f"unmocked: {url}")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as c:
        await _run_streaming_llamacpp(
            c, _make_state(),
            {"task_id": "L1", "model": "m", "messages": [], "stream": True},
        )

    assert seen["upstream_url"] == "http://llama.test:8080/v1/chat/completions"
    assert seen["upstream_auth"] == "Bearer secret"
    # final done frame carries the streamed token
    assert any(p.get("done") for p in seen["stream_posts"])
    joined = "".join(p.get("chunk", "") for p in seen["stream_posts"])
    assert "hi" in joined
