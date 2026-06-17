"""
D106 — hub-side tool-support guard.

A streaming chat request that carries `tools` but routes to a vLLM / MLX /
llama.cpp backend silently drops the tool schema (those streaming agent paths
build the upstream body without `tools`), so the model replies with plain
prose — the no-op D105 fixed on the request side. The guard rejects that
combination with a 400 instead of returning a useless completion. Non-streaming
forwards tools on every backend (D099/D102); Ollama forwards on both paths
(D094); so only `tools + streaming + non-Ollama backend` is rejected.
"""
import asyncio

import pytest
from fastapi import HTTPException

from lib.hub import server, tasks as tasks_mod
from lib.hub.models import Node, ResourceCaps


def _node(*, ollama=None, vllm=None, mlx=None, llamacpp=None):
    res = ResourceCaps(
        cpu_cores=4, ram_gb=64, os_name="linux",
        ollama_available=bool(ollama), ollama_models=ollama or [],
        vllm_available=bool(vllm), vllm_models=vllm or [],
        mlx_available=bool(mlx), mlx_models=mlx or [],
        llamacpp_available=bool(llamacpp), llamacpp_models=llamacpp or [],
        streaming_capable=True,
    )
    return Node(node_id="n1", owner_id="o", resources=res, last_seen=0.0)


# --- _resolve_backend precedence (must mirror the agent) ---

def test_resolve_backend_precedence_vllm_wins_over_ollama():
    n = _node(ollama=["m"], vllm=["m"])
    assert server._resolve_backend(n, "m") == "vllm"


def test_resolve_backend_mlx_then_llamacpp_then_ollama():
    assert server._resolve_backend(_node(mlx=["m"]), "m") == "mlx"
    assert server._resolve_backend(_node(llamacpp=["m"]), "m") == "llamacpp"
    assert server._resolve_backend(_node(ollama=["m"]), "m") == "ollama"


def test_resolve_backend_unknown_model_defaults_ollama():
    assert server._resolve_backend(_node(vllm=["other"]), "m") == "ollama"


# --- the guard, exercised through _route_inference ---

def _req(model, *, tools=None):
    return server.InferenceRequest(owner_id="o", model=model, messages=[{"role": "user", "content": "hi"}], tools=tools)


def _route(req, node, stream, monkeypatch):
    monkeypatch.setattr(server, "_select_node", lambda owner, model, pred: node)

    async def _noop_queue(node_id, task):
        return None
    monkeypatch.setattr(tasks_mod, "queue_task_for_node", _noop_queue)
    return asyncio.run(server._route_inference(req, stream=stream))


_TOOLS = [{"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {}}}}]


@pytest.mark.parametrize("backend_kw", ["vllm", "mlx", "llamacpp"])
def test_guard_rejects_streaming_tools_on_non_ollama(monkeypatch, backend_kw):
    node = _node(**{backend_kw: ["coder"]})
    with pytest.raises(HTTPException) as ei:
        _route(_req("coder", tools=_TOOLS), node, stream=True, monkeypatch=monkeypatch)
    assert ei.value.status_code == 400
    assert ei.value.detail["error"]["type"] == "backend_does_not_support_tools"
    assert ei.value.detail["error"]["param"] == "tools"


def test_guard_allows_streaming_tools_on_ollama(monkeypatch):
    node = _node(ollama=["coder"])
    resp = _route(_req("coder", tools=_TOOLS), node, stream=True, monkeypatch=monkeypatch)
    assert resp.node_assigned == "n1"  # past the guard, task queued


def test_guard_allows_nonstreaming_tools_on_vllm(monkeypatch):
    """Non-streaming forwards tools on every backend (D099/D102) — no reject."""
    node = _node(vllm=["coder"])
    resp = _route(_req("coder", tools=_TOOLS), node, stream=False, monkeypatch=monkeypatch)
    assert resp.node_assigned == "n1"


def test_guard_ignores_requests_without_tools(monkeypatch):
    node = _node(vllm=["coder"])
    resp = _route(_req("coder", tools=None), node, stream=True, monkeypatch=monkeypatch)
    assert resp.node_assigned == "n1"  # no tools → guard inert even on vLLM streaming


def test_guard_400_surfaces_through_process_chat_completion(monkeypatch):
    """The 400 from _route_inference must reach the client as 400 (structured),
    not be masked as 503 by the generic HTTPException handler."""
    node = _node(vllm=["coder"])
    monkeypatch.setattr(server.storage, "authenticate_owner", lambda k: "o")
    monkeypatch.setattr(server, "_select_node", lambda owner, model, pred: node)

    async def _noop_queue(node_id, task):
        return None
    monkeypatch.setattr(tasks_mod, "queue_task_for_node", _noop_queue)

    async def go():
        return await server._process_chat_completion(
            "coder", [{"role": "user", "content": "hi"}], "k",
            want_stream=True, tools=_TOOLS,
        )

    task, err_resp, *_ = asyncio.run(go())
    assert task is None and err_resp is not None
    assert err_resp.status_code == 400
