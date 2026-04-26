"""Unit tests for /v1/models dedup + capabilities (D031).

Covers:
  - Same model id on two backends → single entry, max ctx wins
  - Per-model context map overrides node-level scalar (D030)
  - Embedding-only model surfaces with capabilities=["embed"]
  - Chat-only model surfaces with capabilities=["chat"]
  - Model present as both chat and embed surfaces with both capabilities
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lib.hub import server, storage


_TEST_KEY = "models-dedup-test-key"
_TEST_OWNER = "owner_dedup_test"


@pytest.fixture(scope="module", autouse=True)
def hub_and_node():
    """Override the subprocess hub fixture from conftest.py — these tests run
    the FastAPI app in-process via TestClient."""
    yield


def _make_node(
    *,
    ollama_models=(),
    embedding_models=(),
    vllm_models=(),
    mlx_models=(),
    context_size=8192,
    model_context=None,
    node_id="node-dedup-test",
):
    resources = SimpleNamespace(
        ollama_available=bool(ollama_models),
        ollama_models=list(ollama_models),
        embedding_models=list(embedding_models),
        vllm_available=bool(vllm_models),
        vllm_models=list(vllm_models),
        mlx_available=bool(mlx_models),
        mlx_models=list(mlx_models),
        context_size=context_size,
        model_context=model_context or {},
        ram_gb=16,
    )
    return SimpleNamespace(
        node_id=node_id,
        owner_id=_TEST_OWNER,
        resources=resources,
        last_seen=0,
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(storage, "authenticate_owner",
                        lambda key: _TEST_OWNER if key == _TEST_KEY else None)
    monkeypatch.setattr(server, "MODELS_CACHE_TTL", 0.0)  # bypass cache for these tests
    server._models_cache.clear()
    yield TestClient(server.app)
    server._models_cache.clear()


def _auth():
    return {"Authorization": f"Bearer {_TEST_KEY}"}


def test_same_model_two_backends_dedup_max_ctx(client, monkeypatch):
    monkeypatch.setattr(storage, "get_all_nodes", lambda: [
        _make_node(node_id="n1", ollama_models=["llama3"], context_size=4096),
        _make_node(node_id="n2", vllm_models=["llama3"], context_size=8192),
    ])
    r = client.get("/v1/models", headers=_auth())
    assert r.status_code == 200
    data = r.json()["data"]
    by_id = {m["id"]: m for m in data}
    assert list(by_id.keys()) == ["llama3"], f"expected single dedup entry, got {list(by_id)}"
    assert by_id["llama3"]["context_length"] == 8192


def test_per_model_context_overrides_scalar(client, monkeypatch):
    monkeypatch.setattr(storage, "get_all_nodes", lambda: [
        _make_node(
            node_id="n1",
            ollama_models=["llama3"],
            embedding_models=["nomic-embed-text"],
            context_size=8192,
            model_context={"nomic-embed-text": 2048, "llama3": 8192},
        ),
    ])
    r = client.get("/v1/models", headers=_auth())
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert by_id["nomic-embed-text"]["context_length"] == 2048
    assert by_id["llama3"]["context_length"] == 8192


def test_capabilities_chat_only(client, monkeypatch):
    monkeypatch.setattr(storage, "get_all_nodes", lambda: [
        _make_node(ollama_models=["llama3"]),
    ])
    r = client.get("/v1/models", headers=_auth())
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert by_id["llama3"]["capabilities"] == ["chat"]


def test_capabilities_embed_only(client, monkeypatch):
    monkeypatch.setattr(storage, "get_all_nodes", lambda: [
        _make_node(embedding_models=["nomic-embed-text"]),
    ])
    r = client.get("/v1/models", headers=_auth())
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert by_id["nomic-embed-text"]["capabilities"] == ["embed"]


def test_capabilities_both_when_registered_twice(client, monkeypatch):
    monkeypatch.setattr(storage, "get_all_nodes", lambda: [
        _make_node(node_id="n1", ollama_models=["multi-purpose"]),
        _make_node(node_id="n2", embedding_models=["multi-purpose"]),
    ])
    r = client.get("/v1/models", headers=_auth())
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert by_id["multi-purpose"]["capabilities"] == ["chat", "embed"]
