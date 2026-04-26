"""Unit tests for the per-owner TTL cache on GET /v1/models (D027).

The endpoint scans the node registry on every call. Under rapid back-to-back
client traffic (e.g. the qc_eval harness hitting the hub at burst rates), this
keeps /v1/models off the hot path so the event loop has more budget for
inference routing and SSE streaming.

These tests use FastAPI's in-process TestClient and monkeypatch
storage.authenticate_owner + storage.get_all_nodes to isolate cache behavior
from the wider hub.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lib.hub import server, storage


_TEST_KEY = "models-cache-test-key"
_TEST_OWNER = "owner_cache_test"


@pytest.fixture(scope="module", autouse=True)
def hub_and_node():
    """Override the subprocess hub fixture from conftest.py — these tests run
    the FastAPI app in-process via TestClient and do not need the full hub."""
    yield


def _make_node(owner_id: str, ollama_models=(), context_size=8192):
    resources = SimpleNamespace(
        ollama_available=True,
        ollama_models=list(ollama_models),
        vllm_available=False,
        vllm_models=[],
        mlx_available=False,
        mlx_models=[],
        context_size=context_size,
        ram_gb=16,
    )
    return SimpleNamespace(
        node_id="node-cache-test",
        owner_id=owner_id,
        resources=resources,
        last_seen=0,
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(storage, "authenticate_owner",
                        lambda key: _TEST_OWNER if key == _TEST_KEY else None)
    server._models_cache.clear()
    yield TestClient(server.app)
    server._models_cache.clear()


def _auth():
    return {"Authorization": f"Bearer {_TEST_KEY}"}


def test_second_call_within_ttl_hits_cache(client, monkeypatch):
    calls = {"n": 0}

    def _get_all_nodes():
        calls["n"] += 1
        return [_make_node(_TEST_OWNER, ollama_models=["llama3"])]

    monkeypatch.setattr(storage, "get_all_nodes", _get_all_nodes)
    monkeypatch.setattr(server, "MODELS_CACHE_TTL", 10.0)

    r1 = client.get("/v1/models", headers=_auth())
    r2 = client.get("/v1/models", headers=_auth())

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == r2.json()
    assert calls["n"] == 1, "second call within TTL must be served from cache"


def test_expired_entry_rebuilds(client, monkeypatch):
    calls = {"n": 0}

    def _get_all_nodes():
        calls["n"] += 1
        return [_make_node(_TEST_OWNER, ollama_models=[f"llama3-v{calls['n']}"])]

    monkeypatch.setattr(storage, "get_all_nodes", _get_all_nodes)
    monkeypatch.setattr(server, "MODELS_CACHE_TTL", 10.0)

    r1 = client.get("/v1/models", headers=_auth())
    assert r1.status_code == 200

    # Force the cached entry to be expired.
    server._models_cache[_TEST_OWNER] = (0.0, server._models_cache[_TEST_OWNER][1])

    r2 = client.get("/v1/models", headers=_auth())
    assert r2.status_code == 200
    assert calls["n"] == 2
    assert r2.json() != r1.json(), "expired entry must be rebuilt from the registry"


def test_ttl_zero_disables_cache(client, monkeypatch):
    calls = {"n": 0}

    def _get_all_nodes():
        calls["n"] += 1
        return [_make_node(_TEST_OWNER, ollama_models=["llama3"])]

    monkeypatch.setattr(storage, "get_all_nodes", _get_all_nodes)
    monkeypatch.setattr(server, "MODELS_CACHE_TTL", 0.0)

    client.get("/v1/models", headers=_auth())
    client.get("/v1/models", headers=_auth())
    client.get("/v1/models", headers=_auth())

    assert calls["n"] == 3, "MODELS_CACHE_TTL=0 must bypass cache on every call"
    assert server._models_cache == {}, "disabled cache must not populate"


def test_cache_scoped_per_owner(client, monkeypatch):
    other_key = "second-owner-key"
    other_owner = "owner_beta"

    def _auth_two(key):
        if key == _TEST_KEY:
            return _TEST_OWNER
        if key == other_key:
            return other_owner
        return None

    monkeypatch.setattr(storage, "authenticate_owner", _auth_two)

    def _get_all_nodes():
        return [
            _make_node(_TEST_OWNER, ollama_models=["llama3"]),
            _make_node(other_owner, ollama_models=["mistral"]),
        ]

    monkeypatch.setattr(storage, "get_all_nodes", _get_all_nodes)
    monkeypatch.setattr(server, "MODELS_CACHE_TTL", 10.0)

    r_a = client.get("/v1/models", headers={"Authorization": f"Bearer {_TEST_KEY}"}).json()
    r_b = client.get("/v1/models", headers={"Authorization": f"Bearer {other_key}"}).json()

    a_ids = {m["id"] for m in r_a["data"]}
    b_ids = {m["id"] for m in r_b["data"]}
    assert a_ids == {"llama3"}
    assert b_ids == {"mistral"}
    assert set(server._models_cache.keys()) == {_TEST_OWNER, other_owner}
