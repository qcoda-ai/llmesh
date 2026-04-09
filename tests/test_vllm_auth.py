"""
Unit tests for the vLLM bearer-auth header helper added in D014.

The agent must attach `Authorization: Bearer <key>` to vLLM-path requests
when `VLLM_API_KEY` is set, and must NOT attach a header when it is unset
(plain local vLLM is unauthenticated). This is a pure-function test — no
hub or agent startup required.
"""
import importlib

import pytest


def _reload_client(monkeypatch, **env):
    """Reload lib.agent.client with a controlled environment so the
    module-level VLLM_API_KEY constant is re-read."""
    # LLMESH_API_KEY is required at import time — provide a dummy.
    monkeypatch.setenv("LLMESH_API_KEY", "test-key")
    for key in ("VLLM_API_KEY", "VLLM_HOST", "VLLM_HEALTH_PATH"):
        monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    import lib.agent.client as client
    return importlib.reload(client)


def test_vllm_headers_empty_when_no_key(monkeypatch):
    client = _reload_client(monkeypatch)
    assert client._vllm_headers() == {}


def test_vllm_headers_populated_when_key_set(monkeypatch):
    client = _reload_client(monkeypatch, VLLM_API_KEY="sk-litellm-test")
    assert client._vllm_headers() == {
        "Authorization": "Bearer sk-litellm-test"
    }


def test_vllm_health_path_default(monkeypatch):
    client = _reload_client(monkeypatch)
    assert client.VLLM_HEALTH_PATH == "/health"


def test_vllm_health_path_overridable(monkeypatch):
    client = _reload_client(monkeypatch, VLLM_HEALTH_PATH="/health/liveliness")
    assert client.VLLM_HEALTH_PATH == "/health/liveliness"


def test_vllm_check_skips_when_host_unset(monkeypatch):
    """Without VLLM_HOST, check_vllm_available must short-circuit to False
    and never make an HTTP call (so an unset env can't accidentally hit
    a stale URL)."""
    client = _reload_client(monkeypatch)  # no VLLM_HOST
    assert client.check_vllm_available() is False
    assert client.get_vllm_models() == []
