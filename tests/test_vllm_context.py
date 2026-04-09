"""
Unit tests for the vLLM context-window resolution added in D015.

The agent must report a `context_size` to the hub at registration that
reflects the actual maximum context the node can serve, not the legacy
hardcoded `OLLAMA_NUM_CTX` value (which was wrong for vLLM-only nodes
before D015).

Resolution priority:
  1. `VLLM_MAX_CONTEXT` env var (explicit operator override)
  2. `max_model_len` auto-detected from `/v1/models`
  3. `OLLAMA_NUM_CTX` (conservative fallback)

These are pure-function tests — no hub or vLLM startup required.
"""
import importlib

import pytest


def _reload_client(monkeypatch, **env):
    """Reload lib.agent.client with a controlled environment so module-level
    constants like VLLM_MAX_CONTEXT and OLLAMA_NUM_CTX are re-read."""
    monkeypatch.setenv("LLMESH_API_KEY", "test-key")
    for key in (
        "VLLM_HOST",
        "VLLM_API_KEY",
        "VLLM_HEALTH_PATH",
        "VLLM_MAX_CONTEXT",
        "OLLAMA_NUM_CTX",
    ):
        monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    import lib.agent.client as client
    return importlib.reload(client)


def test_vllm_max_context_env_var_unset_when_zero(monkeypatch):
    """`VLLM_MAX_CONTEXT=0` is treated as unset (the default), not as a
    literal zero-token window."""
    client = _reload_client(monkeypatch, VLLM_MAX_CONTEXT="0")
    assert client.VLLM_MAX_CONTEXT is None


def test_vllm_max_context_env_var_parsed_as_int(monkeypatch):
    client = _reload_client(monkeypatch, VLLM_MAX_CONTEXT="32768")
    assert client.VLLM_MAX_CONTEXT == 32768


# ── _resolve_node_context_size: pure-function logic ────────────────────


def test_vllm_only_with_detected_window_uses_vllm(monkeypatch):
    client = _reload_client(monkeypatch, OLLAMA_NUM_CTX="8192")
    assert client._resolve_node_context_size(
        ollama_active=False, vllm_active=True, vllm_max_context=32768
    ) == 32768


def test_vllm_only_unknown_window_falls_back_to_ollama_default(monkeypatch):
    """When vLLM is up but `/v1/models` did not expose `max_model_len` and
    no explicit override is set, the agent must report a value rather than
    crash. OLLAMA_NUM_CTX is the documented fallback."""
    client = _reload_client(monkeypatch, OLLAMA_NUM_CTX="8192")
    assert client._resolve_node_context_size(
        ollama_active=False, vllm_active=True, vllm_max_context=None
    ) == 8192


def test_ollama_only_uses_ollama_num_ctx(monkeypatch):
    client = _reload_client(monkeypatch, OLLAMA_NUM_CTX="16384")
    assert client._resolve_node_context_size(
        ollama_active=True, vllm_active=False, vllm_max_context=None
    ) == 16384


def test_both_active_takes_max(monkeypatch):
    """A heterogeneous node advertises its largest backend window. The
    per-request num_ctx priority chain (D010) handles per-task selection
    downstream — currently Ollama-only, see D015."""
    client = _reload_client(monkeypatch, OLLAMA_NUM_CTX="8192")
    assert client._resolve_node_context_size(
        ollama_active=True, vllm_active=True, vllm_max_context=32768
    ) == 32768
    # And the inverse — Ollama window larger than vLLM's
    assert client._resolve_node_context_size(
        ollama_active=True, vllm_active=True, vllm_max_context=4096
    ) == 8192


def test_both_active_unknown_vllm_window_uses_ollama_default(monkeypatch):
    """When vLLM is up but its window is unknown, the resolver does not
    silently invent a number — it uses OLLAMA_NUM_CTX."""
    client = _reload_client(monkeypatch, OLLAMA_NUM_CTX="8192")
    assert client._resolve_node_context_size(
        ollama_active=True, vllm_active=True, vllm_max_context=None
    ) == 8192


def test_neither_active_uses_legacy_default(monkeypatch):
    """A node with no backends running still reports a context_size
    (registration must not crash)."""
    client = _reload_client(monkeypatch, OLLAMA_NUM_CTX="8192")
    assert client._resolve_node_context_size(
        ollama_active=False, vllm_active=False, vllm_max_context=None
    ) == 8192


# ── resolve_vllm_max_context: env var override priority ────────────────


def test_resolve_vllm_max_context_env_var_wins_over_autodetect(monkeypatch):
    """If VLLM_MAX_CONTEXT is set, the resolver must return it without
    making any HTTP call to /v1/models. We verify the no-HTTP claim by
    leaving VLLM_HOST unset — _query_vllm_models would otherwise try to
    reach an undefined host."""
    client = _reload_client(monkeypatch, VLLM_MAX_CONTEXT="16384")
    assert client.resolve_vllm_max_context() == 16384


def test_resolve_vllm_max_context_returns_none_when_no_host(monkeypatch):
    """No env var, no VLLM_HOST → no context to resolve. Caller falls back."""
    client = _reload_client(monkeypatch)
    assert client.resolve_vllm_max_context() is None
