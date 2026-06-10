"""
D-009 Phase 1 — tools/tool_choice loud-reject.

Hub used to silently drop unknown OpenAI request fields because the Pydantic
schema didn't declare them (qcoda customer report 2026-06-10). Phase 1 adds
the fields to the schema and returns 400 with `error.type="unsupported_param"`
when present, gated by `OPENAI_TOOLS_ENABLED`. Phase 2 ships real Ollama-backed
tool calling; the v0.21.0 release flipped the default to `true` (D094/D095).
These tests monkeypatch the flag back to `false` to assert the reject path
is still honored when an operator sets the emergency kill-switch.

See `.qcoda/features/feature_openai_tools_harmony.md` for the phase plan.
"""
import os

import pytest
from fastapi.testclient import TestClient

_FIXTURES = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "server_config.json"
)
os.environ.setdefault("LLMESH_CONFIG_PATH", _FIXTURES)
os.environ.setdefault("LLMESH_ALLOW_SAMPLE_KEYS", "1")

from lib.hub import server  # noqa: E402


@pytest.fixture
def client():
    return TestClient(server.app)


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

_AUTH = {"Authorization": "Bearer my_secret_key_1"}
_ANTHROPIC_AUTH = {"x-api-key": "my_secret_key_1"}


# --- Schema acceptance ---


def test_openai_schema_accepts_tools_field():
    """Pydantic must accept tools/tool_choice without 422. Phase 1's whole
    point: the fields are now wire-shape-visible to the hub."""
    req = server.OpenAIRequest(
        model="qwen3-coder:30b",
        messages=[{"role": "user", "content": "hi"}],
        tools=_TOOLS,
        tool_choice="auto",
    )
    assert req.tools == _TOOLS
    assert req.tool_choice == "auto"


def test_openai_schema_accepts_tool_choice_dict():
    """tool_choice can be a string or a {type:function, function:{name:...}} dict."""
    req = server.OpenAIRequest(
        model="qwen3-coder:30b",
        messages=[{"role": "user", "content": "hi"}],
        tool_choice={"type": "function", "function": {"name": "get_current_weather"}},
    )
    assert req.tool_choice == {"type": "function", "function": {"name": "get_current_weather"}}


def test_openai_schema_accepts_no_tools_no_regression():
    """No tools present must not 422 — backward compat."""
    req = server.OpenAIRequest(
        model="qwen3-coder:30b",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.tools is None
    assert req.tool_choice is None


def test_anthropic_schema_accepts_tools_field():
    req = server.AnthropicRequest(
        model="claude-3-opus",
        messages=[{"role": "user", "content": "hi"}],
        tools=_TOOLS,
        tool_choice="auto",
    )
    assert req.tools == _TOOLS


# --- Loud 400 reject when flag off ---


def test_openai_tools_rejected_400_when_flag_off(client, monkeypatch):
    monkeypatch.setattr(server, "OPENAI_TOOLS_ENABLED", False)
    resp = client.post(
        "/v1/chat/completions",
        headers=_AUTH,
        json={
            "model": "qwen3-coder:30b",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": _TOOLS,
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["type"] == "unsupported_param"
    assert body["error"]["param"] == "tools"
    assert "D-009" in body["error"]["message"]


def test_openai_tool_choice_alone_rejected_400(client, monkeypatch):
    """tool_choice without tools still trips the reject — qcoda's exact bug."""
    monkeypatch.setattr(server, "OPENAI_TOOLS_ENABLED", False)
    resp = client.post(
        "/v1/chat/completions",
        headers=_AUTH,
        json={
            "model": "qwen3-coder:30b",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tool_choice": "required",
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["param"] == "tool_choice"


def test_anthropic_tools_rejected_400_when_flag_off(client, monkeypatch):
    monkeypatch.setattr(server, "OPENAI_TOOLS_ENABLED", False)
    resp = client.post(
        "/v1/messages",
        headers=_ANTHROPIC_AUTH,
        json={
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": _TOOLS,
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["type"] == "unsupported_param"


# --- No-regression: requests without tools must pass auth + reach routing ---


def test_openai_without_tools_passes_phase1_gate(client, monkeypatch):
    """Tools-absent request must NOT trip Phase 1 reject — only auth + routing
    matter. We expect 503 (no node available in test env) NOT 400."""
    monkeypatch.setattr(server, "OPENAI_TOOLS_ENABLED", False)
    resp = client.post(
        "/v1/chat/completions",
        headers=_AUTH,
        json={
            "model": "qwen3-coder:30b",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    # 503 = routing failed (no nodes); 200 = some test fixture node served it.
    # Anything but 400 with unsupported_param means Phase 1 didn't trip.
    assert resp.status_code != 400 or resp.json().get("error", {}).get("type") != "unsupported_param"


# --- Flag-on path: Phase 1 reject SHOULD NOT trip ---


def test_openai_tools_pass_phase1_gate_when_flag_on(client, monkeypatch):
    """When OPENAI_TOOLS_ENABLED=true, Phase 1 reject is bypassed. Request
    flows to routing (which will 503 in test env without a node). Phase 2
    will replace the 503 with real tool-call dispatch."""
    monkeypatch.setattr(server, "OPENAI_TOOLS_ENABLED", True)
    resp = client.post(
        "/v1/chat/completions",
        headers=_AUTH,
        json={
            "model": "qwen3-coder:30b",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": _TOOLS,
            "tool_choice": "auto",
        },
    )
    # 503 acceptable (no node in test env). 400 unsupported_param means the
    # Phase-1 gate fired when it shouldn't have.
    if resp.status_code == 400:
        body = resp.json()
        assert body.get("error", {}).get("type") != "unsupported_param", (
            "Phase 1 gate fired when OPENAI_TOOLS_ENABLED=true"
        )
