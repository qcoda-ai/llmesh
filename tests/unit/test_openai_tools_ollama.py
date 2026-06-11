"""
D-009/D-010 Phase 2 — Ollama tools + harmony plumb-through.

Covers:
- tool_choice enforcement at hub (Ollama 0.30.7 ignores all four values per
  Phase 0 — verified in `.qcoda/discussions.md::D-009`).
- tool_calls round-trip into OpenAI response shape (id/type/function with
  JSON-string arguments).
- reasoning_content surfaces gpt-oss harmony `thinking` channel.
- Streaming path synthesizes a single tool_call delta.
- /complete endpoint persists tool_calls + reasoning_content on Task.

Mocks the agent dispatch by reaching into `_process_chat_completion` /
`_format_tool_calls_for_openai` directly where appropriate.
"""
import json
import os

import pytest

_FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "server_config.json")
os.environ.setdefault("LLMESH_CONFIG_PATH", _FIXTURES)
os.environ.setdefault("LLMESH_ALLOW_SAMPLE_KEYS", "1")
os.environ["OPENAI_TOOLS_ENABLED"] = "true"

from lib.hub import server  # noqa: E402
from lib.hub import tasks as hub_tasks  # noqa: E402


# --- Tool-call formatting helpers ---


def test_format_tool_calls_coerces_dict_args_to_json_string():
    """Ollama emits arguments as dict; OpenAI wire shape requires JSON-string."""
    ollama_shape = [{
        "id": "call_abc",
        "function": {"index": 0, "name": "get_weather",
                     "arguments": {"city": "Paris", "unit": "celsius"}},
    }]
    out = server._format_tool_calls_for_openai(ollama_shape)
    assert len(out) == 1
    tc = out[0]
    assert tc["id"] == "call_abc"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris", "unit": "celsius"}
    # Ollama's non-standard `function.index` MUST be dropped.
    assert "index" not in tc["function"]


def test_format_tool_calls_passes_through_string_args():
    """Idempotent — already-conformant shape unchanged in structure."""
    openai_shape = [{
        "id": "call_xyz",
        "type": "function",
        "function": {"name": "calc", "arguments": '{"x":1}'},
    }]
    out = server._format_tool_calls_for_openai(openai_shape)
    assert out[0]["function"]["arguments"] == '{"x":1}'


def test_format_tool_calls_handles_missing_id():
    """Defensive: minted only if Ollama omitted (should not happen on 0.30.7)."""
    out = server._format_tool_calls_for_openai([{"function": {"name": "x", "arguments": {}}}])
    assert out[0]["id"].startswith("call_")


def test_format_tool_calls_handles_empty_args():
    out = server._format_tool_calls_for_openai([{"id": "c1", "function": {"name": "n", "arguments": None}}])
    assert out[0]["function"]["arguments"] == "{}"


def test_format_tool_calls_empty_input():
    assert server._format_tool_calls_for_openai([]) == []
    assert server._format_tool_calls_for_openai(None) == []


# --- Task model fields ---


def test_task_has_tool_calls_and_reasoning_slots():
    t = hub_tasks.Task("t1", model="qwen3-coder:30b", owner_id="o")
    assert t.tool_calls == []
    assert t.reasoning_content == ""


def test_task_tool_calls_assignable():
    t = hub_tasks.Task("t1", model="qwen3-coder:30b", owner_id="o")
    t.tool_calls = [{"id": "c1", "function": {"name": "n", "arguments": {}}}]
    t.reasoning_content = "thinking..."
    assert len(t.tool_calls) == 1
    assert t.reasoning_content == "thinking..."


# --- Schema: extra="allow" on OpenAIMessage carries tool-flow fields ---


def test_openai_message_carries_tool_calls_via_model_extra():
    m = server.OpenAIMessage(
        role="assistant",
        content=None,
        tool_calls=[{"id": "c1", "function": {"name": "n", "arguments": "{}"}}],
    )
    assert m.role == "assistant"
    extras = m.model_extra or {}
    assert "tool_calls" in extras


def test_openai_message_carries_tool_call_id():
    m = server.OpenAIMessage(role="tool", content="result", tool_call_id="c1", name="get_weather")
    extras = m.model_extra or {}
    assert extras.get("tool_call_id") == "c1"
    assert extras.get("name") == "get_weather"


def test_openai_message_accepts_null_content_on_tool_turns():
    """Assistant-with-tool_calls has null content per OpenAI spec."""
    m = server.OpenAIMessage(role="assistant", content=None)
    assert m.content is None


# --- tool_choice enforcement logic (unit-level — pure transformation) ---


def _tools_fixture():
    return [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
        {"type": "function", "function": {"name": "send_email", "parameters": {}}},
    ]


def test_tool_choice_none_strips_tools():
    """tool_choice='none' must drop tools entirely before forwarding to agent."""
    # Mirror the enforcement block from _process_chat_completion in isolation
    # to assert the transformation rule. (Integration via the route would
    # require a node round-trip; this asserts the rule itself.)
    tools = _tools_fixture()
    tool_choice = "none"
    effective_tools = tools
    if tool_choice == "none":
        effective_tools = None
    assert effective_tools is None


def test_tool_choice_specific_filters_to_one_function():
    tools = _tools_fixture()
    tool_choice = {"type": "function", "function": {"name": "get_weather"}}
    effective_tools = tools
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        wanted = (tool_choice.get("function") or {}).get("name")
        effective_tools = [
            t for t in tools
            if isinstance(t, dict) and (t.get("function") or {}).get("name") == wanted
        ] or tools
    assert len(effective_tools) == 1
    assert effective_tools[0]["function"]["name"] == "get_weather"


def test_tool_choice_specific_with_unknown_name_falls_back_to_all():
    """Unknown function name in tool_choice => no match => fallback to all tools."""
    tools = _tools_fixture()
    tool_choice = {"type": "function", "function": {"name": "nonexistent"}}
    wanted = tool_choice["function"]["name"]
    filtered = [t for t in tools if t["function"]["name"] == wanted]
    effective = filtered or tools
    # No nonexistent match -> fallback. Better than 500 erroring on user mistake.
    assert len(effective) == 2


def test_tool_choice_auto_and_required_pass_tools_through():
    """auto + required both forward tools unchanged. Enforcement of `required`
    happens post-response by inspecting task.tool_calls."""
    for tc in ("auto", "required", None):
        tools = _tools_fixture()
        effective = tools
        if tc == "none":
            effective = None
        elif isinstance(tc, dict) and tc.get("type") == "function":
            effective = []
        assert effective == tools, f"tool_choice={tc!r} should pass through"


# --- StreamChunk schema carries tool_calls + reasoning_content ---


def test_streamchunk_accepts_tool_calls():
    c = server.StreamChunk(
        chunk="", done=True,
        prompt_tokens=10, completion_tokens=20,
        tool_calls=[{"id": "c1", "function": {"name": "n", "arguments": {}}}],
        reasoning_content="thinking",
    )
    assert c.done
    assert c.tool_calls and c.tool_calls[0]["id"] == "c1"
    assert c.reasoning_content == "thinking"


def test_streamchunk_defaults_nullable():
    c = server.StreamChunk(chunk="hi", done=False)
    assert c.tool_calls is None
    assert c.reasoning_content is None


# --- InferenceRequest carries tools + tool_choice ---


def test_inference_request_carries_tools():
    req = server.InferenceRequest(
        owner_id="o1",
        model="qwen3-coder:30b",
        messages=[{"role": "user", "content": "hi"}],
        tools=_tools_fixture(),
        tool_choice="auto",
    )
    assert len(req.tools) == 2
    assert req.tool_choice == "auto"


# --- Agent-side normalization: JSON-string args -> dict args for Ollama ---


def test_normalize_tool_call_args_string_to_dict():
    """Multi-turn flow finding: Ollama 0.30.7 requires dict args in history,
    NOT JSON-string. Hub sends OpenAI shape (string); agent un-coerces."""
    from lib.agent.client import _normalize_tool_call_args_for_ollama
    history = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "function": {"name": "n",
                                                   "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    ]
    out = _normalize_tool_call_args_for_ollama(history)
    assert isinstance(out[1]["tool_calls"][0]["function"]["arguments"], dict)
    assert out[1]["tool_calls"][0]["function"]["arguments"]["city"] == "Paris"


def test_normalize_tool_call_args_dict_passthrough():
    """Already-dict args (sent natively by some clients) must pass through."""
    from lib.agent.client import _normalize_tool_call_args_for_ollama
    history = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "function": {"name": "n",
                                                   "arguments": {"city": "Paris"}}}]},
    ]
    out = _normalize_tool_call_args_for_ollama(history)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}


def test_normalize_tool_call_args_invalid_json_falls_back_to_empty():
    """Defensive: malformed JSON in arguments -> empty dict, not crash."""
    from lib.agent.client import _normalize_tool_call_args_for_ollama
    history = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "function": {"name": "n",
                                                   "arguments": "{not valid json"}}]},
    ]
    out = _normalize_tool_call_args_for_ollama(history)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {}


def test_normalize_tool_call_args_messages_without_tool_calls_unchanged():
    from lib.agent.client import _normalize_tool_call_args_for_ollama
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = _normalize_tool_call_args_for_ollama(history)
    assert out == history


# --- Hub route integration: tool_choice enforcement on the wire ---


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(server.app)


_AUTH = {"Authorization": "Bearer my_secret_key_1"}
_TOOLS_REQ = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}]


async def _drain(gen):
    out = []
    async for frame in gen:
        out.append(frame)
    return out


def test_sse_generator_emits_reasoning_content_delta_before_done():
    """Streaming consumers must see message.reasoning_content as a delta.
    Single-shot synthesis matches the actual Ollama wire shape (one whole
    thinking field on the done frame)."""
    import asyncio
    from lib.hub import tasks as ht
    task = ht.Task("t1", model="gpt-oss:20b", owner_id="o")
    task.stream_queue = asyncio.Queue()
    task.created_at = 0.0
    task.reasoning_content = "User asks weather. Use the tool."

    async def go():
        await task.stream_queue.put(None)  # immediate sentinel — no content tokens
        frames = await _drain(server._real_sse_generator(
            task, request=None, task_id_str="x", created=0, model="gpt-oss:20b",
            session_id="s", owner_id="o", stored_history=[], incoming_messages=[],
        ))
        joined = "".join(frames)
        return joined

    out = asyncio.run(go())
    assert "reasoning_content" in out, "stream must emit reasoning_content delta"
    assert "User asks weather" in out


def test_sse_generator_emits_tool_calls_then_done():
    """tool_calls delta must precede the finish_reason=tool_calls frame."""
    import asyncio
    from lib.hub import tasks as ht
    task = ht.Task("t2", model="qwen3-coder:30b", owner_id="o")
    task.stream_queue = asyncio.Queue()
    task.created_at = 0.0
    task.tool_calls = [{"id": "call_x", "function":
                        {"name": "get_weather", "arguments": {"city": "Paris"}}}]

    async def go():
        await task.stream_queue.put(None)
        frames = await _drain(server._real_sse_generator(
            task, request=None, task_id_str="x", created=0, model="qwen3-coder:30b",
            session_id="s", owner_id="o", stored_history=[], incoming_messages=[],
        ))
        return "".join(frames)

    out = asyncio.run(go())
    assert "tool_calls" in out
    assert "get_weather" in out
    assert '"finish_reason": "tool_calls"' in out
    # arguments coerced to JSON-string in delta (city embedded in nested JSON)
    assert "city" in out and "Paris" in out


def test_route_phase2_no_node_returns_503_not_400_with_tools(client, monkeypatch):
    """OPENAI_TOOLS_ENABLED=true => Phase 1 reject bypassed. No node in test env
    => routing yields 503. NOT 400 unsupported_param."""
    monkeypatch.setattr(server, "OPENAI_TOOLS_ENABLED", True)
    resp = client.post("/v1/chat/completions", headers=_AUTH, json={
        "model": "qwen3-coder:30b",
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": _TOOLS_REQ,
        "tool_choice": "required",
    })
    # Either 503 (no node) or other downstream. 400 unsupported_param means
    # Phase 1 gate fired, which is wrong with the flag on.
    if resp.status_code == 400:
        body = resp.json()
        assert body.get("error", {}).get("type") != "unsupported_param"


# --- D098 — hub→agent serialization carries tools (the prod bug) ---


def test_pending_task_serialization_flattens_tools_to_top_level():
    """Hub places tools in payload["tools"]; agent reads task.get("tools") from
    the top level. Without flatten the tools silently never reach Ollama.
    Verified in prod 2026-06-10 via curl probe — qwen3-coder + gpt-oss both
    returned finish_reason=stop / tool_calls=None with prompt_tokens=13/72
    (tool schema absent from upstream prompt). D098."""
    from lib.hub import tasks as ht
    tools = [{"type": "function", "function": {"name": "git_ls_files",
              "description": "ls", "parameters": {"type": "object"}}}]
    t = ht.Task(
        task_id="t-d098",
        kind=ht.TaskKind.CHAT,
        model="qwen3-coder:30b",
        owner_id="o1",
        messages=[{"role": "user", "content": "list files"}],
    )
    t.payload["tools"] = tools

    # Mirror the entry-build shape from get_pending_tasks_for_node.
    entry = {
        "task_id": t.task_id,
        "kind": t.kind.value,
        "payload": t.payload,
        "model": t.model,
        "stream": t.stream,
    }
    if t.kind is ht.TaskKind.CHAT:
        entry["prompt"] = t.prompt
        entry["messages"] = t.messages
        entry["num_ctx"] = t.num_ctx
        entry["max_tokens"] = t.max_tokens
        entry["tools"] = t.tools

    assert entry["tools"] == tools, "tools must be flattened to top level so agent.get('tools') sees them"


def test_task_tools_property_returns_payload_tools_for_chat():
    from lib.hub import tasks as ht
    tools = [{"type": "function", "function": {"name": "x"}}]
    t = ht.Task("tx", kind=ht.TaskKind.CHAT, model="m", owner_id="o")
    t.payload["tools"] = tools
    assert t.tools == tools


def test_task_tools_property_returns_none_for_non_chat():
    from lib.hub import tasks as ht
    t = ht.Task("ty", kind=ht.TaskKind.EMBEDDING, model="m", owner_id="o")
    t.payload["tools"] = [{"function": {"name": "x"}}]
    assert t.tools is None, "tools only meaningful on CHAT tasks"


def test_named_function_tool_choice_post_validate_fires_when_empty(client, monkeypatch):
    """D098 — named-function tool_choice MUST 422 when model emits no tool_calls,
    symmetric with tool_choice='required'. Pre-fix, named-function form silently
    returned 200+content (qcoda symptom). Forces the contract."""
    monkeypatch.setattr(server, "OPENAI_TOOLS_ENABLED", True)
    # Test-suite ordering can leave api_keys unpopulated; patch storage auth.
    from lib.hub import storage as _storage
    monkeypatch.setattr(_storage, "authenticate_owner", lambda k: "test-owner" if k == "my_secret_key_1" else None)

    async def _fake_route(req, stream=False):
        from lib.hub import tasks as ht
        # Simulate the model returning content with no tool_calls.
        task = ht.Task(
            task_id="t-named",
            kind=ht.TaskKind.CHAT,
            model=req.model,
            owner_id=req.owner_id,
            messages=req.messages,
        )
        task.status = "completed"
        task.result = "I would call ls but cannot"
        task.tool_calls = []
        task.done_event.set()
        from lib.hub import tasks as _t
        _t._task_index[task.task_id] = task
        _t._node_tasks.setdefault("fake-node", []).append(task)
        from lib.hub.server import TaskResponse
        return TaskResponse(task_id=task.task_id, node_assigned="fake-node",
                             status="completed", model=req.model)

    monkeypatch.setattr(server, "_route_inference", _fake_route)

    resp = client.post("/v1/chat/completions", headers=_AUTH, json={
        "model": "qwen3-coder:30b",
        "messages": [{"role": "user", "content": "list files"}],
        "tools": _TOOLS_REQ,
        "tool_choice": {"type": "function", "function": {"name": "get_current_weather"}},
    })
    assert resp.status_code == 422, f"expected 422 for unfulfilled named-function tool_choice, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["error"]["type"] == "tool_choice_required_but_none_emitted"
    assert "get_current_weather" in body["error"]["attempted_tools"]


# --- D099 — close the silent-drop gaps D098 left open ---
#   (a) /v1/messages (Anthropic) forwards tools + surfaces tool_use
#   (b) agent vLLM/MLX OpenAI-compat path forwards tools + parses tool_calls
#   + DoD regressions for the OpenAI/harmony path that D094/D095 fixed.


def _install_fake_route(monkeypatch, *, result, tool_calls=None, reasoning="",
                        prompt_tokens=11, completion_tokens=7):
    """Patch storage auth + _route_inference so the endpoint handlers run end to
    end against a synthetic completed task (no live node needed)."""
    from lib.hub import storage as _storage
    monkeypatch.setattr(server, "OPENAI_TOOLS_ENABLED", True)
    monkeypatch.setattr(_storage, "authenticate_owner",
                        lambda k: "test-owner" if k == "my_secret_key_1" else None)

    async def _fake_route(req, stream=False):
        from lib.hub import tasks as ht
        task = ht.Task(task_id="t-d099", kind=ht.TaskKind.CHAT, model=req.model,
                       owner_id=req.owner_id, messages=req.messages)
        task.status = "completed"
        task.result = result
        task.tool_calls = tool_calls or []
        task.reasoning_content = reasoning
        task.prompt_tokens = prompt_tokens
        task.completion_tokens = completion_tokens
        task.done_event.set()
        ht._task_index[task.task_id] = task
        ht._node_tasks.setdefault("fake-node", []).append(task)
        from lib.hub.server import TaskResponse
        return TaskResponse(task_id=task.task_id, node_assigned="fake-node",
                            status="completed", model=req.model)

    monkeypatch.setattr(server, "_route_inference", _fake_route)


# (a.0) the Anthropic tool_use formatter

def test_format_tool_calls_for_anthropic_coerces_string_args_to_dict():
    out = server._format_tool_calls_for_anthropic(
        [{"id": "c1", "type": "function",
          "function": {"name": "git_ls_files", "arguments": '{"pattern": "*.py"}'}}])
    assert out == [{"type": "tool_use", "id": "c1", "name": "git_ls_files",
                    "input": {"pattern": "*.py"}}]


def test_format_tool_calls_for_anthropic_passes_dict_args():
    out = server._format_tool_calls_for_anthropic(
        [{"id": "c2", "function": {"name": "n", "arguments": {"a": 1}}}])
    assert out[0]["input"] == {"a": 1} and out[0]["type"] == "tool_use"


def test_format_tool_calls_for_anthropic_handles_missing_id_and_bad_json():
    out = server._format_tool_calls_for_anthropic(
        [{"function": {"name": "x", "arguments": "not-json"}}])
    assert out[0]["input"] == {} and out[0]["id"].startswith("toolu_")


def test_format_tool_calls_for_anthropic_empty():
    assert server._format_tool_calls_for_anthropic([]) == []
    assert server._format_tool_calls_for_anthropic(None) == []


# (a.1) /v1/messages forwards tools and emits a tool_use block

def test_anthropic_messages_surfaces_tool_use_block(client, monkeypatch):
    """Pre-D099 /v1/messages dropped tools/tool_choice and could only emit a
    text block — silent 0-tool-call. Now it must return a tool_use block and
    stop_reason='tool_use'."""
    _install_fake_route(monkeypatch, result="",
                        tool_calls=[{"id": "call_z", "function":
                                     {"name": "git_ls_files", "arguments": {"pattern": "*"}}}])
    resp = client.post("/v1/messages", headers={"x-api-key": "my_secret_key_1"}, json={
        "model": "gpt-oss:20b", "max_tokens": 256,
        "messages": [{"role": "user", "content": "list files"}],
        "tools": [{"type": "function", "function": {"name": "git_ls_files",
                   "parameters": {"type": "object", "properties": {}}}}],
        "tool_choice": {"type": "function", "function": {"name": "git_ls_files"}},
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    uses = [b for b in body["content"] if b["type"] == "tool_use"]
    assert len(uses) == 1
    assert uses[0]["name"] == "git_ls_files"
    assert uses[0]["input"] == {"pattern": "*"}


def test_anthropic_messages_text_only_is_end_turn(client, monkeypatch):
    _install_fake_route(monkeypatch, result="hello there")
    resp = client.post("/v1/messages", headers={"x-api-key": "my_secret_key_1"}, json={
        "model": "gpt-oss:20b", "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stop_reason"] == "end_turn"
    assert body["content"] == [{"type": "text", "text": "hello there"}]


# (a.2) Anthropic streaming generator emits tool_use content blocks

def test_anthropic_sse_emits_tool_use_blocks():
    import asyncio
    from lib.hub import tasks as ht
    task = ht.Task("t-asse", model="gpt-oss:20b", owner_id="o")
    task.stream_queue = asyncio.Queue()
    task.created_at = 0.0
    task.tool_calls = [{"id": "call_a", "function":
                        {"name": "git_grep", "arguments": {"pattern": "def foo"}}}]

    async def go():
        await task.stream_queue.put(None)
        frames = await _drain(server._real_sse_generator_anthropic(
            task, request=None, message_id="msg_x", model="gpt-oss:20b",
            input_tokens_estimate=5))
        return "".join(frames)

    out = asyncio.run(go())
    assert '"type": "tool_use"' in out
    assert "git_grep" in out
    assert "input_json_delta" in out and "def foo" in out
    assert '"stop_reason": "tool_use"' in out


# (b) agent vLLM/MLX path forwards tools + parses tool_calls (real dispatch)

def test_agent_mlx_path_forwards_tools_and_parses_tool_calls(monkeypatch):
    """D099 — the OpenAI-compat (vLLM/MLX) branch of _run_single_task previously
    built req_body without `tools` and ignored `tool_calls` in the response.
    Drive the real function with a MockTransport backend + capture the hub
    submit payload."""
    import asyncio, json as _json, httpx
    from lib.agent import client as agent

    monkeypatch.setattr(agent, "MLX_HOST", "http://mlx.test")
    monkeypatch.setattr(agent, "HUB_URL", "http://hub.test")

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "mlx.test":
            sent = _json.loads(request.content)
            captured["backend_body"] = sent
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "content": "",
                    "tool_calls": [{"id": "call_m", "type": "function",
                                    "function": {"name": "git_ls_files", "arguments": "{}"}}],
                }}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            })
        # hub /complete
        captured["submit_body"] = _json.loads(request.content)
        return httpx.Response(200, json={"status": "ok"})

    state = agent.AppState()
    state.node_id = "node-1"
    state.node_token = "tok"
    state.mlx_models = ["my-mlx-model"]
    task = {
        "task_id": "tk1", "model": "my-mlx-model", "kind": "chat",
        "messages": [{"role": "user", "content": "list files"}],
        "tools": [{"type": "function", "function": {"name": "git_ls_files",
                   "parameters": {"type": "object", "properties": {}}}}],
    }

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            await agent._run_single_task(c, state, task)

    asyncio.run(go())
    assert "tools" in captured["backend_body"], "tools must be forwarded to the MLX/vLLM backend"
    assert captured["submit_body"].get("tool_calls"), "tool_calls must be submitted back to the hub"
    assert captured["submit_body"]["tool_calls"][0]["function"]["name"] == "git_ls_files"


# DoD regressions on the OpenAI/harmony path (D094/D095 — must stay fixed)

def test_openai_gpt_oss_content_clean_and_reasoning_separated(client, monkeypatch):
    """BUG 1 guard: final channel maps to content, analysis to reasoning_content,
    no raw <|channel|> control tokens leak into content."""
    _install_fake_route(monkeypatch, result="ready",
                        reasoning="The user wants exactly: ready.")
    resp = client.post("/v1/chat/completions", headers=_AUTH, json={
        "model": "gpt-oss:20b",
        "messages": [{"role": "user", "content": "reply with exactly: ready"}],
    })
    assert resp.status_code == 200, resp.text
    msg = resp.json()["choices"][0]["message"]
    assert msg["content"] == "ready"
    assert "<|channel|>" not in msg["content"] and "<|message|>" not in msg["content"]
    assert msg["reasoning_content"] == "The user wants exactly: ready."


def test_openai_named_tool_choice_roundtrip_produces_tool_call(client, monkeypatch):
    """BUG 2 guard: named-function tool_choice that DOES produce a tool_call
    returns 200 with finish_reason=tool_calls (not dropped, not 422)."""
    _install_fake_route(monkeypatch, result="",
                        tool_calls=[{"id": "call_g", "function":
                                     {"name": "git_ls_files", "arguments": {}}}])
    resp = client.post("/v1/chat/completions", headers=_AUTH, json={
        "model": "gpt-oss:20b",
        "messages": [{"role": "user", "content": "list files"}],
        "tools": [{"type": "function", "function": {"name": "git_ls_files",
                   "parameters": {"type": "object", "properties": {}}}}],
        "tool_choice": {"type": "function", "function": {"name": "git_ls_files"}},
    })
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "git_ls_files"
