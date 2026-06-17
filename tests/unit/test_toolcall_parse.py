"""D101 — text-form tool-call normalization (qwen-family on ollama).

Covers the net-new parser `lib/agent/toolcall_parse.py` that graduates the
qcoda dev-time proxy rules into nodemesh's serving path. Pure + I/O-free —
mirrors how `server._format_tool_calls_for_openai` is unit-tested (construct
inputs directly, no live ollama).

Empirical grounding (Verified 2026-06-16, ollama 0.30.8, local node):
- qwen2.5-coder:32b  -> JSON object in content, tool_calls=None (needs parse)
- qwen3-coder:30b    -> native structured tool_calls, content="" (no parse)
- devstral-small-2   -> native structured tool_calls, content="" (no parse)
The XML branch covers the qwen3-coder format on ollama versions that still
leak it (the proxy was built against one); harmless + guarded on 0.30.8.

Maps to spec §6 cases: 1 (JSON), 2 (XML), 3 (multiple), 4/5 (guard no-ops),
6 (fail-open malformed), 7 (idempotency).
"""
import json

from lib.agent.toolcall_parse import extract_tool_calls, should_normalize


# --- §6.1 qwen2.5 JSON-in-content ---


def test_qwen25_json_in_content_extracted():
    """Real qwen2.5-coder:32b emission (Verified 2026-06-16, ollama 0.30.8)."""
    content = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    out = extract_tool_calls(content)
    assert len(out) == 1
    tc = out[0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    # arguments emitted as a JSON STRING (OpenAI wire shape; the hub's
    # _format_tool_calls_for_openai is idempotent on this).
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}
    assert tc["id"].startswith("call_")


def test_qwen25_json_accepts_parameters_key():
    """Some emissions use `parameters` instead of `arguments`."""
    out = extract_tool_calls('{"name": "f", "parameters": {"x": 1}}')
    assert len(out) == 1
    assert json.loads(out[0]["function"]["arguments"]) == {"x": 1}


def test_json_with_surrounding_prose():
    """Balanced-brace extraction finds the call even amid leading/trailing text."""
    content = 'Sure, calling it now: {"name": "edit", "arguments": {"path": "a.py"}} done.'
    out = extract_tool_calls(content)
    assert len(out) == 1
    assert out[0]["function"]["name"] == "edit"


def test_json_nested_object_args_not_split():
    """Nested braces inside arguments must not split the object."""
    content = '{"name": "f", "arguments": {"a": {"b": {"c": 1}}}}'
    out = extract_tool_calls(content)
    assert len(out) == 1
    assert json.loads(out[0]["function"]["arguments"]) == {"a": {"b": {"c": 1}}}


def test_json_brace_inside_string_literal():
    """A `}` inside a string value must not close the object early."""
    content = '{"name": "say", "arguments": {"msg": "use } carefully"}}'
    out = extract_tool_calls(content)
    assert len(out) == 1
    assert json.loads(out[0]["function"]["arguments"]) == {"msg": "use } carefully"}


# --- §6.2 qwen3-coder XML-in-content ---


def test_qwen3_xml_function_extracted():
    content = "<function=get_weather><parameter=city>Paris</parameter></function>"
    out = extract_tool_calls(content)
    assert len(out) == 1
    assert out[0]["function"]["name"] == "get_weather"
    assert json.loads(out[0]["function"]["arguments"]) == {"city": "Paris"}


def test_qwen3_xml_multiple_params():
    content = ("<function=edit><parameter=path>a.py</parameter>"
               "<parameter=line>10</parameter></function>")
    out = extract_tool_calls(content)
    assert len(out) == 1
    assert json.loads(out[0]["function"]["arguments"]) == {"path": "a.py", "line": "10"}


# --- §6.3 multiple calls ---


def test_multiple_json_calls():
    content = ('{"name": "a", "arguments": {"x": 1}} '
               '{"name": "b", "arguments": {"y": 2}}')
    out = extract_tool_calls(content)
    assert len(out) == 2
    assert [c["function"]["name"] for c in out] == ["a", "b"]
    assert out[0]["id"] != out[1]["id"]


def test_multiple_xml_calls():
    content = ("<function=a><parameter=x>1</parameter></function>"
               "<function=b><parameter=y>2</parameter></function>")
    out = extract_tool_calls(content)
    assert len(out) == 2
    assert [c["function"]["name"] for c in out] == ["a", "b"]


def test_xml_preferred_when_both_present():
    """XML form checked first; a JSON blob alongside XML is not double-counted."""
    content = ('<function=a><parameter=x>1</parameter></function>'
               '{"name": "b", "arguments": {}}')
    out = extract_tool_calls(content)
    assert [c["function"]["name"] for c in out] == ["a"]


# --- §6.6 fail-open on malformed / non-calls ---


def test_empty_content_returns_empty():
    assert extract_tool_calls("") == []
    assert extract_tool_calls(None) == []  # type: ignore[arg-type]


def test_plain_prose_returns_empty():
    assert extract_tool_calls("The weather in Paris is sunny today.") == []


def test_json_without_name_ignored():
    """A balanced object lacking `name`+args keys is not a tool call."""
    assert extract_tool_calls('{"foo": "bar", "baz": 1}') == []


def test_malformed_json_does_not_raise():
    """Unparseable brace blob is skipped, never propagated."""
    assert extract_tool_calls('{"name": "f", "arguments": {bad json}}') == []


def test_unbalanced_braces_no_crash():
    assert extract_tool_calls('{"name": "f", "arguments": {') == []


# --- §6.7 idempotency ---


def test_idempotent_extraction():
    content = '{"name": "f", "arguments": {"x": 1}}'
    first = extract_tool_calls(content)
    second = extract_tool_calls(content)
    assert first == second


# --- §6.4 / §6.5 guard (should_normalize) ---


def test_guard_skips_when_tool_calls_present():
    """§6.4 — native structured call (devstral/qwen3-coder) never re-parsed."""
    structured = [{"id": "c1", "function": {"name": "n", "arguments": {}}}]
    assert should_normalize(structured, "ignored", True) is False


def test_guard_skips_when_no_tools_requested():
    """§6.5 — no tools offered → content left untouched."""
    assert should_normalize([], '{"name": "f", "arguments": {}}', False) is False


def test_guard_skips_when_no_content():
    assert should_normalize([], "", True) is False
    assert should_normalize([], None, True) is False


def test_guard_fires_on_text_form_call():
    assert should_normalize([], '{"name": "f", "arguments": {}}', True) is True


def test_guard_composes_to_noop_on_devstral_shape():
    """End-to-end guard semantics: structured present → extract never called."""
    structured = [{"id": "c1", "type": "function",
                   "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]
    content = ""  # native path carries empty content
    if should_normalize(structured, content, True):
        raise AssertionError("guard should have short-circuited")
