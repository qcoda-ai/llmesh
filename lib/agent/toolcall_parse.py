"""Text-form tool-call normalization for Ollama-served qwen-family models (D101).

Ollama leaves **qwen-family** tool calls as TEXT in ``message.content`` instead
of populating structured ``tool_calls``:

- ``qwen2.5-coder`` → JSON object ``{"name": ..., "arguments": {...}}`` in content
- ``qwen3-coder``   → XML ``<function=NAME><parameter=k>v</parameter></function>``
- ``devstral``      → emits structured ``tool_calls`` natively (no fix needed)

Any client that only executes structured ``tool_calls`` (pi, goose, opencode,
MCP clients via Claude Desktop / Chatwise) silently no-ops on the text form.
This module parses the text call out of ``content`` and builds OpenAI-shape
``tool_calls`` so the agent submits a structured call the hub can shape for
both the OpenAI ``/v1/chat/completions`` and Anthropic ``/v1/messages`` surfaces.

This is the **response-in** sibling of D098 (which made ``tools`` *reach* the
agent). Graduated from the qcoda dev-time proxy
``labs/agent_eval/auto_code/toolcall_proxy.py`` (port 8765, A/B proven). Only the
parse *rules* are ported — NOT the standalone proxy process (a dev shim, SPOF).

Pure + deterministic + I/O-free — unit-testable in isolation, mirroring how
``server._format_tool_calls_for_openai`` is tested. Wired at the agent response-in
hook (``lib/agent/client.py`` Ollama non-stream + streaming-accumulated paths).
"""
from __future__ import annotations

import json
import re

# qwen3-coder / hermes-ish XML: <function=NAME><parameter=k>v</parameter></function>
_XML_FN = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_XML_PARAM = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


def should_normalize(tool_calls, content, tools_requested: bool) -> bool:
    """R4 guard — parse text-form calls ONLY when all hold (D101):

    - ``tool_calls`` is empty — never touch a native structured call (protects
      the qwen3-coder/devstral path + idempotency).
    - ``content`` is present — nothing to parse otherwise.
    - ``tools_requested`` — the request actually offered tools. Transitively
      honors the ``OPENAI_TOOLS_ENABLED`` kill-switch: when tools are globally
      off the hub forwards no ``tools`` to the task, so this is False and no
      parsing happens.
    """
    return bool(not tool_calls and content and tools_requested)


def extract_tool_calls(content: str) -> list[dict]:
    """Best-effort parse of text-form tool calls into OpenAI ``tool_calls``.

    Handles (a) qwen2.5 JSON ``{"name":..,"arguments":{..}}`` and
    (b) qwen3-coder XML ``<function=..>``. Returns ``[]`` if none found.

    Emits FINAL OpenAI wire shape directly:
    ``{id, type:"function", function:{name, arguments:<JSON string>}}``.
    ``server._format_tool_calls_for_openai`` is idempotent on this shape (string
    args pass through), and ``_format_tool_calls_for_anthropic`` parses the JSON
    string back to a dict — so both client surfaces work unchanged.

    Never raises: a malformed JSON blob is skipped, not propagated (fail-open
    is the caller's contract — a bad parse must not crash the completion).
    """
    if not content:
        return []
    calls: list[dict] = []

    # (b) XML form — qwen3-coder. Checked first: an XML call never contains a
    # standalone balanced JSON object, so ordering is unambiguous.
    for i, m in enumerate(_XML_FN.finditer(content)):
        name = m.group(1).strip()
        args = {k.strip(): v.strip() for k, v in _XML_PARAM.findall(m.group(2))}
        calls.append(_mk_call(i, name, args))
    if calls:
        return calls

    # (a) JSON form — qwen2.5-coder. Find balanced {...} blobs that look like a
    # call (have a "name" and an "arguments"/"parameters" key).
    for i, blob in enumerate(_json_objects(content)):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj and (
            "arguments" in obj or "parameters" in obj
        ):
            args = obj.get("arguments", obj.get("parameters", {}))
            calls.append(_mk_call(len(calls), obj["name"], args))
    return calls


def _mk_call(i: int, name: str, args) -> dict:
    """Build an OpenAI-shape tool_call dict. ``arguments`` is a JSON string."""
    if not isinstance(args, str):
        args = json.dumps(args)
    return {
        "id": f"call_{i}",
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _json_objects(s: str):
    """Yield top-level ``{...}`` substrings via brace matching (ignores braces
    inside string literals). Not a naive regex — balanced-brace extraction so a
    nested ``arguments`` object does not split the call."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for idx, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield s[start: idx + 1]
