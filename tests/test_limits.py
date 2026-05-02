"""Integration tests for GET /v1/limits and structured 413 bodies (D049).

Reuses the module-scoped hub + FakeNode fixture from conftest.py. The fake
node registers as supporting `llama3.2:3b` (chat, ctx 8192) and
`nomic-embed-text` (embed, ctx 2048), so the per-model block on /v1/limits
should reflect those exactly.
"""
import httpx

from tests.test_anthropic_api import HUB_BASE, API_KEY


def _get_limits(*, api_key: str = API_KEY):
    return httpx.get(
        f"{HUB_BASE}/v1/limits",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=5,
    )


def test_limits_returns_static_caps():
    """Static block: global DoS-defence bounds enforced at request edge."""
    r = _get_limits()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["max_input_bytes"] == 262144
    assert body["max_messages"] == 200
    assert body["max_batch_embeddings"] == 128
    assert body["stream_queue_max"] == 256


def test_limits_includes_per_model_block():
    """Dynamic block: per-model context capacity from connected nodes."""
    r = _get_limits()
    assert r.status_code == 200, r.text
    models = r.json().get("models", {})
    assert "llama3.2:3b" in models, f"chat model missing from limits.models: {models}"
    chat = models["llama3.2:3b"]
    assert chat["context_tokens"] == 8192
    # min(MAX_INPUT_BYTES=262144, 8192*4=32768) → 32768
    assert chat["context_bytes_estimate"] == 32768

    assert "nomic-embed-text" in models, f"embed model missing from limits.models: {models}"
    embed = models["nomic-embed-text"]
    assert embed["context_tokens"] == 2048
    # min(262144, 2048*4=8192) → 8192
    assert embed["context_bytes_estimate"] == 8192


def test_limits_requires_auth():
    r = httpx.get(f"{HUB_BASE}/v1/limits", timeout=5)
    assert r.status_code == 401


def test_limits_invalid_auth():
    r = _get_limits(api_key="not-a-real-key")
    assert r.status_code == 401


def test_chat_oversize_413_structured():
    """D049: chat-completions oversize content emits structured 413 with the
    exact `messages[N].content` field path."""
    big = "a" * 300_000  # over MAX_INPUT_BYTES=262144
    r = httpx.post(
        f"{HUB_BASE}/v1/chat/completions",
        json={"model": "llama3.2:3b", "messages": [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": big},
        ]},
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=15,
    )
    assert r.status_code == 413, r.text
    err = r.json().get("error", {})
    assert err.get("type") == "payload_too_large"
    assert err.get("field") == "messages[2].content"
    assert err.get("limit_bytes") == 262144
    assert err.get("actual_bytes") == 300_000


def test_chat_oversize_messages_count_413_structured():
    """D049: too many messages → structured 413 with field=messages."""
    msgs = [{"role": "user", "content": "x"} for _ in range(250)]  # over MAX_MESSAGES=200
    r = httpx.post(
        f"{HUB_BASE}/v1/chat/completions",
        json={"model": "llama3.2:3b", "messages": msgs},
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=15,
    )
    assert r.status_code == 413, r.text
    err = r.json().get("error", {})
    assert err.get("type") == "payload_too_large"
    assert err.get("field") == "messages"
    assert err.get("limit_bytes") == 200
    assert err.get("actual_bytes") == 250
