"""Integration tests for /v1/embeddings (D028).

Reuses the module-scoped hub + FakeNode fixture from conftest.py. The fake
node returns deterministic synthetic 8-dim vectors for any embedding task,
which is enough to exercise hub routing, payload validation, response shape,
and error paths without a real Ollama install.
"""
import httpx

from tests.test_anthropic_api import HUB_BASE, API_KEY


def _post_embeddings(payload: dict, *, api_key: str = API_KEY):
    return httpx.post(
        f"{HUB_BASE}/v1/embeddings",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )


def test_embeddings_single_string():
    r = _post_embeddings({"model": "nomic-embed-text", "input": "hello world"})
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "nomic-embed-text"
    assert len(body["data"]) == 1
    item = body["data"][0]
    assert item["object"] == "embedding"
    assert item["index"] == 0
    assert isinstance(item["embedding"], list)
    assert len(item["embedding"]) == 8
    assert all(isinstance(x, (int, float)) for x in item["embedding"])
    assert "usage" in body
    assert "prompt_tokens" in body["usage"]
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"]


def test_embeddings_batch_preserves_order():
    inputs = ["alpha", "beta-string", "gamma value here"]
    r = _post_embeddings({"model": "nomic-embed-text", "input": inputs})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 3
    indices = [d["index"] for d in body["data"]]
    assert indices == [0, 1, 2], f"index order broken: {indices}"


def test_embeddings_default_model():
    """Omitting `model` falls back to nomic-embed-text (DEFAULT_EMBEDDING_MODEL)."""
    r = _post_embeddings({"input": "default model test"})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "nomic-embed-text"


def test_embeddings_unknown_model_503():
    r = _post_embeddings({"model": "totally-not-an-embedding-model", "input": "x"})
    assert r.status_code == 503, r.text


def test_embeddings_chat_model_rejected():
    """A chat-only model (in ollama_models, not embedding_models) must 503."""
    r = _post_embeddings({"model": "llama3.2:3b", "input": "x"})
    assert r.status_code == 503, r.text


def test_embeddings_empty_input_400():
    r = _post_embeddings({"model": "nomic-embed-text", "input": ""})
    assert r.status_code == 400, r.text


def test_embeddings_empty_list_400():
    r = _post_embeddings({"model": "nomic-embed-text", "input": []})
    assert r.status_code == 400, r.text


def test_embeddings_oversize_batch_413():
    inputs = ["x"] * 200  # over MAX_BATCH_EMBEDDINGS=128
    r = _post_embeddings({"model": "nomic-embed-text", "input": inputs})
    assert r.status_code == 413, r.text


def test_embeddings_oversize_item_413():
    big = "a" * (40_000)  # over MAX_INPUT_BYTES=32768
    r = _post_embeddings({"model": "nomic-embed-text", "input": big})
    assert r.status_code == 413, r.text


def test_embeddings_missing_auth_401():
    r = httpx.post(f"{HUB_BASE}/v1/embeddings", json={"input": "x"}, timeout=5)
    assert r.status_code == 401


def test_embeddings_invalid_auth_401():
    r = _post_embeddings({"input": "x"}, api_key="not-a-real-key")
    assert r.status_code == 401


def test_v1_models_includes_embed_capability():
    r = httpx.get(f"{HUB_BASE}/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=5)
    assert r.status_code == 200, r.text
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert "nomic-embed-text" in by_id, "embedding model not surfaced in /v1/models"
    assert "embed" in by_id["nomic-embed-text"]["capabilities"]
    assert "llama3.2:3b" in by_id
    assert by_id["llama3.2:3b"]["capabilities"] == ["chat"]
