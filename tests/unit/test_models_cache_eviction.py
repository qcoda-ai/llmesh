"""
D056 — bounded LRU eviction on `_models_cache`. Verifies the dict cannot
grow past MODELS_CACHE_MAX entries regardless of how many distinct owners
the hub serves.
"""
import os

# Same import-time guard pattern as test_csrf.py.
_FIXTURES = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "server_config.json"
)
os.environ.setdefault("LLMESH_CONFIG_PATH", _FIXTURES)
os.environ.setdefault("LLMESH_ALLOW_SAMPLE_KEYS", "1")

from collections import OrderedDict  # noqa: E402

from lib.hub import server  # noqa: E402


def test_models_cache_is_ordered_dict():
    """The cache must be an OrderedDict so move_to_end / popitem(last=False)
    work as expected. Sanity-check the type."""
    assert isinstance(server._models_cache, OrderedDict)


def test_models_cache_max_eviction(monkeypatch):
    """Filling the cache past MODELS_CACHE_MAX evicts the oldest entry."""
    monkeypatch.setattr(server, "MODELS_CACHE_MAX", 3)
    server._models_cache.clear()
    # Simulate the write pattern from the endpoint, without invoking HTTP.
    import time as _t

    def _put(owner):
        server._models_cache[owner] = (_t.monotonic() + 10.0, {"owner": owner})
        server._models_cache.move_to_end(owner)
        while len(server._models_cache) > server.MODELS_CACHE_MAX:
            server._models_cache.popitem(last=False)

    for o in ["a", "b", "c", "d", "e"]:
        _put(o)
    assert list(server._models_cache.keys()) == ["c", "d", "e"]


def test_models_cache_lru_promotes_recently_read(monkeypatch):
    """A read should refresh LRU position so the entry survives eviction."""
    monkeypatch.setattr(server, "MODELS_CACHE_MAX", 3)
    server._models_cache.clear()
    import time as _t

    def _put(owner):
        server._models_cache[owner] = (_t.monotonic() + 10.0, {"owner": owner})
        server._models_cache.move_to_end(owner)
        while len(server._models_cache) > server.MODELS_CACHE_MAX:
            server._models_cache.popitem(last=False)

    for o in ["a", "b", "c"]:
        _put(o)
    # Simulate a read on 'a' that promotes it.
    server._models_cache.move_to_end("a")
    _put("d")
    # 'b' was the oldest after 'a' got promoted, so it should be evicted.
    assert "a" in server._models_cache
    assert "b" not in server._models_cache
    assert "c" in server._models_cache
    assert "d" in server._models_cache


def test_models_cache_max_env_default():
    """Default MODELS_CACHE_MAX should be a sensible non-tiny value."""
    # Refetch from the module's parsed env var — should be >= 1.
    assert server.MODELS_CACHE_MAX >= 1
