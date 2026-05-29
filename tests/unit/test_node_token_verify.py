"""
Unit tests for lib/hub/storage.verify_node_token dual-verify path (D058).

Two acceptance paths must both work and only the right one must accept:
  1. Plaintext compare — normal operation, in-memory node holds plaintext.
  2. Hash compare — restored-from-DB node, plaintext empty, hash populated.
"""
import time

import pytest

from lib.hub import storage
from lib.hub.models import Node, ResourceCaps


def _make_node(node_id="node-A", node_token="", node_token_hash=""):
    return Node(
        node_id=node_id,
        owner_id="alice",
        resources=ResourceCaps(
            cpu_cores=2, ram_gb=8.0, os_name="linux", ollama_available=True,
        ),
        last_seen=time.time(),
        node_token=node_token,
        node_token_hash=node_token_hash,
        fingerprint=node_id,
    )


@pytest.fixture(autouse=True)
def isolate_registry():
    """Each test starts with an empty _nodes dict."""
    storage._nodes.clear()
    yield
    storage._nodes.clear()


def test_plaintext_path_accepts_matching_token():
    node = _make_node(node_token="plain-fixture-tok")
    storage._nodes[node.node_id] = node
    assert storage.verify_node_token("node-A", "plain-fixture-tok") is True


def test_plaintext_path_rejects_wrong_token():
    node = _make_node(node_token="plain-fixture-tok")
    storage._nodes[node.node_id] = node
    assert storage.verify_node_token("node-A", "other-secret") is False


def test_hash_path_accepts_matching_token_after_restore():
    """Restored-from-DB node: plaintext is empty, hash holds sha256(token).
    The running agent still presents the original plaintext; verify must
    hash + compare."""
    token = "the-real-token"
    node = _make_node(node_token="", node_token_hash=storage._hash_token(token))
    storage._nodes[node.node_id] = node
    assert storage.verify_node_token("node-A", token) is True


def test_hash_path_rejects_wrong_token_after_restore():
    token = "the-real-token"
    node = _make_node(node_token="", node_token_hash=storage._hash_token(token))
    storage._nodes[node.node_id] = node
    assert storage.verify_node_token("node-A", "wrong-token") is False


def test_no_credential_returns_false():
    """Both plaintext and hash empty → cannot verify; never accept anything."""
    node = _make_node(node_token="", node_token_hash="")
    storage._nodes[node.node_id] = node
    assert storage.verify_node_token("node-A", "anything") is False
    assert storage.verify_node_token("node-A", "") is False


def test_unknown_node_returns_false():
    assert storage.verify_node_token("does-not-exist", "anything") is False


def test_hash_function_is_sha256_hex():
    """Lock the hash format so cross-process verify keeps working as long as
    every actor uses the same function."""
    import hashlib
    assert storage._hash_token("abc") == hashlib.sha256(b"abc").hexdigest()
