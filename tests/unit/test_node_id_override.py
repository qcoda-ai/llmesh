"""Unit tests for D048 — `LLMESH_NODE_ID` env var override of node fingerprint.

When set and valid, the env value is used verbatim as the node ID. When
unset or invalid, the agent falls back to the salted-hash fingerprint
(D021).
"""

import os
import unittest.mock as mock

import pytest

from lib.agent import client as agent_client


# --- valid override ----------------------------------------------------------

def test_node_id_env_unset_falls_back_to_hash():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LLMESH_NODE_ID", None)
        assert agent_client._resolve_operator_node_id() is None


def test_node_id_env_valid_hostname_used():
    with mock.patch.dict(os.environ, {"LLMESH_NODE_ID": "gpu-host-1"}, clear=False):
        assert agent_client._resolve_operator_node_id() == "gpu-host-1"


def test_node_id_env_valid_with_dots():
    with mock.patch.dict(os.environ, {"LLMESH_NODE_ID": "node.prod.us-east-1"}, clear=False):
        assert agent_client._resolve_operator_node_id() == "node.prod.us-east-1"


def test_node_id_env_valid_underscore_separated():
    with mock.patch.dict(os.environ, {"LLMESH_NODE_ID": "rocky_box_07"}, clear=False):
        assert agent_client._resolve_operator_node_id() == "rocky_box_07"


# --- invalid → fallback ------------------------------------------------------

def test_node_id_env_with_slash_rejected(caplog):
    with caplog.at_level("WARNING", logger="llmesh.agent"):
        with mock.patch.dict(os.environ, {"LLMESH_NODE_ID": "bad/value"}, clear=False):
            assert agent_client._resolve_operator_node_id() is None
    msgs = " ".join(r.message for r in caplog.records)
    assert "LLMESH_NODE_ID" in msgs and "ignored" in msgs


def test_node_id_env_with_space_rejected():
    with mock.patch.dict(os.environ, {"LLMESH_NODE_ID": "host 1"}, clear=False):
        assert agent_client._resolve_operator_node_id() is None


def test_node_id_env_starting_with_dash_rejected():
    """Leading non-alphanumeric not allowed (regex anchors first char)."""
    with mock.patch.dict(os.environ, {"LLMESH_NODE_ID": "-host"}, clear=False):
        assert agent_client._resolve_operator_node_id() is None


def test_node_id_env_too_long_rejected():
    long = "a" * 65
    with mock.patch.dict(os.environ, {"LLMESH_NODE_ID": long}, clear=False):
        assert agent_client._resolve_operator_node_id() is None


def test_node_id_env_max_length_accepted():
    sixty_four = "a" * 64
    with mock.patch.dict(os.environ, {"LLMESH_NODE_ID": sixty_four}, clear=False):
        assert agent_client._resolve_operator_node_id() == sixty_four


def test_node_id_env_empty_string_falls_back():
    with mock.patch.dict(os.environ, {"LLMESH_NODE_ID": ""}, clear=False):
        assert agent_client._resolve_operator_node_id() is None


# --- compute_node_fingerprint integration ------------------------------------

def test_compute_node_fingerprint_uses_env_when_set():
    with mock.patch.dict(os.environ, {"LLMESH_NODE_ID": "my-host-42"}, clear=False):
        assert agent_client.compute_node_fingerprint() == "my-host-42"


def test_compute_node_fingerprint_hashes_when_env_unset():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LLMESH_NODE_ID", None)
        result = agent_client.compute_node_fingerprint()
    assert result.startswith("node_")
    assert len(result) == len("node_") + 16


def test_compute_node_fingerprint_hashes_when_env_invalid(caplog):
    with caplog.at_level("WARNING", logger="llmesh.agent"):
        with mock.patch.dict(os.environ, {"LLMESH_NODE_ID": "in valid"}, clear=False):
            result = agent_client.compute_node_fingerprint()
    assert result.startswith("node_")
    msgs = " ".join(r.message for r in caplog.records)
    assert "ignored" in msgs
