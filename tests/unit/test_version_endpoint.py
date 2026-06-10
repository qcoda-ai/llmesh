"""
D097 — unauthenticated /version endpoint for post-deploy verification.

/health stays version-less (CVE-targeting concern documented inline). /version
is the separate post-deploy signal operators need to confirm a CI deploy
actually landed without an API key. Tradeoff: codebase is public OSS so
version-enumeration adds no real surface beyond `pip index versions llmesh`.
"""
import os
import re

import pytest
from fastapi.testclient import TestClient

_FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "server_config.json")
os.environ.setdefault("LLMESH_CONFIG_PATH", _FIXTURES)
os.environ.setdefault("LLMESH_ALLOW_SAMPLE_KEYS", "1")

from lib.hub import server  # noqa: E402


@pytest.fixture
def client():
    return TestClient(server.app)


def test_version_unauth_returns_200_with_semver(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body
    assert re.match(r"^\d+\.\d+\.\d+", body["version"]), body


def test_version_matches_module_constant(client):
    resp = client.get("/version")
    assert resp.json()["version"] == server.APP_VERSION


def test_version_no_auth_header_needed(client):
    """No Authorization header; endpoint must still return 200."""
    resp = client.get("/version")
    assert resp.status_code == 200


def test_health_remains_versionless(client):
    """D097 explicitly keeps /health version-free. CVE-targeting concern
    documented inline. Future contributor must not bolt version onto /health."""
    resp = client.get("/health")
    body = resp.json()
    assert body == {"status": "ok"}, "health must stay minimal — see /version for version"
