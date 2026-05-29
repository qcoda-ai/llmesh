"""
CSRF (D055) — double-submit-cookie protection on `/login` and
`/dashboard/request_inference`.
"""
import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Provide a config path BEFORE importing the hub module so config.py does not
# warn-log on import. tests/fixtures/server_config.json carries publicly known
# sample keys (D013), so the sample-key guard must be bypassed.
_FIXTURES = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "server_config.json"
)
os.environ.setdefault("LLMESH_CONFIG_PATH", _FIXTURES)
os.environ.setdefault("LLMESH_ALLOW_SAMPLE_KEYS", "1")

from lib.hub import server  # noqa: E402


# --- helper unit tests ---


def test_require_csrf_matched_returns_none():
    assert server._require_csrf("abc", "abc") is None


def test_require_csrf_mismatched_raises_403():
    with pytest.raises(HTTPException) as exc:
        server._require_csrf("abc", "xyz")
    assert exc.value.status_code == 403


def test_require_csrf_missing_form_raises_403():
    with pytest.raises(HTTPException) as exc:
        server._require_csrf(None, "abc")
    assert exc.value.status_code == 403


def test_require_csrf_missing_cookie_raises_403():
    with pytest.raises(HTTPException) as exc:
        server._require_csrf("abc", None)
    assert exc.value.status_code == 403


def test_require_csrf_empty_string_raises_403():
    with pytest.raises(HTTPException) as exc:
        server._require_csrf("", "")
    assert exc.value.status_code == 403


# --- TestClient flows ---


@pytest.fixture
def client():
    return TestClient(server.app)


def test_get_login_sets_csrf_cookie_and_embeds_value(client):
    response = client.get("/login")
    assert response.status_code == 200
    assert server.CSRF_COOKIE_NAME in response.cookies
    cookie_val = response.cookies[server.CSRF_COOKIE_NAME]
    # Hidden input rendered with the same value
    assert f'name="csrf_token" value="{cookie_val}"' in response.text


def test_post_login_without_csrf_token_field_fails():
    """No form field at all → FastAPI returns 422 (missing required form
    field) BEFORE the handler runs. Either 422 or 403 is acceptable; both
    block the request. Use a fresh client to avoid carry-over cookies."""
    fresh = TestClient(server.app)
    response = fresh.post("/login", data={"api_key": "my_secret_key_1"})
    assert response.status_code in (403, 422)


def test_post_login_with_token_but_no_cookie_fails(client):
    """Form token present but no cookie — server cannot validate."""
    # Do NOT call GET first so no cookie is set.
    response = client.post(
        "/login",
        data={"api_key": "my_secret_key_1", "csrf_token": "any-value"},
    )
    assert response.status_code == 403


def test_post_login_mismatched_token_fails(client):
    """Cookie + form value disagree → 403."""
    client.get("/login")  # establish cookie
    cookie_val = client.cookies.get(server.CSRF_COOKIE_NAME)
    assert cookie_val
    response = client.post(
        "/login",
        data={"api_key": "my_secret_key_1", "csrf_token": "different-value"},
    )
    assert response.status_code == 403


def test_post_login_matched_token_proceeds(client):
    """Matching cookie + form token clears CSRF; auth proceeds. The hub
    sets the CSRF cookie with `Secure`, which httpx will not echo back
    over the http://testserver scheme — set the cookie directly on the
    client to simulate a real browser."""
    get_resp = client.get("/login")
    cookie_val = get_resp.cookies[server.CSRF_COOKIE_NAME]
    client.cookies.set(server.CSRF_COOKIE_NAME, cookie_val)
    response = client.post(
        "/login",
        data={"api_key": "my_secret_key_1", "csrf_token": cookie_val},
        follow_redirects=False,
    )
    # 303 redirect to /dashboard on success — proves CSRF passed AND auth
    # passed. The point is no 403.
    assert response.status_code != 403


def test_post_dashboard_inference_without_csrf_blocked():
    """Pre-seed a session so we know any 403 is from CSRF, not auth."""
    fresh = TestClient(server.app)
    test_owner = "test-owner"
    test_session = "test-session-token-csrf"
    server._session_tokens[test_session] = test_owner
    try:
        fresh.cookies.set(server.SESSION_COOKIE_NAME, test_session)
        response = fresh.post(
            "/dashboard/request_inference",
            data={"prompt": "hi", "model": "llama3"},
            follow_redirects=False,
        )
        # 403 from CSRF guard, or 422 from missing required form field.
        # Either blocks; the bug we're guarding against (no protection)
        # would yield 200 / redirect.
        assert response.status_code in (403, 422)
    finally:
        server._session_tokens.pop(test_session, None)
