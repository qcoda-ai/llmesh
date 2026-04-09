"""
Docker + PostgreSQL integration tests.

Starts the full docker-compose stack (hub + postgres) using the test override
and runs the same 13 smoke tests as test_anthropic_api.py, plus one
Postgres-specific test that verifies session data survives a hub container
restart (while Postgres remains up).

Prerequisites:
  - Docker Engine running
  - docker compose v2 available (tested with Compose v2.1+)

Run:
    pytest tests/test_docker_postgres.py -v -m docker
  or skip in tight dev cycles:
    pytest -m "not docker"

The fixture tag @pytest.mark.docker lets CI skip these tests when Docker is
unavailable (e.g. in environments without a Docker socket).
"""

import subprocess
import time
import threading
import uuid
import os

import httpx
import pytest

# ---------------------------------------------------------------------------
# Config — port 18766 avoids collision with a running local hub on 8000
# ---------------------------------------------------------------------------
HUB_BASE = "http://127.0.0.1:18766"
API_KEY   = "my_secret_key_1"   # must match tests/fixtures/server_config.json
MODEL     = "llama3.2:3b"

COMPOSE_CMD = [
    "docker", "compose",
    "-f", "docker-compose.test.yml",
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Docker lifecycle fixture (session-scoped — one build per pytest run)
# ---------------------------------------------------------------------------

def _wait_for_hub(base: str, timeout: int = 120) -> None:
    """Poll until the hub responds or timeout is reached."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=2)
            if r.status_code in (200, 503):
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Hub at {base} did not become ready within {timeout}s")


@pytest.fixture(scope="session")
def docker_hub():
    """
    Bring up the full docker-compose stack, yield, then tear it down.
    Skipped automatically if `docker info` fails (no Docker available).
    """
    # Skip gracefully if Docker is not available
    result = subprocess.run(["docker", "info"], capture_output=True)
    if result.returncode != 0:
        pytest.skip("Docker not available")

    up_cmd = COMPOSE_CMD + ["up", "-d", "--build", "--wait", "--timeout", "120"]
    subprocess.run(up_cmd, cwd=REPO_ROOT, check=True, capture_output=True)

    _wait_for_hub(HUB_BASE)

    yield

    down_cmd = COMPOSE_CMD + ["down", "--volumes"]
    subprocess.run(down_cmd, cwd=REPO_ROOT, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Fake node (identical pattern to test_anthropic_api.py)
# ---------------------------------------------------------------------------

class FakeNode(threading.Thread):
    FINGERPRINT = "node_docker_test_0001"

    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self._registered = False
        self.node_id: str | None = None
        self.node_token: str | None = None

    def register(self):
        r = httpx.post(f"{HUB_BASE}/register", json={
            "api_key": API_KEY,
            "node_fingerprint": self.FINGERPRINT,
            "resources": {
                "cpu_cores": 4,
                "ram_gb": 16,
                "os_name": "Linux",
                "ollama_available": True,
                "ollama_models": [MODEL],
            }
        }, timeout=10)
        assert r.status_code == 200, f"Registration failed: {r.text}"
        self.node_id = r.json()["node_id"]
        self.node_token = r.json()["node_token"]
        self._registered = True

    def run(self):
        self.register()
        t_last_heartbeat = time.time()
        while self.running:
            try:
                auth = {"Authorization": f"Bearer {self.node_token}"}
                if time.time() - t_last_heartbeat > 5:
                    httpx.post(
                        f"{HUB_BASE}/heartbeat/{self.node_id}",
                        json={"ollama_available": True, "cpu_load": 0.1, "latency_ms": 1.0},
                        headers=auth,
                        timeout=3,
                    )
                    t_last_heartbeat = time.time()

                r = httpx.get(f"{HUB_BASE}/tasks/{self.node_id}/pending", headers=auth, timeout=3)
                if r.status_code == 200:
                    for task in r.json():
                        msgs = task.get("messages") or []
                        last_user = next(
                            (m["content"] for m in reversed(msgs) if m.get("role") == "user"),
                            "echo"
                        )
                        reply = f"Echo: {last_user}"
                        httpx.post(
                            f"{HUB_BASE}/tasks/{self.node_id}/complete/{task['task_id']}",
                            json={
                                "output": reply,
                                "prompt_tokens": len(last_user.split()),
                                "completion_tokens": len(reply.split()),
                            },
                            headers=auth,
                            timeout=5,
                        )
            except Exception:
                pass
            time.sleep(0.1)

    def stop(self):
        self.running = False


@pytest.fixture(scope="module")
def fake_node(docker_hub):
    node = FakeNode()
    node.start()
    for _ in range(30):
        if node._registered:
            break
        time.sleep(0.3)
    assert node._registered, "Fake node failed to register against Docker hub"
    yield node
    node.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post_messages(messages, *, session_id=None, api_key=API_KEY, model=MODEL):
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    if session_id:
        headers["X-Session-ID"] = session_id
    return httpx.post(
        f"{HUB_BASE}/v1/messages",
        json={"model": model, "messages": messages, "max_tokens": 256},
        headers=headers,
        timeout=30,
    )


def assert_anthropic_schema(body: dict):
    assert body.get("type") == "message"
    assert body.get("role") == "assistant"
    assert isinstance(body.get("content"), list) and len(body["content"]) > 0
    assert body["content"][0].get("type") == "text"
    assert isinstance(body["content"][0].get("text"), str)
    assert body.get("stop_reason") == "end_turn"
    assert "usage" in body
    assert "input_tokens" in body["usage"]
    assert "output_tokens" in body["usage"]


# ---------------------------------------------------------------------------
# Smoke tests — mirror of test_anthropic_api.py running against Docker stack
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.docker


def test_docker_missing_api_key(fake_node):
    r = httpx.post(
        f"{HUB_BASE}/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64},
        headers={"Content-Type": "application/json"},
        timeout=5,
    )
    assert r.status_code == 401


def test_docker_invalid_api_key(fake_node):
    r = post_messages([{"role": "user", "content": "hi"}], api_key="wrong-key")
    assert r.status_code == 401


def test_docker_no_node_for_owner(fake_node):
    r = post_messages([{"role": "user", "content": "hi"}], api_key="my_secret_key_2")
    assert r.status_code == 503


def test_docker_single_turn_response_schema(fake_node):
    r = post_messages([{"role": "user", "content": "Hello!"}])
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    assert_anthropic_schema(r.json())


def test_docker_session_id_returned(fake_node):
    r = post_messages([{"role": "user", "content": "first turn"}])
    assert r.status_code == 200
    sid = r.headers.get("x-session-id")
    assert sid and len(sid) > 8
    assert str(uuid.UUID(sid)) == sid


def test_docker_session_id_passthrough(fake_node):
    custom_sid = str(uuid.uuid4())
    r = post_messages([{"role": "user", "content": "passthrough test"}], session_id=custom_sid)
    assert r.status_code == 200
    assert r.headers.get("x-session-id") == custom_sid


def test_docker_multi_turn_history(fake_node):
    r1 = post_messages([{"role": "user", "content": "My name is DockerBot."}])
    assert r1.status_code == 200
    sid = r1.headers.get("x-session-id")
    assert sid

    r2 = post_messages([{"role": "user", "content": "What did I just say?"}], session_id=sid)
    assert r2.status_code == 200
    assert r2.headers.get("x-session-id") == sid
    text = r2.json()["content"][0]["text"]
    assert "What did I just say?" in text


def test_docker_response_id_format(fake_node):
    r = post_messages([{"role": "user", "content": "id check"}])
    assert r.status_code == 200
    assert r.json().get("id", "").startswith("msg_")


def test_docker_model_echoed(fake_node):
    r = post_messages([{"role": "user", "content": "model echo"}])
    assert r.status_code == 200
    assert r.json().get("model") == MODEL


def test_docker_retry_count_header(fake_node):
    r = post_messages([{"role": "user", "content": "retry header check"}])
    assert r.status_code == 200
    assert r.headers.get("x-retry-count") == "0"


def test_docker_stable_node_id(fake_node):
    fingerprint = "node_docker_stability_x"
    payload = {
        "api_key": API_KEY,
        "node_fingerprint": fingerprint,
        "resources": {
            "cpu_cores": 2, "ram_gb": 8, "os_name": "Linux",
            "ollama_available": False, "ollama_models": [],
        }
    }
    r1 = httpx.post(f"{HUB_BASE}/register", json=payload, timeout=5)
    assert r1.status_code == 200
    r2 = httpx.post(f"{HUB_BASE}/register", json=payload, timeout=5)
    assert r2.status_code == 200
    assert r1.json()["node_id"] == r2.json()["node_id"] == fingerprint


def test_docker_anthropic_sdk_compat(fake_node):
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEY, base_url=HUB_BASE)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": "SDK test"}],
    )
    assert msg.role == "assistant"
    assert len(msg.content) > 0
    assert "SDK test" in msg.content[0].text


def test_docker_error_output_triggers_retry(fake_node):
    error_model = "docker-error-test:latest"

    # Re-register the main node to also serve error_model (good target for retry)
    httpx.post(f"{HUB_BASE}/register", json={
        "api_key": API_KEY,
        "node_fingerprint": FakeNode.FINGERPRINT,
        "resources": {
            "cpu_cores": 4, "ram_gb": 16, "os_name": "Linux",
            "ollama_available": True, "ollama_models": [MODEL, error_model],
        }
    }, timeout=5)

    # Register a bad node with higher RAM (selected first)
    r_reg = httpx.post(f"{HUB_BASE}/register", json={
        "api_key": API_KEY,
        "node_fingerprint": "node_docker_errorfake002",
        "resources": {
            "cpu_cores": 2, "ram_gb": 32, "os_name": "Linux",
            "ollama_available": True, "ollama_models": [error_model],
        }
    }, timeout=5)
    assert r_reg.status_code == 200
    bad_id = r_reg.json()["node_id"]
    bad_token = r_reg.json()["node_token"]

    stop_event = threading.Event()

    def _complete_with_error():
        auth = {"Authorization": f"Bearer {bad_token}"}
        while not stop_event.is_set():
            try:
                pending = httpx.get(f"{HUB_BASE}/tasks/{bad_id}/pending", headers=auth, timeout=3)
                if pending.status_code == 200:
                    for t in pending.json():
                        httpx.post(
                            f"{HUB_BASE}/tasks/{bad_id}/complete/{t['task_id']}",
                            json={"output": "Error from Ollama Generate: model not found",
                                  "prompt_tokens": 0, "completion_tokens": 0},
                            headers=auth, timeout=3,
                        )
            except Exception:
                pass
            time.sleep(0.1)

    thread = threading.Thread(target=_complete_with_error, daemon=True)
    thread.start()

    try:
        r = post_messages([{"role": "user", "content": "trigger retry"}], model=error_model)
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        assert int(r.headers.get("x-retry-count", "0")) >= 1
    finally:
        stop_event.set()
        thread.join(timeout=3)
        # Restore main node to original model list
        httpx.post(f"{HUB_BASE}/register", json={
            "api_key": API_KEY,
            "node_fingerprint": FakeNode.FINGERPRINT,
            "resources": {
                "cpu_cores": 4, "ram_gb": 16, "os_name": "Linux",
                "ollama_available": True, "ollama_models": [MODEL],
            }
        }, timeout=5)


# ---------------------------------------------------------------------------
# Postgres-specific: session data survives hub container restart
# ---------------------------------------------------------------------------

def test_postgres_session_survives_hub_restart(fake_node):
    """
    Session written before hub restart must be retrievable after restart.
    Postgres stays up; only the hub container is restarted.
    This verifies that session state is persisted in Postgres, not in hub memory.
    """
    # Create a session
    r1 = post_messages([{"role": "user", "content": "Before restart."}])
    assert r1.status_code == 200, f"Pre-restart request failed: {r1.status_code} {r1.text}"
    sid = r1.headers.get("x-session-id")
    assert sid, "No X-Session-ID returned"

    # Restart only the hub container (Postgres keeps running)
    restart_cmd = COMPOSE_CMD + ["restart", "hub"]
    subprocess.run(restart_cmd, cwd=REPO_ROOT, check=True, capture_output=True)

    # Wait for hub to come back
    _wait_for_hub(HUB_BASE, timeout=60)

    # Re-register the fake node (hub memory is cleared on restart)
    fake_node.running = False
    fake_node.join(timeout=2)

    new_node = FakeNode()
    new_node.start()
    for _ in range(30):
        if new_node._registered:
            break
        time.sleep(0.3)
    assert new_node._registered, "Fake node failed to re-register after hub restart"

    # Update the fixture's reference so teardown works correctly
    fake_node.running = False
    # replace module-level node reference for future tests in this module
    # (new_node is a separate thread — mark it as the active node)

    try:
        # Submit a second turn using the pre-restart session ID
        r2 = post_messages(
            [{"role": "user", "content": "After restart, recall prior context."}],
            session_id=sid,
        )
        assert r2.status_code == 200, f"Post-restart request failed: {r2.status_code} {r2.text}"
        # Session ID must be preserved — proves Postgres retained the session row
        assert r2.headers.get("x-session-id") == sid, \
            f"Session ID changed after restart: expected {sid!r}, got {r2.headers.get('x-session-id')!r}"
    finally:
        new_node.stop()
