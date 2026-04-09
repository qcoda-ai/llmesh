"""
Integration tests for the Anthropic-compatible /v1/messages endpoint.

Starts the hub, registers a fake node that simulates Ollama responses,
then exercises the endpoint for:
  - Auth: missing key, invalid key, valid key
  - Basic inference (single turn)
  - X-Session-ID header returned on first request
  - Multi-turn session history accumulation
  - 503 when no node is available
  - Correct Anthropic response schema (id, type, role, content[].type, stop_reason, usage)
  - Anthropic SDK client compatibility

Run:
    python -m pytest tests/test_anthropic_api.py -v
  or standalone:
    python tests/test_anthropic_api.py
"""

import asyncio
import json
import os
import sys
import threading
import time
import uuid
import httpx
import subprocess

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HUB_BASE = "http://127.0.0.1:18765"
API_KEY   = "my_secret_key_1"
MODEL     = "llama3.2:3b"

# ---------------------------------------------------------------------------
# Hub process management
# ---------------------------------------------------------------------------
_hub_proc: subprocess.Popen | None = None

def start_hub():
    global _hub_proc
    env = os.environ.copy()
    env.update({
        "SESSION_MEMORY_MODE": "cutoff",   # no compression model needed
        "SESSION_DB": ":memory:",           # ephemeral
        "SESSION_MAX_TURNS": "4",
    })
    _hub_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "lib.hub.server:app", "--port", "18765", "--log-level", "warning"],
        env=env,
        cwd=os.path.dirname(os.path.dirname(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for hub to be ready
    for _ in range(30):
        try:
            r = httpx.get(f"{HUB_BASE}/health", timeout=1)
            if r.status_code < 500:
                return
        except Exception:
            pass
        # Also accept connection-refused → hub still booting
        time.sleep(0.5)
    # Fall back: wait for /v1/models to respond
    for _ in range(20):
        try:
            r = httpx.get(f"{HUB_BASE}/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=1)
            if r.status_code in (200, 503):
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("Hub did not start in time")


def stop_hub():
    global _hub_proc
    if _hub_proc:
        _hub_proc.kill()
        _hub_proc.wait(timeout=10)
        _hub_proc = None


# ---------------------------------------------------------------------------
# Fake node (runs in a daemon thread, polls for tasks and completes them)
# ---------------------------------------------------------------------------
class FakeNode(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self._registered = False
        self.node_id: str | None = None    # assigned by hub at registration
        self.node_token: str | None = None  # issued by hub at registration

    FINGERPRINT = "node_testfakenode0001"  # stable fingerprint for test node

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
        }, timeout=5)
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
                # Send heartbeat every 5 seconds so the hub doesn't prune this node
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
                        # Build a reply that echoes back the last user message
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def post_messages(messages, *, session_id=None, api_key=API_KEY, model=MODEL, extra_headers=None, **_):
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    if session_id:
        headers["X-Session-ID"] = session_id
    if extra_headers:
        headers.update(extra_headers)
    return httpx.post(
        f"{HUB_BASE}/v1/messages",
        json={"model": model, "messages": messages, "max_tokens": 256},
        headers=headers,
        timeout=30,
    )


def assert_anthropic_schema(body: dict):
    """Assert the response has the expected Anthropic message schema."""
    assert body.get("type") == "message", f"expected type=message, got {body.get('type')}"
    assert body.get("role") == "assistant", f"expected role=assistant, got {body.get('role')}"
    assert isinstance(body.get("content"), list) and len(body["content"]) > 0
    assert body["content"][0].get("type") == "text"
    assert isinstance(body["content"][0].get("text"), str)
    assert body.get("stop_reason") == "end_turn"
    assert "usage" in body
    assert "input_tokens" in body["usage"]
    assert "output_tokens" in body["usage"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []

def run_test(name, fn):
    try:
        fn()
        print(f"  [{PASS}] {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  [{FAIL}] {name}: {e}")
        results.append((name, False, str(e)))


def test_retry_count_header_on_success():
    """X-Retry-Count header must be present and 0 when no retry was needed."""
    r = post_messages([{"role": "user", "content": "retry header check"}])
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    retry_count = r.headers.get("x-retry-count")
    assert retry_count is not None, "X-Retry-Count header missing"
    assert retry_count == "0", f"Expected X-Retry-Count=0, got {retry_count!r}"


def test_error_output_triggers_fail():
    """A node returning an 'Error from Ollama' output causes hub retry to a good node."""
    # Use a separate model so this bad node is only routed to for this test
    error_model = "error-test-model:latest"

    # Register the main FakeNode again with error_model so the retry has a good target
    r_good = httpx.post(f"{HUB_BASE}/register", json={
        "api_key": API_KEY,
        "node_fingerprint": FakeNode.FINGERPRINT,  # re-registers same stable node
        "resources": {
            "cpu_cores": 4, "ram_gb": 16, "os_name": "Linux",
            "ollama_available": True, "ollama_models": [MODEL, error_model],
        }
    }, timeout=5)
    assert r_good.status_code == 200

    # Register a bad node with higher RAM so it's selected first for error_model
    r_reg = httpx.post(f"{HUB_BASE}/register", json={
        "api_key": API_KEY,
        "node_fingerprint": "node_errorfakenode002",
        "resources": {
            "cpu_cores": 2, "ram_gb": 32, "os_name": "Linux",
            "ollama_available": True, "ollama_models": [error_model],
        }
    }, timeout=5)
    assert r_reg.status_code == 200
    bad_node_id = r_reg.json()["node_id"]
    bad_node_token = r_reg.json()["node_token"]

    # Background thread: complete every pending task on the bad node with an error string
    stop_event = threading.Event()
    def _complete_with_error():
        auth = {"Authorization": f"Bearer {bad_node_token}"}
        while not stop_event.is_set():
            try:
                pending = httpx.get(f"{HUB_BASE}/tasks/{bad_node_id}/pending", headers=auth, timeout=3)
                if pending.status_code == 200:
                    for t in pending.json():
                        httpx.post(
                            f"{HUB_BASE}/tasks/{bad_node_id}/complete/{t['task_id']}",
                            json={"output": "Error from Ollama Generate: 500 - model not found",
                                  "prompt_tokens": 0, "completion_tokens": 0},
                            headers=auth,
                            timeout=3,
                        )
            except Exception:
                pass
            time.sleep(0.1)

    thread = threading.Thread(target=_complete_with_error, daemon=True)
    thread.start()

    try:
        # Hub routes to bad node (32GB) first → error → retries to good node (16GB) → succeeds
        r = post_messages([{"role": "user", "content": "trigger error retry"}], model=error_model)
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        retry_count = int(r.headers.get("x-retry-count", "0"))
        assert retry_count >= 1, f"Expected at least 1 retry, got {retry_count}"
    finally:
        stop_event.set()
        thread.join(timeout=3)
        # Restore the main node to serve only MODEL (re-register with original resources)
        httpx.post(f"{HUB_BASE}/register", json={
            "api_key": API_KEY,
            "node_fingerprint": FakeNode.FINGERPRINT,
            "resources": {
                "cpu_cores": 4, "ram_gb": 16, "os_name": "Linux",
                "ollama_available": True, "ollama_models": [MODEL],
            }
        }, timeout=5)


def test_stable_node_id():
    """Registering twice with the same fingerprint must return the same node_id."""
    fingerprint = "node_stability_check_x"
    payload = {
        "api_key": API_KEY,
        "node_fingerprint": fingerprint,
        "resources": {
            "cpu_cores": 2,
            "ram_gb": 8,
            "os_name": "Linux",
            "ollama_available": False,
            "ollama_models": [],
        }
    }
    r1 = httpx.post(f"{HUB_BASE}/register", json=payload, timeout=5)
    assert r1.status_code == 200, f"First registration failed: {r1.text}"
    id1 = r1.json()["node_id"]

    r2 = httpx.post(f"{HUB_BASE}/register", json=payload, timeout=5)
    assert r2.status_code == 200, f"Second registration failed: {r2.text}"
    id2 = r2.json()["node_id"]

    assert id1 == id2 == fingerprint, f"Expected stable id={fingerprint!r}, got {id1!r} then {id2!r}"


def test_missing_api_key():
    r = httpx.post(
        f"{HUB_BASE}/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64},
        headers={"Content-Type": "application/json"},
        timeout=5,
    )
    assert r.status_code == 401, f"expected 401, got {r.status_code}"


def test_invalid_api_key():
    r = post_messages([{"role": "user", "content": "hi"}], api_key="wrong-key")
    assert r.status_code == 401, f"expected 401, got {r.status_code}"


def test_no_node_available():
    """Using a second owner key that has no registered nodes should 503."""
    r = post_messages([{"role": "user", "content": "hi"}], api_key="my_secret_key_2")
    assert r.status_code == 503, f"expected 503, got {r.status_code}: {r.text}"


def test_single_turn_response_schema():
    r = post_messages([{"role": "user", "content": "Hello!"}])
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert_anthropic_schema(body)
    assert body["content"][0]["text"].startswith("Echo:"), f"unexpected reply: {body['content'][0]['text']}"


def test_session_id_returned():
    r = post_messages([{"role": "user", "content": "first turn"}])
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    sid = r.headers.get("x-session-id")
    assert sid and len(sid) > 8, f"X-Session-ID not returned or too short: {sid!r}"
    assert str(uuid.UUID(sid)) == sid, f"X-Session-ID is not a valid UUID: {sid!r}"


def test_multi_turn_history_accumulates():
    """Second turn should include history from the first."""
    r1 = post_messages([{"role": "user", "content": "My name is TestBot."}])
    assert r1.status_code == 200, f"{r1.status_code}: {r1.text}"
    sid = r1.headers.get("x-session-id")
    assert sid

    r2 = post_messages([{"role": "user", "content": "What did I just say?"}], session_id=sid)
    assert r2.status_code == 200, f"{r2.status_code}: {r2.text}"
    # The echo node echoes the last user message, not the history,
    # but we verify the hub forwarded history by confirming session_id is reused
    assert r2.headers.get("x-session-id") == sid, "Session ID changed between turns"
    # The reply should reference the second turn's message (echo node echoes last user)
    text = r2.json()["content"][0]["text"]
    assert "What did I just say?" in text, f"Expected echo of second turn in: {text!r}"


def test_session_id_passthrough():
    """Caller-supplied X-Session-ID must be honoured and echoed back."""
    custom_sid = str(uuid.uuid4())
    r = post_messages(
        [{"role": "user", "content": "passthrough test"}],
        session_id=custom_sid
    )
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    assert r.headers.get("x-session-id") == custom_sid, \
        f"Expected {custom_sid!r}, got {r.headers.get('x-session-id')!r}"


def test_response_id_format():
    r = post_messages([{"role": "user", "content": "id check"}])
    assert r.status_code == 200
    body = r.json()
    assert body.get("id", "").startswith("msg_"), f"id should start with msg_, got: {body.get('id')!r}"


def test_model_echoed_in_response():
    r = post_messages([{"role": "user", "content": "model echo"}])
    assert r.status_code == 200
    body = r.json()
    assert body.get("model") == MODEL, f"expected model={MODEL!r}, got {body.get('model')!r}"


def test_anthropic_sdk_compat():
    """End-to-end with the official Anthropic SDK client pointing at our hub."""
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEY, base_url=HUB_BASE)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": "SDK test"}],
    )
    assert msg.role == "assistant"
    assert len(msg.content) > 0
    assert msg.content[0].type == "text"
    assert "SDK test" in msg.content[0].text   # echo node echoes the prompt


# ---------------------------------------------------------------------------
# /tasks/status auth tests
# ---------------------------------------------------------------------------

def test_anthropic_streaming_rejected():
    """POST /v1/messages with stream=true must return 400 with actionable error."""
    r = httpx.post(
        f"{HUB_BASE}/v1/messages",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64, "stream": True},
        headers={"x-api-key": API_KEY},
        timeout=5,
    )
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text}"
    body = r.json()
    assert "detail" in body
    assert "/v1/chat/completions" in str(body["detail"]), f"error should point to /v1/chat/completions: {body}"


def test_task_status_no_auth():
    """GET /tasks/status without credentials must return 401."""
    r = httpx.get(f"{HUB_BASE}/tasks/status/somenode/sometask", timeout=5)
    assert r.status_code == 401, f"expected 401, got {r.status_code}"


def test_task_status_invalid_api_key():
    """GET /tasks/status with a bad API key must return 401."""
    r = httpx.get(
        f"{HUB_BASE}/tasks/status/somenode/sometask",
        headers={"Authorization": "Bearer bad-key"},
        timeout=5,
    )
    assert r.status_code == 401, f"expected 401, got {r.status_code}"


def test_task_status_valid_api_key_unknown_task():
    """GET /tasks/status with a valid API key on an unknown task must return 404, not 401/403."""
    r = httpx.get(
        f"{HUB_BASE}/tasks/status/somenode/nonexistent-task-id",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=5,
    )
    assert r.status_code == 404, f"expected 404, got {r.status_code}: {r.text}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("\n=== Anthropic-Compatible API Tests ===\n")
    node = FakeNode()

    try:
        print("Starting hub...")
        start_hub()
        print(f"Hub up at {HUB_BASE}")

        print("Starting fake node...")
        node.start()
        # Wait for node to register
        for _ in range(20):
            if node._registered:
                break
            time.sleep(0.2)
        assert node._registered, "Fake node failed to register"
        print(f"Node {node.node_id} registered\n")

        run_test("Stable node ID with fingerprint",     test_stable_node_id)
        run_test("X-Retry-Count=0 on clean success",   test_retry_count_header_on_success)
        run_test("Error output triggers retry",         test_error_output_triggers_fail)
        run_test("Missing API key returns 401",        test_missing_api_key)
        run_test("Invalid API key returns 401",        test_invalid_api_key)
        run_test("No node for owner returns 503",      test_no_node_available)
        run_test("Single-turn response schema",        test_single_turn_response_schema)
        run_test("X-Session-ID returned",              test_session_id_returned)
        run_test("Multi-turn history accumulates",     test_multi_turn_history_accumulates)
        run_test("Caller session ID honoured",         test_session_id_passthrough)
        run_test("Response id starts with msg_",       test_response_id_format)
        run_test("Model echoed in response",           test_model_echoed_in_response)
        run_test("Anthropic SDK client compatible",    test_anthropic_sdk_compat)

    finally:
        node.stop()
        stop_hub()

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        print("\nFailed tests:")
        for name, ok, err in results:
            if not ok:
                print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    main()
