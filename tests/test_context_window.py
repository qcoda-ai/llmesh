import os
import sys
import time
import uuid
import httpx
import subprocess
import threading

HUB_BASE = "http://127.0.0.1:18766"
API_KEY  = "my_secret_key_1"
MODEL    = "llama3"

def start_hub(env_overrides=None):
    fixtures_cfg = os.path.join(
        os.path.dirname(__file__), "fixtures", "server_config.json"
    )
    env = os.environ.copy()
    env.update({
        "SESSION_DB": ":memory:",
        "LLMESH_API_KEY": API_KEY,
        "LLMESH_CONFIG_PATH": fixtures_cfg,
        "LLMESH_ALLOW_SAMPLE_KEYS": "1",
    })
    if env_overrides:
        env.update(env_overrides)
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "lib.hub.server:app", "--port", "18766", "--log-level", "error"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for hub
    for _ in range(20):
        try:
            r = httpx.get(f"{HUB_BASE}/health", timeout=1)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError("Hub failed to start")

def test_context_window_behavior():
    print("\n[Test] Starting Hub with DEFAULT_CONTEXT_WINDOW=16384...")
    hub = start_hub({"DEFAULT_CONTEXT_WINDOW": "16384"})
    
    try:
        # 1. Register a node with a specific capacity
        print("[Step 1] Registering node with context_size=32768...")
        reg_payload = {
            "api_key": API_KEY,
            "node_fingerprint": "test_node_ctx",
            "resources": {
                "cpu_cores": 8,
                "ram_gb": 32,
                "os_name": "Darwin",
                "ollama_available": True,
                "ollama_models": [MODEL],
                "context_size": 32768
            }
        }
        r = httpx.post(f"{HUB_BASE}/register", json=reg_payload)
        assert r.status_code == 200
        node_token = r.json()["node_token"]
        node_id = r.json()["node_id"]
        auth = {"Authorization": f"Bearer {node_token}"}

        # 2. Check if the hub reports the correctly registered context_size in /api/nodes
        print("[Step 2] Verifying dashboard API reports context_size...")
        r = httpx.get(f"{HUB_BASE}/api/nodes", headers={"Authorization": f"Bearer {API_KEY}"})
        nodes = r.json()
        target = next(n for n in nodes if n["node_id_full"] == "test_node_ctx")
        assert target["context_size"] == 32768, f"Expected 32768, got {target['context_size']}"

        # 3. Create an inference request (without num_ctx) and check if it gets the HUB DEFAULT
        print("[Step 3] Requesting inference (expecting hub default 16384)...")
        req_payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}]
        }
        
        # Fire and don't wait (it will block)
        def fire_inf():
            try:
                httpx.post(f"{HUB_BASE}/v1/chat/completions", json=req_payload, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=2)
            except: pass
        
        threading.Thread(target=fire_inf, daemon=True).start()
        time.sleep(1) # wait for hub to process
        
        pending_r = httpx.get(f"{HUB_BASE}/tasks/{node_id}/pending", headers=auth)
        assert pending_r.status_code == 200
        pending_tasks = pending_r.json()
        assert len(pending_tasks) > 0
        assert pending_tasks[0]["num_ctx"] == 16384, f"Expected default 16384, got {pending_tasks[0]['num_ctx']}"
        print(f"✅ Received task with num_ctx={pending_tasks[0]['num_ctx']}")

        # 4. Create an inference request (WITH num_ctx) and check if it OVERRIDES
        print("[Step 4] Requesting inference with explicit num_ctx=4096 (override)...")
        req_payload_override = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "num_ctx": 4096
        }
        
        threading.Thread(target=lambda: httpx.post(f"{HUB_BASE}/v1/chat/completions", json=req_payload_override, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=1), daemon=True).start()
        time.sleep(1)
        
        pending_r = httpx.get(f"{HUB_BASE}/tasks/{node_id}/pending", headers=auth)
        pending_tasks = pending_r.json()
        assert len(pending_tasks) > 0
        assert pending_tasks[0]["num_ctx"] == 4096, f"Expected override 4096, got {pending_tasks[0]['num_ctx']}"
        print(f"✅ Received task with override num_ctx={pending_tasks[0]['num_ctx']}")

        print("\n🎉 All context window tests passed!")

    finally:
        hub.kill()

if __name__ == "__main__":
    try:
        test_context_window_behavior()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
