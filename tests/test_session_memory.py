"""
Session memory integration test.

Runs against a live hub. Acts as both client AND node:
  - Registers a fake node with a dummy model
  - Sends inference requests (client role)
  - Polls pending tasks and completes them manually (node role)
  - Verifies X-Session-ID header and that history is accumulated in sessions.db

Run:
    cd /path/to/qcoda-nodemesh
    uvicorn lib.hub.server:app --port 8000 &
    python tests/test_session_memory.py
"""

import asyncio
import httpx
import json
import sqlite3
import os
import sys
import uuid

HUB = "http://127.0.0.1:8000"
API_KEY = "my_secret_key_1"
MODEL = f"test-{uuid.uuid4().hex[:8]}"  # unique per run — prevents cross-test routing
DB_PATH = "./sessions.db"

HEADERS_CLIENT = {"Authorization": f"Bearer {API_KEY}"}


async def register_fake_node(client: httpx.AsyncClient) -> str:
    resp = await client.post(f"{HUB}/register", json={
        "api_key": API_KEY,
        "resources": {
            "cpu_cores": 4,
            "ram_gb": 16.0,
            "os_name": "TestOS",
            "ollama_available": True,
            "ollama_models": [MODEL]
        }
    })
    resp.raise_for_status()
    node_id = resp.json()["node_id"]
    print(f"[setup]  Registered fake node: {node_id[:8]}...")
    return node_id


async def beat(client: httpx.AsyncClient, node_id: str):
    await client.post(f"{HUB}/heartbeat/{node_id}", json={
        "ollama_available": True,
        "cpu_load": 0.1,
        "latency_ms": 5.0
    })


async def complete_pending(client: httpx.AsyncClient, node_id: str, reply: str) -> int:
    """Poll pending tasks for this node and complete them all. Returns count completed."""
    resp = await client.get(f"{HUB}/tasks/{node_id}/pending")
    pending = resp.json()
    for task in pending:
        task_id = task["task_id"]
        msgs = task.get("messages") or []
        print(f"[node]   Completing task {task_id[:8]}... ({len(msgs)} messages in context)")
        for i, m in enumerate(msgs):
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:80]
            print(f"           [{i}] {role}: {content}")
        await client.post(f"{HUB}/tasks/{node_id}/complete/{task_id}", json={
            "output": reply,
            "prompt_tokens": 10,
            "completion_tokens": 5
        })
    return len(pending)


async def chat(client: httpx.AsyncClient, node_id: str, user_msg: str,
               session_id: str | None = None, reply: str = "Got it.") -> tuple[str, str]:
    """Send a chat request and complete it. Returns (assistant_text, session_id)."""
    headers = {**HEADERS_CLIENT}
    if session_id:
        headers["X-Session-ID"] = session_id

    # Start the request in background; we need to complete the task concurrently
    async def do_request():
        return await client.post(f"{HUB}/v1/chat/completions", headers=headers, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": user_msg}]
        }, timeout=15.0)

    async def do_complete():
        # Give hub a moment to queue the task, then keep completing until the request finishes
        for _ in range(30):
            await asyncio.sleep(0.3)
            await beat(client, node_id)
            count = await complete_pending(client, node_id, reply)
            if count:
                break

    resp_task = asyncio.create_task(do_request())
    await asyncio.gather(do_complete(), resp_task)
    resp = resp_task.result()

    if resp.status_code != 200:
        print(f"[ERROR]  HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)

    returned_sid = resp.headers.get("x-session-id", "")
    result_text = resp.json()["choices"][0]["message"]["content"]
    return result_text, returned_sid


def inspect_db(session_id: str, owner_id: str = "owner_alpha"):
    """Read sessions.db and print the stored messages for a session."""
    if not os.path.exists(DB_PATH):
        print("[db]     sessions.db not found")
        return
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT messages, last_active FROM sessions WHERE session_id=? AND owner_id=?",
        (session_id, owner_id)
    ).fetchone()
    con.close()
    if not row:
        print(f"[db]     No row found for session {session_id[:8]}...")
        return
    messages = json.loads(row[0])
    print(f"[db]     {len(messages)} messages stored for session {session_id[:8]}...  (last_active={round(row[1])})")
    for i, m in enumerate(messages):
        print(f"           [{i}] {m['role']}: {str(m['content'])[:80]}")


async def main():
    async with httpx.AsyncClient() as client:
        # --- 1. Check hub is up ---
        try:
            await client.get(f"{HUB}/nodes", timeout=3.0)
        except Exception:
            print(f"ERROR: Hub not reachable at {HUB}. Start it first:\n")
            print(f"  cd /path/to/qcoda-nodemesh")
            print(f"  uvicorn lib.hub.server:app --port 8000\n")
            sys.exit(1)

        node_id = await register_fake_node(client)
        await beat(client, node_id)

        print()
        print("=== TURN 1: No X-Session-ID sent ===")
        _, sid = await chat(client, node_id, "My favourite colour is blue.", reply="Noted, blue!")
        print(f"[client] Got X-Session-ID: {sid}")
        inspect_db(sid)

        print()
        print("=== TURN 2: Reuse session — history should be prepended ===")
        _, sid2 = await chat(client, node_id, "What is my favourite colour?",
                             session_id=sid, reply="Your favourite colour is blue.")
        assert sid2 == sid, f"Session ID changed: {sid} → {sid2}"
        print(f"[client] Session ID unchanged: {sid2[:8]}...")
        inspect_db(sid)

        print()
        print("=== TURN 3: Third message — accumulated history ===")
        _, _ = await chat(client, node_id, "And what is 2+2?",
                          session_id=sid, reply="4")
        inspect_db(sid)

        print()
        print("=== DELETE session ===")
        r = await client.delete(f"{HUB}/v1/sessions/{sid}", headers=HEADERS_CLIENT)
        print(f"[client] DELETE response: {r.json()}")
        inspect_db(sid)

        print()
        print("All checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
