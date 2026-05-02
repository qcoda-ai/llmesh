"""
Session memory performance test.

Measures hub overhead across:
  - No session (baseline)
  - Short sessions (2-4 turns)
  - Medium sessions (10 turns)
  - Long sessions (30 turns, approaching compression threshold)

Run:
    uvicorn lib.hub.server:app --port 8000 &
    python tests/test_session_perf.py
"""

import asyncio
import httpx
import time
import statistics
import os
import uuid

HUB = "http://127.0.0.1:8000"
API_KEY = "my_secret_key_1"
MODEL = f"test-{uuid.uuid4().hex[:8]}"  # unique per run — prevents cross-test routing
DB_PATH = "./sessions.db"

HEADERS = {"Authorization": f"Bearer {API_KEY}"}


async def register_node(client: httpx.AsyncClient) -> str:
    resp = await client.post(f"{HUB}/register", json={
        "api_key": API_KEY,
        "resources": {
            "cpu_cores": 4, "ram_gb": 16.0, "os_name": "TestOS",
            "ollama_available": True, "ollama_models": [MODEL]
        }
    })
    return resp.json()["node_id"]


async def beat(client: httpx.AsyncClient, node_id: str):
    await client.post(f"{HUB}/heartbeat/{node_id}", json={
        "ollama_available": True, "cpu_load": 0.1, "latency_ms": 5.0
    })


async def complete_next(client: httpx.AsyncClient, node_id: str, reply: str = "ok") -> int:
    resp = await client.get(f"{HUB}/tasks/{node_id}/pending")
    for t in resp.json():
        await client.post(f"{HUB}/tasks/{node_id}/complete/{t['task_id']}", json={
            "output": reply, "prompt_tokens": len(str(t.get("messages", ""))) // 4, "completion_tokens": 3
        })
    return len(resp.json())


async def single_turn(client: httpx.AsyncClient, node_id: str,
                      msg: str, session_id: str | None, reply: str = "ok") -> tuple[float, str]:
    headers = {**HEADERS}
    if session_id:
        headers["X-Session-ID"] = session_id

    t0 = time.perf_counter()

    async def req():
        return await client.post(f"{HUB}/v1/chat/completions", headers=headers, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": msg}]
        }, timeout=20.0)

    async def complete():
        for _ in range(50):
            await asyncio.sleep(0.2)
            await beat(client, node_id)
            if await complete_next(client, node_id, reply):
                break

    r_task = asyncio.create_task(req())
    await asyncio.gather(complete(), r_task)
    resp = r_task.result()
    elapsed = (time.perf_counter() - t0) * 1000

    sid = resp.headers.get("x-session-id", "")
    return elapsed, sid


async def run_scenario(client: httpx.AsyncClient, node_id: str, label: str,
                       turns: int, use_session: bool) -> list[float]:
    print(f"\n--- {label} ({turns} turn{'s' if turns > 1 else ''}, session={'yes' if use_session else 'no'}) ---")
    session_id = None
    times = []
    for i in range(turns):
        msg = f"Turn {i+1}: " + ("x " * (i * 3 + 5))  # grows slightly each turn
        elapsed, session_id = await single_turn(client, node_id, msg, session_id if use_session else None)
        times.append(elapsed)
        label_turn = f"  turn {i+1:2d}"
        if use_session and session_id:
            label_turn += f"  [{session_id[:8]}]"
        print(f"{label_turn}  {elapsed:6.1f} ms")
    return times


def summarise(label: str, times: list[float]):
    if not times:
        return
    print(f"  {label}: mean={statistics.mean(times):.1f}ms  "
          f"median={statistics.median(times):.1f}ms  "
          f"p95={sorted(times)[int(len(times)*0.95)]:.1f}ms  "
          f"max={max(times):.1f}ms")


async def main():
    async with httpx.AsyncClient() as client:
        try:
            await client.get(f"{HUB}/nodes", timeout=3.0)
        except Exception:
            print(f"Hub not reachable at {HUB}")
            return

        node_id = await register_node(client)
        await beat(client, node_id)
        print(f"Node: {node_id[:8]}...")

        # 1. Baseline: single turn, no session
        baseline = await run_scenario(client, node_id, "Baseline (no session)", 5, use_session=False)

        # 2. Short session: 4 turns
        short = await run_scenario(client, node_id, "Short session", 4, use_session=True)

        # 3. Medium session: 10 turns
        medium = await run_scenario(client, node_id, "Medium session", 10, use_session=True)

        # 4. Long session: 30 turns (crosses SESSION_MAX_TURNS=20 threshold, triggers compression)
        long_ = await run_scenario(client, node_id, "Long session (crosses compression threshold)", 30, use_session=True)

        # 5. Parallel independent sessions (concurrency)
        print("\n--- Parallel: 5 independent sessions, 3 turns each ---")
        t0 = time.perf_counter()
        tasks_parallel = [
            run_scenario(client, node_id, f"  session-{i}", 3, use_session=True)
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks_parallel)
        total_parallel = (time.perf_counter() - t0) * 1000
        parallel_flat = [t for r in results for t in r]
        print(f"  wall time for all 5 sessions: {total_parallel:.0f} ms")

        # Check db size
        if os.path.exists(DB_PATH):
            size_kb = os.path.getsize(DB_PATH) / 1024
            print(f"\nsessions.db size: {size_kb:.1f} KB")

        print("\n=== Summary ===")
        summarise("Baseline (no session, 5x)", baseline)
        summarise("Short session (4 turns)", short)
        summarise("Medium session (10 turns)", medium)
        summarise("Long session (30 turns)", long_)
        summarise("Parallel sessions (15 turns total)", parallel_flat)

        hub_overhead = statistics.mean(short) - statistics.mean(baseline)
        print(f"\nEst. session overhead vs baseline: {hub_overhead:+.1f} ms/turn")

    # Cleanup
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


if __name__ == "__main__":
    asyncio.run(main())
