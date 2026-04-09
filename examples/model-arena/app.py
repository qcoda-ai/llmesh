"""
Model Arena — A/B test local LLMs across your network with LLMesh.

A FastAPI app that sends the same prompt to two different models
through your LLMesh hub, streams both responses side by side,
and shows latency + token counts for each.

Usage:
    # Set your LLMesh hub URL and API key
    export LLMESH_HUB_URL="http://localhost:8000"
    export LLMESH_API_KEY="your-api-key"

    # Run the arena
    uvicorn app:app --port 5000

    # Open http://localhost:5000
"""

import asyncio
import json
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.local", override=True)

HUB_URL = os.getenv("LLMESH_HUB_URL", "http://localhost:8000")
API_KEY = os.getenv("LLMESH_API_KEY", "")

app = FastAPI(title="Model Arena", version="0.1.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _headers():
    return {"Authorization": f"Bearer {API_KEY}"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"hub_url": HUB_URL},
    )


@app.get("/api/models")
async def list_models():
    """Proxy the LLMesh /v1/models endpoint with graceful error handling."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{HUB_URL}/v1/models", headers=_headers())
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        return {"models": [], "error": f"Cannot reach hub at {HUB_URL}"}
    except httpx.HTTPStatusError as e:
        return {"models": [], "error": f"Hub returned {e.response.status_code}"}
    except httpx.TimeoutException:
        return {"models": [], "error": f"Hub at {HUB_URL} timed out"}

    models = [m["id"] for m in data.get("data", [])]
    if not models:
        return {"models": [], "error": "Hub is reachable but no models available — check that nodes are connected"}
    return {"models": sorted(models)}


@app.post("/api/arena")
async def arena(request: Request):
    """
    Run two models against the same prompt, stream results as SSE.

    Expects JSON body:
        {
            "prompt": "...",
            "model_a": "llama3.2:3b",
            "model_b": "mistral:7b",
            "max_tokens": 1024
        }

    Streams SSE events:
        data: {"side": "a"|"b", "type": "token"|"done"|"error", ...}
    """
    body = await request.json()
    prompt = body["prompt"]
    model_a = body["model_a"]
    model_b = body["model_b"]
    max_tokens = body.get("max_tokens", 1024)

    async def generate():
        results = {"a": {}, "b": {}}

        async def run_model(side, model):
            start = time.monotonic()
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": True,
            }

            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream(
                        "POST",
                        f"{HUB_URL}/v1/chat/completions",
                        json=payload,
                        headers={**_headers(), "Accept": "text/event-stream"},
                    ) as resp:
                        resp.raise_for_status()
                        usage = {}
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            choices = chunk.get("choices", [])
                            if not choices:
                                continue

                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            finish = choices[0].get("finish_reason")

                            if "usage" in chunk and chunk["usage"]:
                                usage = chunk["usage"]

                            if content:
                                yield f"data: {json.dumps({'side': side, 'type': 'token', 'content': content})}\n\n"

                elapsed = round((time.monotonic() - start) * 1000)
                results[side] = {
                    "model": model,
                    "elapsed_ms": elapsed,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                yield f"data: {json.dumps({'side': side, 'type': 'done', **results[side]})}\n\n"

            except httpx.HTTPStatusError as e:
                yield f"data: {json.dumps({'side': side, 'type': 'error', 'message': f'HTTP {e.response.status_code}: {e.response.text[:200]}'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'side': side, 'type': 'error', 'message': str(e)[:200]})}\n\n"

        # Run both models concurrently, interleaving their SSE output.
        queue = asyncio.Queue()
        done_count = 0

        async def stream_to_queue(side, model):
            async for event in run_model(side, model):
                await queue.put(event)
            await queue.put(None)  # sentinel

        task_a = asyncio.create_task(stream_to_queue("a", model_a))
        task_b = asyncio.create_task(stream_to_queue("b", model_b))

        sentinels = 0
        while sentinels < 2:
            event = await queue.get()
            if event is None:
                sentinels += 1
                continue
            yield event

        await task_a
        await task_b

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/health")
async def health():
    """Check connectivity to the LLMesh hub and report node status."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{HUB_URL}/health")
            hub_status = resp.json()

            nodes_resp = await client.get(f"{HUB_URL}/nodes", headers=_headers())
            nodes_resp.raise_for_status()
            nodes = nodes_resp.json()

        return {
            "arena": "ok",
            "hub": hub_status,
            "hub_url": HUB_URL,
            "nodes_connected": len(nodes),
        }
    except httpx.ConnectError:
        return {"arena": "ok", "hub": "unreachable", "hub_url": HUB_URL, "error": f"Cannot connect to {HUB_URL}"}
    except Exception as e:
        return {"arena": "ok", "hub": "error", "hub_url": HUB_URL, "error": str(e)}
