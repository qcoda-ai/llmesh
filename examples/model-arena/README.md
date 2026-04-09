# Model Arena

A/B test local LLMs across your network with LLMesh.

Send the same prompt to two different models through your LLMesh hub, see both responses stream side by side, and compare latency and token usage.

## Why

You have multiple models running on different machines. Which one is better for your use case? Model Arena lets you find out — without changing your infrastructure, without paying for cloud APIs, and without running manual tests in two terminal windows.

## Quick Start

### Prerequisites

- A running LLMesh hub (`uvicorn lib.hub.server:app --port 8000`)
- At least one LLMesh agent connected with two or more models available
- Python 3.10+

### Setup

```bash
cd examples/model-arena

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your hub URL and API key

# Run
uvicorn app:app --port 5000
```

Open [http://localhost:5000](http://localhost:5000).

### Usage

1. Select two models from the dropdowns (populated from your LLMesh hub)
2. Enter a prompt
3. Hit **Run Arena** (or press Enter)
4. Watch both responses stream in side by side
5. Compare latency, token counts, and output quality
6. Round history accumulates at the bottom for comparison across prompts

## Architecture

```
Browser  -->  Model Arena (FastAPI :5000)  -->  LLMesh Hub (:8000)  -->  Node A (Ollama)
                                                                   -->  Node B (vLLM)
```

The Arena app never talks to model backends directly. It uses the LLMesh hub's OpenAI-compatible API (`/v1/chat/completions` with streaming). This means:

- The Arena works identically whether your hub is on localhost or a remote staging server
- You can add/remove compute nodes without touching the Arena
- Token tracking happens automatically in the LLMesh dashboard

## Docker Setup

Run the full stack — Postgres, LLMesh hub, and Model Arena — with one command:

```bash
cd examples/model-arena
docker compose up
```

This starts:

| Service | Port | Role |
|---------|------|------|
| `postgres` | 5432 (internal) | Session persistence |
| `hub` | 8000 | LLMesh inference broker |
| `arena` | 5000 | Model Arena UI |

Then start LLMesh agents on your machines pointing at the hub:

```bash
LLMESH_API_KEY="arena-demo-key" \
HUB_URL="http://<hub-machine-ip>:8000" \
python -m lib.agent.client
```

Open [http://localhost:5000](http://localhost:5000) and start comparing models.

The `server_config.json` in this directory contains a demo API key (`arena-demo-key`). For production use, replace it with your own keys.

### What's in the stack

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│   Browser   │────▶│   Model Arena    │────▶│  LLMesh Hub    │
│             │     │  (FastAPI :5000)  │     │  (:8000)       │
└─────────────┘     └──────────────────┘     └───────┬────────┘
                                                     │
                                              ┌──────┴──────┐
                                              │  Postgres   │
                                              │  (sessions) │
                                              └─────────────┘
                                                     ▲
                            ┌────────────────────────┼────────────────────────┐
                            │                        │                        │
                     ┌──────┴──────┐          ┌──────┴──────┐          ┌──────┴──────┐
                     │  Agent      │          │  Agent      │          │  Agent      │
                     │  (laptop)   │          │  (GPU box)  │          │  (Mac Mini) │
                     └─────────────┘          └─────────────┘          └─────────────┘
```

Agents run on bare metal — that's the point. They sit on the machines where your GPUs and models live. The containerized hub and arena don't need GPUs; they just route and display.

## Portable Dev Environment Demo

This app demonstrates LLMesh's environment portability. Try this:

1. **Local**: Run hub + agent + arena all on your laptop
2. **Add a machine**: Start an agent on a second machine pointing at your hub. The arena now routes to both — no code change.
3. **Push to staging**: Point `LLMESH_HUB_URL` at a shared hub server. Deploy the arena anywhere. Same code, same behavior.

The only thing that changes between environments is one URL.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLMESH_HUB_URL` | `http://localhost:8000` | LLMesh hub address |
| `LLMESH_API_KEY` | (none) | API key from hub's `server_config.json` |

## License

MIT — same as LLMesh.
