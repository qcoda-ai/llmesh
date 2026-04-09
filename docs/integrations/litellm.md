# LiteLLM Compatibility (Community Integration)

> **Status: user-supported integration, not a first-class backend.**
> LLMesh's `vllm` backend can talk to any OpenAI-compatible endpoint that
> accepts a bearer token, which includes
> [LiteLLM Proxy](https://github.com/BerriAI/litellm). Pointing an LLMesh
> agent at LiteLLM works, but it is not a tested release surface — LiteLLM
> ships breaking changes frequently and you are responsible for catching
> drift in their `/v1/models` and `/health` endpoints.

## What this is

LLMesh agents detect a `vllm` backend by probing `$VLLM_HOST/health` and
`$VLLM_HOST/v1/models`. Both vanilla vLLM and LiteLLM Proxy expose the
OpenAI-compatible surface those probes need, so the same agent code can
register against either one. The only differences are:

1. LiteLLM requires `Authorization: Bearer <key>` on every request.
2. LiteLLM's `/health` endpoint runs a real upstream model probe and is
   too heavy for a 2-second liveness check; you want `/health/liveliness`
   instead.

The agent supports both via two opt-in env vars added in D014:

| Env var            | Default        | Purpose |
|--------------------|----------------|---------|
| `VLLM_HOST`        | unset          | Base URL of the OpenAI-compatible server (vLLM, LiteLLM, anything else that speaks the spec). |
| `VLLM_API_KEY`     | unset          | Bearer token attached to all requests against `VLLM_HOST`. Leave unset for plain local vLLM. |
| `VLLM_HEALTH_PATH` | `/health`      | Path the agent probes for liveness. Set to `/health/liveliness` for LiteLLM. |
| `VLLM_MAX_CONTEXT` | _(auto)_       | Optional explicit context window override. **LiteLLM users almost always need this** — see "Context window" below. |

## Architectural caveats (read these before wiring it up)

LLMesh's value proposition is **hardware-aware routing across self-hosted
GPU nodes**. Pointing an agent at a remote LiteLLM proxy gives up the
properties that make LLMesh useful:

- The agent reports the hardware fingerprint of the *box running the
  agent*, not the GPU(s) behind LiteLLM. The hub's routing decisions for
  this node are based on irrelevant data.
- Adds a network hop. Requests flow `client → hub → agent → LiteLLM →
  upstream provider`. Five hops, three of which can fail independently.
- Heartbeat health is coupled to LiteLLM's liveness, not to the upstream
  model's actual availability.
- Model names returned by LiteLLM (whatever you defined in its
  `model_list`) end up in `state.vllm_models`, mixed in with any real vLLM
  nodes you have. Naming collisions are your problem to avoid.
- If LiteLLM does cost tracking, virtual keys, or budget enforcement, the
  hub has zero visibility into any of it.

If your real goal is "let LLMesh users call cloud-hosted models," consider
pointing your clients directly at LiteLLM and using LLMesh only for the
self-hosted nodes you actually own. The tools solve adjacent problems and
do not need to be stacked.

That said, valid reasons to wire LiteLLM up as an LLMesh "node" include:

- Giving the hub an always-available fallback model when all your real
  GPU nodes are offline or saturated.
- Quick integration testing with a known-good upstream.
- Exposing a small set of cloud models to LLMesh clients without writing
  a new backend type.

## Setup

### 1. Configure LiteLLM Proxy

Run LiteLLM with at least one model in its `config.yaml`:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: llama3-70b-local
    litellm_params:
      model: openai/meta-llama/Llama-3-70B-Instruct
      api_base: http://gpu-box.internal:8001/v1
      api_key: not-needed

general_settings:
  master_key: sk-litellm-master-replace-me
```

Start it (the default port is 4000):

```bash
litellm --config config.yaml --port 4000
```

Sanity-check from another shell — both the liveness probe and the model
list must respond:

```bash
LITELLM_KEY=sk-litellm-master-replace-me

curl -fsS https://litellm.example.com/litellm/health/liveliness
curl -fsS -H "Authorization: Bearer $LITELLM_KEY" \
    https://litellm.example.com/litellm/v1/models
```

If either fails, fix it before moving on. The LLMesh agent will silently
mark the backend unavailable and there will be nothing useful in its logs.

### 2. Configure the LLMesh agent

In the agent's `.env` (or shell environment):

```bash
HUB_URL=http://hub.internal:8000
LLMESH_API_KEY=your-llmesh-owner-key

# Point the vllm backend at LiteLLM
VLLM_HOST=https://litellm.example.com/litellm
VLLM_API_KEY=sk-litellm-master-replace-me
VLLM_HEALTH_PATH=/health/liveliness

# Disable Ollama and MLX detection if this node is purely a LiteLLM proxy.
# (Leave them set to detect both local and remote backends from one agent.)
OLLAMA_HOST=
MLX_HOST=
```

A virtual key from LiteLLM works just as well as the master key — and is
the right choice if you want per-LLMesh-node spend tracking on the
LiteLLM side.

### 3. Start the agent

```bash
source .venv/bin/activate
python -m lib.agent.client
```

You should see `vllm_available=True` and the model list reported by
LiteLLM in the registration log:

```
vLLM models found: ['gpt-4o-mini', 'llama3-70b-local']
Registered as node node_xxxx
```

If you see `vllm_available=False`:

- Confirm `VLLM_HEALTH_PATH=/health/liveliness` (without it, LiteLLM
  returns either 401 or a slow probe response on plain `/health`).
- Confirm `VLLM_API_KEY` matches a key LiteLLM accepts.
- Hit the same URL the agent hits, with the same header, from the same
  host. If `curl` works and the agent doesn't, double-check that
  `python -m lib.agent.client` is reading the env vars you think it is
  (the agent uses `python-dotenv`, which only loads `.env` from the
  current working directory).

### 4. Call a model from the hub

The model name a client passes to the LLMesh hub must match exactly what
LiteLLM advertised in `/v1/models`:

```bash
curl -X POST http://hub.internal:8000/v1/chat/completions \
    -H "Authorization: Bearer your-llmesh-owner-key" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "gpt-4o-mini",
      "messages": [{"role": "user", "content": "hello"}]
    }'
```

The hub will route the request to the agent that registered with that
model name in its `vllm_models` list. The agent forwards the request to
LiteLLM with the bearer token attached. LiteLLM forwards upstream.

## Context window

The LLMesh agent reports a `context_size` to the hub at registration so the routing layer can make decisions about which node can serve which request (D010, D015). For vanilla vLLM the agent auto-detects this from the `max_model_len` field on each model card returned by `/v1/models` — vLLM exposes this as a non-standard extension field.

**LiteLLM Proxy generally does not expose `max_model_len`.** LiteLLM's `/v1/models` returns the OpenAI-spec fields only, which means the agent's auto-detection returns `None` and the resolver falls back to `OLLAMA_NUM_CTX` (default `8192`). The hub then thinks your LiteLLM-backed node can only serve 8k contexts even when the upstream models support much larger windows. Routing decisions for that node will be wrong.

**The fix is `VLLM_MAX_CONTEXT`.** Set it explicitly to the largest context window the upstream models behind LiteLLM can serve:

```bash
# In the agent's .env
VLLM_HOST=https://litellm.example.com
VLLM_API_KEY=sk-litellm-virtual-key
VLLM_HEALTH_PATH=/health/liveliness
VLLM_MAX_CONTEXT=128000   # gpt-4o-mini's actual window, for example
```

If your LiteLLM proxy fronts a mix of models with different context windows (gpt-4o at 128k, claude-3-5-sonnet at 200k, llama-3-70b at 8k), set `VLLM_MAX_CONTEXT` to the **maximum** across all of them — the hub uses this as the node's "maximum capability" for routing. Per-request context-window enforcement against vLLM/LiteLLM is not supported (the OpenAI Chat Completions API has no field for it; see D015 for the full analysis), so a client request asking for an 8k window when routed through a 128k LiteLLM node will not be clamped down on the vLLM path. This is a known limitation, not a bug.

When the agent starts, it prints the resolved vLLM window and its source so you can verify your configuration:

```
vLLM context window: 128000 (source: VLLM_MAX_CONTEXT)
```

If you instead see the warning:

```
vLLM context window: unknown — `/v1/models` did not expose `max_model_len`
and `VLLM_MAX_CONTEXT` is unset. The hub will see this node as a 8192-token
node. Set `VLLM_MAX_CONTEXT` to fix routing decisions.
```

…it means you're hitting the LiteLLM auto-detect gap and need to set the env var. This warning is intentionally loud; do not ignore it.

## Maintenance

LiteLLM ships breaking changes frequently. If LLMesh stops detecting
LiteLLM as available after a LiteLLM upgrade:

1. Re-run the two `curl` checks from setup step 1. If they pass, the
   problem is the agent — open an issue.
2. If `/health/liveliness` is gone, find LiteLLM's new liveness path and
   set `VLLM_HEALTH_PATH` accordingly.
3. If `/v1/models` changed shape, the agent's `get_vllm_models()` parser
   in `lib/agent/client.py` may need updating — it expects the OpenAI
   format `{"data": [{"id": "..."}, ...]}`.

LLMesh does not pin a specific LiteLLM version. If you depend on this
integration for anything important, pin LiteLLM yourself in your
deployment.
