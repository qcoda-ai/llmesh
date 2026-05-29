# Image Generation (v0.2)

LLMesh supports OpenAI-compatible image generation via the local `mflux` backend on Apple Silicon Macs. v1 ships with the following constraints (see `.qcoda/decisions.md::D064`):

- **Apple Silicon Macs only.** Intel Macs, Linux, and Windows are not supported in v1. Future versions may add ComfyUI / Draw Things / A1111 drivers.
- **Operator-explicit model installs.** Models are never auto-downloaded. You run `llmesh-agent install-image-model <id>` once per model.
- **No content filter shipped.** LLMesh does not run safety classification on generated images. Operator policy applies. This is an explicit decision (D064) — for SOHO deployments with shared devices, consider this carefully before exposing image gen.
- **No model substitution.** If a client requests a model not installed on any node, the hub returns `404 model_not_available`.

---

## Quick start

### 1. Install mflux on the agent host

```bash
pip install mflux huggingface_hub
```

mflux is a soft dependency — LLMesh agent runs without it, but won't advertise image-gen capability unless it's importable.

### 2. Authenticate with Hugging Face (one-time)

FLUX weights are hosted on Hugging Face as **gated repos** — they're free + Apache-2.0 (for `flux-schnell`) but require a one-time access acknowledgement.

1. Create a free account at <https://huggingface.co>.
2. Visit <https://huggingface.co/black-forest-labs/FLUX.1-schnell> (and `/FLUX.1-dev` if you also want that one). Click **"Agree and access repository"** once.
3. Authenticate locally:

```bash
hf auth login
```

(Or set `HF_TOKEN=<your-read-scope-token>` in `.env` or your shell. The legacy `huggingface-cli login` command was deprecated in favour of `hf auth login` — both ship with the `huggingface_hub` package, but only `hf` works on current versions.)

This step is per-machine, not per-install. The agent and CLI both pick up the same auth.

### 3. Install an image model

```bash
python scripts/install_image_model.py list
python scripts/install_image_model.py install flux-schnell --yes
```

Models in the v1 registry:

| model_id | family | size | min VRAM (UMA) | license |
|---|---|---|---|---|
| `flux-schnell` | FLUX | ~32 GB | 16 GB | Free for personal & commercial use (Apache-2.0) |
| `flux-dev` | FLUX | ~32 GB | 16 GB | Non-commercial only (FLUX-1-dev-NC) |

v1 is **FLUX-only** after mflux 0.17.5 verification — mflux does not support SDXL or SD 1.5. Those backends are deferred to v2.

The installer pulls the full diffusers-layout repo (~32 GB across 23 files: transformer shards, two text encoders, VAE, tokenizers) — NOT the single-file `flux1-schnell.safetensors` checkpoint. mflux expects the multi-file layout; the single-file BFL format is not supported. Downloads use `curl` per file with resume support (`curl -C -`) so an interrupted install picks up where it left off on re-run. See `.qcoda/decisions.md::D071`.

Non-permissive licenses (`flux-dev`) require `--accept-license` to confirm you've read and accept the license terms.

### 4. Restart the agent

The agent advertises image capability at startup. Restart to pick up new models.

### 5. Generate

**Via dashboard** — open the hub dashboard. If at least one node advertises image capability, a green "Generate Image" card appears with prompt + model + size + speed fields.

**Via OpenAI-compatible client** (Chatwise, Raycast AI, OpenWebUI, etc.):

```bash
curl -X POST http://your-hub:8000/v1/images/generations \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"flux-schnell","prompt":"a red apple on a wooden table","n":1,"size":"1024x1024"}'
```

Returns:

```json
{"created":1779953600,"data":[{"b64_json":"iVBORw0KGgoAAAANS..."}]}
```

---

## Request parameters

| Param | Type | Default | Notes |
|---|---|---|---|
| `model` | string | — | Required. Must match an installed model on at least one node. |
| `prompt` | string | — | Required. Max 8 KB (`MAX_IMAGE_PROMPT_BYTES`). |
| `negative_prompt` | string | — | Optional. Forwarded to backend when supported. |
| `size` | string | `"square"` | `"square"` / `"portrait"` / `"landscape"` (mapped to model-native pixels) or a concrete `"WxH"`. |
| `n` | int | `1` | 1–4 (`MAX_IMAGES_PER_REQUEST`). |
| `seed` | int | random | For reproducibility. |
| `quality` | string | `"draft"` | `"draft"` (~20 steps, fast) / `"quality"` (~40 steps, slower). |
| `response_format` | string | `"b64_json"` | Only `b64_json` in v1. URL mode deferred to v2. |

---

## Operator notes

### Content filter

**LLMesh does not ship a content filter.** Output is whatever the model produces. Generated content can include sensitive, harmful, or copyright-violating material depending on prompt and model.

If your deployment exposes image gen to shared devices (e.g. a household with children using the same Mac), consider:

- Not enabling image gen
- Using only models trained without harmful content (research the model's training set + community track record)
- Adding your own classifier in front of the dashboard

This may change in a future release; see D064 limitations.

### Disk + bandwidth

- FLUX models are ~32 GB each (full diffusers-layout repo). The installer checks free disk and refuses to install if post-install would leave less than 10 GB (override with `--force`).
- Initial download from Hugging Face can be slow on home connections (30-60 min for FLUX on a typical residential link). Resume works — re-run the install command and already-downloaded files are skipped.
- `LLMESH_IMAGE_MODELS_DIR` overrides the cache location (default `~/.llmesh/models/image/`).

### Configuring the models directory

Set the path in any of three places (system env wins over `.env` wins over default):

**System environment variable** (shell profile, launchd plist `EnvironmentVariables`, systemd `Environment=`):

```bash
export LLMESH_IMAGE_MODELS_DIR=/Volumes/Sandisk1.5T/llmesh/models
```

**Project `.env` file** (in repo root, alongside `pyproject.toml`):

```
# .env
LLMESH_IMAGE_MODELS_DIR=/Volumes/Sandisk1.5T/llmesh/models
```

Both the agent (`lib/agent/client.py`) and the install CLI (`scripts/install_image_model.py`) load this file at startup, so the value stays consistent between download time and runtime. The agent loads the `.env` at the repo root explicitly (works regardless of where launchd invokes the agent from). The CLI does the same. System env vars take precedence over `.env` per python-dotenv default behaviour.

**Default** (`~/.llmesh/models/image/`) is used if neither is set. On Macs where `~` lives on a near-full Data volume, point this at an external drive instead — the install CLI refuses to install if post-install free disk would drop below 10 GB.

### Memory pressure

- mflux runs in-process with the agent. Loaded model weights stay resident.
- The agent loads quantized (`quantize=8`) — effective working set is roughly half the on-disk size.
- On 16 GB Macs, FLUX is at the routing floor and may fail or be very slow under any other load. 32 GB+ is the comfortable minimum; 64 GB+ runs both FLUX-schnell and FLUX-dev without paging pressure.

### `HF_HUB_OFFLINE=1`

The agent sets this env var during inference so mflux won't silently fetch missing weights at runtime. If you delete a model file mid-session, the next request errors loud instead of consuming bandwidth.

### Performance

Approximate wall-time on a Mac M1 Ultra 64 GB (your mileage will vary):

| Model | Draft (~4 steps) | Quality (~20 steps) |
|---|---|---|
| `flux-schnell` 1024² | 6-10 s | (schnell is step-distilled; more steps don't improve output) |
| `flux-dev` 1024² | 8-15 s | 30-50 s |

---

## What's not in v1

Deferred to v2 or later (per D064):

- ComfyUI / A1111 / Draw Things drivers (cross-platform image gen)
- GUI install flow for models (CLI only in v1)
- Bundled NSFW classifier
- URL response mode + hub-side ephemeral image store
- Per-owner default settings in `server_config.json`
- LoRA selection, ControlNet, img2img, inpainting, upscaling
- Progress streaming (SSE preview frames)
- Server-side image history
- Tool use / structured output in image responses

---

## Troubleshooting

**"No node advertises image_available with model X"** — install the model on at least one agent host and restart the agent. Check `python scripts/install_image_model.py installed`.

**Dashboard "Generate Image" card missing** — no nodes for your owner advertise image capability. Same fix as above.

**HTTP 404 with `model_not_available`** — same root cause. Response includes `available` list of models you CAN reach.

**HTTP 400 with `n` out of bounds** — adjust to 1–4.

**HTTP 413 `payload_too_large` on `prompt`** — prompt exceeds 8 KB. Reduce.

**HTTP 504 timeout** — image gen exceeded `IMAGE_TASK_TIMEOUT_S` (default 300s). FLUX-dev on 16 GB Macs may hit this. Use a faster model or larger machine.

**Agent log shows "mflux not available"** — `pip install mflux` on the agent host, restart agent.

---

## References

- `.qcoda/decisions.md::D064` — scope-lock decision (FLUX-only, mflux backend)
- `.qcoda/decisions.md::D071` — install layout fix (diffusers multi-file + curl-per-file)
- `.qcoda/discussions.md::D-003` — exploration (superseded)
- `.qcoda/labs/LAB-005-image-gen-feasibility/` — verification harness
- `lib/hub/server.py` — `/v1/images/generations` endpoint + `_route_image`
- `lib/agent/image_driver_mflux.py` — mflux in-process driver
- `lib/agent/image_model_registry.py` — agent-side registry
- `lib/hub/image_registry.py` — hub-side routing-aware registry
- `scripts/install_image_model.py` — operator CLI
