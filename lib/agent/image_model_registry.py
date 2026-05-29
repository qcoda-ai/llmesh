"""
Agent-side image model registry (D064 + D071).

**Pruned to FLUX-only after mflux 0.17.5 verification (2026-05-28).**
mflux is a FLUX-family-only library — it does NOT support SDXL or SD 1.5.
Those models require a different backend (Diffusers in-process, ComfyUI
HTTP, A1111 HTTP) which is deferred to v2 per D064.

**File layout (D071, 2026-05-29).** mflux expects the diffusers-style
multi-file HF repo layout, NOT the single-file BFL `flux1-schnell.safetensors`
checkpoint. The install CLI downloads `repo_files` (the diffusers-layout
file set) via curl-per-file to bypass `hf_hub_download` hangs on large
gated repos (see `.qcoda/lessons.md` 2026-05-28). mflux's `Flux1(model_path=...)`
constructor is then handed the install directory directly.

Used for:
  1. **Verifying installed weights** at startup — operator-installed model
     directories are matched against this list by `model_id`; all
     `repo_files` must be present for the model to count as installed.
  2. **CLI install** (`llmesh-agent install-image-model <id>`) — curls
     each entry in `repo_files` from the HF repo, preserving sub-directory
     structure, captures license consent.
  3. **Routing hints** — `min_vram_gb` so the agent can decline a task
     locally if its UMA-available memory is below threshold.

Hub-side `lib/hub/image_registry.py` mirrors this for routing. Keep both
files in sync when adding/removing entries.

NEVER auto-downloaded. Operator runs the install CLI explicitly.
`HF_HUB_OFFLINE=1` is set during inference so mflux cannot silently fetch
missing weights.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentImageModel:
    """Operator-facing model metadata + download coordinates."""
    model_id: str               # canonical id used everywhere (registry key)
    family: str                 # FLUX-only in v1
    hf_repo: str                # Hugging Face repo id
    repo_files: tuple[str, ...] # files inside the repo (diffusers layout) mflux needs
    size_gb: float              # approximate total weight size
    min_vram_gb: int            # routing floor (UMA on Apple Silicon)
    license_label: str          # plain-English summary
    license_id: str             # canonical id
    license_url: str            # link to full text
    license_permissive: bool    # True → default --yes; False → require --accept-license


# Diffusers-layout file set mflux needs for both FLUX-schnell and FLUX-dev.
# Sourced from `FluxWeightDefinition.get_download_patterns()` + tokenizer
# defs in mflux 0.17.5, cross-referenced against the live HF tree listing
# for `black-forest-labs/FLUX.1-schnell` and `.../FLUX.1-dev` on 2026-05-29.
# Both repos share the same file layout.
_FLUX_DIFFUSERS_FILES: tuple[str, ...] = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model-00001-of-00002.safetensors",
    "text_encoder_2/model-00002-of-00002.safetensors",
    "text_encoder_2/model.safetensors.index.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer.json",
    "tokenizer_2/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)


# v1 registry: FLUX-schnell (Apache-2.0, permissive) and FLUX-dev (NC license).
# mflux supports both via `ModelConfig.schnell()` / `ModelConfig.dev()`.
# SDXL / SDXL-Turbo / SD 1.5 are NOT in this registry because mflux 0.17.5
# cannot run them; reintroducing them requires shipping a Diffusers-in-process
# or HTTP-backend driver, deferred to v2.
_REGISTRY: dict[str, AgentImageModel] = {
    "flux-schnell": AgentImageModel(
        model_id="flux-schnell",
        family="FLUX",
        hf_repo="black-forest-labs/FLUX.1-schnell",
        repo_files=_FLUX_DIFFUSERS_FILES,
        size_gb=32.0,
        min_vram_gb=16,
        license_label="Free for personal & commercial use",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/LICENSE.md",
        license_permissive=True,
    ),
    "flux-dev": AgentImageModel(
        model_id="flux-dev",
        family="FLUX",
        hf_repo="black-forest-labs/FLUX.1-dev",
        repo_files=_FLUX_DIFFUSERS_FILES,
        size_gb=32.0,
        min_vram_gb=16,
        license_label="Non-commercial only",
        license_id="FLUX-1-dev-NC",
        license_url="https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md",
        license_permissive=False,
    ),
}


def get(model_id: str) -> AgentImageModel | None:
    return _REGISTRY.get(model_id)


def known_model_ids() -> list[str]:
    return list(_REGISTRY.keys())


def all_models() -> list[AgentImageModel]:
    return list(_REGISTRY.values())
