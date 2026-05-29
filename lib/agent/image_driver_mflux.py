"""
mflux in-process image generation driver (D064).

Agent-side driver that imports `mflux` as a Python module and runs inference
in-process. No HTTP, no separate process. Apple Silicon only — mflux is MLX-
backed and refuses to run on Intel Macs / Linux / Windows.

**Verified against mflux 0.17.5 (2026-05-28).** mflux is FLUX-only; it does
NOT support SDXL or SD 1.5. The agent registry was pruned to FLUX entries
only — SDXL/SD1.5 backends require a different driver (Diffusers, ComfyUI,
A1111) which is deferred to v2 per D064.

Real API surface:
    from mflux.models.flux.variants.txt2img.flux import Flux1
    from mflux.models.common.config.model_config import ModelConfig

    flux = Flux1(model_config=ModelConfig.schnell(), quantize=8)
    out = flux.generate_image(
        seed=42, prompt="...",
        num_inference_steps=4, height=1024, width=1024,
        guidance=4.0, negative_prompt=None,
    )
    pil_image = out.image   # PIL.Image.Image

Operators install model weights via `llmesh-agent install-image-model <id>`
(see `scripts/install_image_model.py`). mflux's own loader handles the HF
cache layout; we override `HF_HUB_OFFLINE=1` during inference so it cannot
silently fetch missing weights from Hugging Face.

Soft-import policy: `mflux` is NOT a hard dependency of the agent. If import
fails, `mflux_available()` returns False and the agent advertises
`image_available=False` to the hub. Operators install on demand:
    pip install mflux huggingface_hub
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("llmesh.agent.image")


# Standardized model cache directory. Override via env so operators on
# external drives can point elsewhere. Operator-facing config:
# `LLMESH_IMAGE_MODELS_DIR` in `.env` or system env.
def _models_dir() -> Path:
    return Path(os.environ.get(
        "LLMESH_IMAGE_MODELS_DIR",
        str(Path.home() / ".llmesh" / "models" / "image"),
    ))


_mflux_module: Any = None
_mflux_loaded_model: dict[str, Any] = {}  # model_id → loaded Flux1 instance


def mflux_available() -> bool:
    """Return True if mflux can be imported in this process. Cached after
    first call. Result is sticky for the process lifetime — operators who
    pip-install mflux after the agent started must restart the agent."""
    global _mflux_module
    if _mflux_module is not None:
        return _mflux_module is not False
    try:
        import mflux  # type: ignore  # noqa: F401
        _mflux_module = mflux
        return True
    except Exception as exc:
        logger.debug("mflux import failed: %s — image-gen disabled on this node", exc)
        _mflux_module = False
        return False


def model_install_dir(model_id: str) -> Path:
    """Filesystem path where a registry entry's diffusers-layout files live.

    `<LLMESH_IMAGE_MODELS_DIR>/<family>/<model_id>/`. Matches the layout
    produced by `scripts/install_image_model.py` (D071) and consumed by
    mflux's `Flux1(model_path=...)`.
    """
    from . import image_model_registry as reg
    entry = reg.get(model_id)
    if entry is None:
        raise KeyError(f"unknown model_id: {model_id!r}")
    return _models_dir() / entry.family / model_id


def missing_repo_files(model_id: str) -> list[str]:
    """Return the list of registry `repo_files` not present on disk for
    `model_id`. Empty list = complete install."""
    from . import image_model_registry as reg
    entry = reg.get(model_id)
    if entry is None:
        return []
    model_dir = model_install_dir(model_id)
    if not model_dir.is_dir():
        return list(entry.repo_files)
    return [f for f in entry.repo_files if not (model_dir / f).exists()]


def discover_installed_models() -> list[str]:
    """Scan `LLMESH_IMAGE_MODELS_DIR` for fully-installed FLUX model weights.

    D071: a model counts as installed only when ALL its diffusers-layout
    `repo_files` are present. Partial installs (interrupted curl, missing
    encoder shard) report as not-installed so the agent does not advertise
    a broken model to the hub.

    Unknown directories (operator-installed LoRAs, custom checkpoints) are
    ignored in v1 — they would land in the registry-with-`unverified:true`
    flow once that path is built (D-003 §"Registry drift" recommendation
    (b), deferred per D064).
    """
    from . import image_model_registry as reg

    base = _models_dir()
    if not base.exists():
        return []
    found: list[str] = []
    for model_id in reg.known_model_ids():
        entry = reg.get(model_id)
        if entry is None:
            continue
        model_dir = base / entry.family / model_id
        if not model_dir.is_dir():
            continue
        if all((model_dir / fname).exists() for fname in entry.repo_files):
            found.append(model_id)
    return found


def estimate_uma_gb() -> float:
    """Apple Silicon unified-memory available estimate, in GB.

    On Apple Silicon, system memory IS GPU memory (UMA). Returns total
    system RAM × 0.75 as a conservative ceiling (MacOS reserves ~25% for
    system + other apps). On non-Apple-Silicon platforms, returns 0.0 so
    hub-side routing skips this node."""
    import platform
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return 0.0
    try:
        import psutil
        total_gb = psutil.virtual_memory().total / (1024.0 ** 3)
        return round(total_gb * 0.75, 2)
    except Exception:
        return 0.0


def _model_config_for(model_id: str):
    """Map our registry model_id → mflux ModelConfig classmethod.

    mflux's `ModelConfig.schnell()` and `ModelConfig.dev()` are the two
    FLUX entrypoints we ship in v1. Other variants (kontext, redux,
    controlnet, depth, etc.) are deferred."""
    from mflux.models.common.config.model_config import ModelConfig  # type: ignore
    if model_id == "flux-schnell":
        return ModelConfig.schnell()
    if model_id == "flux-dev":
        return ModelConfig.dev()
    raise RuntimeError(
        f"Model {model_id!r} is not supported by mflux in v1. "
        f"mflux only supports FLUX variants. SDXL/SD1.5 require a "
        f"different backend (Diffusers, ComfyUI, A1111) — deferred to v2."
    )


def _load_pipeline(model_id: str):
    """Load mflux Flux1 pipeline for a model id, caching the loaded weights
    so subsequent inferences reuse the same handles.

    D071: pass `model_path=<install dir>` to bypass mflux's default HF-cache
    lookup. Without this, mflux ignores `LLMESH_IMAGE_MODELS_DIR` and tries
    to populate `~/.cache/huggingface/` via the hung `hf_hub_download` path.
    Verifies the install is complete before invoking `Flux1` so a partial
    install surfaces a clear error instead of an mid-load mflux exception.
    """
    if model_id in _mflux_loaded_model:
        return _mflux_loaded_model[model_id]
    if not mflux_available():
        raise RuntimeError("mflux not available on this node (import failed)")

    missing = missing_repo_files(model_id)
    if missing:
        install_dir = model_install_dir(model_id)
        raise RuntimeError(
            f"Image model {model_id!r} is not fully installed at {install_dir}. "
            f"{len(missing)} file(s) missing — first: {missing[0]}. "
            f"Run `python scripts/install_image_model.py install {model_id}` to complete."
        )

    os.environ["HF_HUB_OFFLINE"] = "1"  # D064: never silently pull weights

    from mflux.models.flux.variants.txt2img.flux import Flux1  # type: ignore
    model_config = _model_config_for(model_id)
    pipeline = Flux1(
        model_config=model_config,
        quantize=8,
        model_path=str(model_install_dir(model_id)),
    )
    _mflux_loaded_model[model_id] = pipeline
    return pipeline


async def run_image_task(payload: dict, model: str) -> list[str]:
    """Execute one image-gen task. Returns list[str] of base64-encoded PNGs.

    Payload shape (set by hub `_route_image`):
      prompt, negative_prompt, size ("WxH" concrete), n, seed, steps,
      quality.

    Runs the inference in an asyncio thread so the agent's poll loop and
    heartbeat coroutines stay responsive. Each generated image is encoded
    as base64 PNG before being returned to the hub.
    """
    if not mflux_available():
        raise RuntimeError("mflux not available; cannot run image task")

    prompt = str(payload.get("prompt", ""))
    negative_prompt = payload.get("negative_prompt")
    size = str(payload.get("size", "1024x1024"))
    n = int(payload.get("n", 1))
    seed = payload.get("seed")
    steps = int(payload.get("steps", 20))

    width, height = (int(x) for x in size.split("x"))

    def _run_one(seed_value: int) -> str:
        """Synchronous mflux inference of a single image. Returns base64 PNG."""
        pipeline = _load_pipeline(model)
        out = pipeline.generate_image(
            seed=seed_value,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
        )
        pil_image = out.image  # PIL.Image.Image
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    loop = asyncio.get_running_loop()
    results: list[str] = []
    for i in range(n):
        s = (seed + i) if seed is not None else int((time.time_ns() + i) & 0xFFFFFFFF)
        b64 = await loop.run_in_executor(None, _run_one, s)
        results.append(b64)
    return results
