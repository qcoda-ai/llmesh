"""
Hub-side image model registry (D064).

**Pruned to FLUX-only after mflux 0.17.5 verification (2026-05-28).**
SDXL / SD 1.5 require a different backend than mflux and are deferred to v2.

Mirrors `lib/agent/image_model_registry.py` for the v1 model set. Hub uses
this for:

  1. Routing: filter candidate nodes whose advertised `vram_gb < min_vram_gb`
     for the requested model BEFORE the D054 weighted score applies.
  2. `/v1/limits` and `/v1/models` capability surfaces: advertise the
     per-model `min_vram_gb` + plain-English license label so clients can
     pre-clamp / pre-display.

Hub does NOT touch model files; agent-side registry holds the SHA256 + HF
coords. Keep this hub copy in sync with the agent copy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HubImageModel:
    """Hub-visible attributes of an image-gen model."""
    model_id: str
    family: str             # FLUX-only in v1
    min_vram_gb: int        # routing filter
    native_size: str        # "1024x1024" — FLUX is square-native
    license_label: str
    license_id: str


_REGISTRY: dict[str, HubImageModel] = {
    "flux-schnell": HubImageModel(
        model_id="flux-schnell",
        family="FLUX",
        min_vram_gb=16,
        native_size="1024x1024",
        license_label="Free for personal & commercial use",
        license_id="Apache-2.0",
    ),
    "flux-dev": HubImageModel(
        model_id="flux-dev",
        family="FLUX",
        min_vram_gb=16,
        native_size="1024x1024",
        license_label="Non-commercial only",
        license_id="FLUX-1-dev-NC",
    ),
}


def get(model_id: str) -> HubImageModel | None:
    return _REGISTRY.get(model_id)


def known_model_ids() -> list[str]:
    return list(_REGISTRY.keys())


def all_models() -> list[HubImageModel]:
    return list(_REGISTRY.values())


# Operator-facing size → (width, height) string. FLUX defaults to 1024×1024;
# operator picks orientation only. Concrete WxH passthrough is also honored.
_SIZE_MAP_BY_FAMILY: dict[str, dict[str, str]] = {
    "FLUX": {"square": "1024x1024", "portrait": "1024x1792", "landscape": "1792x1024"},
}


def resolve_size(model_id: str, operator_token: str) -> str:
    """Map operator-facing 'square'/'portrait'/'landscape' to a concrete WxH
    string per the model's family. If `operator_token` is already a concrete
    'WxH', it is returned unchanged. Falls back to '1024x1024' if the model
    family is unknown."""
    if "x" in operator_token and operator_token.split("x")[0].isdigit():
        return operator_token
    m = _REGISTRY.get(model_id)
    family = m.family if m else "FLUX"
    return _SIZE_MAP_BY_FAMILY.get(family, _SIZE_MAP_BY_FAMILY["FLUX"]).get(
        operator_token, "1024x1024",
    )


def quality_to_steps(quality: str) -> int:
    """Operator quality token → step count.

    FLUX-schnell is a distilled model that produces good output in 4 steps;
    adding more does not improve quality. FLUX-dev benefits from 20-50 steps.
    Because we cannot vary the step count by model from the hub side without
    extra request shape, v1 ships a single mapping (draft=4, quality=20)
    that matches schnell defaults exactly and is the minimum reasonable for
    dev. The agent driver may override per-model if needed.

    Future: per-model `default_steps` field on the registry entry, used to
    map quality tier → steps differently per model.
    """
    return 20 if quality == "quality" else 4
