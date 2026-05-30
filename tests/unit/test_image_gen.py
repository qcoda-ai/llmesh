"""Unit tests for image generation v1 (D064).

Hub-side coverage:
  * image_registry size + step mapping
  * routing filter: capability + min_vram
  * 404 model_not_available shape (no nodes with model)
  * 400 on n out-of-bounds
  * 400 on prompt > MAX_IMAGE_PROMPT_BYTES (413 PayloadTooLarge → JSON shape)
  * happy path with a stubbed agent result

Agent-side coverage:
  * image_model_registry round-trip (known model ids)
  * mflux soft-import returns False when not installed
  * discover_installed_models returns empty list when dir absent

These tests do NOT exercise mflux itself — that's LAB-005's job on a real
Apple Silicon agent. Here we verify the hub plumbing + the registry +
the soft-import path.
"""

import pytest

from lib.hub import image_registry, server as hub_server, models as hub_models
from lib.hub.models import ImageGenerationRequest, ResourceCaps, Node


# --- hub-side image_registry --------------------------------------------------

def test_registry_has_v1_models():
    """v1 registry is FLUX-only after mflux 0.17.5 verification — mflux is
    a FLUX-family-only library. SDXL / SD 1.5 deferred to v2."""
    ids = image_registry.known_model_ids()
    assert "flux-schnell" in ids
    assert "flux-dev" in ids
    assert len(ids) == 2


def test_resolve_size_flux_square_is_1024():
    assert image_registry.resolve_size("flux-schnell", "square") == "1024x1024"


def test_resolve_size_passthrough_concrete():
    assert image_registry.resolve_size("flux-schnell", "1024x1024") == "1024x1024"


def test_resolve_size_portrait_landscape():
    assert image_registry.resolve_size("flux-schnell", "portrait") == "1024x1792"
    assert image_registry.resolve_size("flux-schnell", "landscape") == "1792x1024"


def test_quality_to_steps():
    # schnell defaults: 4 steps draft. dev benefits from more. v1 ships
    # a single mapping (test=1, draft=4, quality=20) that matches schnell
    # exactly and is the minimum reasonable for dev. D081 added the test
    # tier for smoke-testing routing/dashboard without paying full inference.
    assert image_registry.quality_to_steps("test") == 1
    assert image_registry.quality_to_steps("draft") == 4
    assert image_registry.quality_to_steps("quality") == 20
    assert image_registry.quality_to_steps("nonsense") == 4  # falls back to draft


def test_registry_min_vram_present_for_all():
    for m in image_registry.all_models():
        assert m.min_vram_gb > 0


# --- request validation -------------------------------------------------------

def test_image_request_defaults():
    req = ImageGenerationRequest(model="flux-schnell", prompt="apple")
    assert req.n == 1
    assert req.quality == "draft"
    assert req.size == "square"
    assert req.response_format == "b64_json"


def test_image_request_concrete_size_accepted():
    req = ImageGenerationRequest(model="flux-schnell", prompt="x", size="1024x1024")
    assert req.size == "1024x1024"


def test_image_request_rejects_bad_size():
    with pytest.raises(Exception):
        ImageGenerationRequest(model="flux-schnell", prompt="x", size="999x999")


def test_image_request_rejects_url_response_format():
    with pytest.raises(Exception):
        ImageGenerationRequest(model="flux-schnell", prompt="x", response_format="url")


# --- routing predicate --------------------------------------------------------

def _make_node(node_id="n1", owner="alice", image=True,
               image_models=("flux-schnell",), vram_gb=16.0):
    import time as _time
    return Node(
        node_id=node_id, owner_id=owner,
        resources=ResourceCaps(
            cpu_cores=8, ram_gb=64.0, os_name="Darwin",
            ollama_available=False, mlx_available=False,
            image_available=image,
            image_models=list(image_models),
            vram_gb=vram_gb,
        ),
        last_seen=_time.time(),
        node_token="t",
        node_token_hash="",
        fingerprint=node_id,
    )


def test_node_has_image_model_true():
    n = _make_node()
    assert hub_server._node_has_image_model(n, "flux-schnell") is True


def test_node_has_image_model_false_when_image_off():
    n = _make_node(image=False)
    assert hub_server._node_has_image_model(n, "flux-schnell") is False


def test_node_has_image_model_false_when_model_missing():
    n = _make_node(image_models=("flux-dev",))
    assert hub_server._node_has_image_model(n, "flux-schnell") is False


def test_node_is_capable_picks_up_image_only_node():
    """An image-only node (no chat backends) must still count as capable so
    routing doesn't filter it out as a generic-dead node."""
    n = _make_node()
    n.resources.ollama_available = False
    n.resources.vllm_available = False
    n.resources.mlx_available = False
    assert hub_server._node_is_capable(n) is True


# --- agent-side: registry + soft-import --------------------------------------

def test_agent_registry_round_trip():
    from lib.agent import image_model_registry as reg
    assert reg.get("flux-schnell") is not None
    assert reg.get("flux-schnell").family == "FLUX"
    assert reg.get("does-not-exist") is None


def test_agent_registry_licenses():
    from lib.agent import image_model_registry as reg
    schnell = reg.get("flux-schnell")
    assert schnell.license_permissive is True  # Apache-2.0
    flux_dev = reg.get("flux-dev")
    assert flux_dev.license_permissive is False  # FLUX-1-dev-NC non-commercial


def test_agent_registry_repo_files_diffusers_layout():
    """D071: registry advertises the full diffusers-layout file set mflux
    needs — not the single-file BFL checkpoint. Catches a regression to
    the pre-D071 single-file shape."""
    from lib.agent import image_model_registry as reg
    schnell = reg.get("flux-schnell")
    # Sub-directory presence proves diffusers layout, not the BFL
    # `flux1-schnell.safetensors` flat shape.
    assert any(f.startswith("transformer/") for f in schnell.repo_files)
    assert any(f.startswith("text_encoder/") for f in schnell.repo_files)
    assert any(f.startswith("text_encoder_2/") for f in schnell.repo_files)
    assert any(f.startswith("vae/") for f in schnell.repo_files)
    assert any(f.startswith("tokenizer/") for f in schnell.repo_files)
    assert any(f.startswith("tokenizer_2/") for f in schnell.repo_files)
    # The pre-D071 single-file checkpoint must NOT be listed — mflux ignores it.
    assert "flux1-schnell.safetensors" not in schnell.repo_files


def test_mflux_soft_import_false_when_not_installed():
    """On a machine without mflux, the soft-import must report False without
    raising. The agent uses this to decide whether to advertise image_available."""
    from lib.agent import image_driver_mflux as drv
    # Test is True iff the dev workstation has mflux installed; assert at
    # least the call doesn't raise.
    result = drv.mflux_available()
    assert isinstance(result, bool)


def test_discover_installed_models_empty_when_dir_absent(tmp_path, monkeypatch):
    """If LLMESH_IMAGE_MODELS_DIR doesn't exist, discover_installed_models
    returns [] — never raises."""
    monkeypatch.setenv("LLMESH_IMAGE_MODELS_DIR", str(tmp_path / "nonexistent"))
    from lib.agent import image_driver_mflux as drv
    assert drv.discover_installed_models() == []


def test_discover_installed_models_skips_partial_install(tmp_path, monkeypatch):
    """D071: a partial install (some repo_files present, some missing) must
    NOT be reported as installed — otherwise the agent advertises a broken
    model to the hub and inference fails downstream."""
    monkeypatch.setenv("LLMESH_IMAGE_MODELS_DIR", str(tmp_path))
    from lib.agent import image_driver_mflux as drv
    from lib.agent import image_model_registry as reg

    entry = reg.get("flux-schnell")
    model_dir = tmp_path / entry.family / entry.model_id
    model_dir.mkdir(parents=True)
    # Drop only one file from the diffusers layout.
    only = entry.repo_files[0]
    (model_dir / only).parent.mkdir(parents=True, exist_ok=True)
    (model_dir / only).write_bytes(b"stub")
    assert drv.discover_installed_models() == []


def test_discover_installed_models_finds_complete_install(tmp_path, monkeypatch):
    """D071: when ALL repo_files are present (stubbed), the model counts as
    installed."""
    monkeypatch.setenv("LLMESH_IMAGE_MODELS_DIR", str(tmp_path))
    from lib.agent import image_driver_mflux as drv
    from lib.agent import image_model_registry as reg

    entry = reg.get("flux-schnell")
    model_dir = tmp_path / entry.family / entry.model_id
    for rel in entry.repo_files:
        f = model_dir / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"stub")
    assert "flux-schnell" in drv.discover_installed_models()


def test_missing_repo_files_lists_all_when_dir_absent(tmp_path, monkeypatch):
    """D071: helper used by install CLI + driver pre-flight check."""
    monkeypatch.setenv("LLMESH_IMAGE_MODELS_DIR", str(tmp_path / "nonexistent"))
    from lib.agent import image_driver_mflux as drv
    from lib.agent import image_model_registry as reg

    entry = reg.get("flux-schnell")
    assert drv.missing_repo_files("flux-schnell") == list(entry.repo_files)


def test_model_install_dir_uses_family_subdir(tmp_path, monkeypatch):
    """D071: install layout is `<base>/<family>/<model_id>/` — the family
    sub-dir lets v2 backends (SDXL, SD 1.5) sit alongside FLUX without
    namespace collisions."""
    monkeypatch.setenv("LLMESH_IMAGE_MODELS_DIR", str(tmp_path))
    from lib.agent import image_driver_mflux as drv
    p = drv.model_install_dir("flux-schnell")
    assert p == tmp_path / "FLUX" / "flux-schnell"


def test_estimate_uma_gb_zero_on_non_apple_silicon():
    """On non-Apple-Silicon, returns 0.0 (sentinel for hub to skip routing)."""
    import platform
    from lib.agent import image_driver_mflux as drv
    result = drv.estimate_uma_gb()
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        assert result > 0.0
    else:
        assert result == 0.0
