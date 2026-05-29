#!/usr/bin/env python3
"""
llmesh-agent install-image-model <model_id>  (D064 + D071)

Operator-explicit weight download for image-gen models. NEVER auto-runs from
the agent — operator invokes this script directly. Prints size + license +
post-install disk-free; requires `--accept-license` for non-permissive
entries; requires `--yes` to skip the size/disk confirmation.

Usage:
    python scripts/install_image_model.py list
    python scripts/install_image_model.py install <model_id> [--yes] [--accept-license]
    python scripts/install_image_model.py installed
    python scripts/install_image_model.py prune --unused           # listing only; v1 doesn't auto-delete

`LLMESH_IMAGE_MODELS_DIR` overrides the default `~/.llmesh/models/image/`.

D071: install uses curl-per-file over the registry `repo_files` list to
populate the diffusers-layout directory mflux expects. Bypasses the
`huggingface_hub.hf_hub_download` hang documented in `.qcoda/lessons.md`
(2026-05-28) for large gated FLUX repos. Resumable via curl `-C -`.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


sys.path.insert(0, str(_repo_root()))

# Load .env from the repo root if present so the operator can set
# `LLMESH_IMAGE_MODELS_DIR` (and other LLMESH_* vars) in one place that
# both the agent runtime and this CLI consult. System env still wins over
# .env per python-dotenv default behaviour. See `docs/image_gen.md`.
try:
    from dotenv import load_dotenv  # type: ignore
    _env_path = _repo_root() / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    # python-dotenv missing → operator can still set system env vars.
    pass

from lib.agent import image_model_registry as reg  # noqa: E402
from lib.agent import image_driver_mflux as drv  # noqa: E402


def _models_dir() -> Path:
    return Path(os.environ.get(
        "LLMESH_IMAGE_MODELS_DIR",
        str(Path.home() / ".llmesh" / "models" / "image"),
    ))


def _install_log_path() -> Path:
    base = Path(os.environ.get("LLMESH_STATE_DIR", str(Path.home() / ".llmesh")))
    return base / "install.log"


def _append_install_log(line: str) -> None:
    p = _install_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    with p.open("a", encoding="utf-8") as f:
        f.write(f"{ts}  {line}\n")


def _disk_free_gb(path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(path.parent if not path.exists() else path))
    return round(usage.free / (1024.0 ** 3), 2)


def cmd_list(_args) -> int:
    """Print the registry — model_id, family, size, min VRAM, license."""
    print(f"{'MODEL_ID':22s} {'FAMILY':12s} {'SIZE':>6s} {'VRAM':>4s}  LICENSE")
    print(f"{'-'*22:22s} {'-'*12:12s} {'-'*6:>6s} {'-'*4:>4s}  {'-'*40}")
    for m in reg.all_models():
        print(f"{m.model_id:22s} {m.family:12s} {m.size_gb:>5.1f}G {m.min_vram_gb:>3d}G  {m.license_label}")
    return 0


def cmd_installed(_args) -> int:
    """Print which models are present in LLMESH_IMAGE_MODELS_DIR."""
    base = _models_dir()
    installed = drv.discover_installed_models()
    print(f"Image models dir: {base}")
    if not installed:
        print("(none installed)")
        return 0
    for mid in installed:
        m = reg.get(mid)
        if m:
            print(f"  {mid:22s} family={m.family:12s} size={m.size_gb}G")
        else:
            print(f"  {mid:22s} (NOT in registry)")
    return 0


def _resolve_hf_token() -> str | None:
    """Resolve a Hugging Face read-scope token from env or local config.

    Order: `HF_TOKEN` env > `HUGGING_FACE_HUB_TOKEN` env > `~/.cache/huggingface/token`
    (written by `hf auth login`). Returns None if nothing is set; the caller
    surfaces the gated-repo guidance.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(var)
        if v:
            return v.strip()
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        try:
            return token_file.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None


def _curl_download(url: str, target: Path, token: str | None) -> tuple[bool, str]:
    """Resumable curl-per-file fetch. Returns (success, stderr-tail).

    Uses `-C -` for resume, `-f` for HTTP error fail-fast, `-L` follow
    redirects, `--retry 3 --retry-delay 2` for transient retries, and
    `-#` for compact per-file progress bar (visible on operator's TTY).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-L", "-f", "-C", "-",
        "--retry", "3", "--retry-delay", "2",
        "--connect-timeout", "30",
        "-#",
        "-o", str(target),
    ]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    cmd.append(url)
    try:
        # Stream curl's stderr (progress + errors) through to operator TTY.
        proc = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        return False, "curl not found on PATH (install curl or use a system that ships it)"
    if proc.returncode != 0:
        return False, f"curl exit code {proc.returncode}"
    return True, ""


def cmd_install(args) -> int:
    """Download + verify + install one model from the registry.

    D071: curl-per-file over the registry's `repo_files` list. Each file
    is placed at `<target_dir>/<repo_file>`, preserving the diffusers
    sub-directory layout mflux expects (`transformer/`, `text_encoder/`,
    `text_encoder_2/`, `vae/`, `tokenizer/`, `tokenizer_2/`, `scheduler/`).
    Files already present (full size on disk) are skipped; partial files
    are resumed via `curl -C -`.
    """
    m = reg.get(args.model_id)
    if m is None:
        print(f"ERROR: unknown model_id {args.model_id!r}", file=sys.stderr)
        print(f"Known: {', '.join(reg.known_model_ids())}", file=sys.stderr)
        return 2

    base = _models_dir()
    target_dir = base / m.family / m.model_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight: print everything the operator needs to consent to.
    print("=" * 64)
    print(f"Model:         {m.model_id}")
    print(f"Family:        {m.family}")
    print(f"HF repo:       {m.hf_repo}")
    print(f"Files:         {len(m.repo_files)} (diffusers layout, multi-file)")
    print(f"Approx size:   {m.size_gb:.1f} GB")
    print(f"Min VRAM:      {m.min_vram_gb} GB")
    print(f"Install path:  {target_dir}")
    print(f"Disk free:     {_disk_free_gb(target_dir):.1f} GB")
    post_install_free = _disk_free_gb(target_dir) - m.size_gb
    print(f"After install: {post_install_free:.1f} GB free (approx)")
    if post_install_free < 10.0:
        print("WARN: post-install free disk would drop below 10 GB.", file=sys.stderr)
        if not args.force:
            print("Refusing to install. Pass --force to override.", file=sys.stderr)
            return 3
    print()
    print(f"License:       {m.license_label}")
    print(f"               ({m.license_id})")
    print(f"License URL:   {m.license_url}")
    print()

    # License gate.
    if not m.license_permissive and not args.accept_license:
        print(f"ERROR: {m.license_id} is non-permissive. Pass --accept-license to confirm "
              "you have read the license text and accept its terms.", file=sys.stderr)
        return 4

    # Confirm gate.
    if not args.yes:
        ans = input("Proceed with install? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return 0

    token = _resolve_hf_token()
    if token is None:
        print(
            "WARN: no Hugging Face token found (checked HF_TOKEN, HUGGING_FACE_HUB_TOKEN, "
            "~/.cache/huggingface/token). Gated repos will fail with 401.",
            file=sys.stderr,
        )

    # Skip files already on disk with non-zero size. Curl will resume partials
    # but a complete file (matching repo file size exactly) can be skipped
    # entirely. We don't HEAD-check size here to keep the install offline-tolerant
    # — operators can `rm` any suspect file and re-run.
    def _is_complete(p: Path) -> bool:
        return p.exists() and p.stat().st_size > 0

    print()
    print(f"Downloading {len(m.repo_files)} file(s) from {m.hf_repo} ...")
    t0 = time.time()
    failures: list[str] = []
    for idx, rel in enumerate(m.repo_files, start=1):
        target = target_dir / rel
        if _is_complete(target):
            print(f"  [{idx}/{len(m.repo_files)}] {rel}  (already present, skip)")
            continue
        url = f"https://huggingface.co/{m.hf_repo}/resolve/main/{rel}"
        print(f"  [{idx}/{len(m.repo_files)}] {rel}")
        ok, err = _curl_download(url, target, token)
        if not ok:
            print(f"    FAILED: {err}", file=sys.stderr)
            failures.append(rel)
            # First failure on a gated repo with no token → surface guidance once.
            if token is None and not failures[:-1]:
                print(
                    f"\nThis is likely a Hugging Face gated repo. FLUX weights require a free\n"
                    "Hugging Face account and a one-time access acceptance:\n"
                    "  1. Create an account at https://huggingface.co\n"
                    f"  2. Visit https://huggingface.co/{m.hf_repo} and click `Agree and access`\n"
                    "  3. Authenticate locally:\n"
                    "       hf auth login\n"
                    "     OR set the env var HF_TOKEN to your read-scope token\n"
                    "     (`huggingface-cli` was deprecated; current package ships `hf` only)\n"
                    "  4. Re-run this install command (already-downloaded files will skip)\n",
                    file=sys.stderr,
                )

    elapsed = time.time() - t0
    if failures:
        print(
            f"\n{len(failures)}/{len(m.repo_files)} file(s) failed after {elapsed:.1f}s. "
            f"Re-run to resume; first failure: {failures[0]}",
            file=sys.stderr,
        )
        return 6
    print(f"Done in {elapsed:.1f}s.")

    _append_install_log(
        f"install model={m.model_id} family={m.family} license={m.license_id} "
        f"size_gb={m.size_gb} target={target_dir}"
    )
    print()
    print(f"OK {m.model_id} installed at {target_dir}")
    print("Restart the agent for the new model to become advertised to the hub.")
    return 0


def cmd_prune(_args) -> int:
    """v1: list installed models. Auto-prune deferred to v2."""
    base = _models_dir()
    if not base.exists():
        print(f"No image models dir at {base}")
        return 0
    print("v1 lists installed models only. Delete manually if needed.")
    return cmd_installed(_args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="install_image_model",
        description="Install image-generation model weights for the LLMesh agent (D064).",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    p_list = subs.add_parser("list", help="Print the model registry")
    p_list.set_defaults(func=cmd_list)

    p_installed = subs.add_parser("installed", help="List installed models")
    p_installed.set_defaults(func=cmd_installed)

    p_install = subs.add_parser("install", help="Download a model from the registry")
    p_install.add_argument("model_id")
    p_install.add_argument("--yes", action="store_true",
                           help="Skip interactive confirm")
    p_install.add_argument("--accept-license", action="store_true",
                           help="Required for non-permissive licenses (SAIL-NC, FLUX-1-dev-NC)")
    p_install.add_argument("--force", action="store_true",
                           help="Bypass post-install free-disk check (<10GB free)")
    p_install.set_defaults(func=cmd_install)

    p_prune = subs.add_parser("prune", help="List installed models (no auto-delete in v1)")
    p_prune.add_argument("--unused", action="store_true", help="(v1 no-op)")
    p_prune.set_defaults(func=cmd_prune)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
