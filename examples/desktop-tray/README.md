# Example: Desktop Tray Wrapper

> **This is an example, not a v0 release feature.** It demonstrates how to
> wrap the headless `lib/agent/client.py` polling agent in a native menu-bar
> / system-tray icon. It is unpolished and unsupported. See decision **D012**
> in `.qcoda/decisions.md`.

A `pystray`-based wrapper that lets a user start/stop an LLMesh agent node
from their menu bar (macOS) or system tray (Windows) without keeping a
terminal open. The full build/packaging guide lives in
[`docs/desktop_client.md`](../../docs/desktop_client.md).

## Quick start

From the project root, in a venv:

```bash
pip install '.[desktop]'
python examples/desktop-tray/app.py
```

This example imports `lib.agent.client`, so it must be run from the project
root (or with the project installed via `pip install .`).

## Configuration

Either set `HUB_URL` and `LLMESH_API_KEY` as environment variables, or place
a `.env` file containing them next to `app.py`.

## Building a native binary

See [`docs/desktop_client.md`](../../docs/desktop_client.md) for the full
PyInstaller build guide (macOS `.app`, Windows `.exe`, GitHub Actions CI).
