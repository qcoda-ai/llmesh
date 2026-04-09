# LLMesh Desktop Tray — Example Build Guide

> **Status: example, not a v0 release feature.** The desktop tray client lives
> at `examples/desktop-tray/` and is provided as a reference for wrapping the
> headless agent in a native menu-bar/system-tray icon. It is not part of the
> supported release surface. If you need a polished cross-platform desktop
> binary, treat this as a starting point, not a finished product.

The example wraps `lib/agent/client.py` (the polling agent) in a `pystray`
icon so a user can start/stop the node from their menu bar without an open
terminal. It uses two libraries:

- **`pystray`** — native OS system tray icon and menu
- **`PyInstaller`** — bundles the Python project into standalone binaries
  (`.app` for macOS, `.exe` for Windows)

## Prerequisites

Install LLMesh in a venv with the desktop and build extras:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install '.[desktop,build]'
```

This pulls in `pystray`, `Pillow`, the macOS-only `pyobjc-*` packages (skipped
automatically on Linux/Windows via platform markers), and the PyInstaller
toolchain.

## Running for Development

To test the tray app directly from source without building:

```bash
python examples/desktop-tray/app.py
```

Run from the project root — the script inserts the project root into
`sys.path` so it can `import lib.agent.client`.

*Note: depending on your OS security settings, the terminal running this
script may request network or accessibility permissions to run in the
background.*

## Building the Executable

PyInstaller bundles the script, the Python runtime, and all installed
packages into a distributable format. PyInstaller **cannot cross-compile**
(you cannot build a Windows `.exe` on a Mac), so you must run the build
command on the target operating system.

### Compiling on macOS (`.app`)

1. Navigate to the LLMesh project root.
2. Run PyInstaller with `--windowed` and include the `icon.png` resource:
   ```bash
   pyinstaller --name "LLMesh" --windowed \
       --add-data "examples/desktop-tray/icon.png:." \
       --noconfirm examples/desktop-tray/app.py
   ```
   *If you want a native `.icns` icon for the Dock/Finder, see the Advanced
   section.*
3. After compilation, the macOS application is at `dist/LLMesh.app`.
4. Drag `LLMesh.app` into your `/Applications` folder.

### Compiling on Windows (`.exe`)

1. Navigate to the LLMesh project root in Command Prompt or PowerShell.
2. Run PyInstaller with `--icon` and include the `icon.png` resource:
   ```bash
   pyinstaller --name "LLMesh" --noconsole --onefile ^
       --icon "examples/desktop-tray/icon.ico" ^
       --add-data "examples/desktop-tray/icon.png;." ^
       --noconfirm examples/desktop-tray/app.py
   ```
3. The Windows executable is at `dist/LLMesh.exe`.

## Cross-Platform Building (Building `.exe` on a Mac)

`PyInstaller` is **not a cross-compiler**. To generate a Windows `.exe`, run
the build on a Windows host. Two approaches:

### 1. GitHub Actions (Recommended)

Automate the build with CI. Example `.github/workflows/build.yml`:

```yaml
name: Build Desktop Apps
on: [push]
jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install '.[desktop,build]'
      - name: Build with PyInstaller
        run: |
          if [ "${{ matrix.os }}" = "windows-latest" ]; then
            pyinstaller --name "LLMesh" --noconsole --onefile \
                --add-data "examples/desktop-tray/icon.png;." \
                --noconfirm examples/desktop-tray/app.py
          else
            pyinstaller --name "LLMesh" --windowed \
                --add-data "examples/desktop-tray/icon.png:." \
                --noconfirm examples/desktop-tray/app.py
          fi
        shell: bash
      - name: Upload Artifact
        uses: actions/upload-artifact@v4
        with:
          name: LLMesh-${{ matrix.os }}
          path: dist/
```

### 2. Virtual Machines

Use Parallels, VMware, or VirtualBox to run a Windows instance on your Mac.
Inside the VM, clone the repo and run the standard Windows build command.

## Runtime Configuration

The compiled application has no setup wizard. It looks for its configuration
in two places:

1. **System environment variables** — set `HUB_URL` and `LLMESH_API_KEY`
   globally on the OS.
2. **Local `.env` file** — place a `.env` containing `HUB_URL` and
   `LLMESH_API_KEY` next to `LLMesh.exe` or inside the `LLMesh.app` bundle.
