# macOS Launch Agent Setup

Run the LLMesh node agent automatically at login using macOS `launchd`.

---

## Overview

`launchd` is the native macOS service manager. A **Launch Agent** is a small XML plist file that tells `launchd` what to run, when to start it, and what to do if it exits.

- No third-party tools required
- Survives reboots — starts automatically on every login
- Restarts the agent automatically if it crashes (`KeepAlive`)
- Logs stdout and stderr to files for debugging

---

## Where the Plist Must Live

`launchd` only scans specific directories:

| Directory | Scope |
|---|---|
| `~/Library/LaunchAgents/` | Your user, runs at your login (recommended) |
| `/Library/LaunchAgents/` | All users, runs when any user logs in |
| `/Library/LaunchDaemons/` | Runs as root at boot, before login |

The agent script itself can live anywhere. Only the plist must be in one of the above directories.

---

## The Plist File

A template is provided at `com.qcoda.mesh.plist.example` in the repository root. Copy it, replace the placeholders, and install:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.qcoda.mesh</string>

    <key>ProgramArguments</key>
    <array>
        <string>/path/to/your/venv/bin/python</string>
        <string>/path/to/llmesh-repo/lib/agent/client.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/path/to/llmesh-repo</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/tmp/com.qcoda.mesh.log</string>

    <key>StandardErrorPath</key>
    <string>/tmp/com.qcoda.mesh.err</string>
</dict>
</plist>
```

### Key Settings

| Key | Purpose |
|---|---|
| `Label` | Unique identifier — matches the filename |
| `ProgramArguments` | Python binary (from your venv) + the agent script path |
| `WorkingDirectory` | Sets CWD so `.env` and relative paths resolve correctly |
| `RunAtLoad` | Start immediately when the agent is loaded |
| `KeepAlive` | Restart the agent automatically if it exits or crashes |
| `StandardOutPath` | Where `print()` / stdout output is written |
| `StandardErrorPath` | Where errors and tracebacks are written |

---

## Required Environment Variables

The agent needs two environment variables to connect to the hub:

| Variable | Purpose |
|---|---|
| `HUB_URL` | The URL of your LLMesh hub (e.g. `http://localhost:4000`) |
| `LLMESH_API_KEY` | API key used to authenticate with the hub |

You can set these in one of two ways:

1. **Project `.env` file** — add them to the `.env` file in your project directory. As long as `WorkingDirectory` in the plist points to your project root, the agent will pick them up automatically via `python-dotenv`.

2. **Plist `EnvironmentVariables`** — define them directly in the plist by adding an `EnvironmentVariables` block inside the top-level `<dict>`:

```xml
<key>EnvironmentVariables</key>
<dict>
    <key>HUB_URL</key>
    <string>http://localhost:4000</string>
    <key>LLMESH_API_KEY</key>
    <string>your-api-key-here</string>
</dict>
```

If both are present, the plist environment variables take precedence over the `.env` file.

---

## Step-by-Step Installation

### Step 1 — Create the Plist

```bash
cp com.qcoda.mesh.plist.example ~/Library/LaunchAgents/com.qcoda.mesh.plist
```

Edit the file and replace all `/path/to/...` placeholders with your actual paths:

```bash
nano ~/Library/LaunchAgents/com.qcoda.mesh.plist
```

### Step 2 — Validate the Plist

```bash
plutil ~/Library/LaunchAgents/com.qcoda.mesh.plist
```

Expected output: `com.qcoda.mesh.plist: OK`

Fix any errors before proceeding. A malformed plist will silently fail to load.

### Step 3 — Load the Agent

```bash
launchctl load ~/Library/LaunchAgents/com.qcoda.mesh.plist
```

This registers the agent with `launchd` and starts it immediately.

### Step 4 — Verify It Is Running

```bash
launchctl list | grep qcoda
```

Interpret the output:

| Output | Meaning |
|---|---|
| `123  0  com.qcoda.mesh` | Running — `123` is the PID |
| `-    0  com.qcoda.mesh` | Exited cleanly (exit code 0) |
| `-    2  com.qcoda.mesh` | Crashed — check the error log |

### Step 5 — Watch the Logs

Open two terminal tabs:

```bash
tail -f /tmp/com.qcoda.mesh.log   # stdout
tail -f /tmp/com.qcoda.mesh.err   # stderr / errors
```

### Step 6 — Test KeepAlive Behaviour

```bash
# Find the PID
launchctl list | grep qcoda

# Kill the process
kill <PID>

# Wait a few seconds, then verify it restarted
launchctl list | grep qcoda
```

`launchd` should restart the agent automatically within a few seconds.

---

## Managing the Agent

| Action | Command |
|---|---|
| Load / start | `launchctl load ~/Library/LaunchAgents/com.qcoda.mesh.plist` |
| Unload / stop | `launchctl unload ~/Library/LaunchAgents/com.qcoda.mesh.plist` |
| Check status | `launchctl list \| grep qcoda` |
| View stdout log | `tail -f /tmp/com.qcoda.mesh.log` |
| View error log | `tail -f /tmp/com.qcoda.mesh.err` |

Always unload the agent before editing the plist, then reload it after saving changes.

---

## Startup Race Condition: Ollama

If Ollama is also managed by `launchd`, the LLMesh agent may start before Ollama is ready. The agent handles this automatically: it checks for available backends before registering and retries with exponential backoff (5s, 10s, 20s, 40s, 60s) for up to 10 attempts (~6 minutes). If no backends are found after all retries, the agent exits and `KeepAlive` will respawn it.

Check the stdout log to see retry progress:

```
No models found on any backend (attempt 1/10). Retrying in 5s...
No models found on any backend (attempt 2/10). Retrying in 10s...
```

---

## Troubleshooting

### Agent Shows a Non-Zero Exit Code

Check the error log first:

```bash
cat /tmp/com.qcoda.mesh.err
```

Common causes:

- Wrong script path in the plist — double-check spelling and case
- Wrong Python path — run `which python3` or check your venv (`ls /path/to/your/venv/bin/python`)
- Missing dependency — the venv may not have all packages installed
- Bad `.env` variable — a required env var is missing or malformed

### Agent Does Not Start at Login

- Confirm the plist is in `~/Library/LaunchAgents/` (not a subdirectory)
- Confirm the filename ends in `.plist`
- Run `plutil` on the file to check for XML syntax errors
- Verify the agent is loaded: `launchctl list | grep qcoda`

### Using a Virtual Environment

Always point `ProgramArguments` to the venv's Python binary, not the system `python3`:

```bash
# Find your venv Python
ls /path/to/llmesh-repo/.venv/bin/python

# In the plist:
<string>/path/to/llmesh-repo/.venv/bin/python</string>
```

### The .env File Is Not Being Read

Ensure `WorkingDirectory` is set to the directory containing your `.env` file. With `WorkingDirectory` set, `python-dotenv`'s `load_dotenv()` resolves relative to that path.
