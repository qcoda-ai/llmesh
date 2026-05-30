#!/bin/bash
# LLMesh server-side deploy script. Lives at ~/deploy.sh on your-deploy-host.example.com,
# symlinked to /opt/llmesh/app/deploy/deploy.sh. Adapted from
# ../qcoda/deploy/deploy.sh.
#
# Server-side prerequisites (one-time, see docs/cicd_setup_circleci.md):
#   - llmesh user exists with /bin/bash shell + ~/.ssh/authorized_keys
#   - repo cloned to /opt/llmesh/app (D087, relocates from /home for SELinux)
#   - /home/llmesh/deploy.sh symlinked to /opt/llmesh/app/deploy/deploy.sh
#   - sudoers exception for the llmesh user to manage the llmesh service
#   - /etc/systemd/system/llmesh.service is a copy (or symlink) of
#     /opt/llmesh/app/deploy/llmesh.service. On SELinux Enforcing systems,
#     `cp + restorecon` is required (the symlink can't pick up the right
#     systemd_unit_file_t context). Both shapes work; deploy.sh detects drift.
#
# CircleCI invokes this as: ssh llmesh@<host> 'bash -l -c "cd ~ && ./deploy.sh main"'
set -e

# ====================== LOGGING SETUP WITH ROTATION ======================
LOG_FILE="deploy.log"
MAX_LOGS=10

for i in $(seq $((MAX_LOGS-1)) -1 1); do
    if [ -f "${LOG_FILE}.${i}" ]; then
        mv "${LOG_FILE}.${i}" "${LOG_FILE}.$((i+1))" 2>/dev/null || true
    fi
done
if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "${LOG_FILE}.1"
fi
exec > >(tee -a "$LOG_FILE") 2>&1

echo "====================================================================="
echo "Deployment started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running as user: $(whoami)"
echo "Branch: ${1:-main}"
echo "====================================================================="

echo "Current shell: $SHELL"
echo "PATH: $PATH"

# ==================== SAFE SUDO CHECK ====================
# `sudo -n -l <cmd>` dry-runs the sudoers rule — exits 0 iff the rule permits.
# `daemon-reload` is required so systemd picks up unit-file edits from
# deploy/llmesh.service after git pull.
echo "Testing sudo access for service management..."
if sudo -n -l /usr/bin/systemctl start llmesh >/dev/null 2>&1 \
   && sudo -n -l /usr/bin/systemctl stop llmesh >/dev/null 2>&1 \
   && sudo -n -l /usr/bin/systemctl restart llmesh >/dev/null 2>&1 \
   && sudo -n -l /usr/bin/systemctl is-active llmesh >/dev/null 2>&1 \
   && sudo -n -l /usr/bin/systemctl status llmesh >/dev/null 2>&1 \
   && sudo -n -l /usr/bin/systemctl daemon-reload >/dev/null 2>&1; then
    echo "Sudo OK (all required systemctl rules permitted)."
else
    echo "Sudo FAILED — check sudoers for the llmesh user."
    echo "Run as llmesh to debug: sudo -n -l 2>&1"
    exit 1
fi
# ========================================================

APP_DIR="/opt/llmesh/app"
BRANCH="${1:-main}"
VENV_DIR=".venv"

cd "$APP_DIR"

echo "Pulling latest code from branch: $BRANCH..."
git fetch origin
git reset --hard "origin/$BRANCH"
echo "HEAD now at: $(git rev-parse --short HEAD) — $(git log -1 --format='%s')"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip
pip install -e .

# Hub-only install profile. Agent host installs would add [desktop] separately.

# Verify gunicorn is in the venv — the systemd unit launches gunicorn, not
# uvicorn. If a future pyproject.toml edit drops gunicorn from the runtime
# deps the service would fail with a confusing ExecStart "file not found"
# message. Catch it here.
if ! [ -x "$APP_DIR/$VENV_DIR/bin/gunicorn" ]; then
    echo "ERROR: gunicorn not installed in $VENV_DIR/bin/."
    echo "pyproject.toml must include gunicorn under [project.dependencies]."
    exit 1
fi

# ==================== SYSTEMD UNIT DRIFT CHECK ====================
# git pull updates deploy/llmesh.service in the repo, but systemd loads the
# unit from /etc/systemd/system/llmesh.service. If the operator cp'd it once
# instead of symlinking, future unit-file changes silently don't take effect.
SYSTEM_UNIT="/etc/systemd/system/llmesh.service"
REPO_UNIT="$APP_DIR/deploy/llmesh.service"
if [ -L "$SYSTEM_UNIT" ]; then
    LINK_TARGET="$(readlink -f "$SYSTEM_UNIT")"
    if [ "$LINK_TARGET" != "$(readlink -f "$REPO_UNIT")" ]; then
        echo "ERROR: $SYSTEM_UNIT is a symlink but points elsewhere: $LINK_TARGET"
        echo "       Expected: $(readlink -f "$REPO_UNIT")"
        exit 1
    fi
    echo "systemd unit is symlinked → repo (good)."
elif [ -f "$SYSTEM_UNIT" ]; then
    if ! diff -q "$SYSTEM_UNIT" "$REPO_UNIT" >/dev/null 2>&1; then
        echo "WARN: $SYSTEM_UNIT differs from $REPO_UNIT — unit file changes won't take effect."
        diff "$SYSTEM_UNIT" "$REPO_UNIT" || true
        echo "Recommended: replace the copy with a symlink:"
        echo "  sudo rm $SYSTEM_UNIT"
        echo "  sudo ln -s $REPO_UNIT $SYSTEM_UNIT"
        echo "  sudo systemctl daemon-reload"
        exit 1
    fi
    echo "systemd unit is a static copy + matches repo (consider symlinking)."
else
    echo "ERROR: $SYSTEM_UNIT does not exist. First-time setup needed:"
    echo "  sudo ln -s $REPO_UNIT $SYSTEM_UNIT"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable llmesh"
    exit 1
fi
# ==================================================================

echo "Restarting service..."
# Tolerate non-zero on stop/start so the diagnostic block below runs.
# With Type=exec, `systemctl start` returns the unit's exit code — gunicorn
# import error → non-zero → set -e would kill the script before we get to
# print journal lines. `|| true` defers the failure decision to the
# is-active gate, which always runs.
set +e
if sudo /usr/bin/systemctl is-active llmesh >/dev/null 2>&1; then
    sudo /usr/bin/systemctl stop llmesh
    sleep 2
fi
sudo /usr/bin/systemctl daemon-reload
sudo /usr/bin/systemctl start llmesh
START_RC=$?
set -e

# Wait for startup (uvicorn typically <3s; compression model load can stretch this).
sleep 5

if ! sudo /usr/bin/systemctl is-active llmesh >/dev/null 2>&1; then
    echo "ERROR: Service failed to start (systemctl start rc=$START_RC). Last 50 journal lines:"
    sudo journalctl -u llmesh -n 50 --no-pager || true
    exit 1
fi

# Health endpoint must answer 200.
if ! curl -fsS http://localhost:8003/health >/dev/null; then
    echo "ERROR: /health did not return 200. Last 30 journal lines:"
    sudo journalctl -u llmesh -n 30 --no-pager || true
    exit 1
fi

# Verify gunicorn worker count matches the unit's `-w N`. Catches the case
# where master booted but workers crashed during lifespan startup. We pin
# this at 1 in deploy/llmesh.service (single-instance hub constraint —
# see decisions.md::D086). Any value other than 1 here means either the
# unit was hand-edited (operator drift) or workers are dying.
EXPECTED_WORKERS=1
# `pgrep -c` always prints a count AND returns non-zero when 0 matches.
# Past surprises:
#   (1) `|| echo 0` doubled the output (`0\n0`) → broke the integer check.
#   (2) `$(pgrep -c ...)` with `set -e` active propagates pgrep's non-zero
#       exit, killing the script silently on hosts where gunicorn workers
#       don't match the proctitle regex (no `setproctitle` in the venv).
# Belt-and-braces: capture inside `set +e`, then default + arithmetic-coerce.
set +e
RUNNING_WORKERS=$(pgrep -cf 'gunicorn:.*worker' 2>/dev/null)
RUNNING_WORKERS=${RUNNING_WORKERS:-0}
RUNNING_WORKERS=$((RUNNING_WORKERS + 0))
set -e
echo "Expected gunicorn workers: $EXPECTED_WORKERS  Running: $RUNNING_WORKERS"
if [ "$RUNNING_WORKERS" -lt "$EXPECTED_WORKERS" ]; then
    echo "WARN: $RUNNING_WORKERS of $EXPECTED_WORKERS gunicorn workers visible to pgrep."
    echo "      Service is up (health gate passed); proc-name match may be off."
    echo "      Investigate via: pgrep -af gunicorn"
    # Don't exit non-zero — service is up, just under-provisioned (or invisible to pgrep).
fi

echo "Deployment OK — service active + /health 200 + workers=$RUNNING_WORKERS."

# ==================== AGENT RESTART (CONDITIONAL) ====================
# If the host also runs `meshclient.service` (the agent), restart it so it
# picks up the new code in /opt/llmesh/app/.venv. Conditional on the unit
# existing — hub-only hosts skip this block. Sudoers exception per D091.
if systemctl list-unit-files meshclient.service >/dev/null 2>&1; then
    # `sudo -n` (non-interactive) → if the sudoers rule doesn't permit
    # `systemctl restart meshclient`, sudo exits non-zero immediately
    # instead of prompting for a password (which CircleCI's non-tty
    # session can't answer — deploy would hang until no_output_timeout
    # tripped). Operator extends sudoers per docs/cicd_setup_circleci.md
    # §1.13. Until then, agent restart is skipped with a clear warning.
    if sudo -n /usr/bin/systemctl is-active meshclient >/dev/null 2>&1 \
       || sudo -n true >/dev/null 2>&1; then
        echo "Agent (meshclient) detected — restarting..."
        sudo -n /usr/bin/systemctl restart meshclient
        sleep 3
        if sudo -n /usr/bin/systemctl is-active meshclient >/dev/null 2>&1; then
            echo "Agent restart OK — meshclient active."
        else
            echo "WARN: meshclient failed to start. Last 30 journal lines:"
            sudo -n journalctl -u meshclient -n 30 --no-pager || true
            # Don't exit non-zero — hub deploy succeeded; agent failure is
            # operator-visible and recoverable without rolling the hub back.
        fi
    else
        echo "WARN: sudoers rule for systemctl restart meshclient is missing."
        echo "      Extend /etc/sudoers.d/llmesh-deploy per docs/cicd_setup_circleci.md §1.13."
        echo "      Skipping agent restart — operator-recoverable, hub deploy succeeded."
    fi
else
    echo "No agent unit on this host — skipping agent restart."
fi
# ====================================================================
