# CI/CD Setup — CircleCI + bare-metal systemd

How to bring up auto-deploy from Bitbucket `main` → CircleCI → `your-deploy-host.example.com` for LLMesh. Mirrors the pattern used by `../qcoda` and `../talential-python-backend`. Single environment, no staging slot.

**Provenance:** `.qcoda/decisions.md::D085`.

---

## What you're building

```
git push main  ──►  CircleCI test job  ──►  CircleCI deploy job  ──►  ssh llmesh@<host>
                    (pytest + ledger        (mirrors siblings —          ──► ~/deploy.sh main
                     validator + gitleaks)   add_ssh_keys + ssh-keyscan)  ──► git pull, pip install -e .,
                                                                              systemctl restart llmesh,
                                                                              curl /health
```

PRs run the test job only — they cannot deploy.

---

## Part 1 — Server-side setup (`your-deploy-host.example.com`)

### 1.1 Create the runtime user

Run as root (or via your normal sudo path).

```bash
useradd --create-home --shell /bin/bash llmesh
```

The `llmesh` user owns the code, the venv, and runs the systemd service.

### 1.2 Clone the repo into `/opt/llmesh/app`

`/opt` is the standard location for non-distro-package services on RHEL-family. SELinux's default policy lets systemd-managed services read `/opt`; it does NOT let them touch `/home` (D087, learned the hard way on Rocky Linux 9 with SELinux Enforcing). For Debian/Ubuntu without SELinux either path works, but `/opt` keeps the docs consistent across distros.

```bash
mkdir -p /opt/llmesh
chown llmesh:llmesh /opt/llmesh
sudo -u llmesh bash -c '
  cd /opt/llmesh
  git clone https://andrew_schwabe@bitbucket.org/andrew_schwabe/qcoda-nodemesh.git app
'
```

The local directory ends up at `/opt/llmesh/app`. The `/home/llmesh/deploy.sh` symlink (next step) is what CircleCI actually invokes.

### 1.3 Symlink the deploy script + drop the venv stub

```bash
sudo -u llmesh bash -c '
  ln -sf /opt/llmesh/app/deploy/deploy.sh /home/llmesh/deploy.sh
  cd /opt/llmesh/app
  python3.11 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e .
'
```

Confirm: `ls -la /home/llmesh/deploy.sh` should be a symlink pointing into the repo. Future `git pull` updates the script automatically.

### 1.4 Drop the production `.env`

`.env` lives at `/opt/llmesh/app/.env` (gitignored). Mirror your dev `.env` minus debugging flags. At minimum:

```
LLMESH_API_KEY=<don't actually put one here — this is for the agent only>
SESSION_DB=/opt/llmesh/app/sessions.db
TASK_DB=/opt/llmesh/app/tasks.db
SESSION_BACKEND=sqlite
# Optional: POSTGRES_DSN=... if you're running the production Postgres path
# Optional: VLLM_HOST=..., MLX_HOST=... per backend setup
```

Plus the hub's API-key store goes into `server_config.json` (also at `/opt/llmesh/app/server_config.json` — gitignored). The hub refuses to start with sample keys (D013); generate real ones.

### 1.5 Install the systemd unit (copy + restorecon on SELinux; symlink elsewhere)

The unit launches **gunicorn supervising a single uvicorn worker** (see `decisions.md::D086`). `pip install -e .` from §1.3 pulls gunicorn in as a runtime dependency, so no separate install step is needed. The unit is pinned at `-w 1` — **do not raise it**; the hub holds in-memory authoritative state (node registry, task queue, models cache) that does not survive multi-worker. Multi-worker support is a future architecture change.

**On RHEL-family (Rocky, RHEL, CentOS Stream, Alma) with SELinux Enforcing — use copy + restorecon:**

```bash
sudo cp /opt/llmesh/app/deploy/llmesh.service /etc/systemd/system/llmesh.service
sudo restorecon -v /etc/systemd/system/llmesh.service
sudo systemctl daemon-reload
sudo systemctl enable llmesh
sudo systemctl start llmesh
sudo systemctl status llmesh
```

`restorecon` writes the SELinux context (`system_u:object_r:systemd_unit_file_t:s0`) systemd expects. A symlink target in `/opt/llmesh/app/` carries `default_t` or similar, which systemd policy refuses to follow.

**On Debian/Ubuntu (no SELinux) — symlink works and auto-tracks `git pull` updates:**

```bash
sudo ln -s /opt/llmesh/app/deploy/llmesh.service /etc/systemd/system/llmesh.service
sudo systemctl daemon-reload
sudo systemctl enable llmesh
sudo systemctl start llmesh
sudo systemctl status llmesh
```

The deploy script handles both shapes. If you used the copy, future `deploy/llmesh.service` edits in the repo require a re-copy: `deploy.sh` aborts on drift and prints the remediation. If you used the symlink, `git pull` updates the unit automatically and `systemctl daemon-reload` picks it up.

Logs land in the journal (`journalctl -u llmesh -f`) — gunicorn writes access + error logs to stdout/stderr per the unit's `--access-logfile - --error-logfile -` flags.

### 1.6 Sudoers rule for the llmesh user

```bash
EDITOR=nano visudo -f /etc/sudoers.d/llmesh-deploy
```

`visudo` re-parses on save and refuses bad syntax — `EDITOR=nano` only swaps the UI. `Ctrl+O` saves, `Ctrl+X` exits.

Paste (one line, no backslash continuations — keeps the entry greppable):

```
llmesh ALL=(root) NOPASSWD: /usr/bin/systemctl start llmesh, /usr/bin/systemctl stop llmesh, /usr/bin/systemctl restart llmesh, /usr/bin/systemctl is-active llmesh, /usr/bin/systemctl status llmesh, /usr/bin/systemctl daemon-reload
```

Path-pinned, command-pinned. No wildcards, no `ALL`. The `daemon-reload` line is required so unit-file edits from `git pull` actually load.

Smoke test as the llmesh user:

```bash
sudo -u llmesh sudo -n -l /usr/bin/systemctl restart llmesh
# → should print the matched sudoers rule, NOT prompt for a password
```

### 1.7 Generate the CircleCI deploy SSH key

On your admin workstation (NOT the server):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/llmesh_circleci -C "circleci llmesh deploy" -N ""
```

`-N ""` skips the passphrase — CircleCI can't enter one. The pair:

- `~/.ssh/llmesh_circleci` (private) → upload to CircleCI in §2.2.
- `~/.ssh/llmesh_circleci.pub` (public) → append to server in the next step.

### 1.8 Install the public key on the server

```bash
ssh root@your-deploy-host.example.com 'cat >> /home/llmesh/.ssh/authorized_keys' < ~/.ssh/llmesh_circleci.pub
ssh root@your-deploy-host.example.com 'chown -R llmesh:llmesh /home/llmesh/.ssh && chmod 600 /home/llmesh/.ssh/authorized_keys'
```

(Or `ssh-copy-id` if you prefer — same effect.)

Note: this is a less-locked-down approach than the Bitbucket Pipelines `forced command + restrict` walkthrough in `.qcoda/strategy/cicd_circleci_vs_bitbucket_pipelines.md` §3. CircleCI's SSH step expects to run a literal `ssh ... 'command'`, so a forced-command `authorized_keys` entry doesn't fit cleanly. If you want the same lockdown, add `from="<circleci-runner-CIDR>"` to the `authorized_keys` line and tighten the sudoers rule further. Sibling repos (`qcoda`, `talential`) don't bother with `from=`; pick your hardening level.

### 1.9 Confirm SSH from your workstation

```bash
ssh -i ~/.ssh/llmesh_circleci llmesh@your-deploy-host.example.com 'echo ok; whoami'
# → ok
# → llmesh
```

If you get prompted for a password, the public key isn't installed correctly. If you get `Permission denied (publickey)`, sshd may be configured to reject ed25519 — check `/etc/ssh/sshd_config` for `PubkeyAcceptedAlgorithms` / `PubkeyAcceptedKeyTypes`.

### 1.10 Install nginx in front (HTTPS termination + edge hardening)

The systemd unit binds gunicorn to `0.0.0.0:8003` directly. For any deployment exposed beyond a trusted LAN you want nginx terminating HTTPS, enforcing the rate limit + scanner-trap, and forwarding `X-Forwarded-*` headers to the hub. The config template at `deploy/llmesh-nginx.conf` is adapted verbatim from `../qcoda/deploy/qcoda-nginx.conf` — same security headers, same scanner-trap, plus SSE-specific proxy tuning (`proxy_buffering off` for `/v1/chat/completions`, `/v1/messages`, dashboard stream endpoints; `proxy_read_timeout 600s` to outlast `STREAM_CHUNK_TIMEOUT=300s`).

```bash
# Install nginx
sudo apt update && sudo apt install -y nginx

# Edit the config — replace your-deploy-host.example.com with the hostname you actually run.
sudo cp /opt/llmesh/app/deploy/llmesh-nginx.conf /etc/nginx/conf.d/llmesh.conf
sudo vim /etc/nginx/conf.d/llmesh.conf
# Find every `your-deploy-host.example.com` and swap for your real hostname.

# Test + reload
sudo nginx -t
sudo systemctl reload nginx
```

Verify nginx forwards to the hub without HTTPS:

```bash
curl -fsS http://your-deploy-host.example.com/health
# → {"status":"ok"}
```

### 1.11 HTTPS with Let's Encrypt (certbot)

`certbot --nginx` will rewrite `/etc/nginx/conf.d/llmesh.conf` in-place to add the `listen 443 ssl`, cert paths, and HTTP→HTTPS redirect block. The template leaves those lines as commented placeholders so the first run is straightforward.

```bash
sudo apt install -y certbot python3-certbot-nginx

# Replace <hostname> with the real one. certbot will provision the cert AND
# rewrite the nginx config.
sudo certbot --nginx -d your-deploy-host.example.com --redirect --agree-tos -m you@example.com --non-interactive

# Verify the rewrite
sudo nginx -t
sudo systemctl reload nginx

# Smoke
curl -fsS https://your-deploy-host.example.com/health
```

certbot installs a systemd timer for auto-renewal — `sudo systemctl list-timers | grep certbot` should show `certbot.timer` running twice a day.

After certbot edits the file, the in-repo template will diverge from the on-disk config. **Do not** revert it via `git pull` — the certbot edits are operator-side, not version-controlled. If the template needs updating in the repo, edit `deploy/llmesh-nginx.conf` and re-apply by hand on the server. (We don't symlink nginx configs from the repo because certbot would write into the symlinked file.)

### 1.13 (Optional) Agent on the same box

Many deployments run the agent (`meshclient.service`) on the same host as the hub — convenient when the box also runs vLLM / Ollama / MLX. The agent unit + env template live in the repo so deploys keep both in sync (D091).

**One-time setup as root:**

```bash
sudo cp /opt/llmesh/app/deploy/meshclient.service /etc/systemd/system/meshclient.service
sudo restorecon -v /etc/systemd/system/meshclient.service
```

**Agent env (`.env.agent` — separate from the hub's `.env`):**

```bash
sudo -u llmesh cp /opt/llmesh/app/deploy/.env.agent.example /opt/llmesh/app/.env.agent
sudo chmod 600 /opt/llmesh/app/.env.agent
sudo -u llmesh nano /opt/llmesh/app/.env.agent
```

Edit at minimum:
- `LLMESH_API_KEY` — must match an entry in `/opt/llmesh/app/server_config.json` on the hub side.
- `HUB_URL` — `http://127.0.0.1:8003` for a co-located hub, or `https://<host>` for an nginx-fronted remote hub.
- `LLMESH_NODE_ID` — operator label per D048 regex.
- One backend block: vLLM / Ollama / MLX (uncomment the relevant lines).
- `VLLM_API_KEY` if vLLM runs behind a bearer-auth proxy (LiteLLM Proxy, hardened reverse proxy).

**Sudoers — extend the rule from §1.6:**

```bash
sudo EDITOR=nano visudo -f /etc/sudoers.d/llmesh-deploy
```

Replace the existing rule with the agent-extended single-line version:

```
llmesh ALL=(root) NOPASSWD: /usr/bin/systemctl start llmesh, /usr/bin/systemctl stop llmesh, /usr/bin/systemctl restart llmesh, /usr/bin/systemctl is-active llmesh, /usr/bin/systemctl status llmesh, /usr/bin/systemctl daemon-reload, /usr/bin/systemctl start meshclient, /usr/bin/systemctl stop meshclient, /usr/bin/systemctl restart meshclient, /usr/bin/systemctl is-active meshclient, /usr/bin/systemctl status meshclient
```

**Enable + start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable meshclient
sudo systemctl start meshclient
sudo systemctl status meshclient --no-pager
sudo journalctl -u meshclient -n 30 --no-pager
```

Expected: agent registers with the hub within ~5 s; the dashboard's Nodes table shows the new entry on the next refresh.

**Auto-restart on deploy.** `deploy/deploy.sh` detects `meshclient.service` on the host (via `systemctl list-unit-files`) and restarts it after the hub `/health` gate passes. Hub-only hosts skip this block.

### 1.12 (Optional) Restrict gunicorn to loopback once nginx is in front

The default unit binds `gunicorn -b 0.0.0.0:8003` so the hub is reachable directly for diagnostics. Once nginx is the only intended entry point, either:

- **Firewall option (recommended):** drop public traffic to `8000` at iptables / ufw. `sudo ufw deny 8000/tcp`. Direct access still works from the loopback.
- **Bind option:** edit `deploy/llmesh.service` to use `-b 127.0.0.1:8003` and re-deploy. More aggressive but breaks the `curl http://<host>:8003/health` smoke from outside the box; you'll only be able to hit `/health` through nginx (`curl https://<host>/health`).

Either is fine. The firewall option keeps the diagnostic path open from inside the box.

---

## Part 2 — CircleCI project setup

### 2.1 Create the project

1. Log into CircleCI with the same Bitbucket identity that owns `andrew_schwabe/qcoda-nodemesh`.
2. Add Project → pick `qcoda-nodemesh`.
3. When prompted for an existing config, point at `.circleci/config.yml` (already committed in this repo).
4. Do **not** trigger a build yet — the SSH key fingerprint isn't pinned.

### 2.2 Upload the deploy SSH key

1. Project Settings → SSH Keys → Additional SSH Keys → Add SSH Key.
2. Hostname: `your-deploy-host.example.com` (the value is informational; CircleCI matches by fingerprint).
3. Private key: paste the contents of `~/.ssh/llmesh_circleci` (the **private** half, including the BEGIN/END markers).
4. Save. CircleCI displays the SHA256 fingerprint immediately after upload.
5. Copy the fingerprint.

### 2.3 Pin the fingerprint in `.circleci/config.yml`

Open `.circleci/config.yml` in this repo. Find:

```yaml
fingerprints:
  - "SHA256:REPLACE_ME_AFTER_UPLOADING_DEPLOY_KEY"
```

Replace `REPLACE_ME_AFTER_UPLOADING_DEPLOY_KEY` with the fingerprint from §2.2. Commit + push the change — that push will trigger the first real build.

### 2.4 Set environment variables

Project Settings → Environment Variables → Add Variable.

| Name | Value | Notes |
|---|---|---|
| `SSH_PORT` | `22` (or your custom port) | Used by `ssh -p $SSH_PORT` |
| `SERVER_IP` | `your-deploy-host.example.com` | Used by `ssh ... $SERVER_IP` |

If you ever switch to IP-allowlisting (`from="..."` in `authorized_keys`), grab CircleCI's outbound IP ranges from `https://circleci.com/docs/ip-ranges/`.

### 2.5 First build

Push any commit to `main`. CircleCI:

1. Runs `test`: installs deps, runs `pytest tests/unit/` (~2s), `scripts/validate_decisions.py` (~50ms), gitleaks scan (~1–3s).
2. If `test` passes, runs `deploy`: `add_ssh_keys` injects the key, `ssh-keyscan` pins the host, `ssh llmesh@<host> './deploy.sh main'` runs the server-side script.
3. `Verify deployment` step `ssh`es again and checks `systemctl is-active llmesh` + `curl /health`.

A failure on any of the three test steps blocks the deploy. A failure inside `deploy.sh` (e.g. systemd drift check trips, pip install fails, health check fails) returns non-zero, CircleCI marks the step red, the running service is left in whatever state `deploy.sh` aborted at.

---

## Part 3 — Post-deploy operator checklist

After CircleCI goes green:

- [ ] CircleCI shows two green jobs.
- [ ] `ssh llmesh@<host> 'sudo systemctl is-active llmesh'` → `active`.
- [ ] `ssh llmesh@<host> 'sudo journalctl -u llmesh -n 100'` shows clean startup, no traceback.
- [ ] `curl -fsS https://<your-public-hostname>/health` returns 200 (or `http://your-deploy-host.example.com:8003/health` if you haven't wired nginx yet).
- [ ] Dashboard loads — log in, click around the Stats tab, confirm the TTFT chart populates if you have streaming traffic.
- [ ] If the change touched env vars, confirm they're in `/opt/llmesh/app/.env` (out-of-band — secrets aren't in git).

A failure on any of these is a rollback trigger.

---

## Part 4 — Rollback

### Light rollback — revert the commit

```bash
git revert <bad-sha>
git push main main
# CircleCI deploys the revert
```

### Full rollback — reset to a tag

For high-risk changes, cut a tag before merging:

```bash
git tag pre-<change>
git push main pre-<change>
```

If the deploy goes sideways:

```bash
git checkout main
git reset --hard pre-<change>
git push --force-with-lease main main
```

`--force-with-lease` refuses if remote has moved since your last fetch — safer than `--force`.

### Manual server-side rollback (CI not available)

```bash
ssh llmesh@<host>
cd ~/llmesh
git reset --hard <last-known-good-sha>
./deploy.sh main   # rebuilds venv + restarts service from that SHA
```

---

## Part 5 — Hardening additions you can add later

These are **not** wired up by default — they're the next layer if/when threat model demands them.

1. **`from="<CIDR>"` on the deploy public key** — restricts which IPs can use the key. Pull current CircleCI ranges from their docs.
2. **Forced-command SSH** — replace the open shell access with a key that can only run `~/deploy.sh main`, no shell, no other commands. Pattern documented in `.qcoda/strategy/cicd_circleci_vs_bitbucket_pipelines.md` §3.3 (written for the Bitbucket Pipelines path but the SSH side is identical).
3. **Nginx in front + Let's Encrypt** — terminate HTTPS, proxy to `127.0.0.1:8003`. The repo has `docs/nginx_deployment.md` with the config sketch.
4. **Multi-env (`staging` + `main`)** — duplicate the `deploy` job, branch-filter on `staging` vs `main`, point at separate `STAGING_IP` / `PROD_IP` env vars. Sibling pattern in `../qcoda/.circleci/config.yml` is the template.
5. **PR-trigger test job** — already covered by the `test` job not being gated on a branch; PRs automatically run it.

---

## Quick reference

| Thing | Path |
|---|---|
| CI config | `.circleci/config.yml` |
| Server deploy script | `deploy/deploy.sh` → `~/deploy.sh` on the server |
| systemd unit | `deploy/llmesh.service` → `/etc/systemd/system/llmesh.service` |
| nginx config template | `deploy/llmesh-nginx.conf` → `/etc/nginx/conf.d/llmesh.conf` (copy, not symlink — certbot edits it) |
| Comparison doc (CircleCI vs Bitbucket Pipelines) | `.qcoda/strategy/cicd_circleci_vs_bitbucket_pipelines.md` |
| Hub access + error logs | `journalctl -u llmesh -f` (gunicorn writes to stdout/stderr → journald) |
| nginx logs | `/var/log/nginx/llmesh.access.log`, `/var/log/nginx/llmesh.error.log` |
| Deploy logs | `/home/llmesh/deploy.log` (+ rotated `.1` … `.10`) |
| Server repo | `/opt/llmesh/app` |
| Server venv | `/opt/llmesh/app/.venv` |

---

## Decision history

- **2026-05-30** — D085: pick CircleCI over Bitbucket Pipelines for sibling consistency, single environment (your-deploy-host.example.com), bare-metal under systemd, branch `main` only, gates = pytest + validate_decisions + gitleaks. See `.qcoda/decisions.md::D085`.
- **2026-05-30** — D086: refactor systemd unit to gunicorn-supervised uvicorn worker (pinned at `-w 1` per the hub's single-instance constraint), add `deploy/llmesh-nginx.conf` with SSE-tuned proxy + scanner-trap + Let's Encrypt placeholders. See `.qcoda/decisions.md::D086`.
