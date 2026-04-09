# Nginx Deployment Guide

Deploy the LLMesh Hub on a public server (AWS EC2, DigitalOcean Droplet, etc.) with HTTPS via Nginx as a TLS-terminating reverse proxy.

Nginx handles TLS only. Authentication is managed by the hub itself:
- **API clients** (agents, SDKs) authenticate via `Authorization: Bearer <key>` or `x-api-key` headers, validated against `server_config.json`.
- **Dashboard** uses a form-based login at `/login` with a session cookie. No additional auth layer at the nginx level is needed or correct — adding Basic Auth at the proxy level would break API clients.

---

## 1. Install Nginx and Certbot

On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx
```

---

## 2. Configure Nginx

Create a site configuration (replace `yourdomain.com` with your domain):

```bash
sudo nano /etc/nginx/sites-available/llmesh
```

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # ── Streaming: token chunk ingestion and SSE inference responses ──────────
    # proxy_buffering MUST be off for these paths. Without it, nginx holds the
    # entire response in memory until the upstream closes the connection — the
    # client sees nothing until inference completes, defeating the purpose of
    # streaming entirely.
    #
    # Place this block BEFORE the general location / so nginx uses it first.
    location ~ ^/(tasks/[^/]+/[^/]+/stream|v1/chat/completions|v1/messages) {
        proxy_pass             http://127.0.0.1:8000;
        proxy_set_header       Host              $host;
        proxy_set_header       X-Real-IP         $remote_addr;
        proxy_set_header       X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header       X-Forwarded-Proto $scheme;
        proxy_buffering        off;
        proxy_cache            off;
        proxy_http_version     1.1;
        proxy_set_header       Connection        "";
        proxy_read_timeout     660s;
        proxy_send_timeout     660s;
    }

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        # Required for long-running inference requests
        proxy_read_timeout 660s;
        proxy_send_timeout 660s;
    }
}
```

Enable the config:
```bash
sudo ln -s /etc/nginx/sites-available/llmesh /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## 3. Enable HTTPS

```bash
sudo certbot --nginx -d yourdomain.com
```

Certbot will modify the config to add TLS and redirect HTTP → HTTPS. Choose yes to the redirect prompt.

---

## 4. Start the Hub

On the server, with your `server_config.json` in place:

```bash
uvicorn lib.hub.server:app --host 127.0.0.1 --port 8000
```

Bind to `127.0.0.1` so the hub is only reachable through nginx, not directly on the public interface.

For production, run via a systemd service or a process manager like `supervisor`.

---

## 5. Connect an Agent

Point the agent at your public domain:

```bash
LLMESH_API_KEY="your_api_key" HUB_URL="https://yourdomain.com" python -m lib.agent.client
```

The agent's `LLMESH_API_KEY` is sent as `Authorization: Bearer <key>` on every request — the hub validates it against `server_config.json`. No changes to agent code are needed.

---

## 6. Access the Dashboard

Visit `https://yourdomain.com/dashboard`. You will be redirected to `/login` and prompted for your API key. After login a session cookie is set for the browser session.

---

## Timeouts

The default nginx `proxy_read_timeout` is 60 seconds. LLM inference can take much longer. The config above sets 660s (11 minutes) to match the hub's internal 600s inference timeout. Adjust based on your models and hardware.
