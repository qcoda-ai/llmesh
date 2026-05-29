"""
Hub configuration loader.

Reads server_config.json from the project root. Values from the file are used
as defaults — explicit environment variables always take precedence.

If server_config.json is not found, falls back to the legacy api_keys.json for
API key loading (with a deprecation warning). All other settings fall back to
their built-in defaults.
"""

import json
import logging
import os

logger = logging.getLogger("llmesh.hub.config")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# `LLMESH_CONFIG_PATH` env var lets test harnesses (and operators) point the
# hub at an alternate config file without touching the repo root. Used by the
# integration test suite to load tests/fixtures/server_config.json.
_CONFIG_PATH = os.environ.get("LLMESH_CONFIG_PATH") or os.path.join(_PROJECT_ROOT, "server_config.json")
_LEGACY_KEYS_PATH = os.path.join(_PROJECT_ROOT, "api_keys.json")

# Publicly known API key placeholders. If any of these are still present in the
# live server_config.json, the hub refuses to start — running with credentials
# that appear in the public repo is a security risk, not a misconfiguration.
#
# Sources of "publicly known":
#   - change-me-key-1/2       → server_config.example.json (shipped in repo)
#   - my_secret_key_1/2       → tests/fixtures/server_config.json (shipped in repo)
#
# Only public placeholders belong in this set; do NOT add real or rotated keys
# here in plaintext. See decisions.md D013, D020.
_SAMPLE_API_KEYS = frozenset({
    "change-me-key-1", "change-me-key-2",
    "my_secret_key_1", "my_secret_key_2",
})

# Test-only escape hatch. The Docker/Postgres test stack bind-mounts
# tests/fixtures/server_config.json as the hub config, which intentionally
# contains publicly known keys. Setting this env var to "1" bypasses the
# sample-key guard. Must NEVER be set in production — and is loudly logged
# when active so misuse is visible in startup logs. See decisions.md D020.
_ALLOW_SAMPLE_KEYS_ENV = "LLMESH_ALLOW_SAMPLE_KEYS"

# Populated at import time
api_keys: dict[str, str] = {}


def _reject_sample_keys(loaded: dict[str, str], source: str) -> None:
    """Refuse to run if the loaded api_keys still contain any sample placeholder.

    These keys are committed to the public repo via server_config.example.json,
    so leaving them active means anyone can call the hub. Treat as a security
    risk and abort startup loudly rather than degrade silently.
    """
    leaked = sorted(set(loaded) & _SAMPLE_API_KEYS)
    if not leaked:
        return
    if os.environ.get(_ALLOW_SAMPLE_KEYS_ENV) == "1":
        logger.warning(
            "%s=1 — sample API key guard bypassed. Loaded publicly known key(s) %s from %s. "
            "This is a TEST-ONLY mode. Never set this in production.",
            _ALLOW_SAMPLE_KEYS_ENV, leaked, source,
        )
        return
    raise RuntimeError(
        f"SECURITY: refusing to start. {source} still contains publicly "
        f"known sample API key(s): {leaked}. These values are shipped in "
        f"the public repo (server_config.example.json and/or "
        f"tests/fixtures/server_config.json). Replace every api_keys entry "
        f"with a unique secret before starting the hub."
    )


def _load() -> None:
    global api_keys

    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH, "r") as f:
            cfg = json.load(f)

        api_keys = cfg.get("api_keys", {})
        _reject_sample_keys(api_keys, "server_config.json")

        # Propagate file values as env var defaults (explicit env vars win)
        session = cfg.get("session", {})
        _setdefault("SESSION_BACKEND",       session.get("backend"))
        _setdefault("SESSION_DB",            session.get("db"))
        _setdefault("SESSION_TTL_SECONDS",   session.get("ttl_seconds"))
        _setdefault("SESSION_MAX_TURNS",     session.get("max_turns"))
        _setdefault("SESSION_MEMORY_MODE",   session.get("memory_mode"))
        _setdefault("SESSION_COMPRESS_MODEL",session.get("compress_model"))

        # Task persistence (D003 / D053). Falls back to SESSION_DB when unset.
        task = cfg.get("task", {})
        _setdefault("TASK_DB",               task.get("db"))
        _setdefault("TASK_TTL_SECONDS",      task.get("ttl_seconds"))

        # Routing scoring weights (D054). Penalise queue depth + CPU load
        # so the highest-RAM node does not capture every request.
        routing = cfg.get("routing", {})
        _setdefault("ROUTING_QUEUE_PENALTY", routing.get("queue_penalty"))
        _setdefault("ROUTING_CPU_PENALTY",   routing.get("cpu_penalty"))

        compress = cfg.get("compress", {})
        _setdefault("COMPRESS_MODEL_REPO",  compress.get("model_repo"))
        _setdefault("COMPRESS_MODEL_FILE",  compress.get("model_file"))
        _setdefault("COMPRESS_MODEL_CTX",   compress.get("context_size"))
        _setdefault("COMPRESS_N_THREADS",   compress.get("n_threads"))

        metrics = cfg.get("metrics", {})
        _setdefault("METRICS_RETENTION_DAYS",          metrics.get("retention_days_events"))
        _setdefault("SNAPSHOT_RETENTION_DAYS",         metrics.get("retention_days_snapshots"))

        inference = cfg.get("inference", {})
        _setdefault("DEFAULT_CONTEXT_WINDOW",          inference.get("default_context_window"))

        logger.info("Loaded config from server_config.json (%d API key(s)).", len(api_keys))

    elif os.path.exists(_LEGACY_KEYS_PATH):
        logger.warning("api_keys.json is deprecated. Rename to server_config.json (see docs/server_config.md).")
        with open(_LEGACY_KEYS_PATH, "r") as f:
            api_keys = json.load(f)
        _reject_sample_keys(api_keys, "api_keys.json")
        logger.info("Loaded %d API key(s) from legacy api_keys.json.", len(api_keys))

    else:
        logger.warning("%s not found. Authentication will fail.", _CONFIG_PATH)


def _setdefault(key: str, value) -> None:
    """Set env var default from config file value, only if not already set and value is not None."""
    if value is not None and key not in os.environ:
        os.environ[key] = str(value)


_load()
