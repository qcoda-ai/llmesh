"""
Security tests for lib/hub/config.py.

The hub must refuse to start if `server_config.json` (or the legacy
`api_keys.json`) still contains any publicly known API key placeholder:
  - `change-me-key-1`/`change-me-key-2` from `server_config.example.json`
  - `my_secret_key_1`/`my_secret_key_2` from `tests/fixtures/server_config.json`

All of these values are shipped in the public repo, so leaving them active is
a credential leak, not a misconfiguration. The guard can be bypassed with
`LLMESH_ALLOW_SAMPLE_KEYS=1` for the Docker test stack only.

See decisions.md D013, D020.
"""
import pytest

from lib.hub import config


@pytest.fixture(autouse=True)
def _clear_bypass_env(monkeypatch):
    """Ensure the bypass env var is unset for every test by default.
    The single test that exercises the bypass sets it explicitly.
    """
    monkeypatch.delenv("LLMESH_ALLOW_SAMPLE_KEYS", raising=False)


def test_reject_sample_keys_raises_on_change_me_key_1():
    with pytest.raises(RuntimeError, match="SECURITY"):
        config._reject_sample_keys(
            {"change-me-key-1": "owner_alpha"}, "server_config.json"
        )


def test_reject_sample_keys_raises_on_change_me_key_2():
    with pytest.raises(RuntimeError, match="SECURITY"):
        config._reject_sample_keys(
            {"change-me-key-2": "owner_beta"}, "server_config.json"
        )


def test_reject_sample_keys_raises_when_mixed_with_real_keys():
    """A real key alongside a sample key must still abort startup."""
    with pytest.raises(RuntimeError, match="change-me-key-1"):
        config._reject_sample_keys(
            {
                "real-secret-abcdef": "owner_alpha",
                "change-me-key-1": "owner_beta",
            },
            "server_config.json",
        )


def test_reject_sample_keys_passes_on_clean_config():
    # Must not raise
    config._reject_sample_keys(
        {"real-secret-abcdef": "owner_alpha"}, "server_config.json"
    )


def test_reject_sample_keys_passes_on_empty_config():
    # No keys at all is a different failure mode (auth will fail) but not a
    # *security* failure — this guard should not raise.
    config._reject_sample_keys({}, "server_config.json")


@pytest.mark.parametrize("leaked_key", [
    "my_secret_key_1",
    "my_secret_key_2",
])
def test_reject_sample_keys_raises_on_test_fixture_keys(leaked_key):
    """Keys shipped via tests/fixtures/server_config.json are also publicly
    known and must trigger the guard."""
    with pytest.raises(RuntimeError, match="SECURITY"):
        config._reject_sample_keys({leaked_key: "owner_alpha"}, "server_config.json")


def test_bypass_env_var_allows_sample_keys(monkeypatch, caplog):
    """LLMESH_ALLOW_SAMPLE_KEYS=1 must bypass the guard but log a loud warning.

    D051 swapped the original `print()` for `logger.warning()`, so the message
    lands in caplog (the `llmesh.hub.config` logger), not stdout.
    """
    import logging
    monkeypatch.setenv("LLMESH_ALLOW_SAMPLE_KEYS", "1")
    with caplog.at_level(logging.WARNING, logger="llmesh.hub.config"):
        # Must not raise
        config._reject_sample_keys(
            {"my_secret_key_1": "owner_alpha"}, "tests/fixtures/server_config.json"
        )
    msg = caplog.text
    assert "LLMESH_ALLOW_SAMPLE_KEYS=1" in msg
    assert "TEST-ONLY" in msg


def test_bypass_env_var_only_accepts_exactly_1(monkeypatch):
    """Truthy-ish values other than '1' must NOT bypass the guard."""
    for val in ("true", "yes", "0", ""):
        monkeypatch.setenv("LLMESH_ALLOW_SAMPLE_KEYS", val)
        with pytest.raises(RuntimeError, match="SECURITY"):
            config._reject_sample_keys(
                {"my_secret_key_1": "owner_alpha"}, "server_config.json"
            )


def test_error_message_names_the_offending_keys():
    """The RuntimeError must identify which sample keys leaked, so the operator
    knows exactly what to replace."""
    with pytest.raises(RuntimeError) as exc_info:
        config._reject_sample_keys(
            {"change-me-key-1": "a", "change-me-key-2": "b"},
            "server_config.json",
        )
    msg = str(exc_info.value)
    assert "change-me-key-1" in msg
    assert "change-me-key-2" in msg
    assert "server_config.json" in msg
