"""Unit tests for StreamBatcher (D041).

Pure-async tests — no hub, no httpx, no agent. Validates triggers, adaptive
auto-tune math, fixed-mode parity, race safety, memory cap, and CF-5 done
durability.
"""

import asyncio
import os
import time
import unittest.mock as mock

import pytest

from lib.agent.streaming_batcher import (
    StreamBatcher,
    StreamBatcherAborted,
    resolve_batcher_config,
    _env_fixed,
)


def _capture():
    """Record flush_callback calls. Returns (callback, calls list)."""
    calls: list[dict] = []

    async def cb(*, chunk, done, **meta):
        calls.append({"chunk": chunk, "done": done, **meta})

    return cb, calls


# --- 1. size trigger ----------------------------------------------------------

@pytest.mark.asyncio
async def test_size_trigger_fires_at_batch_size():
    cb, calls = _capture()
    b = StreamBatcher(cb, initial_size=5, time_cap_ms=10_000, fixed_size=5)
    for i in range(5):
        await b.add(f"t{i}")
    assert len(calls) == 1
    assert calls[0]["chunk"] == "t0t1t2t3t4"
    assert calls[0]["done"] is False


# --- 2. time trigger ----------------------------------------------------------

@pytest.mark.asyncio
async def test_time_trigger_fires_after_time_cap():
    cb, calls = _capture()
    b = StreamBatcher(cb, initial_size=100, time_cap_ms=50, fixed_size=100)
    await b.add("a")
    await asyncio.sleep(0.06)
    await b.add("b")  # this add triggers the time-cap flush
    assert len(calls) == 1
    assert calls[0]["chunk"] == "ab"


# --- 3. done trigger ---------------------------------------------------------

@pytest.mark.asyncio
async def test_done_with_buffer_combines_into_single_post():
    cb, calls = _capture()
    b = StreamBatcher(cb, initial_size=100, time_cap_ms=10_000, fixed_size=100)
    await b.add("hello")
    await b.add("world")
    await b.flush(done=True, tokens_p=10, tokens_c=2)
    assert len(calls) == 1
    assert calls[0]["chunk"] == "helloworld"
    assert calls[0]["done"] is True
    assert calls[0]["tokens_p"] == 10
    assert calls[0]["tokens_c"] == 2


@pytest.mark.asyncio
async def test_done_with_empty_buffer_emits_done_only():
    cb, calls = _capture()
    b = StreamBatcher(cb, initial_size=10, time_cap_ms=10_000)
    await b.flush(done=True, tokens_p=0, tokens_c=0)
    assert len(calls) == 1
    assert calls[0]["chunk"] == ""
    assert calls[0]["done"] is True


# --- 4. auto-tune up ----------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_tune_grows_under_high_tps():
    cb, _ = _capture()
    b = StreamBatcher(cb, initial_size=2, time_cap_ms=10_000, target_pps=10,
                      min_size=1, max_size=50)
    # Synthetic high TPS: 100 tokens, fast inter-arrival.
    base = time.monotonic()
    for i in range(60):
        # Force window into high-TPS regime by manipulating timestamps.
        await b.add(f"t{i}")
    # After many flushes, size should have grown above initial.
    assert b.current_size > 2


# --- 5. auto-tune down --------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_tune_shrinks_under_low_tps():
    cb, _ = _capture()
    b = StreamBatcher(cb, initial_size=10, time_cap_ms=20, target_pps=10,
                      min_size=1, max_size=50)
    for i in range(20):
        await b.add(f"t{i}")
        await asyncio.sleep(0.03)  # slow: ~30 tps
    # Size should have shrunk toward min.
    assert b.current_size < 10


# --- 6. smoothing — no overshoot ---------------------------------------------

@pytest.mark.asyncio
async def test_smoothing_does_not_overshoot_max():
    cb, _ = _capture()
    b = StreamBatcher(cb, initial_size=5, time_cap_ms=10_000, target_pps=10,
                      min_size=1, max_size=20)
    # Pump many tokens fast — auto-tune should stay within max_size.
    for i in range(200):
        await b.add(f"t{i}")
    assert b.current_size <= 20


# --- 7. empty stream — done with no tokens ------------------------------------

@pytest.mark.asyncio
async def test_empty_stream_emits_one_done_post():
    cb, calls = _capture()
    b = StreamBatcher(cb, initial_size=10, time_cap_ms=10_000)
    await b.flush(done=True)
    assert len(calls) == 1
    assert calls[0] == {"chunk": "", "done": True}


# --- 8. window pruning --------------------------------------------------------

@pytest.mark.asyncio
async def test_window_pruning_drops_old_entries(monkeypatch):
    cb, _ = _capture()
    b = StreamBatcher(cb, initial_size=100, time_cap_ms=10_000, fixed_size=100)
    # Stuff old timestamps into the window directly, then add → prune fires.
    now = time.monotonic()
    b._window.extend([now - 10.0, now - 9.0, now - 8.0])
    await b.add("recent")
    # All entries older than 5s should be gone.
    assert all(ts >= now - 5.0 for ts in b._window)


# --- 9. time-cap timer fires on stall -----------------------------------------

@pytest.mark.asyncio
async def test_timer_loop_flushes_stalled_buffer():
    cb, calls = _capture()
    b = StreamBatcher(cb, initial_size=100, time_cap_ms=50, fixed_size=100)
    b.start_timer()
    try:
        await b.add("only-token")
        # No more adds — timer should fire after time_cap_ms.
        await asyncio.sleep(0.15)
        assert any(c["chunk"] == "only-token" for c in calls)
    finally:
        await b.close()


# --- 11. concurrent batchers isolated ----------------------------------------

@pytest.mark.asyncio
async def test_concurrent_batchers_have_isolated_state():
    cb_a, calls_a = _capture()
    cb_b, calls_b = _capture()
    a = StreamBatcher(cb_a, initial_size=2, time_cap_ms=10_000, fixed_size=2)
    b = StreamBatcher(cb_b, initial_size=3, time_cap_ms=10_000, fixed_size=3)
    await a.add("a1"); await a.add("a2")
    await b.add("b1"); await b.add("b2"); await b.add("b3")
    assert len(calls_a) == 1 and calls_a[0]["chunk"] == "a1a2"
    assert len(calls_b) == 1 and calls_b[0]["chunk"] == "b1b2b3"


# --- 12. CF-2 race / lock -----------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_add_and_timer_no_lost_or_doubled_tokens():
    cb, calls = _capture()
    b = StreamBatcher(cb, initial_size=10, time_cap_ms=20, fixed_size=10)
    b.start_timer()
    try:
        # Use a bracket separator so substring matches are unambiguous.
        await asyncio.gather(*[b.add(f"[t{i}]") for i in range(50)])
        await asyncio.sleep(0.05)
        await b.flush(done=True)
    finally:
        await b.close()

    seen = "".join(c["chunk"] for c in calls)
    for i in range(50):
        marker = f"[t{i}]"
        assert seen.count(marker) == 1, f"token {marker} appears {seen.count(marker)} times"


# --- 13. STREAM_BATCH_FIXED=20 stable ----------------------------------------

@pytest.mark.asyncio
async def test_fixed_size_stays_constant_under_speed_swings():
    cb, _ = _capture()
    b = StreamBatcher(cb, initial_size=5, time_cap_ms=10_000, fixed_size=20,
                      target_pps=10, min_size=1, max_size=100)
    for i in range(100):
        await b.add(f"t{i}")
        if i % 10 == 0:
            await asyncio.sleep(0.001)
    assert b.current_size == 20  # never deviates


# --- 14. STREAM_BATCH_FIXED=1 per-token ---------------------------------------

@pytest.mark.asyncio
async def test_fixed_one_emits_one_post_per_token():
    cb, calls = _capture()
    b = StreamBatcher(cb, initial_size=10, time_cap_ms=10_000, fixed_size=1)
    for i in range(5):
        await b.add(f"t{i}")
    assert len(calls) == 5
    assert [c["chunk"] for c in calls] == ["t0", "t1", "t2", "t3", "t4"]


# --- 15. FIXED + tokens stop → time cap fires --------------------------------

@pytest.mark.asyncio
async def test_fixed_with_stall_uses_time_cap_via_timer():
    cb, calls = _capture()
    b = StreamBatcher(cb, initial_size=10, time_cap_ms=50, fixed_size=10)
    b.start_timer()
    try:
        for i in range(3):
            await b.add(f"t{i}")
        await asyncio.sleep(0.15)  # stall — timer fires
        assert any(c["chunk"] == "t0t1t2" for c in calls)
    finally:
        await b.close()


# --- 16. STREAM_BATCH_FIXED=0 → fallback to adaptive + warning ---------------

def test_env_fixed_zero_returns_none(caplog):
    with mock.patch.dict(os.environ, {"STREAM_BATCH_FIXED": "0"}, clear=False):
        with caplog.at_level("WARNING"):
            result = _env_fixed()
    assert result is None
    assert any("STREAM_BATCH_FIXED" in rec.message for rec in caplog.records)


# --- 17. STREAM_BATCH_FIXED=invalid → fallback + warning ---------------------

def test_env_fixed_invalid_returns_none(caplog):
    with mock.patch.dict(os.environ, {"STREAM_BATCH_FIXED": "abc"}, clear=False):
        with caplog.at_level("WARNING"):
            result = _env_fixed()
    assert result is None
    assert any("STREAM_BATCH_FIXED" in rec.message for rec in caplog.records)


def test_env_fixed_unset_returns_none():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("STREAM_BATCH_FIXED", None)
        assert _env_fixed() is None


def test_env_fixed_valid_returns_int():
    with mock.patch.dict(os.environ, {"STREAM_BATCH_FIXED": "25"}, clear=False):
        assert _env_fixed() == 25


def test_resolve_batcher_config_defaults():
    keep = ["STREAM_BATCH_INITIAL", "STREAM_BATCH_TIME_MS", "STREAM_BATCH_TARGET_PPS",
            "STREAM_BATCH_MIN", "STREAM_BATCH_MAX", "STREAM_BATCH_MAX_BUFFER",
            "STREAM_BATCH_FIXED"]
    with mock.patch.dict(os.environ, {}, clear=False):
        for k in keep:
            os.environ.pop(k, None)
        cfg = resolve_batcher_config()
    assert cfg["initial_size"] == 10
    assert cfg["time_cap_ms"] == 100
    assert cfg["target_pps"] == 10
    assert cfg["min_size"] == 1
    assert cfg["max_size"] == 100
    assert cfg["max_buffer"] == 1000
    assert cfg["fixed_size"] is None


# --- 18. HP-3 slow-hub TPS measurement ---------------------------------------

@pytest.mark.asyncio
async def test_tps_metric_unaffected_by_slow_callback():
    """Slow flush callback must not deflate measured TPS — TPS uses add()
    timestamps only (HP-3). Otherwise batch shrinks → death spiral."""
    slow_calls: list[float] = []

    async def slow_cb(*, chunk, done, **meta):
        slow_calls.append(time.monotonic())
        await asyncio.sleep(0.05)  # 50ms hub latency per POST

    b = StreamBatcher(slow_cb, initial_size=5, time_cap_ms=10_000,
                      target_pps=10, min_size=1, max_size=100)
    # Pump fast tokens through slow callback. add()→add() inter-arrival is
    # near-zero (synchronous loop), so window shows high TPS regardless of
    # POST latency.
    for i in range(40):
        await b.add(f"t{i}")
    # If TPS were flush-to-flush, batch would have shrunk; instead it should
    # have grown to take advantage of fast token rate.
    assert b.current_size >= 5


# --- 19. HP-5 memory cap → abort ----------------------------------------------

@pytest.mark.asyncio
async def test_buffer_cap_aborts_when_callback_blocked():
    """Hub-unreachable simulation: callback hangs, buffer must stop at cap."""
    blocked = asyncio.Event()
    started = asyncio.Event()

    async def hung_cb(*, chunk, done, **meta):
        started.set()
        await blocked.wait()

    b = StreamBatcher(hung_cb, initial_size=10_000, time_cap_ms=10_000,
                      max_buffer=20)
    # Spawn a task that keeps adding until aborted.
    async def driver():
        for i in range(100):
            await b.add(f"t{i}")

    with pytest.raises(StreamBatcherAborted, match="buffer cap"):
        await driver()
    blocked.set()


# --- 20. CF-5 last-batch-fails → done frame still fires ----------------------

@pytest.mark.asyncio
async def test_last_batch_failure_does_not_block_done_frame():
    """If a mid-stream flush raises, buffer is cleared first — a subsequent
    flush(done=True) still emits the done frame from an empty buffer."""
    fail_once = {"n": 0}
    succeeded: list[dict] = []

    async def cb(*, chunk, done, **meta):
        if not done and fail_once["n"] == 0:
            fail_once["n"] += 1
            raise RuntimeError("simulated hub 5xx on mid-stream flush")
        succeeded.append({"chunk": chunk, "done": done, **meta})

    b = StreamBatcher(cb, initial_size=2, time_cap_ms=10_000, fixed_size=2)
    with pytest.raises(RuntimeError):
        await b.add("t0"); await b.add("t1")  # triggers failing flush
    # Buffer is empty (cleared before await). Done frame must still fire.
    await b.flush(done=True, tokens_p=5, tokens_c=2)
    assert len(succeeded) == 1
    assert succeeded[0] == {"chunk": "", "done": True, "tokens_p": 5, "tokens_c": 2}


# --- bonus: validation -------------------------------------------------------

def test_constructor_rejects_invalid_min_size():
    with pytest.raises(ValueError, match="min_size"):
        StreamBatcher(lambda **k: None, min_size=0)


def test_constructor_rejects_max_lt_min():
    with pytest.raises(ValueError, match="max_size"):
        StreamBatcher(lambda **k: None, min_size=10, max_size=5)


def test_constructor_rejects_zero_fixed_size():
    with pytest.raises(ValueError, match="fixed_size"):
        StreamBatcher(lambda **k: None, fixed_size=0)


# --- close idempotent --------------------------------------------------------

@pytest.mark.asyncio
async def test_close_is_idempotent():
    cb, _ = _capture()
    b = StreamBatcher(cb, initial_size=10, time_cap_ms=10_000)
    b.start_timer()
    await b.close()
    await b.close()  # second close must not raise


@pytest.mark.asyncio
async def test_add_after_close_raises():
    cb, _ = _capture()
    b = StreamBatcher(cb, initial_size=10, time_cap_ms=10_000)
    await b.close()
    with pytest.raises(StreamBatcherAborted, match="after close"):
        await b.add("late")


# --- stats surface -----------------------------------------------------------

@pytest.mark.asyncio
async def test_stats_reports_mode_and_counts():
    cb, _ = _capture()
    b = StreamBatcher(cb, initial_size=2, time_cap_ms=10_000, fixed_size=2)
    await b.add("a"); await b.add("b")
    s = b.stats
    assert s["mode"] == "fixed"
    assert s["tokens"] == 2
    assert s["flushes"] == 1
    assert s["current_size"] == 2

    cb2, _ = _capture()
    b2 = StreamBatcher(cb2, initial_size=10, time_cap_ms=10_000)
    assert b2.stats["mode"] == "adaptive"
