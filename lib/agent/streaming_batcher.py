"""StreamBatcher — adaptive token batching for agent → hub streaming (D041).

Used by streaming agent paths (initially _run_streaming_vllm in D040) to buffer
tokens before POSTing to the hub /stream endpoint, capping POST rate under fast
backends (vLLM 50-200 tok/s) without sacrificing dashboard latency.

Three flush triggers — earliest fires:
  1. Buffer reaches current batch size.
  2. STREAM_BATCH_TIME_MS elapsed since last flush (latency cap).
  3. Done event — stream finished, drain via flush(done=True).

Modes:
  - Adaptive (default): TPS-driven sliding window recomputes target batch size
    each flush. Smoothed by 0.7*old + 0.3*new.
  - Fixed (STREAM_BATCH_FIXED=N): batch pinned to N, auto-tune disabled,
    time cap still honored.

Per D041: TPS metric uses add() timestamps (not flush-to-flush) so slow hub
POST latency does not deflate measured speed and trigger a death-spiral
shrink. asyncio.Lock around add()/flush() so the optional stall-case timer
task and the stream loop cannot race on self._buffer.
"""

import asyncio
import logging
import os
import time
from collections import deque
from typing import Awaitable, Callable, Optional

LOG = logging.getLogger(__name__)


# --- env var resolution ------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        LOG.warning("ignoring invalid %s=%r, using default %d", name, raw, default)
        return default


def _env_fixed(name: str = "STREAM_BATCH_FIXED") -> Optional[int]:
    """Resolve fixed batch size from env. Returns None if unset/invalid/<=0
    (caller falls back to adaptive mode + warning logged for invalid)."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        n = int(raw)
    except ValueError:
        LOG.warning("ignoring invalid %s=%r, falling back to adaptive mode", name, raw)
        return None
    if n <= 0:
        LOG.warning("ignoring %s=%d (must be >= 1), falling back to adaptive mode", name, n)
        return None
    return n


def resolve_batcher_config() -> dict:
    """Read all STREAM_BATCH_* env vars and return constructor kwargs."""
    return {
        "initial_size": _env_int("STREAM_BATCH_INITIAL", 10),
        "time_cap_ms": _env_int("STREAM_BATCH_TIME_MS", 100),
        "target_pps": _env_int("STREAM_BATCH_TARGET_PPS", 10),
        "min_size": _env_int("STREAM_BATCH_MIN", 1),
        "max_size": _env_int("STREAM_BATCH_MAX", 100),
        "max_buffer": _env_int("STREAM_BATCH_MAX_BUFFER", 1000),
        "fixed_size": _env_fixed(),
    }


# --- exceptions --------------------------------------------------------------

class StreamBatcherAborted(Exception):
    """Raised when the batcher must abort (memory cap exceeded or use after close)."""


# --- batcher -----------------------------------------------------------------

FlushCallback = Callable[..., Awaitable[None]]


class StreamBatcher:
    """Per-task token buffer with size/time/done triggers and adaptive sizing.

    Single-coroutine workloads (the common case) only need add() + flush().
    For long-running streams that may stall (no tokens for > time_cap_ms),
    call start_timer() to spawn a background flusher; close() cancels it.
    """

    def __init__(
        self,
        flush_callback: FlushCallback,
        *,
        initial_size: int = 10,
        time_cap_ms: int = 100,
        target_pps: int = 10,
        min_size: int = 1,
        max_size: int = 100,
        max_buffer: int = 1000,
        fixed_size: Optional[int] = None,
    ):
        if min_size < 1:
            raise ValueError(f"min_size must be >= 1, got {min_size}")
        if max_size < min_size:
            raise ValueError(f"max_size ({max_size}) must be >= min_size ({min_size})")
        if fixed_size is not None and fixed_size < 1:
            raise ValueError(f"fixed_size must be >= 1 or None, got {fixed_size}")

        self._flush_callback = flush_callback
        self._fixed = fixed_size
        self._size = fixed_size if fixed_size else initial_size
        self._size = max(min_size, min(max_size, self._size))
        self._time_cap_ms = time_cap_ms
        self._target_pps = max(1, target_pps)
        self._min_size = min_size
        self._max_size = max_size
        self._max_buffer = max_buffer

        self._buffer: list[str] = []
        self._window: deque[float] = deque()
        self._last_flush = time.monotonic()
        self._lock = asyncio.Lock()
        self._timer_task: Optional[asyncio.Task] = None
        self._closed = False
        self._flush_count = 0
        self._token_count = 0

    # --- public API ----------------------------------------------------------

    async def add(self, token: str) -> None:
        """Append a token. Triggers size or time flush as needed."""
        if self._closed:
            raise StreamBatcherAborted("add() called after close")
        async with self._lock:
            self._buffer.append(token)
            self._token_count += 1
            now = time.monotonic()
            self._window.append(now)
            self._prune_window(now)

            if len(self._buffer) > self._max_buffer:
                self._closed = True
                LOG.error("StreamBatcher: buffer cap %d exceeded, aborting", self._max_buffer)
                raise StreamBatcherAborted(f"buffer cap {self._max_buffer} exceeded")

            if len(self._buffer) >= self._size:
                await self._flush_locked(done=False)
            elif (now - self._last_flush) * 1000 >= self._time_cap_ms:
                await self._flush_locked(done=False)

    async def flush(self, *, done: bool = False, **meta) -> None:
        """Flush remaining buffer. Pass done=True for the final frame.

        When done=True, the last batch + done are combined into a single
        POST when possible (CF-5). If only the done sentinel is needed
        (buffer empty), a single POST with empty chunk and done=True fires.
        """
        async with self._lock:
            await self._flush_locked(done=done, **meta)

    async def close(self) -> None:
        """Cancel timer task and mark closed. Idempotent."""
        self._closed = True
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            try:
                await self._timer_task
            except (asyncio.CancelledError, Exception):
                pass
            self._timer_task = None

    def start_timer(self) -> None:
        """Start the stall-case timer. Required only for streams that may
        leave the buffer non-empty without arriving tokens for >time_cap_ms."""
        if self._timer_task is None or self._timer_task.done():
            self._timer_task = asyncio.create_task(self._timer_loop())

    @property
    def current_size(self) -> int:
        return self._size

    @property
    def stats(self) -> dict:
        return {
            "current_size": self._size,
            "flushes": self._flush_count,
            "tokens": self._token_count,
            "mode": "fixed" if self._fixed else "adaptive",
        }

    # --- internals -----------------------------------------------------------

    async def _flush_locked(self, *, done: bool, **meta) -> None:
        if not self._buffer and not done:
            return

        if done:
            chunk = "".join(self._buffer)
            self._buffer.clear()
            self._last_flush = time.monotonic()
            self._flush_count += 1
            await self._flush_callback(chunk=chunk, done=True, **meta)
            return

        chunk = "".join(self._buffer)
        self._buffer.clear()
        self._last_flush = time.monotonic()
        self._flush_count += 1
        try:
            await self._flush_callback(chunk=chunk, done=False)
        finally:
            if self._fixed is None:
                self._retune()

    def _retune(self) -> None:
        """Recompute adaptive batch size from add() timestamps in window."""
        if len(self._window) < 5:
            return
        first, last = self._window[0], self._window[-1]
        elapsed = max(0.05, last - first)
        tps = len(self._window) / elapsed
        raw_target = round(tps / self._target_pps)
        target = max(self._min_size, min(self._max_size, raw_target))
        new = round(0.7 * self._size + 0.3 * target)
        new = max(self._min_size, min(self._max_size, new))
        if abs(new - self._size) >= max(1, self._size // 2):
            LOG.info(
                "StreamBatcher: size %d → %d (tps=%.1f, target=%d)",
                self._size, new, tps, target,
            )
        self._size = new

    def _prune_window(self, now: float) -> None:
        cutoff = now - 5.0
        while self._window and self._window[0] < cutoff:
            self._window.popleft()

    async def _timer_loop(self) -> None:
        try:
            interval = max(0.01, self._time_cap_ms / 1000.0 / 2)
            while not self._closed:
                await asyncio.sleep(interval)
                async with self._lock:
                    if self._closed:
                        return
                    if not self._buffer:
                        continue
                    now = time.monotonic()
                    if (now - self._last_flush) * 1000 >= self._time_cap_ms:
                        await self._flush_locked(done=False)
        except asyncio.CancelledError:
            return
