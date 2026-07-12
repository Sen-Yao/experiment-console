from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .config import BridgeConfig


class ProcessLike(Protocol):
    returncode: int | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


PopenFactory = Callable[..., ProcessLike]


@dataclass
class TunnelSnapshot:
    state: str
    pid: int | None
    starts: int
    reconnects: int
    consecutive_probe_failures: int
    next_start_at: float
    last_exit_code: int | None
    last_error: str | None


class TunnelSupervisor:
    def __init__(
        self,
        config: BridgeConfig,
        *,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        self.config = config
        self.popen_factory = popen_factory
        self.process: ProcessLike | None = None
        self.next_start_at = 0.0
        self.backoff_seconds = config.reconnect_initial_seconds
        self.starts = 0
        self.reconnects = 0
        self.consecutive_probe_failures = 0
        self.last_exit_code: int | None = None
        self.last_error: str | None = None
        self.state = "stopped"
        self.started_at: float | None = None

    def ensure_running(self, now: float) -> bool:
        if self.process is not None:
            exit_code = self.process.poll()
            if exit_code is None:
                return True
            self.last_exit_code = exit_code
            self.process = None
            self.started_at = None
            self.state = "backoff"
            self.last_error = f"ssh tunnel exited with code {exit_code}"
            self._schedule_retry(now)
        if now < self.next_start_at:
            return False
        try:
            self.process = self.popen_factory(
                self.config.ssh_command(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except OSError as exc:
            self.state = "backoff"
            self.last_error = f"cannot start ssh tunnel: {exc}"
            self._schedule_retry(now)
            return False
        self.starts += 1
        self.started_at = now
        self.state = "starting"
        self.last_error = None
        return True

    def note_probe(self, healthy: bool, now: float) -> None:
        if healthy:
            self.consecutive_probe_failures = 0
            self.backoff_seconds = self.config.reconnect_initial_seconds
            self.next_start_at = 0.0
            self.state = "healthy"
            return
        if (
            self.state == "starting"
            and self.started_at is not None
            and now - self.started_at <= self.config.connect_timeout_seconds + 2
        ):
            return
        self.consecutive_probe_failures += 1
        if self.process is not None and self.consecutive_probe_failures >= self.config.tunnel_probe_failures_before_restart:
            self.force_reconnect(now, reason="Console health probe repeatedly failed")

    def force_reconnect(self, now: float, *, reason: str, immediate: bool = False) -> None:
        if self.process is not None:
            self._stop_process()
        self.process = None
        self.started_at = None
        self.reconnects += 1
        self.state = "backoff"
        self.last_error = reason
        self.consecutive_probe_failures = 0
        if immediate:
            self.next_start_at = now
        else:
            self._schedule_retry(now)

    def close(self) -> None:
        if self.process is not None:
            self._stop_process()
        self.process = None
        self.started_at = None
        self.state = "stopped"

    def snapshot(self) -> TunnelSnapshot:
        pid = getattr(self.process, "pid", None) if self.process is not None else None
        return TunnelSnapshot(
            state=self.state,
            pid=pid,
            starts=self.starts,
            reconnects=self.reconnects,
            consecutive_probe_failures=self.consecutive_probe_failures,
            next_start_at=self.next_start_at,
            last_exit_code=self.last_exit_code,
            last_error=self.last_error,
        )

    def _schedule_retry(self, now: float) -> None:
        self.next_start_at = now + self.backoff_seconds
        self.backoff_seconds = min(self.config.reconnect_max_seconds, self.backoff_seconds * 2)

    def _stop_process(self) -> None:
        assert self.process is not None
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3.0)


class SleepDetector:
    """Detect suspension or a long scheduler gap using wall and monotonic clocks."""

    def __init__(self, threshold_seconds: float, *, monotonic=time.monotonic, wall_clock=time.time) -> None:
        self.threshold_seconds = threshold_seconds
        self.monotonic = monotonic
        self.wall_clock = wall_clock
        self._last_monotonic = monotonic()
        self._last_wall = wall_clock()

    def observe(self) -> bool:
        current_monotonic = self.monotonic()
        current_wall = self.wall_clock()
        monotonic_delta = max(0.0, current_monotonic - self._last_monotonic)
        wall_delta = max(0.0, current_wall - self._last_wall)
        self._last_monotonic = current_monotonic
        self._last_wall = current_wall
        return (
            max(monotonic_delta, wall_delta) > self.threshold_seconds
            or abs(wall_delta - monotonic_delta) > self.threshold_seconds
        )
