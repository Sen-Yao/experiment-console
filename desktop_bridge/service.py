from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

from .app_server import AppServerError, CodexAppServerSession, ThreadBusy
from .config import BridgeConfig
from .console import ConsoleClient, ConsoleError
from .state import StatusStore
from .tunnel import SleepDetector, TunnelSupervisor


@dataclass
class TickResult:
    healthy: bool = False
    tunnel_running: bool = False
    console_healthy: bool = False
    claimed: int = 0
    delivered: int = 0
    acked: int = 0
    deferred: int = 0
    last_error: str | None = None


class BridgeService:
    def __init__(
        self,
        config: BridgeConfig,
        *,
        tunnel: TunnelSupervisor | None = None,
        console: ConsoleClient | None = None,
        app_session_factory=CodexAppServerSession,
        status_store: StatusStore | None = None,
        monotonic=time.monotonic,
        wall_clock=time.time,
        sleeper=time.sleep,
    ) -> None:
        self.config = config
        self.tunnel = tunnel or TunnelSupervisor(config)
        self.console = console or ConsoleClient(config)
        self.app_session_factory = app_session_factory
        self.status_store = status_store or StatusStore(config.status_file)
        self.monotonic = monotonic
        self.sleeper = sleeper
        self.sleep_detector = SleepDetector(
            config.sleep_jump_seconds, monotonic=monotonic, wall_clock=wall_clock
        )
        self.next_poll_at = 0.0

    def tick(self, *, force_poll: bool = False) -> TickResult:
        now = self.monotonic()
        result = TickResult()
        if self.sleep_detector.observe():
            self.tunnel.force_reconnect(now, reason="sleep detected", immediate=True)
            self.next_poll_at = now
        result.tunnel_running = self.tunnel.ensure_running(now)
        if not result.tunnel_running:
            result.last_error = self.tunnel.snapshot().last_error
            self._write_status(result)
            return result
        try:
            self.console.health()
            result.console_healthy = True
            self.tunnel.note_probe(True, now)
        except ConsoleError as exc:
            self.tunnel.note_probe(False, now)
            result.last_error = str(exc)
            self._write_status(result)
            return result
        if not force_poll and now < self.next_poll_at:
            result.healthy = True
            self._write_status(result)
            return result
        self.next_poll_at = now + self.config.poll_interval_seconds
        try:
            events = self.console.claim()
            result.claimed = len(events)
            if events:
                with self.app_session_factory(self.config) as session:
                    for event in events:
                        try:
                            session.deliver(
                                event.task_id,
                                event.message(),
                                event_id=event.event_id,
                            )
                            result.delivered += 1
                            self.console.ack(event)
                            result.acked += 1
                        except (ThreadBusy, AppServerError, ConsoleError) as exc:
                            result.deferred += 1
                            result.last_error = str(exc)
            result.healthy = True
        except (ConsoleError, AppServerError, ValueError) as exc:
            result.last_error = str(exc)
        self._write_status(result)
        return result

    def _write_status(self, result: TickResult) -> None:
        self.status_store.write(asdict(result))

    def run(self, should_stop: Callable[[], bool]) -> None:
        while not should_stop():
            self.tick()
            self.sleeper(self.config.loop_interval_seconds)

    def run_once(self) -> TickResult:
        return self.tick(force_poll=True)

    def close(self) -> None:
        self.tunnel.close()
