from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Callable

from .app_server import AppServerError, CodexAppServerSession, ThreadBusy
from .config import BridgeConfig
from .models import PaneSnapshot, WakeEvent
from .state import EventStore, EventStoreFull, StatusStore
from .tmux import TmuxClient, TmuxError, classify_event


@dataclass
class TickResult:
    healthy: bool = False
    remote_healthy: bool = False
    watched_sessions: int = 0
    invalid_sessions: int = 0
    observed_events: int = 0
    pending: int = 0
    delivered: int = 0
    deferred: int = 0
    orphaned: int = 0
    last_error: str | None = None


class BridgeService:
    def __init__(
        self,
        config: BridgeConfig,
        *,
        tmux: TmuxClient | None = None,
        app_session_factory=CodexAppServerSession,
        status_store: StatusStore | None = None,
        event_store: EventStore | None = None,
        monotonic=time.monotonic,
        wall_clock=time.time,
        sleeper=time.sleep,
    ) -> None:
        self.config = config
        self.tmux = tmux or TmuxClient(config)
        self.app_session_factory = app_session_factory
        self.status_store = status_store or StatusStore(config.status_file)
        self.event_store = event_store or EventStore(
            config.event_file, max_records=config.max_event_records
        )
        self.monotonic = monotonic
        self.wall_clock = wall_clock
        self.sleeper = sleeper
        self.next_poll_at = 0.0
        self.last_result = TickResult()

    def tick(self, *, force_poll: bool = False) -> TickResult:
        now = self.monotonic()
        result = TickResult()
        if not force_poll and now < self.next_poll_at:
            result = replace(self.last_result)
            self._write_status(result)
            return result
        self.next_poll_at = now + self.config.poll_interval_seconds

        try:
            sessions = self.tmux.sessions()
            result.remote_healthy = True
            result.watched_sessions = len(sessions)
            result.invalid_sessions = self.tmux.invalid_sessions
        except TmuxError as exc:
            result.last_error = str(exc)
            self.last_result = result
            self._write_status(result)
            return result

        observed_at = int(self.wall_clock())
        active_event_ids = {
            f"tmux:{session.generation}:{event_type}"
            for session in sessions
            for event_type in ("attention", "terminal")
        }
        self.event_store.reconcile(active_event_ids)
        for session in sessions:
            event = classify_event(session, observed_at)
            if event is not None:
                try:
                    if self.event_store.add(event):
                        result.observed_events += 1
                except EventStoreFull as exc:
                    result.last_error = str(exc)

        pending = self.event_store.pending(self.config.max_events_per_poll)
        result.pending = len(pending)
        if not pending:
            result.healthy = result.last_error is None
            self.last_result = result
            self._write_status(result)
            return result

        try:
            with self.app_session_factory(self.config) as session:
                for event in pending:
                    try:
                        outcome = session.deliver(
                            event.session.thread_id,
                            self._message(event),
                            event_id=event.event_id,
                        )
                    except (ThreadBusy, AppServerError) as exc:
                        self.event_store.mark_error(event.event_id, str(exc))
                        result.deferred += 1
                        result.last_error = str(exc)
                        continue
                    if outcome.status == "delivered" and outcome.turn_id:
                        self.event_store.mark_delivered(event.event_id, outcome.turn_id)
                        result.delivered += 1
                    elif outcome.status == "orphaned":
                        reason = outcome.reason or "not_deliverable"
                        self.event_store.mark_orphaned(event.event_id, reason)
                        result.orphaned += 1
                    else:
                        result.deferred += 1
        except AppServerError as exc:
            result.last_error = str(exc)

        result.healthy = result.remote_healthy and result.last_error is None
        self.last_result = result
        self._write_status(result)
        return result

    def _message(self, event: WakeEvent) -> str:
        session = event.session
        elapsed = max(0, event.observed_at - session.started_at)
        lines = [
            "An HCCS tmux experiment session needs inspection.",
            f"event_id: {event.event_id}",
            f"event_type: {event.event_type}",
            f"reason: {event.reason}",
            f"host: {self.config.ssh_target}",
            f"session: {session.session_name}",
            f"generation: {session.generation}",
            f"investigation: {session.investigation_id}",
            f"started_at: {self._timestamp(session.started_at)}",
            f"elapsed_seconds: {elapsed}",
            f"expected_seconds: {session.expected_seconds}",
            f"attention_after: {self._timestamp(session.attention_after)}",
            "panes:",
        ]
        for pane in session.panes:
            state = (
                f"dead(exit={pane.exit_status})" if pane.dead else "running"
            )
            lines.append(
                f"- {pane.pane_id} index={pane.pane_index} state={state} "
                f"command={pane.current_command or 'unknown'}"
            )

        capture_targets = self._capture_targets(event)
        for pane in capture_targets:
            try:
                output = self.tmux.capture_pane(pane.pane_id).strip()
            except TmuxError as exc:
                output = f"capture unavailable: {exc}"
            lines.extend([f"recent_output[{pane.pane_id}]:", output or "(empty)"])

        lines.extend(
            [
                "This event reports tmux process state only; it does not prove W&B or result readiness.",
                "Re-probe tmux, GPU/process state, W&B, run manifest, and artifacts before deciding to wait, retry, or run tmux kill-pane.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _capture_targets(event: WakeEvent) -> list[PaneSnapshot]:
        failed = [
            pane
            for pane in event.session.panes
            if pane.dead and pane.exit_status not in (None, 0)
        ]
        if failed:
            return failed
        if event.event_type == "terminal":
            return list(event.session.panes)
        return [pane for pane in event.session.panes if not pane.dead]

    @staticmethod
    def _timestamp(value: int) -> str:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()

    def _write_status(self, result: TickResult) -> None:
        self.status_store.write(asdict(result))

    def run(self, should_stop: Callable[[], bool]) -> None:
        while not should_stop():
            self.tick()
            self.sleeper(self.config.loop_interval_seconds)

    def run_once(self) -> TickResult:
        return self.tick(force_poll=True)

    def close(self) -> None:
        return None
