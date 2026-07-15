from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Protocol

from .app_server import (
    ActiveGoalConflict,
    AppServerError,
    AppServerOffline,
    CodexAppServerSession,
    DeliveryInspection,
    ThreadBusy,
)
from .config import BridgeConfig
from .console import ConsoleClient, ConsoleContractError, ConsoleUnavailable
from .models import OutboxEvent
from .state import DeliveryState, StatusStore
from .tunnel import SleepDetector, TunnelSupervisor


class AppSession(Protocol):
    def deliver(self, thread_id: str, text: str, *, client_message_id: str) -> str: ...

    def inspect_delivery(self, thread_id: str, *, client_message_id: str) -> DeliveryInspection: ...


AppSessionFactory = Callable[[BridgeConfig], AbstractContextManager[AppSession]]


@dataclass
class TickResult:
    status: str = "ok"
    tunnel_running: bool = False
    console_healthy: bool = False
    poll_attempted: bool = False
    claimed: int = 0
    delivered: int = 0
    acked: int = 0
    deferred: int = 0
    rejected: int = 0
    turns_started: int = 0
    sleep_reconnect: bool = False
    app_server_state: str = "not_needed"
    last_error: str | None = None


class BridgeService:
    def __init__(
        self,
        config: BridgeConfig,
        *,
        tunnel: TunnelSupervisor | None = None,
        console: ConsoleClient | None = None,
        state: DeliveryState | None = None,
        status_store: StatusStore | None = None,
        app_session_factory: AppSessionFactory = CodexAppServerSession,
        sleep_detector: SleepDetector | None = None,
        monotonic=time.monotonic,
        wall_clock=time.time,
        sleeper=time.sleep,
    ) -> None:
        self.config = config
        self.tunnel = tunnel or TunnelSupervisor(config)
        self.state = state or DeliveryState(
            config.state_file,
            acked_retention_seconds=config.acked_event_retention_seconds,
            max_acked_event_ids=config.max_acked_event_ids,
            wall_clock=wall_clock,
        )
        self.console = console or ConsoleClient(config, authority_state=self.state)
        self.status_store = status_store or StatusStore(config.status_file)
        self.app_session_factory = app_session_factory
        self.monotonic = monotonic
        self.wall_clock = wall_clock
        self.sleeper = sleeper
        self.sleep_detector = sleep_detector or SleepDetector(
            config.sleep_jump_seconds,
            monotonic=monotonic,
            wall_clock=wall_clock,
        )
        self.next_poll_at = 0.0
        self.last_result = TickResult(status="starting")

    def tick(self, *, force_poll: bool = False) -> TickResult:
        now = self.monotonic()
        result = TickResult()
        if self.sleep_detector.observe():
            self.tunnel.force_reconnect(
                now,
                reason="sleep or long scheduler gap detected",
                immediate=True,
            )
            self.next_poll_at = now
            result.sleep_reconnect = True
        result.tunnel_running = self.tunnel.ensure_running(now)
        if not result.tunnel_running:
            result.status = "degraded"
            result.last_error = self.tunnel.snapshot().last_error
            self._publish_status(result)
            return result

        result.console_healthy = self.console.health()
        tunnel_transport_healthy = getattr(self.console, "last_transport_ok", result.console_healthy)
        self.tunnel.note_probe(tunnel_transport_healthy, now)
        if not result.console_healthy:
            result.status = "degraded"
            result.last_error = getattr(self.console, "last_health_error", None) or "Console health probe failed"
            self._publish_status(result)
            return result

        if not force_poll and now < self.next_poll_at:
            self._publish_status(result)
            return result
        result.poll_attempted = True
        self.next_poll_at = now + self.config.poll_interval_seconds
        try:
            events = self.console.claim_events()
        except (ConsoleUnavailable, ConsoleContractError) as exc:
            result.status = "degraded"
            result.last_error = str(exc)
            self._publish_status(result)
            return result
        result.claimed = len(events)
        if not events:
            self._publish_status(result)
            return result

        pending_by_thread: dict[str, list[OutboxEvent]] = defaultdict(list)
        recovery_groups: dict[tuple[str, str], list[OutboxEvent]] = defaultdict(list)
        for event in events:
            delivery_status = self.state.status(event.event_id)
            if delivery_status in {"delivered", "acked"}:
                if self._ack_recorded(event, result):
                    result.acked += 1
                else:
                    result.deferred += 1
                continue
            if delivery_status in {"inflight", "uncertain"}:
                record = self.state.record(event.event_id) or {}
                client_message_id = record.get("client_message_id")
                recorded_thread_id = record.get("thread_id")
                if not isinstance(client_message_id, str) or not client_message_id or recorded_thread_id != event.thread_id:
                    self.state.mark_uncertain(
                        [event],
                        client_message_id=str(client_message_id or "missing"),
                        reason="delivery journal record is malformed or targets another thread",
                        ambiguous=True,
                    )
                    result.deferred += 1
                    result.last_error = f"ambiguous delivery journal for event {event.event_id}"
                    continue
                if record.get("ambiguous") is True:
                    result.deferred += 1
                    result.last_error = f"event {event.event_id} remains in ambiguous delivery state"
                    continue
                recovery_groups[(event.thread_id, client_message_id)].append(event)
                continue
            if not event.actionable:
                result.rejected += 1
                result.deferred += 1
                result.last_error = f"Console returned non-actionable claimed event {event.event_id}"
                continue
            pending_by_thread[event.thread_id].append(event)

        if recovery_groups or pending_by_thread:
            self._process_groups(recovery_groups, pending_by_thread, result)
        if result.last_error and result.status == "ok":
            result.status = "degraded"
        self._publish_status(result)
        return result

    def run(self, stop_requested: Callable[[], bool]) -> None:
        try:
            while not stop_requested():
                self.tick()
                self.sleeper(self.config.loop_interval_seconds)
        finally:
            self.tunnel.close()
            stopped = TickResult(status="stopped")
            self._publish_status(stopped)

    def run_once(self) -> TickResult:
        deadline = self.monotonic() + self.config.connect_timeout_seconds + 2
        result = self.tick(force_poll=True)
        while (
            not result.console_healthy
            and not getattr(self.console, "last_transport_ok", False)
            and self.monotonic() < deadline
        ):
            remaining = deadline - self.monotonic()
            self.sleeper(min(self.config.loop_interval_seconds, remaining))
            result = self.tick(force_poll=True)
        return result

    def close(self) -> None:
        self.tunnel.close()
        self._publish_status(TickResult(status="stopped"))

    def _process_groups(
        self,
        recovery_groups: dict[tuple[str, str], list[OutboxEvent]],
        pending_groups: dict[str, list[OutboxEvent]],
        result: TickResult,
    ) -> None:
        accounted_deferred: set[str] = set()
        try:
            context = self.app_session_factory(self.config)
            with context as app:
                result.app_server_state = "connected"
                for (thread_id, client_message_id), events in recovery_groups.items():
                    if not self._recover_group(
                        app,
                        thread_id,
                        events,
                        client_message_id=client_message_id,
                        result=result,
                    ):
                        accounted_deferred.update(event.event_id for event in events)
                for thread_id, events in pending_groups.items():
                    client_message_id = self._client_message_id(thread_id, events)
                    try:
                        self.state.mark_inflight(events, client_message_id=client_message_id)
                        app.deliver(
                            thread_id,
                            self._build_handoff(events),
                            client_message_id=client_message_id,
                        )
                    except (ThreadBusy, ActiveGoalConflict) as exc:
                        self.state.clear_attempt(events)
                        result.app_server_state = "busy"
                        result.deferred += len(events)
                        accounted_deferred.update(event.event_id for event in events)
                        result.last_error = str(exc)
                        continue
                    except AppServerOffline as exc:
                        self.state.mark_uncertain(
                            events,
                            client_message_id=client_message_id,
                            reason=str(exc),
                        )
                        raise
                    except (AppServerError, ConsoleContractError) as exc:
                        self.state.mark_uncertain(
                            events,
                            client_message_id=client_message_id,
                            reason=str(exc),
                        )
                        result.app_server_state = "error"
                        result.deferred += len(events)
                        accounted_deferred.update(event.event_id for event in events)
                        result.last_error = str(exc)
                        continue
                    self.state.mark_uncertain(
                        events,
                        client_message_id=client_message_id,
                        reason="turn/start accepted; awaiting persisted UserMessage.clientId",
                    )
                    result.app_server_state = "connected"
                    result.turns_started += 1
                    result.deferred += len(events)
                    accounted_deferred.update(event.event_id for event in events)
        except AppServerOffline as exc:
            result.app_server_state = "offline"
            remaining = sum(
                1
                for events in [*recovery_groups.values(), *pending_groups.values()]
                for event in events
                if self.state.status(event.event_id) not in {"delivered", "acked"}
                and event.event_id not in accounted_deferred
            )
            result.deferred += remaining
            result.last_error = str(exc)

    def _recover_group(
        self,
        app: AppSession,
        thread_id: str,
        events: list[OutboxEvent],
        *,
        client_message_id: str,
        result: TickResult,
    ) -> bool:
        try:
            inspection = app.inspect_delivery(thread_id, client_message_id=client_message_id)
        except AppServerOffline:
            raise
        except AppServerError as exc:
            result.deferred += len(events)
            result.last_error = str(exc)
            return False
        if inspection.ambiguous:
            self.state.mark_uncertain(
                events,
                client_message_id=client_message_id,
                reason=(
                    f"ambiguous thread history for client id: count={inspection.client_message_count}, "
                    f"status={inspection.thread_status}"
                ),
                ambiguous=True,
            )
            result.deferred += len(events)
            result.last_error = f"ambiguous delivery history for {client_message_id}"
            return False
        if inspection.found:
            self.state.mark_delivered(events, client_message_id=client_message_id)
            result.delivered += len(events)
            for event in events:
                if self._ack_recorded(event, result):
                    result.acked += 1
                else:
                    result.deferred += 1
            return True
        current_event_ids = {event.event_id for event in events}
        expected_groups = {
            tuple(sorted(recorded_ids))
            for event in events
            for record in [self.state.record(event.event_id) or {}]
            for recorded_ids in [record.get("group_event_ids")]
            if isinstance(recorded_ids, list) and all(isinstance(item, str) for item in recorded_ids)
        }
        if len(expected_groups) > 1:
            self.state.mark_uncertain(
                events,
                client_message_id=client_message_id,
                reason="delivery journal contains conflicting event groups",
                ambiguous=True,
            )
            result.deferred += len(events)
            result.last_error = f"conflicting delivery groups for {client_message_id}"
            return False
        expected_event_ids = set(next(iter(expected_groups))) if expected_groups else current_event_ids
        if current_event_ids != expected_event_ids:
            result.deferred += len(events)
            result.last_error = (
                f"partial delivery group for {client_message_id}; waiting for all journaled event ids"
            )
            return False
        minimum_age = min(
            (self.state.attempt_age_seconds(event.event_id) or 0.0) for event in events
        )
        if minimum_age < self.config.inflight_retry_grace_seconds:
            result.deferred += len(events)
            result.last_error = f"delivery confirmation grace is still active for {client_message_id}"
            return False
        if inspection.thread_status not in {"idle", "notLoaded"}:
            result.deferred += len(events)
            result.last_error = (
                f"delivery not found but thread status is {inspection.thread_status}; automatic retry deferred"
            )
            return False
        try:
            self.state.mark_inflight(events, client_message_id=client_message_id)
            app.deliver(
                thread_id,
                self._build_handoff(events),
                client_message_id=client_message_id,
            )
        except (ThreadBusy, ActiveGoalConflict) as exc:
            result.deferred += len(events)
            result.last_error = str(exc)
            return False
        except (AppServerOffline, AppServerError, ConsoleContractError) as exc:
            self.state.mark_uncertain(
                events,
                client_message_id=client_message_id,
                reason=str(exc),
            )
            if isinstance(exc, AppServerOffline):
                raise
            result.deferred += len(events)
            result.last_error = str(exc)
            return False
        self.state.mark_uncertain(
            events,
            client_message_id=client_message_id,
            reason="turn/start retry accepted; awaiting persisted UserMessage.clientId",
        )
        result.turns_started += 1
        result.deferred += len(events)
        return False

    def _ack_recorded(self, event: OutboxEvent, result: TickResult) -> bool:
        try:
            acked = self.console.ack_event(event)
        except (ConsoleUnavailable, ConsoleContractError) as exc:
            result.last_error = str(exc)
            return False
        if not acked:
            result.last_error = f"Console did not acknowledge event {event.event_id}"
            return False
        self.state.mark_acked(event.event_id)
        return True

    def _build_handoff(self, events: Iterable[OutboxEvent]) -> str:
        rows = [
            {
                "event_id": event.event_id,
                "kind": event.kind,
                "job_id": event.job_id,
                "summary": event.summary,
                "created_at": event.created_at,
                "payload": event.payload,
            }
            for event in events
        ]
        encoded = json.dumps({"events": rows}, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > self.config.max_handoff_bytes:
            compact_rows: list[dict[str, Any]] = []
            for row in rows:
                payload_json = json.dumps(row["payload"], ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                compact = dict(row)
                compact["payload"] = {
                    "omitted_by_bridge": True,
                    "sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
                }
                compact_rows.append(compact)
            encoded = json.dumps(
                {"events": compact_rows}, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            )
        if len(encoded.encode("utf-8")) > self.config.max_handoff_bytes:
            raise ConsoleContractError("actionable event handoff exceeds max_handoff_bytes after compaction")
        return (
            "Experiment Console emitted state changes for this existing research task. "
            "Use each payload's action_required field to distinguish attention from recovery. "
            "Handle the merged handoff below. Do not create a heartbeat or keep a Goal active merely "
            "to wait for external experiments; the Console will emit another event when action is needed.\n\n"
            f"```json\n{encoded}\n```"
        )

    @staticmethod
    def _client_message_id(thread_id: str, events: Iterable[OutboxEvent]) -> str:
        joined = "\0".join([thread_id, *sorted(event.event_id for event in events)])
        return f"experiment-console-{hashlib.sha256(joined.encode('utf-8')).hexdigest()}"

    def _publish_status(self, result: TickResult) -> None:
        tunnel = asdict(self.tunnel.snapshot())
        payload: dict[str, Any] = {
            "status": result.status,
            "healthy": result.status == "ok" and result.tunnel_running and result.console_healthy,
            "pid": os.getpid(),
            "updated_at": self.wall_clock(),
            "consumer_id": self.config.consumer_id,
            "tunnel": tunnel,
            "delivery_state": self.state.summary(),
            "last_tick": asdict(result),
        }
        self.status_store.write(payload)
        self.last_result = result
