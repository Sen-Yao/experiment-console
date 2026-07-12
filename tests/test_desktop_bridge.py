from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desktop_bridge.app_server import (
    ActiveGoalConflict,
    AppServerOffline,
    CodexAppServerSession,
    DeliveryInspection,
    ThreadBusy,
)
from desktop_bridge.config import BridgeConfig
from desktop_bridge.console import ConsoleClient, ConsoleContractError
from desktop_bridge.models import OutboxEvent
from desktop_bridge.service import BridgeService
from desktop_bridge.state import AlreadyRunning, DeliveryState, InstanceLock, StatusStore
from desktop_bridge.tunnel import SleepDetector, TunnelSnapshot, TunnelSupervisor


def config(tmp_path: Path, **overrides) -> BridgeConfig:
    values = {
        "ssh_target": "Yggdrasil",
        "state_file": str(tmp_path / "state.json"),
        "status_file": str(tmp_path / "status.json"),
        "tunnel_probe_failures_before_restart": 2,
        "reconnect_initial_seconds": 1,
        "reconnect_max_seconds": 8,
    }
    values.update(overrides)
    return BridgeConfig.from_mapping(values)


def event(event_id: str, *, thread_id: str = "thr_1", actionable: bool = True) -> OutboxEvent:
    return OutboxEvent(
        event_id=event_id,
        thread_id=thread_id,
        kind="result_ready",
        summary=f"result {event_id} is ready",
        actionable=actionable,
        payload={"job_id": "job_1", "state": "result_ready"},
        lease_token=f"lease_{event_id}",
        created_at="2026-07-12T12:00:00Z",
        job_id="job_1",
    )


class FakeProcess:
    next_pid = 1000

    def __init__(self) -> None:
        self.returncode = None
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode or 0

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_tunnel_restarts_after_ssh_exit_with_exponential_backoff(tmp_path):
    processes = [FakeProcess(), FakeProcess()]
    commands = []
    spawn_options = []

    def spawn(command, **kwargs):
        commands.append(command)
        spawn_options.append(kwargs)
        return processes[len(commands) - 1]

    bridge_config = config(tmp_path)
    tunnel = TunnelSupervisor(bridge_config, popen_factory=spawn)
    assert tunnel.ensure_running(0.0)
    assert "ExitOnForwardFailure=yes" in commands[0]
    assert "IdentitiesOnly=yes" in commands[0]
    assert "ServerAliveInterval=15" in commands[0]
    assert "start_new_session" not in spawn_options[0]
    processes[0].returncode = 255
    assert not tunnel.ensure_running(0.5)
    assert not tunnel.ensure_running(1.49)
    assert tunnel.ensure_running(1.5)
    assert len(commands) == 2


def test_tunnel_restarts_a_lingering_process_after_network_probe_failures(tmp_path):
    process = FakeProcess()
    tunnel = TunnelSupervisor(config(tmp_path), popen_factory=lambda *args, **kwargs: process)
    assert tunnel.ensure_running(0.0)
    tunnel.note_probe(True, 0.5)
    tunnel.note_probe(False, 1.0)
    assert not process.terminated
    tunnel.note_probe(False, 2.0)
    assert process.terminated
    assert tunnel.snapshot().state == "backoff"
    assert tunnel.snapshot().reconnects == 1
    assert tunnel.snapshot().next_start_at == 3.0


def test_tunnel_does_not_kill_a_slow_connection_during_connect_timeout_grace(tmp_path):
    process = FakeProcess()
    tunnel = TunnelSupervisor(config(tmp_path), popen_factory=lambda *args, **kwargs: process)
    assert tunnel.ensure_running(0.0)
    tunnel.note_probe(False, 1.0)
    tunnel.note_probe(False, 6.0)
    tunnel.note_probe(False, 12.0)
    assert not process.terminated
    tunnel.note_probe(False, 12.1)
    assert not process.terminated
    tunnel.note_probe(False, 13.0)
    assert process.terminated


def test_sleep_time_jump_forces_reconnect_signal():
    clock = SimpleNamespace(monotonic=100.0, wall=1000.0)
    detector = SleepDetector(
        90.0,
        monotonic=lambda: clock.monotonic,
        wall_clock=lambda: clock.wall,
    )
    clock.monotonic += 2.0
    clock.wall += 302.0
    assert detector.observe()
    clock.monotonic += 2.0
    clock.wall += 2.0
    assert not detector.observe()


class FakeLineTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def receive(self, timeout):
        if not self.responses:
            raise AppServerOffline("no response")
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def test_app_server_uses_official_initialize_resume_and_turn_start_protocol(tmp_path):
    transport = FakeLineTransport(
        [
            {"id": 1, "result": {"userAgent": "test"}},
            {"id": 2, "result": {"thread": {"id": "thr_1", "status": {"type": "notLoaded"}}}},
            {"id": 3, "result": {"goal": {"threadId": "thr_1", "status": "paused"}}},
            {"id": 4, "result": {"thread": {"id": "thr_1", "status": {"type": "idle"}}}},
            {"id": 5, "result": {"turn": {"id": "turn_1", "status": "inProgress"}}},
        ]
    )
    session = CodexAppServerSession(config(tmp_path), transport_factory=lambda command: transport)
    with session:
        assert session.deliver("thr_1", "handoff", client_message_id="msg_1") == "turn_1"
    assert [message["method"] for message in transport.sent] == [
        "initialize",
        "initialized",
        "thread/resume",
        "thread/goal/get",
        "thread/read",
        "turn/start",
    ]
    assert transport.sent[-1]["params"] == {
        "threadId": "thr_1",
        "input": [{"type": "text", "text": "handoff"}],
        "clientUserMessageId": "msg_1",
    }
    assert transport.closed


def test_app_server_does_not_resume_or_start_a_busy_thread(tmp_path):
    transport = FakeLineTransport(
        [
            {"id": 1, "result": {"userAgent": "test"}},
            {
                "id": 2,
                "result": {"thread": {"id": "thr_1", "status": {"type": "active", "activeFlags": []}}},
            },
        ]
    )
    session = CodexAppServerSession(config(tmp_path), transport_factory=lambda command: transport)
    with session, pytest.raises(ThreadBusy):
        session.deliver("thr_1", "handoff", client_message_id="msg_1")
    assert [message["method"] for message in transport.sent][-1] == "thread/resume"


def test_app_server_defers_thread_with_active_goal_without_clearing_it(tmp_path):
    transport = FakeLineTransport(
        [
            {"id": 1, "result": {"userAgent": "test"}},
            {"id": 2, "result": {"thread": {"id": "thr_1", "status": {"type": "idle"}}}},
            {"id": 3, "result": {"goal": {"threadId": "thr_1", "status": "active"}}},
        ]
    )
    session = CodexAppServerSession(config(tmp_path), transport_factory=lambda command: transport)
    with session, pytest.raises(ActiveGoalConflict):
        session.deliver("thr_1", "handoff", client_message_id="msg_1")
    assert [message["method"] for message in transport.sent][-2:] == ["thread/resume", "thread/goal/get"]
    assert all(message["method"] != "thread/goal/clear" for message in transport.sent)


def test_app_server_reconciles_delivery_from_user_message_client_id(tmp_path):
    transport = FakeLineTransport(
        [
            {"id": 1, "result": {"userAgent": "test"}},
            {
                "id": 2,
                "result": {
                    "thread": {
                        "id": "thr_1",
                        "status": {"type": "idle"},
                        "turns": [
                            {
                                "id": "turn_1",
                                "items": [
                                    {
                                        "type": "userMessage",
                                        "id": "item_1",
                                        "clientId": "msg_1",
                                        "content": [],
                                    }
                                ],
                            }
                        ],
                    }
                },
            },
        ]
    )
    session = CodexAppServerSession(config(tmp_path), transport_factory=lambda command: transport)
    with session:
        inspection = session.inspect_delivery("thr_1", client_message_id="msg_1")
    assert inspection.found
    assert transport.sent[-1] == {
        "method": "thread/read",
        "id": 2,
        "params": {"threadId": "thr_1", "includeTurns": True},
    }


class FakeTunnel:
    def __init__(self):
        self.state = "healthy"
        self.reconnect_reasons = []
        self.closed = False

    def ensure_running(self, now):
        return True

    def note_probe(self, healthy, now):
        self.state = "healthy" if healthy else "starting"

    def force_reconnect(self, now, *, reason, immediate=False):
        self.reconnect_reasons.append(reason)

    def close(self):
        self.closed = True
        self.state = "stopped"

    def snapshot(self):
        return TunnelSnapshot(
            state=self.state,
            pid=123,
            starts=1,
            reconnects=len(self.reconnect_reasons),
            consecutive_probe_failures=0,
            next_start_at=0.0,
            last_exit_code=None,
            last_error=None,
        )


class FakeConsole:
    def __init__(self, events):
        self.events = list(events)
        self.acks = []
        self.ack_fails = False

    def health(self):
        return True

    def claim_events(self):
        return list(self.events)

    def ack_event(self, claimed_event):
        self.acks.append((claimed_event.event_id, claimed_event.lease_token))
        return not self.ack_fails


class FakeApp:
    def __init__(self, error=None, *, persist_delivery=True):
        self.error = error
        self.persist_delivery = persist_delivery
        self.calls = []
        self.persisted_client_ids = set()
        self.inspection_override = None

    def deliver(self, thread_id, text, *, client_message_id):
        self.calls.append((thread_id, text, client_message_id))
        if self.error:
            raise self.error
        if self.persist_delivery:
            self.persisted_client_ids.add(client_message_id)
        return "turn_accepted"

    def inspect_delivery(self, thread_id, *, client_message_id):
        if self.inspection_override is not None:
            return self.inspection_override
        count = int(client_message_id in self.persisted_client_ids)
        return DeliveryInspection(count, "idle" if not count else "active")


class FakeAppFactory:
    def __init__(self, app=None, enter_error=None):
        self.app = app or FakeApp()
        self.enter_error = enter_error
        self.opens = 0

    def __call__(self, bridge_config):
        factory = self

        class Context:
            def __enter__(self):
                factory.opens += 1
                if factory.enter_error:
                    raise factory.enter_error
                return factory.app

            def __exit__(self, exc_type, exc, traceback):
                return False

        return Context()


def make_service(tmp_path, events, factory):
    bridge_config = config(tmp_path, inflight_retry_grace_seconds=5)
    console = FakeConsole(events)
    clock = SimpleNamespace(value=1000.0)
    state = DeliveryState(
        bridge_config.state_file,
        acked_retention_seconds=bridge_config.acked_event_retention_seconds,
        max_acked_event_ids=bridge_config.max_acked_event_ids,
        wall_clock=lambda: clock.value,
    )
    service = BridgeService(
        bridge_config,
        tunnel=FakeTunnel(),
        console=console,
        state=state,
        status_store=StatusStore(bridge_config.status_file),
        app_session_factory=factory,
        monotonic=lambda: 10.0,
        wall_clock=lambda: clock.value,
        sleeper=lambda seconds: None,
    )
    service.test_clock = clock
    return service, console, state


def test_empty_poll_never_connects_to_app_server_or_starts_model(tmp_path):
    factory = FakeAppFactory()
    service, console, state = make_service(tmp_path, [], factory)
    result = service.tick(force_poll=True)
    assert result.claimed == 0
    assert result.turns_started == 0
    assert factory.opens == 0
    assert console.acks == []


def test_service_replaces_tunnel_before_polling_after_sleep(tmp_path):
    factory = FakeAppFactory()
    service, console, state = make_service(tmp_path, [], factory)

    class Slept:
        def observe(self):
            return True

    service.sleep_detector = Slept()
    result = service.tick(force_poll=True)
    assert result.sleep_reconnect
    assert service.tunnel.reconnect_reasons == ["sleep or long scheduler gap detected"]
    assert result.poll_attempted


def test_events_for_one_thread_are_merged_and_persistently_deduplicated(tmp_path):
    events = [event("evt_1"), event("evt_2")]
    factory = FakeAppFactory()
    service, console, state = make_service(tmp_path, events, factory)
    first = service.tick(force_poll=True)
    assert first.turns_started == 1
    assert first.acked == 0
    assert len(factory.app.calls) == 1
    assert "evt_1" in factory.app.calls[0][1]
    assert "evt_2" in factory.app.calls[0][1]
    assert state.status("evt_1") == "uncertain"
    second = service.tick(force_poll=True)
    assert second.turns_started == 0
    assert second.acked == 2
    assert len(factory.app.calls) == 1
    assert factory.opens == 2
    assert state.status("evt_1") == "acked"
    third = service.tick(force_poll=True)
    assert third.turns_started == 0
    assert len(factory.app.calls) == 1
    assert console.acks == [
        ("evt_1", "lease_evt_1"),
        ("evt_2", "lease_evt_2"),
        ("evt_1", "lease_evt_1"),
        ("evt_2", "lease_evt_2"),
    ]


def test_ack_failure_after_accepted_turn_retries_ack_without_new_turn(tmp_path):
    factory = FakeAppFactory()
    service, console, state = make_service(tmp_path, [event("evt_1")], factory)
    console.ack_fails = True
    first = service.tick(force_poll=True)
    assert first.turns_started == 1
    assert first.acked == 0
    assert state.status("evt_1") == "uncertain"
    second = service.tick(force_poll=True)
    assert second.turns_started == 0
    assert state.status("evt_1") == "delivered"
    console.ack_fails = False
    third = service.tick(force_poll=True)
    assert third.turns_started == 0
    assert len(factory.app.calls) == 1
    assert state.status("evt_1") == "acked"


@pytest.mark.parametrize(
    "factory, expected_state",
    [
        (FakeAppFactory(app=FakeApp(error=ThreadBusy("busy"))), None),
        (FakeAppFactory(app=FakeApp(error=ActiveGoalConflict("active Goal"))), None),
        (FakeAppFactory(enter_error=AppServerOffline("Desktop is offline")), None),
    ],
)
def test_busy_active_goal_or_offline_app_server_retains_events_without_ack(tmp_path, factory, expected_state):
    service, console, state = make_service(tmp_path, [event("evt_1")], factory)
    result = service.tick(force_poll=True)
    assert result.turns_started == 0
    assert result.deferred == 1
    assert console.acks == []
    assert state.status("evt_1") == expected_state


def test_non_actionable_event_never_invokes_app_server_or_ack(tmp_path):
    factory = FakeAppFactory()
    service, console, state = make_service(tmp_path, [event("evt_1", actionable=False)], factory)
    result = service.tick(force_poll=True)
    assert result.rejected == 1
    assert factory.opens == 0
    assert console.acks == []


def test_crash_after_turn_start_is_reconciled_without_duplicate_turn(tmp_path):
    claimed = event("evt_1")
    factory = FakeAppFactory()
    service, console, state = make_service(tmp_path, [claimed], factory)
    client_message_id = service._client_message_id(claimed.thread_id, [claimed])
    state.mark_inflight([claimed], client_message_id=client_message_id)
    factory.app.persisted_client_ids.add(client_message_id)

    result = service.tick(force_poll=True)

    assert result.turns_started == 0
    assert result.acked == 1
    assert factory.app.calls == []
    assert state.status("evt_1") == "acked"


def test_turn_start_timeout_with_persisted_message_recovers_without_resubmit(tmp_path):
    class TimeoutAfterPersist(FakeApp):
        def deliver(self, thread_id, text, *, client_message_id):
            self.calls.append((thread_id, text, client_message_id))
            self.persisted_client_ids.add(client_message_id)
            raise AppServerOffline("turn/start response timed out")

    claimed = event("evt_1")
    app = TimeoutAfterPersist()
    factory = FakeAppFactory(app=app)
    service, console, state = make_service(tmp_path, [claimed], factory)

    timed_out = service.tick(force_poll=True)
    recovered = service.tick(force_poll=True)

    assert timed_out.acked == 0
    assert recovered.acked == 1
    assert len(app.calls) == 1
    assert state.status("evt_1") == "acked"


def test_missing_inflight_delivery_retries_only_after_grace_and_idle(tmp_path):
    claimed = event("evt_1")
    factory = FakeAppFactory(app=FakeApp(persist_delivery=True))
    service, console, state = make_service(tmp_path, [claimed], factory)
    client_message_id = service._client_message_id(claimed.thread_id, [claimed])
    state.mark_inflight([claimed], client_message_id=client_message_id)

    before_grace = service.tick(force_poll=True)
    assert before_grace.turns_started == 0
    assert factory.app.calls == []
    assert console.acks == []

    service.test_clock.value += 6
    retried = service.tick(force_poll=True)
    assert retried.turns_started == 1
    assert len(factory.app.calls) == 1
    assert factory.app.calls[0][2] == client_message_id
    assert console.acks == []

    confirmed = service.tick(force_poll=True)
    assert confirmed.acked == 1
    assert len(factory.app.calls) == 1


def test_partial_inflight_group_is_never_retried_with_missing_event_payload(tmp_path):
    first_event = event("evt_1")
    missing_event = event("evt_2")
    factory = FakeAppFactory(app=FakeApp(persist_delivery=False))
    service, console, state = make_service(tmp_path, [first_event], factory)
    client_message_id = service._client_message_id(first_event.thread_id, [first_event, missing_event])
    state.mark_inflight([first_event, missing_event], client_message_id=client_message_id)
    service.test_clock.value += 6

    result = service.tick(force_poll=True)

    assert result.turns_started == 0
    assert factory.app.calls == []
    assert console.acks == []
    assert "partial delivery group" in result.last_error


def test_ambiguous_client_id_history_never_acks_or_retries(tmp_path):
    claimed = event("evt_1")
    app = FakeApp()
    app.inspection_override = DeliveryInspection(2, "idle", ambiguous=True)
    factory = FakeAppFactory(app=app)
    service, console, state = make_service(tmp_path, [claimed], factory)
    client_message_id = service._client_message_id(claimed.thread_id, [claimed])
    state.mark_inflight([claimed], client_message_id=client_message_id)

    first = service.tick(force_poll=True)
    second = service.tick(force_poll=True)

    assert first.acked == second.acked == 0
    assert app.calls == []
    assert console.acks == []
    assert state.status("evt_1") == "uncertain"
    assert state.record("evt_1")["ambiguous"] is True


def test_console_client_uses_claim_lease_and_idempotent_ack_contract(tmp_path, monkeypatch):
    token_file = tmp_path / "console_api_token"
    token_file.write_text("bridge-secret\n", encoding="utf-8")
    bridge_config = config(
        tmp_path,
        consumer_id="bridge-a",
        poll_limit=7,
        lease_seconds=90,
        console_token_file=str(token_file),
    )
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return self.payload

    def fake_urlopen(request, timeout):
        calls.append(request)
        if request.get_method() == "POST":
            return Response({"status": "ok", "event_id": "evt/1", "acked": True, "idempotent": False})
        return Response(
            {
                "authority_role": "authoritative",
                "instance_id": "yggdrasil-production",
                "ledger_id": "ledger-production-1",
                "events": [
                    {
                        "event_id": "evt/1",
                        "thread_id": "thr_1",
                        "kind": "result_ready",
                        "summary": "ready",
                        "actionable": True,
                        "payload": {"job_id": "job_1"},
                        "created_at": "2026-07-12T12:00:00Z",
                        "lease": {"consumer_id": "bridge-a", "expires_at": "later", "token": "lease-1"},
                    }
                ]
            }
        )

    monkeypatch.setattr("desktop_bridge.console.urlopen", fake_urlopen)
    client = ConsoleClient(bridge_config)
    claimed = client.claim_events()
    assert [item.event_id for item in claimed] == ["evt/1"]
    query = parse_qs(urlsplit(calls[0].full_url).query)
    assert query == {"consumer_id": ["bridge-a"], "limit": ["7"], "lease_seconds": ["90"]}
    assert calls[0].headers["Authorization"] == "Bearer bridge-secret"
    assert client.ack_event(claimed[0])
    assert calls[1].full_url.endswith("/api/bridge/events/evt%2F1/ack")
    assert json.loads(calls[1].data) == {
        "consumer_id": "bridge-a",
        "expected_ledger_id": "ledger-production-1",
        "lease_token": "lease-1",
    }


def test_claim_revalidates_pinned_ledger_before_returning_events(tmp_path, monkeypatch):
    bridge_config = config(tmp_path)
    state = DeliveryState(
        bridge_config.state_file,
        acked_retention_seconds=bridge_config.acked_event_retention_seconds,
        max_acked_event_ids=bridge_config.max_acked_event_ids,
    )
    state.pin_authority(
        authority_role="authoritative",
        instance_id="yggdrasil-production",
        ledger_id="ledger-1",
    )
    payload = {
        "authority_role": "authoritative",
        "instance_id": "yggdrasil-production",
        "ledger_id": "ledger-2",
        "events": [],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr("desktop_bridge.console.urlopen", lambda request, timeout: Response())
    client = ConsoleClient(bridge_config, authority_state=state)
    with pytest.raises(ConsoleContractError, match="authority pin mismatch"):
        client.claim_events()


def test_console_health_rejects_local_dev_and_pins_stable_production_ledger(tmp_path, monkeypatch):
    bridge_config = config(tmp_path)
    state = DeliveryState(
        bridge_config.state_file,
        acked_retention_seconds=bridge_config.acked_event_retention_seconds,
        max_acked_event_ids=bridge_config.max_acked_event_ids,
    )
    payload = {
        "status": "ok",
        "authority_role": "authoritative",
        "instance_id": "local-development",
        "ledger_id": "ledger-local",
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr("desktop_bridge.console.urlopen", lambda request, timeout: Response())
    client = ConsoleClient(bridge_config, authority_state=state)
    assert not client.health()
    assert client.last_transport_ok
    assert state.authority_pin() is None

    payload.update(instance_id="yggdrasil-production", ledger_id="ledger-production-1")
    assert client.health()
    assert state.authority_pin() == {
        "authority_role": "authoritative",
        "instance_id": "yggdrasil-production",
        "ledger_id": "ledger-production-1",
    }
    assert client.health()

    payload["ledger_id"] = "ledger-production-2"
    assert not client.health()
    assert client.last_transport_ok
    assert "authority pin mismatch" in client.last_health_error


def test_instance_lock_prevents_concurrent_bridge_processes(tmp_path):
    path = tmp_path / "bridge.lock"
    with InstanceLock(path):
        with pytest.raises(AlreadyRunning):
            with InstanceLock(path):
                pass
    with InstanceLock(path):
        assert path.read_text().strip().isdigit()
