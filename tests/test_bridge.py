from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import pytest

from desktop_bridge.app_server import (
    AppServerError,
    CodexAppServerSession,
    DeliveryOutcome,
)
from desktop_bridge.config import BridgeConfig
from desktop_bridge.models import PaneSnapshot, SessionSnapshot
from desktop_bridge.service import BridgeService
from desktop_bridge.state import EventStore, EventStoreFull
from desktop_bridge.tmux import classify_event


class FakeTmux:
    def __init__(self, sessions):
        self._sessions = sessions
        self.invalid_sessions = 0
        self.captures = []

    def sessions(self):
        return list(self._sessions)

    def capture_pane(self, pane_id):
        self.captures.append(pane_id)
        return f"last output for {pane_id}"


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.deliveries = []

    def deliver(self, task_id, text, *, event_id):
        self.deliveries.append((task_id, event_id, text))
        return self.outcomes.pop(0)


class MemoryStatus:
    def __init__(self):
        self.value = None

    def write(self, payload):
        self.value = payload


def snapshot(*, failed=False, live=False):
    panes = [
        PaneSnapshot(
            pane_id="%1",
            pane_index=0,
            dead=not live,
            exit_status=1 if failed else (None if live else 0),
            current_command="python",
            created_at=100,
        )
    ]
    return SessionSnapshot(
        session_id="$1",
        session_name="codex-validation",
        thread_id="019f8223-7124-7722-aea2-05ee78f79ef4",
        generation="generation-1",
        investigation_id="2026-07-21-control-plane-validation",
        started_at=100,
        expected_seconds=20,
        attention_after=120,
        panes=tuple(panes),
    )


def factory_for(session):
    @contextmanager
    def factory(_config):
        yield session

    return factory


def test_bridge_delivers_terminal_event_once_across_restart(tmp_path):
    config = BridgeConfig(
        ssh_target="HCCS-25",
        status_file=tmp_path / "status",
        event_file=tmp_path / "events",
        lock_file=tmp_path / "lock",
    )
    session = FakeSession([DeliveryOutcome("delivered", turn_id="turn-1")])
    first = BridgeService(
        config,
        tmux=FakeTmux([snapshot()]),
        app_session_factory=factory_for(session),
        status_store=MemoryStatus(),
        event_store=EventStore(config.event_file),
        wall_clock=lambda: 130,
    ).run_once()
    assert first.healthy is True
    assert first.delivered == 1
    assert session.deliveries[0][0:2] == (
        "019f8223-7124-7722-aea2-05ee78f79ef4",
        "tmux:generation-1:terminal",
    )
    assert "does not prove W&B or result readiness" in session.deliveries[0][2]

    restarted_session = FakeSession([])
    second = BridgeService(
        config,
        tmux=FakeTmux([snapshot()]),
        app_session_factory=factory_for(restarted_session),
        status_store=MemoryStatus(),
        event_store=EventStore(config.event_file),
        wall_clock=lambda: 131,
    ).run_once()
    assert second.delivered == 0
    assert restarted_session.deliveries == []


def test_event_store_prunes_closed_events_only_after_generation_disappears(tmp_path):
    store = EventStore(tmp_path / "events")
    event = classify_event(snapshot(), 130)
    assert event is not None
    store.add(event)
    store.mark_delivered(event.event_id, "turn-1")

    assert store.reconcile({event.event_id}) == 0
    assert event.event_id in store._read()["events"]
    assert store.reconcile(set()) == 1
    assert store._read()["events"] == {}


def test_event_store_preserves_pending_and_fails_closed_at_limit(tmp_path):
    store = EventStore(tmp_path / "events", max_records=1)
    first = classify_event(snapshot(), 130)
    second = classify_event(snapshot(), 130)
    assert first is not None and second is not None
    store.add(first)
    second = replace(second, event_id="tmux:other:terminal")

    assert store.reconcile(set()) == 0
    with pytest.raises(EventStoreFull, match="record limit"):
        store.add(second)
    assert [event.event_id for event in store.pending(10)] == [first.event_id]


def test_bridge_defers_active_goal_then_delivers_when_blocked(tmp_path):
    config = BridgeConfig(
        ssh_target="HCCS-25",
        status_file=tmp_path / "status",
        event_file=tmp_path / "events",
        lock_file=tmp_path / "lock",
    )
    deferred = FakeSession([DeliveryOutcome("deferred", reason="goal_active")])
    result = BridgeService(
        config,
        tmux=FakeTmux([snapshot(failed=True, live=False)]),
        app_session_factory=factory_for(deferred),
        status_store=MemoryStatus(),
        event_store=EventStore(config.event_file),
        wall_clock=lambda: 110,
    ).run_once()
    assert result.deferred == 1
    assert len(EventStore(config.event_file).pending(10)) == 1

    blocked = FakeSession([DeliveryOutcome("delivered", turn_id="turn-2")])
    result = BridgeService(
        config,
        tmux=FakeTmux([snapshot(failed=True, live=False)]),
        app_session_factory=factory_for(blocked),
        status_store=MemoryStatus(),
        event_store=EventStore(config.event_file),
        wall_clock=lambda: 111,
    ).run_once()
    assert result.delivered == 1
    assert EventStore(config.event_file).pending(10) == []


def test_bridge_delivers_at_most_one_event_per_poll(tmp_path):
    config = BridgeConfig(
        ssh_target="HCCS-25",
        status_file=tmp_path / "status",
        event_file=tmp_path / "events",
        lock_file=tmp_path / "lock",
    )
    first_snapshot = snapshot()
    second_snapshot = replace(
        first_snapshot,
        session_id="$2",
        session_name="codex-validation-2",
        generation="generation-2",
    )
    first_session = FakeSession(
        [
            DeliveryOutcome("delivered", turn_id="turn-1"),
            DeliveryOutcome("delivered", turn_id="turn-should-not-start"),
        ]
    )
    first = BridgeService(
        config,
        tmux=FakeTmux([first_snapshot, second_snapshot]),
        app_session_factory=factory_for(first_session),
        status_store=MemoryStatus(),
        event_store=EventStore(config.event_file),
        wall_clock=lambda: 130,
    ).run_once()

    assert first.delivered == 1
    assert len(first_session.deliveries) == 1
    assert [event.event_id for event in EventStore(config.event_file).pending(10)] == [
        "tmux:generation-2:terminal"
    ]

    second_session = FakeSession(
        [DeliveryOutcome("delivered", turn_id="turn-2")]
    )
    second = BridgeService(
        config,
        tmux=FakeTmux([first_snapshot, second_snapshot]),
        app_session_factory=factory_for(second_session),
        status_store=MemoryStatus(),
        event_store=EventStore(config.event_file),
        wall_clock=lambda: 131,
    ).run_once()

    assert second.delivered == 1
    assert second_session.deliveries[0][1] == "tmux:generation-2:terminal"
    assert EventStore(config.event_file).pending(10) == []


def test_bridge_marks_complete_goal_event_orphaned(tmp_path):
    config = BridgeConfig(
        ssh_target="HCCS-25",
        status_file=tmp_path / "status",
        event_file=tmp_path / "events",
        lock_file=tmp_path / "lock",
    )
    session = FakeSession([DeliveryOutcome("orphaned", reason="goal_complete")])
    result = BridgeService(
        config,
        tmux=FakeTmux([snapshot()]),
        app_session_factory=factory_for(session),
        status_store=MemoryStatus(),
        event_store=EventStore(config.event_file),
        wall_clock=lambda: 130,
    ).run_once()
    assert result.orphaned == 1
    assert EventStore(config.event_file).pending(10) == []


class ScriptedTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def receive(self, _timeout):
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def make_config(tmp_path):
    return BridgeConfig(
        ssh_target="HCCS-25",
        status_file=tmp_path / "status",
        event_file=tmp_path / "events",
        lock_file=tmp_path / "lock",
    )


def test_app_server_delivers_only_to_blocked_goal(tmp_path):
    transport = ScriptedTransport(
        [
            {"id": 1, "result": {}},
            {
                "id": 2,
                "result": {
                    "data": [
                        {
                            "id": "task-1234",
                            "status": {"type": "idle"},
                        }
                    ]
                },
            },
            {
                "id": 3,
                "result": {"goal": {"status": "blocked"}},
            },
            {
                "id": 4,
                "result": {"thread": {"status": {"type": "idle"}}},
            },
            {"id": 5, "result": {"turn": {"id": "turn-1"}}},
        ]
    )
    with CodexAppServerSession(
        make_config(tmp_path), transport_factory=lambda _command: transport
    ) as session:
        outcome = session.deliver("task-1234", "wake", event_id="event-1")
    assert outcome == DeliveryOutcome("delivered", turn_id="turn-1")
    methods = [message.get("method") for message in transport.sent]
    assert methods == [
        "initialize",
        "initialized",
        "thread/list",
        "thread/goal/get",
        "thread/resume",
        "turn/start",
    ]
    assert transport.sent[-1]["params"]["clientUserMessageId"] == "event-1"


@pytest.mark.parametrize("goal_status", ["active", "paused", "usageLimited", "budgetLimited"])
def test_app_server_defers_nonblocked_goal(tmp_path, goal_status):
    transport = ScriptedTransport(
        [
            {"id": 1, "result": {}},
            {
                "id": 2,
                "result": {
                    "data": [
                        {"id": "task-1234", "status": {"type": "idle"}}
                    ]
                },
            },
            {"id": 3, "result": {"goal": {"status": goal_status}}},
        ]
    )
    with CodexAppServerSession(
        make_config(tmp_path), transport_factory=lambda _command: transport
    ) as session:
        outcome = session.deliver("task-1234", "wake", event_id="event-1")
    assert outcome.status == "deferred"
    assert all(message.get("method") != "turn/start" for message in transport.sent)


@pytest.mark.parametrize("goal", [None, {"status": "complete"}])
def test_app_server_orphans_missing_or_complete_goal(tmp_path, goal):
    transport = ScriptedTransport(
        [
            {"id": 1, "result": {}},
            {
                "id": 2,
                "result": {
                    "data": [
                        {"id": "task-1234", "status": {"type": "idle"}}
                    ]
                },
            },
            {"id": 3, "result": {"goal": goal}},
        ]
    )
    with CodexAppServerSession(
        make_config(tmp_path), transport_factory=lambda _command: transport
    ) as session:
        outcome = session.deliver("task-1234", "wake", event_id="event-1")
    assert outcome.status == "orphaned"


def test_app_server_does_not_resume_archived_thread(tmp_path):
    transport = ScriptedTransport(
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"data": []}},
            {
                "id": 3,
                "result": {
                    "data": [
                        {"id": "task-1234", "status": {"type": "idle"}}
                    ]
                },
            },
        ]
    )
    with CodexAppServerSession(
        make_config(tmp_path), transport_factory=lambda _command: transport
    ) as session:
        outcome = session.deliver("task-1234", "wake", event_id="event-1")
    assert outcome == DeliveryOutcome("orphaned", reason="thread_archived")
    assert all(message.get("method") != "thread/resume" for message in transport.sent)


def test_app_server_transport_closes_when_initialization_fails(tmp_path):
    transport = ScriptedTransport(
        [{"id": 1, "error": {"message": "initialization failed"}}]
    )
    with pytest.raises(AppServerError, match="initialize failed"):
        CodexAppServerSession(
            make_config(tmp_path), transport_factory=lambda _command: transport
        )
    assert transport.closed is True
