from __future__ import annotations

from contextlib import contextmanager

import pytest

from desktop_bridge.app_server import AppServerError, CodexAppServerSession
from desktop_bridge.config import BridgeConfig
from desktop_bridge.models import OutboxEvent
from desktop_bridge.service import BridgeService


class FakeTunnel:
    def ensure_running(self, now):
        return True

    def note_probe(self, healthy, now):
        pass

    def force_reconnect(self, now, *, reason, immediate=False):
        pass

    def close(self):
        pass

    def snapshot(self):
        return type("Snapshot", (), {"last_error": None})()


class FakeConsole:
    def __init__(self, event):
        self.event = event
        self.acked = []

    def health(self):
        pass

    def claim(self):
        return [self.event]

    def ack(self, event):
        self.acked.append(event.event_id)


class FakeSession:
    def __init__(self):
        self.deliveries = []

    def deliver(self, task_id, text, *, event_id):
        self.deliveries.append((task_id, event_id, text))
        return "turn-1"


class MemoryStatus:
    def __init__(self):
        self.value = None

    def write(self, payload):
        self.value = payload


def test_bridge_delivers_stable_event_id_and_acks(tmp_path):
    config = BridgeConfig(
        ssh_target="yggdrasil",
        status_file=tmp_path / "status",
        lock_file=tmp_path / "lock",
    )
    event = OutboxEvent(
        event_id="terminal:job_1",
        job_id="job_1",
        task_id="task_1",
        event_type="job_terminal",
        payload={"status": "succeeded", "exit_code": 0},
        lease_token="lease",
    )
    console = FakeConsole(event)
    session = FakeSession()

    @contextmanager
    def factory(_config):
        yield session

    bridge = BridgeService(
        config,
        tunnel=FakeTunnel(),
        console=console,
        app_session_factory=factory,
        status_store=MemoryStatus(),
    )
    result = bridge.run_once()
    assert result.healthy is True
    assert result.delivered == result.acked == 1
    assert session.deliveries[0][:2] == ("task_1", "terminal:job_1")
    assert console.acked == ["terminal:job_1"]


def test_app_server_transport_closes_when_initialization_fails(tmp_path):
    class FailingTransport:
        def __init__(self):
            self.closed = False

        def send(self, _message):
            pass

        def receive(self, _timeout):
            return {"id": 1, "error": {"message": "initialization failed"}}

        def close(self):
            self.closed = True

    transport = FailingTransport()
    config = BridgeConfig(
        ssh_target="yggdrasil",
        status_file=tmp_path / "status",
        lock_file=tmp_path / "lock",
    )
    with pytest.raises(AppServerError, match="initialize failed"):
        CodexAppServerSession(
            config, transport_factory=lambda _command: transport
        )
    assert transport.closed is True
