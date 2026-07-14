from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from experiment_console import api
from experiment_console.config import Settings
from experiment_console.models import JobRecord, JobStatus
from experiment_console.monitor import MonitorWorker
from experiment_console.store import (
    ConsoleStore,
    WakeEventLedgerMismatch,
    WakeEventLeaseConflict,
)


def make_store(tmp_path: Path) -> ConsoleStore:
    return ConsoleStore(tmp_path / "console.sqlite3", tmp_path / "audit.jsonl")


def test_wake_event_ack_requires_current_ledger_and_exact_opaque_lease(tmp_path):
    store = make_store(tmp_path)
    event, _ = store.enqueue_wake_event(
        dedupe_key="job:ready:one",
        job_id="job",
        thread_id="thread-1",
        kind="result_ready",
        summary="ready",
        payload={},
    )
    claimed = store.claim_wake_events(consumer_id="bridge-a", lease_seconds=60)
    lease_token = claimed[0]["lease"]["token"]
    ledger_id = store.metadata("ledger_id")

    with pytest.raises(WakeEventLeaseConflict):
        store.ack_wake_event(
            event["event_id"],
            consumer_id="bridge-a",
            expected_ledger_id=ledger_id,
            lease_token="lease_wrong",
        )
    with pytest.raises(WakeEventLeaseConflict):
        store.ack_wake_event(
            event["event_id"],
            consumer_id="bridge-b",
            expected_ledger_id=ledger_id,
            lease_token=lease_token,
        )
    with pytest.raises(WakeEventLedgerMismatch):
        store.ack_wake_event(
            event["event_id"],
            consumer_id="bridge-a",
            expected_ledger_id="ledger_replaced",
            lease_token=lease_token,
        )

    acked, idempotent = store.ack_wake_event(
        event["event_id"],
        consumer_id="bridge-a",
        expected_ledger_id=ledger_id,
        lease_token=lease_token,
    )
    assert acked["acked_at"] and idempotent is False


def test_poison_events_without_thread_are_rejected_and_never_claimed(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="thread_id"):
        store.enqueue_wake_event(
            dedupe_key="poison:new",
            job_id="job",
            thread_id=None,
            kind="attention",
            summary="cannot route",
            payload={},
        )

    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO wake_events(
                event_id, dedupe_key, job_id, thread_id, kind, summary, payload_json, created_at
            ) VALUES ('evt_legacy_poison', 'poison:legacy', 'job', '', 'attention', 'legacy', '{}',
                      '2026-07-12T00:00:00+00:00')
            """
        )
    assert store.claim_wake_events(consumer_id="bridge-a", lease_seconds=60) == []


def test_bearer_auth_protects_bridge_artifact_and_runner_routes(tmp_path, monkeypatch):
    token_file = tmp_path / "console_api_token"
    token_file.write_text("test-console-token\n", encoding="utf-8")
    monkeypatch.setattr(api.settings, "console_api_token_file", token_file)
    monkeypatch.setattr(api.settings, "authority_role", "local-development")
    client = TestClient(api.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["console_api_auth_configured"] is True
    assert client.get("/api/bridge/events?consumer_id=bridge-a").status_code == 401
    assert client.get("/api/artifacts/missing/download").status_code == 401
    assert client.post("/api/intents/preview", json={}).status_code == 401
    assert client.post("/api/runner/status", json={"job_id": "missing"}).status_code == 401
    assert client.get("/api/hosts/gpus?host=-oProxyCommand%3Dtouch_pwned").status_code == 401
    assert client.get(
        "/api/bridge/events?consumer_id=bridge-a",
        headers={"Authorization": "Bearer wrong"},
    ).status_code == 401

    response = client.get(
        "/api/bridge/events?consumer_id=bridge-a",
        headers={"Authorization": "Bearer test-console-token"},
    )
    assert response.status_code == 200
    assert response.json()["ledger_id"] == api.service.store.metadata("ledger_id")
    assert isinstance(response.json()["events"], list)
    invalid_host = client.get(
        "/api/hosts/gpus?host=-oProxyCommand%3Dtouch_pwned",
        headers={"Authorization": "Bearer test-console-token"},
    )
    assert invalid_host.status_code == 503
    assert "not an option" in invalid_host.json()["detail"]


def test_authoritative_api_fails_closed_when_token_secret_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(api.settings, "console_api_token_file", tmp_path / "missing-token")
    monkeypatch.setattr(api.settings, "authority_role", "authoritative")
    client = TestClient(api.app)
    response = client.get("/api/bridge/events?consumer_id=bridge-a")
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]
    short_token = tmp_path / "short-token"
    short_token.write_text("too-short\n", encoding="utf-8")
    monkeypatch.setattr(api.settings, "console_api_token_file", short_token)
    assert client.get("/api/bridge/events?consumer_id=bridge-a").status_code == 503


def test_monitor_exception_event_exposes_only_stable_redacted_metadata(tmp_path):
    settings = Settings(
        state_dir=tmp_path,
        monitor_worker_enabled=True,
        monitor_worker_poll_seconds=1,
        monitor_lease_seconds=10,
    )
    store = make_store(tmp_path)
    store.upsert_job(JobRecord(
        job_id="job-monitor-error",
        name="monitor-error",
        status=JobStatus.running,
    ))
    store.upsert_monitor_schedule(
        job_id="job-monitor-error",
        interval_seconds=1,
        timeout_seconds=5,
        thread_id="thread-1",
    )
    with store._connect() as connection:
        connection.execute(
            "UPDATE monitor_schedules SET next_run_at = '2020-01-01T00:00:00+00:00' "
            "WHERE job_id = 'job-monitor-error'"
        )

    def fail_tick(job_id):
        raise RuntimeError("token=abcdefghijklmnop secret remote/path")

    worker = MonitorWorker(SimpleNamespace(
        store=store,
        settings=settings,
        due_agent_reconcile_job_ids=lambda *, limit: [],
        monitor_tick=fail_tick,
    ))
    worker.run_once()
    events = store.claim_wake_events(consumer_id="bridge-a", lease_seconds=60)

    assert len(events) == 1
    assert "dedupe_key" not in api._public_wake_event(events[0])
    assert events[0]["payload"]["classification"] == "monitor_invariant_error"
    assert events[0]["payload"]["error_type"] == "RuntimeError"
    assert len(events[0]["payload"]["error_fingerprint"]) == 64
    serialized = str(events[0]) + str(worker.status())
    assert "abcdefghijklmnop" not in serialized
    assert "secret remote/path" not in serialized
