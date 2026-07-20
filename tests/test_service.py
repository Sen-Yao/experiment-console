from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from experiment_console.models import JobStatus, RemoteObservation, RunRequest, utc_now
from experiment_console.remote import RemoteError
from experiment_console.service import ResourceUnavailable
from experiment_console.store import ConsoleStore, IdempotencyConflict, ResourceBusy


def request(**updates) -> RunRequest:
    payload = {
        "request_id": "req_12345678",
        "task_id": "task-1",
        "profile": "test",
        "cwd": "/work/project",
        "argv": ["python", "train.py"],
        "env": {"MODE": "test"},
        "gpu_indices": [0],
        "total_runs": 10,
        "name": "demo",
    }
    payload.update(updates)
    return RunRequest.model_validate(payload)


def test_run_is_idempotent_and_locks_gpu(service):
    first, replayed = service.run(request())
    assert replayed is False
    assert first.status == JobStatus.running
    assert service.store.locked_gpus("test") == {0: first.job_id}

    second, replayed = service.run(request())
    assert replayed is True
    assert second.job_id == first.job_id

    with pytest.raises(IdempotencyConflict):
        service.run(request(argv=["python", "other.py"]))


def test_gpu_conflicts_and_remote_availability_are_rejected(service):
    first, _ = service.run(request())
    with pytest.raises(ResourceBusy):
        service.run(request(request_id="req_abcdefgh", task_id="task-2"))
    with pytest.raises(ResourceUnavailable):
        service.run(request(request_id="req_ijklmnop", gpu_indices=[1]))
    assert service.store.locked_gpus("test") == {0: first.job_id}


def test_progress_eta_terminal_event_and_outbox(service, fake_remote):
    job, _ = service.run(request())
    fake_remote.observations[job.job_id] = RemoteObservation(
        state="running",
        observed_at=utc_now(),
        pid=job.remote_pid,
        pgid=job.remote_pgid,
        start_ticks=job.remote_start_ticks,
        completed_runs=2,
        total_runs=10,
        progress_message="2/10",
    )
    status = service.status(job.job_id)
    assert status["completed_runs"] == 2
    assert status["eta_seconds"] == int(status["elapsed_seconds"] / 2 * 8)

    fake_remote.observations[job.job_id] = RemoteObservation(
        state="succeeded",
        observed_at=utc_now(),
        pid=job.remote_pid,
        pgid=job.remote_pgid,
        start_ticks=job.remote_start_ticks,
        exit_code=0,
        completed_runs=10,
        total_runs=10,
    )
    terminal = service.status(job.job_id)
    assert terminal["status"] == "succeeded"
    assert service.store.locked_gpus("test") == {}

    events = service.store.claim_outbox("bridge", 20, 60)
    assert len(events) == 1
    assert events[0]["event_id"] == f"terminal:{job.job_id}"
    assert service.store.ack_outbox(
        events[0]["event_id"], "bridge", events[0]["lease_token"]
    )
    assert service.store.claim_outbox("bridge", 20, 60) == []


def test_eta_uses_job_creation_time(service):
    job, _ = service.run(request())
    current = datetime.now(timezone.utc)
    created_at = (current - timedelta(seconds=120)).isoformat(timespec="seconds")
    started_at = (current - timedelta(seconds=10)).isoformat(timespec="seconds")
    view = service.public_job(
        job.model_copy(
            update={
                "created_at": created_at,
                "started_at": started_at,
                "completed_runs": 1,
                "total_runs": 2,
            }
        )
    )
    assert 120 <= view["elapsed_seconds"] <= 121
    assert view["eta_seconds"] == view["elapsed_seconds"]
    assert view["eta_basis"]["created_at"] == created_at


def test_logs_fetch_and_cancel(service, fake_remote):
    job, _ = service.run(request())
    log = service.logs(job.job_id, stream="stdout", offset=0, limit=9, tail=True)
    assert log.text.endswith("line two\n")

    data, chunk = service.fetch(job.job_id, path="result.json", offset=0, limit=6)
    assert data == b"result"
    assert chunk.eof is False

    cancelled = service.cancel(job.job_id, reason="user requested")
    assert cancelled["status"] == "cancelled"
    assert fake_remote.cancelled == [job.job_id]
    assert service.store.locked_gpus("test") == {}


def test_schema_contains_only_v3_tables(service):
    with sqlite3.connect(service.settings.sqlite_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            if not row[0].startswith("sqlite_")
        }
    assert tables == {"metadata", "jobs", "events", "resource_locks", "outbox"}


def test_store_rejects_incompatible_existing_ledger(tmp_path):
    sqlite_path = tmp_path / "console-v3.sqlite3"
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('schema_version', '2')"
        )
        connection.execute("CREATE TABLE legacy_jobs(job_id TEXT PRIMARY KEY)")
    with pytest.raises(RuntimeError, match="refusing to open a non-v3 ledger"):
        ConsoleStore(sqlite_path)


def test_store_rejects_v3_ledger_with_a_non_v3_table_set(tmp_path):
    sqlite_path = tmp_path / "console-v3.sqlite3"
    ConsoleStore(sqlite_path)
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            "CREATE TABLE legacy_operations(operation_id TEXT PRIMARY KEY)"
        )
    with pytest.raises(RuntimeError, match="incompatible v3 ledger"):
        ConsoleStore(sqlite_path)


def test_transient_observation_error_preserves_remote_process_identity(
    service, fake_remote
):
    job, _ = service.run(request())
    identity = (job.remote_pid, job.remote_pgid, job.remote_start_ticks)

    def unavailable(*_args, **_kwargs):
        raise RemoteError("temporary SSH failure")

    fake_remote.inspect = unavailable
    refreshed = service.refresh(job.job_id)
    assert refreshed.status == JobStatus.running
    assert (
        refreshed.remote_pid,
        refreshed.remote_pgid,
        refreshed.remote_start_ticks,
    ) == identity
    assert refreshed.last_error == "temporary SSH failure"


def test_monitor_retries_a_cancel_that_initially_cannot_reach_remote(
    service, fake_remote
):
    job, _ = service.run(request())
    working_cancel = fake_remote.cancel

    def unavailable(*_args, **_kwargs):
        raise RemoteError("remote request is not visible yet")

    fake_remote.cancel = unavailable
    cancelling = service.cancel(job.job_id)
    assert cancelling["status"] == "cancelling"

    fake_remote.cancel = working_cancel
    refreshed = service.refresh(job.job_id)
    assert refreshed.status == JobStatus.cancelled
    assert fake_remote.cancelled == [job.job_id]


def test_secret_like_environment_keys_are_rejected():
    with pytest.raises(ValueError, match="secret-like"):
        request(env={"WANDB_API_KEY": "must-not-enter-ledger"})
