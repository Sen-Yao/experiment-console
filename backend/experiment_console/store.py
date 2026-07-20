from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from uuid import uuid4

from .models import ACTIVE_JOB_STATUSES, JobRecord, JobStatus, TERMINAL_JOB_STATUSES, utc_now


V3_TABLES = {"metadata", "jobs", "events", "resource_locks", "outbox"}


class IdempotencyConflict(ValueError):
    pass


class ResourceBusy(ValueError):
    def __init__(self, profile: str, gpu_indices: list[int]):
        self.profile = profile
        self.gpu_indices = gpu_indices
        super().__init__(f"GPU already locked on {profile}: {','.join(map(str, gpu_indices))}")


class ConsoleStore:
    def __init__(self, sqlite_path):
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.sqlite_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            existing_tables = {
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            if existing_tables:
                version = None
                if "metadata" in existing_tables:
                    try:
                        row = connection.execute(
                            "SELECT value FROM metadata WHERE key='schema_version'"
                        ).fetchone()
                    except sqlite3.DatabaseError:
                        row = None
                    version = str(row["value"]) if row else None
                if version != "3":
                    raise RuntimeError(
                        "refusing to open a non-v3 ledger; archive it and start with an empty state directory"
                    )
                if existing_tables != V3_TABLES:
                    raise RuntimeError(
                        "refusing to repair or extend an incompatible v3 ledger; "
                        "archive it and start with an empty state directory"
                    )
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    request_hash TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    argv_json TEXT NOT NULL,
                    env_json TEXT NOT NULL,
                    gpu_indices_json TEXT NOT NULL,
                    total_runs INTEGER,
                    completed_runs INTEGER NOT NULL DEFAULT 0,
                    name TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    remote_pid INTEGER,
                    remote_pgid INTEGER,
                    remote_start_ticks TEXT,
                    exit_code INTEGER,
                    progress_message TEXT,
                    last_observed_at TEXT,
                    last_error TEXT,
                    cancel_requested_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    detail_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id, event_id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_terminal_event_once
                    ON events(job_id, event_type)
                    WHERE event_type IN ('succeeded', 'failed', 'cancelled');
                CREATE TABLE IF NOT EXISTS resource_locks (
                    profile TEXT NOT NULL,
                    gpu_index INTEGER NOT NULL,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    locked_at TEXT NOT NULL,
                    PRIMARY KEY(profile, gpu_index)
                );
                CREATE INDEX IF NOT EXISTS idx_locks_job ON resource_locks(job_id);
                CREATE TABLE IF NOT EXISTS outbox (
                    event_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_token TEXT,
                    lease_until TEXT,
                    acked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_outbox_claim
                    ON outbox(acked_at, lease_until, created_at);
                """
            )
            row = connection.execute("SELECT value FROM metadata WHERE key='ledger_id'").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('ledger_id', ?)",
                    (f"ledger_v3_{uuid4().hex}",),
                )
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('schema_version', '3') "
                "ON CONFLICT(key) DO UPDATE SET value='3'"
            )

    def metadata(self, key: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else None

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            request_id=row["request_id"],
            request_hash=row["request_hash"],
            task_id=row["task_id"],
            profile=row["profile"],
            cwd=row["cwd"],
            argv=json.loads(row["argv_json"]),
            env=json.loads(row["env_json"]),
            gpu_indices=json.loads(row["gpu_indices_json"]),
            total_runs=row["total_runs"],
            completed_runs=int(row["completed_runs"] or 0),
            name=row["name"],
            status=JobStatus(row["status"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            remote_pid=row["remote_pid"],
            remote_pgid=row["remote_pgid"],
            remote_start_ticks=row["remote_start_ticks"],
            exit_code=row["exit_code"],
            progress_message=row["progress_message"],
            last_observed_at=row["last_observed_at"],
            last_error=row["last_error"],
            cancel_requested_at=row["cancel_requested_at"],
        )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._job_from_row(row) if row else None

    def get_by_request_id(self, request_id: str) -> JobRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE request_id=?", (request_id,)
            ).fetchone()
        return self._job_from_row(row) if row else None

    def active_jobs(self) -> list[JobRecord]:
        placeholders = ",".join("?" for _ in ACTIVE_JOB_STATUSES)
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM jobs WHERE status IN ({placeholders}) ORDER BY created_at",
                [item.value for item in ACTIVE_JOB_STATUSES],
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def create_job(self, job: JobRecord) -> tuple[JobRecord, bool]:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_row = connection.execute(
                "SELECT * FROM jobs WHERE request_id=?", (job.request_id,)
            ).fetchone()
            if existing_row:
                existing = self._job_from_row(existing_row)
                if existing.request_hash != job.request_hash:
                    connection.rollback()
                    raise IdempotencyConflict(
                        f"request_id already belongs to job {existing.job_id} with a different payload"
                    )
                connection.commit()
                return existing, True
            conflicts = connection.execute(
                "SELECT gpu_index FROM resource_locks WHERE profile=? AND gpu_index IN ({})".format(
                    ",".join("?" for _ in job.gpu_indices) or "NULL"
                ),
                [job.profile, *job.gpu_indices],
            ).fetchall()
            if conflicts:
                connection.rollback()
                raise ResourceBusy(job.profile, [int(row["gpu_index"]) for row in conflicts])
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, request_id, request_hash, task_id, profile, cwd, argv_json, env_json,
                    gpu_indices_json, total_runs, completed_runs, name, status, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job.job_id,
                    job.request_id,
                    job.request_hash,
                    job.task_id,
                    job.profile,
                    job.cwd,
                    json.dumps(job.argv, ensure_ascii=False),
                    json.dumps(job.env, ensure_ascii=False),
                    json.dumps(job.gpu_indices),
                    job.total_runs,
                    job.completed_runs,
                    job.name,
                    job.status.value,
                    job.created_at,
                ),
            )
            for gpu_index in job.gpu_indices:
                connection.execute(
                    "INSERT INTO resource_locks(profile,gpu_index,job_id,locked_at) VALUES(?,?,?,?)",
                    (job.profile, gpu_index, job.job_id, job.created_at),
                )
            connection.execute(
                "INSERT INTO events(job_id,event_type,created_at,detail_json) VALUES(?,?,?,?)",
                (job.job_id, "created", job.created_at, "{}"),
            )
            connection.commit()
            return job, False

    def update_observation(
        self, job_id: str, observation: dict[str, Any], *, error: str | None = None
    ) -> JobRecord:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row:
                connection.rollback()
                raise KeyError(f"job not found: {job_id}")
            job = self._job_from_row(row)
            remote_state = str(observation.get("state") or "unknown")
            if remote_state == "running":
                status = (
                    JobStatus.cancelling
                    if job.status == JobStatus.cancelling
                    else JobStatus.running
                )
            elif remote_state == "succeeded":
                status = JobStatus.succeeded
            elif remote_state == "cancelled":
                status = JobStatus.cancelled
            elif remote_state in {"failed", "lost"}:
                status = JobStatus.failed
            else:
                status = JobStatus.unknown
            observed_at = str(observation.get("observed_at") or utc_now())
            completed = observation.get("completed_runs")
            total = observation.get("total_runs")
            if completed is not None:
                completed = max(0, int(completed))
            if total is not None:
                total = max(1, int(total))
            finished_at = observed_at if status in TERMINAL_JOB_STATUSES else job.finished_at
            last_error = error or (
                "remote supervisor disappeared" if remote_state == "lost" else None
            )
            previous_status = job.status
            connection.execute(
                """
                UPDATE jobs SET status=?, started_at=COALESCE(started_at, ?), finished_at=?,
                    remote_pid=COALESCE(?, remote_pid), remote_pgid=COALESCE(?, remote_pgid),
                    remote_start_ticks=COALESCE(?, remote_start_ticks), exit_code=COALESCE(?, exit_code),
                    completed_runs=COALESCE(?, completed_runs), total_runs=COALESCE(?, total_runs),
                    progress_message=COALESCE(?, progress_message), last_observed_at=?, last_error=?
                WHERE job_id=?
                """,
                (
                    status.value,
                    observed_at if observation.get("pid") else None,
                    finished_at,
                    observation.get("pid"),
                    observation.get("pgid"),
                    observation.get("start_ticks"),
                    observation.get("exit_code"),
                    completed,
                    total,
                    observation.get("progress_message"),
                    observed_at,
                    last_error,
                    job_id,
                ),
            )
            if status in TERMINAL_JOB_STATUSES:
                connection.execute("DELETE FROM resource_locks WHERE job_id=?", (job_id,))
                event_id = f"terminal:{job_id}"
                payload = {
                    "event_id": event_id,
                    "job_id": job_id,
                    "task_id": job.task_id,
                    "status": status.value,
                    "exit_code": observation.get("exit_code"),
                    "finished_at": observed_at,
                }
                connection.execute(
                    "INSERT OR IGNORE INTO events(job_id,event_type,created_at,detail_json) VALUES(?,?,?,?)",
                    (
                        job_id,
                        status.value,
                        observed_at,
                        json.dumps(payload, separators=(",", ":")),
                    ),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO outbox(event_id,job_id,task_id,event_type,payload_json,created_at)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (
                        event_id,
                        job_id,
                        job.task_id,
                        "job_terminal",
                        json.dumps(payload, separators=(",", ":")),
                        observed_at,
                    ),
                )
            elif status != previous_status:
                connection.execute(
                    "INSERT INTO events(job_id,event_type,created_at,detail_json) VALUES(?,?,?,?)",
                    (job_id, status.value, observed_at, "{}"),
                )
            connection.commit()
        result = self.get_job(job_id)
        if result is None:
            raise KeyError(job_id)
        return result

    def request_cancel(self, job_id: str, reason: str | None = None) -> JobRecord:
        stamp = utc_now()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row:
                connection.rollback()
                raise KeyError(f"job not found: {job_id}")
            job = self._job_from_row(row)
            if job.status in TERMINAL_JOB_STATUSES:
                connection.commit()
                return job
            connection.execute(
                "UPDATE jobs SET status=?, cancel_requested_at=? WHERE job_id=?",
                (JobStatus.cancelling.value, stamp, job_id),
            )
            connection.execute(
                "INSERT INTO events(job_id,event_type,created_at,detail_json) VALUES(?,?,?,?)",
                (job_id, "cancel_requested", stamp, json.dumps({"reason": reason})),
            )
            connection.commit()
        result = self.get_job(job_id)
        if result is None:
            raise KeyError(job_id)
        return result

    def locked_gpus(self, profile: str) -> dict[int, str]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT gpu_index,job_id FROM resource_locks WHERE profile=?", (profile,)
            ).fetchall()
        return {int(row["gpu_index"]): str(row["job_id"]) for row in rows}

    def claim_outbox(
        self, consumer_id: str, limit: int, lease_seconds: int
    ) -> list[dict[str, Any]]:
        now_dt = datetime.now(timezone.utc)
        now_value = now_dt.isoformat(timespec="seconds")
        lease_until = (now_dt + timedelta(seconds=lease_seconds)).isoformat(timespec="seconds")
        claimed: list[dict[str, Any]] = []
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT * FROM outbox
                WHERE acked_at IS NULL AND (lease_until IS NULL OR lease_until < ? OR lease_owner = ?)
                ORDER BY created_at LIMIT ?
                """,
                (now_value, consumer_id, limit),
            ).fetchall()
            for row in rows:
                token = uuid4().hex
                connection.execute(
                    "UPDATE outbox SET lease_owner=?,lease_token=?,lease_until=? WHERE event_id=?",
                    (consumer_id, token, lease_until, row["event_id"]),
                )
                claimed.append(
                    {
                        "event_id": row["event_id"],
                        "job_id": row["job_id"],
                        "task_id": row["task_id"],
                        "event_type": row["event_type"],
                        "payload": json.loads(row["payload_json"]),
                        "created_at": row["created_at"],
                        "lease_token": token,
                        "lease_until": lease_until,
                    }
                )
            connection.commit()
        return claimed

    def ack_outbox(self, event_id: str, consumer_id: str, lease_token: str) -> bool:
        stamp = utc_now()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE outbox SET acked_at=?, lease_owner=NULL, lease_token=NULL, lease_until=NULL
                WHERE event_id=? AND lease_owner=? AND lease_token=? AND acked_at IS NULL
                """,
                (stamp, event_id, consumer_id, lease_token),
            )
        return cursor.rowcount == 1
