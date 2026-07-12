from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import AuditEvent, IntentRecord, IntentStatus, JobRecord, JobStatus, TERMINAL_JOB_STATUSES, now_iso
from .redaction import redact_value
from .state import validate_job_transition


class ConsoleStore:
    def __init__(self, sqlite_path: Path, audit_path: Path, *, audit_max_bytes: int = 10 * 1024 * 1024, audit_backup_count: int = 5):
        self.sqlite_path = sqlite_path
        self.audit_path = audit_path
        self.audit_max_bytes = max(1024, audit_max_bytes)
        self.audit_backup_count = max(1, audit_backup_count)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._named_locks_guard = threading.Lock()
        self._named_locks: dict[str, threading.RLock] = {}
        self._named_lock_state = threading.local()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS intents (
                    intent_id TEXT PRIMARY KEY,
                    idempotency_key TEXT,
                    intent TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    confirmation_phrase TEXT NOT NULL,
                    confirmed_at TEXT,
                    executed_at TEXT,
                    plan_json TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_intents_idempotency ON intents(idempotency_key) WHERE idempotency_key IS NOT NULL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    operation_id TEXT,
                    idempotency_key TEXT,
                    entity TEXT,
                    project TEXT,
                    sweep_id TEXT,
                    config_path TEXT,
                    remote_host TEXT,
                    remote_cwd TEXT,
                    conda_env TEXT,
                    agent_pids_json TEXT NOT NULL,
                    operation_log_json TEXT NOT NULL DEFAULT '[]',
                    monitor_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            for column, ddl in [
                ("operation_id", "ALTER TABLE jobs ADD COLUMN operation_id TEXT"),
                ("idempotency_key", "ALTER TABLE jobs ADD COLUMN idempotency_key TEXT"),
                ("operation_log_json", "ALTER TABLE jobs ADD COLUMN operation_log_json TEXT NOT NULL DEFAULT '[]'"),
            ]:
                existing = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
                if column not in existing:
                    conn.execute(ddl)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_launch_identity ON jobs(name, config_path, remote_host, remote_cwd)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_sweep ON jobs(entity, project, sweep_id)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS monitor_schedules (
                    job_id TEXT PRIMARY KEY,
                    interval_seconds INTEGER NOT NULL,
                    timeout_seconds INTEGER NOT NULL,
                    thread_id TEXT,
                    active INTEGER NOT NULL DEFAULT 1,
                    next_run_at TEXT NOT NULL,
                    last_started_at TEXT,
                    last_finished_at TEXT,
                    last_classification TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_monitor_schedules_due ON monitor_schedules(active, next_run_at)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS leases (
                    lease_name TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wake_events (
                    event_id TEXT PRIMARY KEY,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    job_id TEXT NOT NULL,
                    thread_id TEXT,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    lease_token TEXT,
                    acked_at TEXT,
                    acked_by TEXT
                )
            """)
            wake_event_columns = {row["name"] for row in conn.execute("PRAGMA table_info(wake_events)").fetchall()}
            if "lease_token" not in wake_event_columns:
                conn.execute("ALTER TABLE wake_events ADD COLUMN lease_token TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_wake_events_pending ON wake_events(acked_at, created_at)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS source_observations (
                    job_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    reconcile_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    freshness TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(job_id, source)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            row = conn.execute("SELECT value FROM metadata WHERE key = 'ledger_id'").fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO metadata(key, value, updated_at) VALUES ('ledger_id', ?, ?)",
                    (f"ledger_{uuid4().hex}", now_iso()),
                )

    def write_audit(self, event: AuditEvent) -> AuditEvent:
        cleaned = event.model_copy(update={"detail": redact_value(compact_audit_value(event.detail))})
        encoded = cleaned.model_dump_json() + "\n"
        with self._lock:
            self._rotate_audit_if_needed(len(encoded.encode("utf-8")))
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
        return cleaned

    def _rotate_audit_if_needed(self, incoming_bytes: int) -> None:
        try:
            current_bytes = self.audit_path.stat().st_size
        except FileNotFoundError:
            return
        if current_bytes + incoming_bytes <= self.audit_max_bytes:
            return
        oldest = self.audit_path.with_name(f"{self.audit_path.name}.{self.audit_backup_count}")
        if oldest.exists():
            oldest.unlink()
        for index in range(self.audit_backup_count - 1, 0, -1):
            source = self.audit_path.with_name(f"{self.audit_path.name}.{index}")
            if source.exists():
                source.replace(self.audit_path.with_name(f"{self.audit_path.name}.{index + 1}"))
        self.audit_path.replace(self.audit_path.with_name(f"{self.audit_path.name}.1"))

    def read_audit(self, limit: int = 100) -> list[AuditEvent]:
        if not self.audit_path.exists():
            return []
        lines = self.audit_path.read_text(encoding="utf-8").splitlines()
        return [AuditEvent.model_validate_json(line) for line in lines[-limit:] if line.strip()]

    def metadata(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def save_intent_if_absent(self, intent: IntentRecord) -> tuple[IntentRecord, bool]:
        with self._lock:
            if intent.idempotency_key:
                existing = self.find_intent_by_idempotency_key(intent.idempotency_key)
                if existing:
                    return existing, True
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO intents (
                        intent_id, idempotency_key, intent, status, payload_json,
                        requested_by, confirmation_phrase, confirmed_at, executed_at,
                        plan_json, result_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    intent.intent_id,
                    intent.idempotency_key,
                    intent.intent.value,
                    intent.status.value,
                    json.dumps(redact_value(intent.payload), ensure_ascii=False),
                    intent.requested_by,
                    intent.confirmation_phrase,
                    intent.confirmed_at,
                    intent.executed_at,
                    intent.plan.model_dump_json(),
                    json.dumps(redact_value(intent.result), ensure_ascii=False) if intent.result is not None else None,
                    intent.created_at,
                    intent.updated_at,
                ))
            return intent, False

    def _intent_from_row(self, row: sqlite3.Row) -> IntentRecord:
        return IntentRecord.model_validate({
            "intent_id": row["intent_id"],
            "idempotency_key": row["idempotency_key"],
            "intent": row["intent"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"]),
            "requested_by": row["requested_by"],
            "confirmation_phrase": row["confirmation_phrase"],
            "confirmed_at": row["confirmed_at"],
            "executed_at": row["executed_at"],
            "plan": json.loads(row["plan_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    def get_intent(self, intent_id: str) -> IntentRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM intents WHERE intent_id = ?", (intent_id,)).fetchone()
        return self._intent_from_row(row) if row else None

    def find_intent_by_idempotency_key(self, key: str) -> IntentRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM intents WHERE idempotency_key = ?", (key,)).fetchone()
        return self._intent_from_row(row) if row else None

    def update_intent(self, intent: IntentRecord) -> IntentRecord:
        intent.updated_at = now_iso()
        with self._lock, self._connect() as conn:
            conn.execute("""
                UPDATE intents
                SET status = ?, confirmed_at = ?, executed_at = ?, result_json = ?, updated_at = ?
                WHERE intent_id = ?
            """, (
                intent.status.value,
                intent.confirmed_at,
                intent.executed_at,
                json.dumps(redact_value(intent.result), ensure_ascii=False) if intent.result is not None else None,
                intent.updated_at,
                intent.intent_id,
            ))
        return intent

    def list_jobs(self) -> list[JobRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._job_from_row(row) for row in rows]

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._job_from_row(row) if row else None

    def find_job_by_idempotency_key(self, key: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE idempotency_key = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (key,),
            ).fetchone()
        return self._job_from_row(row) if row else None

    def find_job_by_launch_identity(
        self,
        *,
        name: str,
        config_path: str,
        remote_host: str,
        remote_cwd: str,
        kind: str | None = None,
    ) -> JobRecord | None:
        jobs = self.find_jobs_by_launch_identity(
            name=name,
            config_path=config_path,
            remote_host=remote_host,
            remote_cwd=remote_cwd,
        )
        if kind:
            return next((job for job in jobs if infer_job_kind(job) == kind), None)
        return jobs[0] if jobs else None

    def find_jobs_by_launch_identity(
        self,
        *,
        name: str,
        config_path: str,
        remote_host: str,
        remote_cwd: str,
    ) -> list[JobRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE name = ? AND config_path = ? AND remote_host = ? AND remote_cwd = ?
                ORDER BY created_at DESC
                """,
                (name, config_path, remote_host, remote_cwd),
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def find_job_by_sweep(self, entity: str, project: str, sweep_id: str) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE entity = ? AND project = ? AND sweep_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (entity, project, sweep_id),
            ).fetchone()
        return self._job_from_row(row) if row else None

    def find_other_job_by_sweep(
        self,
        entity: str,
        project: str,
        sweep_id: str,
        *,
        exclude_job_id: str,
    ) -> JobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE entity = ? AND project = ? AND sweep_id = ? AND job_id != ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (entity, project, sweep_id, exclude_job_id),
            ).fetchone()
        return self._job_from_row(row) if row else None

    def list_queue_group_jobs(self, queue_group: str) -> list[JobRecord]:
        jobs = [
            job
            for job in self.list_jobs()
            if isinstance(job.monitor.get("queue"), dict)
            and job.monitor["queue"].get("queue_group") == queue_group
        ]
        return sorted(jobs, key=lambda job: job.created_at)

    def list_queued_jobs(self, queue_group: str | None = None) -> list[JobRecord]:
        jobs = [
            job
            for job in self.list_jobs()
            if job.status == JobStatus.queued
            and isinstance(job.monitor.get("queue"), dict)
            and (queue_group is None or job.monitor["queue"].get("queue_group") == queue_group)
        ]
        return sorted(jobs, key=lambda job: job.created_at)

    def queue_groups(self) -> list[str]:
        groups = {
            str(job.monitor["queue"].get("queue_group"))
            for job in self.list_jobs()
            if isinstance(job.monitor.get("queue"), dict) and job.monitor["queue"].get("queue_group")
        }
        return sorted(groups)

    def active_queue_blocker(self, queue_group: str, *, exclude_job_id: str | None = None) -> JobRecord | None:
        for job in self.list_queue_group_jobs(queue_group):
            if exclude_job_id and job.job_id == exclude_job_id:
                continue
            if job.status in TERMINAL_JOB_STATUSES or job.status == JobStatus.queued:
                continue
            if infer_job_kind(job) == "sweep":
                return job
        return None

    def next_queued_job(self, queue_group: str) -> JobRecord | None:
        jobs = self.list_queued_jobs(queue_group)
        return jobs[0] if jobs else None

    def _job_from_row(self, row: sqlite3.Row) -> JobRecord:
        keys = set(row.keys())
        return JobRecord.model_validate({
            "job_id": row["job_id"],
            "name": row["name"],
            "status": row["status"],
            "operation_id": row["operation_id"] if "operation_id" in keys else None,
            "idempotency_key": row["idempotency_key"] if "idempotency_key" in keys else None,
            "entity": row["entity"],
            "project": row["project"],
            "sweep_id": row["sweep_id"],
            "config_path": row["config_path"],
            "remote_host": row["remote_host"],
            "remote_cwd": row["remote_cwd"],
            "conda_env": row["conda_env"],
            "agent_pids": json.loads(row["agent_pids_json"] or "[]"),
            "operation_log": json.loads(row["operation_log_json"] or "[]") if "operation_log_json" in keys else [],
            "monitor": json.loads(row["monitor_json"] or "{}"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    def upsert_job(self, job: JobRecord) -> JobRecord:
        job.updated_at = now_iso()
        with self._lock, self._connect() as conn:
            self._upsert_job_conn(conn, job)
        return job

    def upsert_job_with_monitor_schedule(
        self,
        job: JobRecord,
        *,
        interval_seconds: int,
        timeout_seconds: int,
        thread_id: str,
    ) -> JobRecord:
        now = datetime.now(timezone.utc)
        stamp = now.isoformat(timespec="seconds")
        next_run_at = (now + timedelta(seconds=max(1, interval_seconds))).isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""
                INSERT INTO monitor_schedules(
                    job_id, interval_seconds, timeout_seconds, thread_id, active,
                    next_run_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    interval_seconds = excluded.interval_seconds,
                    timeout_seconds = excluded.timeout_seconds,
                    thread_id = excluded.thread_id,
                    active = 1,
                    next_run_at = excluded.next_run_at,
                    updated_at = excluded.updated_at
            """, (
                job.job_id,
                interval_seconds,
                timeout_seconds,
                thread_id,
                next_run_at,
                stamp,
                stamp,
            ))
            row = conn.execute("SELECT * FROM monitor_schedules WHERE job_id = ?", (job.job_id,)).fetchone()
            schedule = dict(row) if row else {}
            job.monitor["monitor_schedule"] = {
                key: schedule.get(key)
                for key in (
                    "job_id", "interval_seconds", "timeout_seconds", "thread_id", "active",
                    "next_run_at", "last_started_at", "last_finished_at", "last_classification",
                    "last_error", "updated_at",
                )
                if key in schedule
            }
            job.updated_at = stamp
            self._upsert_job_conn(conn, job)
        return job

    def _upsert_job_conn(self, conn: sqlite3.Connection, job: JobRecord) -> None:
        conn.execute("""
                INSERT INTO jobs (
                    job_id, name, status, operation_id, idempotency_key, entity, project, sweep_id, config_path,
                    remote_host, remote_cwd, conda_env, agent_pids_json, operation_log_json,
                    monitor_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    name = excluded.name,
                    status = excluded.status,
                    operation_id = excluded.operation_id,
                    idempotency_key = excluded.idempotency_key,
                    entity = excluded.entity,
                    project = excluded.project,
                    sweep_id = excluded.sweep_id,
                    config_path = excluded.config_path,
                    remote_host = excluded.remote_host,
                    remote_cwd = excluded.remote_cwd,
                    conda_env = excluded.conda_env,
                    agent_pids_json = excluded.agent_pids_json,
                    operation_log_json = excluded.operation_log_json,
                    monitor_json = excluded.monitor_json,
                    updated_at = excluded.updated_at
            """, (
                job.job_id,
                job.name,
                job.status.value,
                job.operation_id,
                job.idempotency_key,
                job.entity,
                job.project,
                job.sweep_id,
                job.config_path,
                job.remote_host,
                job.remote_cwd,
                job.conda_env,
                json.dumps(job.agent_pids, ensure_ascii=False),
                json.dumps(redact_value(job.operation_log), ensure_ascii=False),
                json.dumps(redact_value(job.monitor), ensure_ascii=False),
                job.created_at,
                job.updated_at,
            ))

    def update_job_status(self, job_id: str, status: JobStatus, monitor: dict[str, Any] | None = None) -> JobRecord:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(f"job not found: {job_id}")
        validate_job_transition(job.status, status)
        job.status = status
        if monitor:
            job.monitor.update(monitor)
        return self.upsert_job(job)

    def reconcile_job(
        self,
        job: JobRecord,
        status: JobStatus,
        monitor: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> JobRecord:
        """Atomically commit one authoritative reconcile and its source observations."""
        if job.status != status:
            if not (job.status == JobStatus.finished and status in {JobStatus.running, JobStatus.finalizing, JobStatus.attention, JobStatus.unknown}):
                validate_job_transition(job.status, status)
            job.status = status
        job.monitor.update(monitor)
        job.updated_at = now_iso()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._upsert_job_conn(conn, job)
            for observation in observations:
                conn.execute("""
                    INSERT INTO source_observations (
                        job_id, source, reconcile_id, observed_at, freshness, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id, source) DO UPDATE SET
                        reconcile_id = excluded.reconcile_id,
                        observed_at = excluded.observed_at,
                        freshness = excluded.freshness,
                        payload_json = excluded.payload_json
                """, (
                    job.job_id,
                    str(observation["source"]),
                    str(observation["reconcile_id"]),
                    str(observation["observed_at"]),
                    str(observation["freshness"]),
                    json.dumps(redact_value(observation.get("payload") or {}), ensure_ascii=False),
                ))
        return job

    def source_observations(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM source_observations WHERE job_id = ? ORDER BY source",
                (job_id,),
            ).fetchall()
        return [
            {
                "job_id": row["job_id"],
                "source": row["source"],
                "reconcile_id": row["reconcile_id"],
                "observed_at": row["observed_at"],
                "freshness": row["freshness"],
                "payload": json.loads(row["payload_json"] or "{}"),
            }
            for row in rows
        ]

    def acquire_lease(self, lease_name: str, owner_id: str, ttl_seconds: int) -> bool:
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=max(1, ttl_seconds))).isoformat(timespec="seconds")
        now_text = now.isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT owner_id, expires_at FROM leases WHERE lease_name = ?", (lease_name,)).fetchone()
            if row and row["owner_id"] != owner_id and _parse_timestamp(row["expires_at"]) > now:
                return False
            conn.execute("""
                INSERT INTO leases(lease_name, owner_id, expires_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(lease_name) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
            """, (lease_name, owner_id, expires_at, now_text))
        return True

    def lease_owned(self, lease_name: str, owner_id: str) -> bool:
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            row = conn.execute("SELECT owner_id, expires_at FROM leases WHERE lease_name = ?", (lease_name,)).fetchone()
        return bool(row and row["owner_id"] == owner_id and _parse_timestamp(row["expires_at"]) > now)

    def release_lease(self, lease_name: str, owner_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM leases WHERE lease_name = ? AND owner_id = ?", (lease_name, owner_id))

    @contextmanager
    def named_lock(self, name: str, *, ttl_seconds: int = 900):
        with self._named_locks_guard:
            local_lock = self._named_locks.setdefault(name, threading.RLock())
        with local_lock:
            held = getattr(self._named_lock_state, "held", None)
            if held is None:
                held = {}
                self._named_lock_state.held = held
            existing = held.get(name)
            if existing:
                existing["depth"] += 1
                try:
                    yield
                finally:
                    existing["depth"] -= 1
                return
            owner_id = f"{uuid4().hex}:{threading.get_ident()}"
            if not self.acquire_lease(name, owner_id, ttl_seconds):
                raise RuntimeError(f"lock busy: {name}")
            held[name] = {"owner_id": owner_id, "depth": 1}
            try:
                yield
            finally:
                held.pop(name, None)
                self.release_lease(name, owner_id)

    def upsert_monitor_schedule(
        self,
        *,
        job_id: str,
        interval_seconds: int,
        timeout_seconds: int,
        thread_id: str | None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        stamp = now.isoformat(timespec="seconds")
        next_run_at = (now + timedelta(seconds=max(1, interval_seconds))).isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            conn.execute("""
                INSERT INTO monitor_schedules(
                    job_id, interval_seconds, timeout_seconds, thread_id, active,
                    next_run_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    interval_seconds = excluded.interval_seconds,
                    timeout_seconds = excluded.timeout_seconds,
                    thread_id = excluded.thread_id,
                    active = 1,
                    next_run_at = excluded.next_run_at,
                    updated_at = excluded.updated_at
            """, (job_id, interval_seconds, timeout_seconds, thread_id, next_run_at, stamp, stamp))
        return self.get_monitor_schedule(job_id) or {}

    def get_monitor_schedule(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM monitor_schedules WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def disable_monitor_schedule(self, job_id: str) -> tuple[dict[str, Any] | None, bool]:
        existing = self.get_monitor_schedule(job_id)
        if not existing:
            return None, False
        stamp = now_iso()
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE monitor_schedules SET active = 0, updated_at = ? WHERE job_id = ?", (stamp, job_id))
        return self.get_monitor_schedule(job_id), bool(existing.get("active"))

    def due_monitor_schedules(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM monitor_schedules WHERE active = 1 AND next_run_at <= ? ORDER BY next_run_at LIMIT ?",
                (now_iso(), max(1, limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_monitor_started(self, job_id: str) -> None:
        stamp = now_iso()
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE monitor_schedules SET last_started_at = ?, updated_at = ? WHERE job_id = ?", (stamp, stamp, job_id))

    def mark_monitor_finished(self, job_id: str, *, classification: str, error: str | None = None) -> None:
        schedule = self.get_monitor_schedule(job_id)
        if not schedule:
            return
        now = datetime.now(timezone.utc)
        next_run_at = (now + timedelta(seconds=max(1, int(schedule["interval_seconds"])))).isoformat(timespec="seconds")
        stamp = now.isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            conn.execute("""
                UPDATE monitor_schedules
                SET last_finished_at = ?, last_classification = ?, last_error = ?, next_run_at = ?, updated_at = ?
                WHERE job_id = ?
            """, (stamp, classification, error, next_run_at, stamp, job_id))

    def enqueue_wake_event(
        self,
        *,
        dedupe_key: str,
        job_id: str,
        thread_id: str | None,
        kind: str,
        summary: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            raise ValueError("wake events require a non-empty thread_id")
        event_id = f"evt_{uuid4().hex}"
        created_at = now_iso()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM wake_events WHERE dedupe_key = ?", (dedupe_key,)).fetchone()
            if existing:
                return self._wake_event_from_row(existing), False
            conn.execute("""
                INSERT INTO wake_events(
                    event_id, dedupe_key, job_id, thread_id, kind, summary, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (event_id, dedupe_key, job_id, normalized_thread_id, kind, summary, json.dumps(redact_value(payload), ensure_ascii=False), created_at))
            row = conn.execute("SELECT * FROM wake_events WHERE event_id = ?", (event_id,)).fetchone()
        return self._wake_event_from_row(row), True

    def claim_wake_events(self, *, consumer_id: str, limit: int = 20, lease_seconds: int = 60) -> list[dict[str, Any]]:
        _, events = self.claim_wake_events_with_ledger(
            consumer_id=consumer_id,
            limit=limit,
            lease_seconds=lease_seconds,
        )
        return events

    def claim_wake_events_with_ledger(
        self,
        *,
        consumer_id: str,
        limit: int = 20,
        lease_seconds: int = 60,
    ) -> tuple[str, list[dict[str, Any]]]:
        now = datetime.now(timezone.utc)
        now_text = now.isoformat(timespec="seconds")
        lease_expires = (now + timedelta(seconds=max(1, lease_seconds))).isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            ledger_row = conn.execute("SELECT value FROM metadata WHERE key = 'ledger_id'").fetchone()
            ledger_id = str(ledger_row["value"]) if ledger_row else ""
            if not ledger_id:
                raise RuntimeError("wake event claim requires a non-empty ledger_id")
            rows = conn.execute("""
                SELECT * FROM wake_events
                WHERE acked_at IS NULL
                  AND thread_id IS NOT NULL
                  AND TRIM(thread_id) != ''
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ? OR lease_owner = ?)
                ORDER BY created_at
                LIMIT ?
            """, (now_text, consumer_id, max(1, limit))).fetchall()
            ids = [row["event_id"] for row in rows]
            for event_id in ids:
                lease_token = f"lease_{uuid4().hex}"
                conn.execute(
                    "UPDATE wake_events SET lease_owner = ?, lease_expires_at = ?, lease_token = ? WHERE event_id = ?",
                    (consumer_id, lease_expires, lease_token, event_id),
                )
            claimed = [conn.execute("SELECT * FROM wake_events WHERE event_id = ?", (event_id,)).fetchone() for event_id in ids]
        return ledger_id, [self._wake_event_from_row(row) for row in claimed]

    def ack_wake_event(
        self,
        event_id: str,
        *,
        consumer_id: str,
        expected_ledger_id: str,
        lease_token: str,
    ) -> tuple[dict[str, Any], bool]:
        now = datetime.now(timezone.utc)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            ledger_row = conn.execute("SELECT value FROM metadata WHERE key = 'ledger_id'").fetchone()
            current_ledger_id = str(ledger_row["value"]) if ledger_row else ""
            if not current_ledger_id or expected_ledger_id != current_ledger_id:
                raise WakeEventLedgerMismatch(
                    f"wake event ledger mismatch: expected {expected_ledger_id!r}, current {current_ledger_id!r}"
                )
            row = conn.execute("SELECT * FROM wake_events WHERE event_id = ?", (event_id,)).fetchone()
            if not row:
                raise KeyError(f"wake event not found: {event_id}")
            if row["lease_owner"] != consumer_id or row["lease_token"] != lease_token:
                raise WakeEventLeaseConflict("wake event lease owner or token does not match")
            if row["acked_at"]:
                return self._wake_event_from_row(row), True
            lease_expires = _parse_timestamp(row["lease_expires_at"])
            if lease_expires <= now:
                raise WakeEventLeaseConflict("wake event lease expired before ack")
            conn.execute(
                "UPDATE wake_events SET acked_at = ?, acked_by = ? WHERE event_id = ?",
                (now.isoformat(timespec="seconds"), consumer_id, event_id),
            )
            updated = conn.execute("SELECT * FROM wake_events WHERE event_id = ?", (event_id,)).fetchone()
        return self._wake_event_from_row(updated), False

    def _wake_event_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["event_id"],
            "event_id": row["event_id"],
            "dedupe_key": row["dedupe_key"],
            "job_id": row["job_id"],
            "thread_id": row["thread_id"],
            "kind": row["kind"],
            "summary": row["summary"],
            "actionable": True,
            "payload": json.loads(row["payload_json"] or "{}"),
            "created_at": row["created_at"],
            "lease": {
                "consumer_id": row["lease_owner"],
                "expires_at": row["lease_expires_at"],
                "token": row["lease_token"],
            } if row["lease_owner"] else None,
            "acked_at": row["acked_at"],
        }


def infer_job_kind(job: JobRecord) -> str:
    monitor_kind = str((job.monitor or {}).get("kind") or "")
    if monitor_kind == "single_run":
        return "single_run"
    if monitor_kind in {"sweep", "wandb_sweep"}:
        return "sweep"
    if job.sweep_id:
        return "sweep"
    return "unknown"


class WakeEventLeaseConflict(RuntimeError):
    pass


class WakeEventLedgerMismatch(RuntimeError):
    pass


def _parse_timestamp(value: Any) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def compact_audit_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 6:
        return "<max-depth>"
    if isinstance(value, str):
        return value if len(value) <= 4000 else value[:4000] + "...<truncated>"
    if isinstance(value, list):
        compact = [compact_audit_value(item, depth=depth + 1) for item in value[:20]]
        if len(value) > 20:
            compact.append({"truncated_items": len(value) - 20})
        return compact
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in list(value.items())[:80]:
        if key == "status_result" and isinstance(item, dict):
            state = item.get("state") if isinstance(item.get("state"), dict) else {}
            result[key] = {
                "classification": item.get("classification"),
                "job_id": (item.get("job") or {}).get("job_id") if isinstance(item.get("job"), dict) else None,
                "state": compact_audit_value(state, depth=depth + 1),
                "generated_at": item.get("generated_at"),
            }
            continue
        if key in {"runs", "operation_history"} and isinstance(item, list):
            result[f"{key}_count"] = len(item)
            continue
        result[str(key)] = compact_audit_value(item, depth=depth + 1)
    if len(value) > 80:
        result["truncated_keys"] = len(value) - 80
    return result
