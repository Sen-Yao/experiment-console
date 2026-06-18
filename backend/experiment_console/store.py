from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import AuditEvent, IntentRecord, IntentStatus, JobRecord, JobStatus, now_iso
from .redaction import redact_value
from .state import validate_job_transition


class ConsoleStore:
    def __init__(self, sqlite_path: Path, audit_path: Path):
        self.sqlite_path = sqlite_path
        self.audit_path = audit_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
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

    def write_audit(self, event: AuditEvent) -> AuditEvent:
        cleaned = event.model_copy(update={"detail": redact_value(event.detail)})
        with self._lock:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(cleaned.model_dump_json() + "\n")
        return cleaned

    def read_audit(self, limit: int = 100) -> list[AuditEvent]:
        if not self.audit_path.exists():
            return []
        lines = self.audit_path.read_text(encoding="utf-8").splitlines()
        return [AuditEvent.model_validate_json(line) for line in lines[-limit:] if line.strip()]

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
        return job

    def update_job_status(self, job_id: str, status: JobStatus, monitor: dict[str, Any] | None = None) -> JobRecord:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(f"job not found: {job_id}")
        validate_job_transition(job.status, status)
        job.status = status
        if monitor:
            job.monitor.update(monitor)
        return self.upsert_job(job)


def infer_job_kind(job: JobRecord) -> str:
    monitor_kind = str((job.monitor or {}).get("kind") or "")
    if monitor_kind == "single_run":
        return "single_run"
    if monitor_kind in {"sweep", "wandb_sweep"}:
        return "sweep"
    if job.sweep_id:
        return "sweep"
    return "unknown"
