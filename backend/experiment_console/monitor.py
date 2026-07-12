from __future__ import annotations

import hashlib
import subprocess
import threading
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from .command import CommandFailed
from .reconcile import update_sync_consistency
from .redaction import redact_text
from .wandb_client import WandBUnavailable

if TYPE_CHECKING:
    from .service import ConsoleService


class MonitorWorker:
    def __init__(self, service: "ConsoleService"):
        self.service = service
        self.store = service.store
        self.settings = service.settings
        self.owner_id = f"monitor_{uuid4().hex}"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready = False
        self._lease_held = False
        self._last_tick_at: str | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if not self.settings.monitor_worker_enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="experiment-console-monitor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self.store.release_lease("monitor-worker", self.owner_id)
        self._lease_held = False

    def status(self) -> dict[str, Any]:
        running = bool(self._thread and self._thread.is_alive())
        return {
            "enabled": self.settings.monitor_worker_enabled,
            "ready": self._ready if self.settings.monitor_worker_enabled else True,
            "running": running,
            "lease_held": self._lease_held,
            "owner_id": self.owner_id,
            "last_tick_at": self._last_tick_at,
            "last_error": self._last_error,
        }

    def run_once(self) -> list[dict[str, Any]]:
        due_schedules = self.store.due_monitor_schedules(limit=20)
        batch_budget = sum(int(schedule["timeout_seconds"]) + 30 for schedule in due_schedules)
        lease_seconds = max(self.settings.monitor_lease_seconds, self.settings.monitor_worker_poll_seconds * 3, batch_budget)
        self._lease_held = self.store.acquire_lease("monitor-worker", self.owner_id, lease_seconds)
        self._ready = True
        if not self._lease_held:
            return []
        results = []
        for index, schedule in enumerate(due_schedules):
            remaining_budget = sum(int(item["timeout_seconds"]) + 30 for item in due_schedules[index:])
            if not self.store.acquire_lease("monitor-worker", self.owner_id, max(self.settings.monitor_lease_seconds, remaining_budget)):
                self._lease_held = False
                break
            job_id = str(schedule["job_id"])
            job_lease = f"monitor-job:{job_id}"
            if not self.store.acquire_lease(job_lease, self.owner_id, int(schedule["timeout_seconds"]) + 30):
                continue
            if not self.store.lease_owned("monitor-worker", self.owner_id) or not self.store.lease_owned(job_lease, self.owner_id):
                self.store.release_lease(job_lease, self.owner_id)
                self._lease_held = False
                break
            self.store.mark_monitor_started(job_id)
            try:
                result = self.service.monitor_tick(job_id)
                classification = str(result.get("classification") or "healthy")
                self.store.mark_monitor_finished(job_id, classification=classification)
                self._clear_monitor_exception(job_id)
                results.append(result)
                self._last_error = None
            except Exception as exc:
                error_type, error_fingerprint = monitor_error_identity(exc)
                self._last_error = f"monitor_tick_error:{error_type}:{error_fingerprint[:16]}"
                expected_external = is_expected_external_monitor_error(exc)
                consistency = self._record_monitor_exception(job_id, error_type) if expected_external else None
                escalated = bool(consistency and consistency.get("classification") == "sync_error")
                classification = (
                    "external_unavailable"
                    if escalated
                    else "external_unavailable_reconciling"
                    if expected_external
                    else "monitor_invariant_error"
                )
                self.store.mark_monitor_finished(
                    job_id,
                    classification=classification,
                    error=self._last_error,
                )
                job = self.store.get_job(job_id)
                if job:
                    schedule_row = self.store.get_monitor_schedule(job_id) or {}
                    thread_id = str(schedule_row.get("thread_id") or "").strip()
                    if thread_id and (not expected_external or escalated):
                        event_kind = "sync_error" if expected_external else "attention"
                        episode_id = consistency.get("episode_id") if consistency else None
                        self.store.enqueue_wake_event(
                            dedupe_key=f"{job_id}:{event_kind}:{classification}:{episode_id or error_fingerprint}",
                            job_id=job_id,
                            thread_id=thread_id,
                            kind=event_kind,
                            summary=(
                                f"job {job_id} external monitor dependency remains unavailable"
                                if expected_external
                                else f"job {job_id} monitor invariant failed"
                            ),
                            payload={
                                "classification": classification,
                                "error_type": error_type,
                                "error_fingerprint": error_fingerprint,
                                "consistency": consistency,
                            },
                        )
            finally:
                self.store.release_lease(job_lease, self.owner_id)
        self._last_tick_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return results

    def _run(self) -> None:
        self._ready = True
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as exc:
                error_type, error_fingerprint = monitor_error_identity(exc)
                self._last_error = f"monitor_worker_error:{error_type}:{error_fingerprint[:16]}"
            self._stop.wait(max(1, self.settings.monitor_worker_poll_seconds))

    def _record_monitor_exception(self, job_id: str, error_type: str) -> dict[str, Any] | None:
        with self.store.named_lock(f"job:{job_id}"):
            job = self.store.get_job(job_id)
            if not job:
                return None
            state = update_sync_consistency(
                job.monitor.get("monitor_exception_consistency"),
                [f"external_unavailable:{error_type}"],
                observed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                consecutive_threshold=self.settings.monitor_external_error_consecutive_threshold,
                grace_seconds=self.settings.monitor_external_error_grace_seconds,
            )
            job.monitor["monitor_exception_consistency"] = state
            self.store.upsert_job(job)
            return state

    def _clear_monitor_exception(self, job_id: str) -> None:
        with self.store.named_lock(f"job:{job_id}"):
            job = self.store.get_job(job_id)
            if not job or not isinstance(job.monitor.get("monitor_exception_consistency"), dict):
                return
            job.monitor["monitor_exception_consistency"] = update_sync_consistency(
                job.monitor.get("monitor_exception_consistency"),
                [],
                observed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                consecutive_threshold=self.settings.monitor_external_error_consecutive_threshold,
                grace_seconds=self.settings.monitor_external_error_grace_seconds,
            )
            self.store.upsert_job(job)


def monitor_error_identity(exc: Exception) -> tuple[str, str]:
    error_type = exc.__class__.__name__
    redacted = redact_text(str(exc))
    fingerprint = hashlib.sha256(f"{error_type}:{redacted}".encode("utf-8")).hexdigest()
    return error_type, fingerprint


def is_expected_external_monitor_error(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            WandBUnavailable,
            CommandFailed,
            subprocess.TimeoutExpired,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    )
