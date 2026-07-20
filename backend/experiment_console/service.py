from __future__ import annotations

import base64
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from .config import ServerProfile, Settings
from .models import (
    ACTIVE_JOB_STATUSES,
    TERMINAL_JOB_STATUSES,
    FileChunk,
    JobRecord,
    JobStatus,
    LogChunk,
    RemoteObservation,
    RunRequest,
    utc_now,
)
from .remote import RemoteError, RemoteExecutor
from .store import ConsoleStore


class UnknownProfile(KeyError):
    pass


class ResourceUnavailable(ValueError):
    pass


class ConsoleService:
    def __init__(
        self,
        settings: Settings,
        *,
        store: ConsoleStore | None = None,
        remote: RemoteExecutor | None = None,
        profiles: dict[str, ServerProfile] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or ConsoleStore(settings.sqlite_path)
        self.remote = remote or RemoteExecutor(settings)
        self.profiles = profiles or settings.load_profiles()
        self._job_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def profile(self, name: str) -> ServerProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise UnknownProfile(f"unknown server profile: {name}") from exc

    def _job_lock(self, job_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._job_locks.setdefault(job_id, threading.Lock())

    @staticmethod
    def _request_hash(request: RunRequest) -> str:
        payload = request.model_dump(mode="json", exclude={"request_id"})
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _validate_cwd(cwd: str, profile: ServerProfile) -> None:
        path = PurePosixPath(cwd)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("cwd must be an absolute remote path without '..'")
        allowed = [PurePosixPath(root) for root in profile.allowed_roots]
        if not any(path == root or root in path.parents for root in allowed):
            raise ValueError("cwd is outside the server profile allowed roots")

    def resources(self, profile_name: str | None = None) -> dict[str, Any]:
        names = [profile_name] if profile_name else sorted(self.profiles)
        result: dict[str, Any] = {"profiles": []}
        for name in names:
            profile = self.profile(name)
            locks = self.store.locked_gpus(name)
            try:
                remote = self.remote.resources(name, profile)
                gpus = []
                for item in remote.get("gpus") or []:
                    gpu = dict(item)
                    index = int(gpu["index"])
                    gpu["locked_by_job_id"] = locks.get(index)
                    gpu["available"] = bool(gpu.get("available")) and index not in locks
                    gpus.append(gpu)
                entry = {
                    "name": name,
                    "status": "ok",
                    "observed_at": remote.get("observed_at"),
                    "gpus": gpus,
                }
            except RemoteError as exc:
                entry = {
                    "name": name,
                    "status": "unavailable",
                    "error": str(exc),
                    "gpus": [],
                }
            result["profiles"].append(entry)
        return result

    def _verify_requested_gpus(self, request: RunRequest) -> None:
        if not request.gpu_indices:
            return
        snapshot = self.resources(request.profile)["profiles"][0]
        if snapshot["status"] != "ok":
            raise ResourceUnavailable(f"cannot verify GPU resources: {snapshot.get('error')}")
        by_index = {int(item["index"]): item for item in snapshot["gpus"]}
        unavailable = [
            index
            for index in request.gpu_indices
            if index not in by_index or not bool(by_index[index].get("available"))
        ]
        if unavailable:
            raise ResourceUnavailable(
                f"requested GPUs are unavailable on {request.profile}: "
                f"{','.join(map(str, unavailable))}"
            )

    def run(self, request: RunRequest) -> tuple[JobRecord, bool]:
        profile = self.profile(request.profile)
        self._validate_cwd(request.cwd, profile)
        existing = self.store.get_by_request_id(request.request_id)
        request_hash = self._request_hash(request)
        if existing:
            if existing.request_hash != request_hash:
                from .store import IdempotencyConflict

                raise IdempotencyConflict(
                    f"request_id already belongs to job {existing.job_id} with a different payload"
                )
            return existing, True
        locked = self.store.locked_gpus(request.profile)
        conflicts = [index for index in request.gpu_indices if index in locked]
        if conflicts:
            from .store import ResourceBusy

            raise ResourceBusy(request.profile, conflicts)
        self._verify_requested_gpus(request)
        job = JobRecord(
            job_id=f"job_{uuid4().hex}",
            request_id=request.request_id,
            request_hash=request_hash,
            task_id=request.task_id,
            profile=request.profile,
            cwd=request.cwd,
            argv=request.argv,
            env=request.env,
            gpu_indices=request.gpu_indices,
            total_runs=request.total_runs,
            name=request.name,
            status=JobStatus.starting,
            created_at=utc_now(),
        )
        job, replayed = self.store.create_job(job)
        if replayed:
            return job, True
        try:
            observation = self.remote.launch(job, profile)
            job = self._apply_observation(job, observation)
        except RemoteError as exc:
            job = self.store.update_observation(
                job.job_id,
                {"state": "unknown", "observed_at": utc_now()},
                error=str(exc),
            )
        return job, False

    def _apply_observation(
        self, job: JobRecord, observation: RemoteObservation
    ) -> JobRecord:
        payload = observation.model_dump(mode="json")
        if observation.state == "lost" and job.status == JobStatus.cancelling:
            payload["state"] = "cancelled"
        return self.store.update_observation(job.job_id, payload)

    def refresh(self, job_id: str) -> JobRecord:
        with self._job_lock(job_id):
            job = self.require_job(job_id)
            if job.status in TERMINAL_JOB_STATUSES:
                return job
            try:
                if job.status == JobStatus.cancelling:
                    observation = self.remote.cancel(job, self.profile(job.profile))
                else:
                    observation = self.remote.inspect(job, self.profile(job.profile))
            except RemoteError as exc:
                retained_state = (
                    "running"
                    if job.status in {JobStatus.running, JobStatus.cancelling}
                    else "unknown"
                )
                return self.store.update_observation(
                    job.job_id,
                    {"state": retained_state, "observed_at": utc_now()},
                    error=str(exc),
                )
            return self._apply_observation(job, observation)

    def status(self, job_id: str, *, refresh: bool = True) -> dict[str, Any]:
        job = self.require_job(job_id)
        if refresh and job.status in ACTIVE_JOB_STATUSES:
            job = self.refresh(job_id)
        return self.public_job(job)

    def logs(
        self,
        job_id: str,
        *,
        stream: str,
        offset: int,
        limit: int,
        tail: bool = False,
    ) -> LogChunk:
        job = self.require_job(job_id)
        bounded = min(max(1, limit), self.settings.max_log_chunk_bytes)
        return self.remote.logs(
            job,
            self.profile(job.profile),
            stream=stream,
            offset=max(0, offset),
            limit=bounded,
            tail=tail,
        )

    def fetch(
        self, job_id: str, *, path: str, offset: int, limit: int
    ) -> tuple[bytes, FileChunk]:
        job = self.require_job(job_id)
        bounded = min(max(1, limit), self.settings.max_fetch_chunk_bytes)
        chunk = self.remote.fetch(
            job,
            self.profile(job.profile),
            path=path,
            offset=max(0, offset),
            limit=bounded,
        )
        try:
            data = base64.b64decode(chunk.data_base64, validate=True)
        except ValueError as exc:
            raise RemoteError("remote file chunk is not valid base64") from exc
        if chunk.next_offset - chunk.offset != len(data):
            raise RemoteError("remote file chunk offsets do not match its payload")
        return data, chunk

    def cancel(self, job_id: str, *, reason: str | None = None) -> dict[str, Any]:
        with self._job_lock(job_id):
            job = self.store.request_cancel(job_id, reason)
            if job.status in TERMINAL_JOB_STATUSES:
                return self.public_job(job)
            try:
                observation = self.remote.cancel(job, self.profile(job.profile))
            except RemoteError as exc:
                job = self.store.update_observation(
                    job.job_id,
                    {"state": "running", "observed_at": utc_now()},
                    error=str(exc),
                )
                return self.public_job(job)
            job = self._apply_observation(job, observation)
            return self.public_job(job)

    def require_job(self, job_id: str) -> JobRecord:
        job = self.store.get_job(job_id)
        if not job:
            raise KeyError(f"job not found: {job_id}")
        return job

    @staticmethod
    def _parse_time(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def public_job(self, job: JobRecord) -> dict[str, Any]:
        data = job.model_dump(mode="json")
        start_value = job.created_at
        end = self._parse_time(job.finished_at) if job.finished_at else datetime.now(timezone.utc)
        elapsed = max(0, int((end - self._parse_time(start_value)).total_seconds()))
        eta_seconds = None
        if job.total_runs and job.completed_runs > 0 and job.completed_runs < job.total_runs:
            eta_seconds = int(
                (elapsed / job.completed_runs) * (job.total_runs - job.completed_runs)
            )
        data["elapsed_seconds"] = elapsed
        data["eta_seconds"] = eta_seconds
        data["eta_basis"] = {
            "created_at": start_value,
            "completed_runs": job.completed_runs,
            "total_runs": job.total_runs,
            "formula": (
                "elapsed_seconds / completed_runs * remaining_runs"
                if eta_seconds is not None
                else None
            ),
        }
        data.pop("env", None)
        data.pop("request_hash", None)
        return data

    def monitor_once(self) -> dict[str, int]:
        checked = terminal = errors = 0
        for job in self.store.active_jobs():
            checked += 1
            try:
                refreshed = self.refresh(job.job_id)
                if refreshed.status in TERMINAL_JOB_STATUSES:
                    terminal += 1
            except Exception:
                errors += 1
        return {"checked": checked, "terminal": terminal, "errors": errors}
