from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .config import Settings
from .models import (
    AuditEvent,
    ConfirmRequest,
    ExecuteResponse,
    IntentPreviewRequest,
    IntentRecord,
    IntentStatus,
    IntentType,
    JobRecord,
    JobStatus,
    LaunchSweepPayload,
    RecoverAgentsPayload,
    StatusQueryPayload,
    StopJobPayload,
    ValidateConfigPayload,
    assert_local_config_path,
    new_id,
    parse_payload,
)
from .planner import build_plan, confirmation_phrase
from .redaction import redact_value
from .ssh import SSHExecutor
from .store import ConsoleStore
from .validation import validate_experiment_config
from .wandb_client import WandBClient, WandBUnavailable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConsoleService:
    def __init__(
        self,
        settings: Settings | None = None,
        store: ConsoleStore | None = None,
        wandb: WandBClient | None = None,
        ssh: SSHExecutor | None = None,
    ):
        self.settings = settings or Settings()
        self.store = store or ConsoleStore(self.settings.sqlite_path, self.settings.audit_path)
        self.wandb = wandb or WandBClient(self.settings)
        self.ssh = ssh or SSHExecutor(self.settings)

    def preview(self, request: IntentPreviewRequest) -> tuple[IntentRecord, bool]:
        parse_payload(request.intent, request.payload)
        intent_id = new_id("intent", request.intent.value)
        plan = build_plan(request.intent, request.payload, self.settings)
        record = IntentRecord(
            intent_id=intent_id,
            intent=request.intent,
            payload=request.payload,
            requested_by=request.requested_by,
            idempotency_key=request.idempotency_key,
            confirmation_phrase=confirmation_phrase(intent_id, request.intent),
            plan=plan,
        )
        stored, replay = self.store.save_intent_if_absent(record)
        self.store.write_audit(AuditEvent(
            event_type="intent_replayed" if replay else "intent_previewed",
            intent_id=stored.intent_id,
            message="Returned existing intent." if replay else "Intent preview created.",
            detail={"intent": stored.intent.value, "plan": stored.plan.model_dump(mode="json")},
        ))
        return stored, replay

    def confirm(self, intent_id: str, request: ConfirmRequest) -> IntentRecord:
        intent = self._require_intent(intent_id)
        if intent.status not in {IntentStatus.previewed, IntentStatus.confirmed}:
            raise ValueError(f"intent cannot be confirmed from status {intent.status.value}")
        if request.confirmation_phrase != intent.confirmation_phrase:
            self.store.write_audit(AuditEvent(
                event_type="intent_confirm_rejected",
                intent_id=intent.intent_id,
                message="Confirmation phrase mismatch.",
                detail={"provided": request.confirmation_phrase},
            ))
            raise ValueError("confirmation phrase mismatch")
        intent.status = IntentStatus.confirmed
        intent.confirmed_at = utc_now()
        self.store.update_intent(intent)
        self.store.write_audit(AuditEvent(
            event_type="intent_confirmed",
            intent_id=intent.intent_id,
            message="Intent confirmed by phrase.",
            detail={"intent": intent.intent.value},
        ))
        return intent

    def execute(self, intent_id: str) -> ExecuteResponse:
        intent = self._require_intent(intent_id)
        if intent.plan.risk_level in {"remote_side_effect", "destructive"} and intent.status != IntentStatus.confirmed:
            raise ValueError("real side-effect intent must be confirmed before execution")
        if intent.status in {IntentStatus.executing, IntentStatus.succeeded}:
            raise ValueError(f"intent cannot be executed from status {intent.status.value}")
        intent.status = IntentStatus.executing
        self.store.update_intent(intent)
        self.store.write_audit(AuditEvent(
            event_type="intent_execute_started",
            intent_id=intent.intent_id,
            message="Intent execution started.",
            detail={"intent": intent.intent.value},
        ))
        try:
            result, job = self._execute_intent(intent)
            intent.status = IntentStatus.succeeded
            intent.executed_at = utc_now()
            intent.result = redact_value(result)
            self.store.update_intent(intent)
            self.store.write_audit(AuditEvent(
                event_type="intent_execute_succeeded",
                intent_id=intent.intent_id,
                job_id=job.job_id if job else None,
                message="Intent execution succeeded.",
                detail={"result": result},
            ))
            return ExecuteResponse(intent=intent, job=job)
        except Exception as exc:
            intent.status = IntentStatus.failed
            intent.executed_at = utc_now()
            intent.result = {"error": str(exc)}
            self.store.update_intent(intent)
            self.store.write_audit(AuditEvent(
                event_type="intent_execute_failed",
                intent_id=intent.intent_id,
                message="Intent execution failed.",
                detail={"error": str(exc)},
            ))
            raise

    def _execute_intent(self, intent: IntentRecord) -> tuple[dict[str, Any], JobRecord | None]:
        payload = parse_payload(intent.intent, intent.payload)
        if isinstance(payload, ValidateConfigPayload):
            path = assert_local_config_path(payload.config_path)
            return validate_experiment_config(path, payload.profile), None
        if isinstance(payload, LaunchSweepPayload):
            return self._launch_sweep(payload)
        if isinstance(payload, StatusQueryPayload):
            return self._status(payload), None
        if isinstance(payload, StopJobPayload):
            return self._stop(payload)
        if isinstance(payload, RecoverAgentsPayload):
            return self._recover(payload)
        raise ValueError(f"unsupported intent: {intent.intent}")

    def _launch_sweep(self, payload: LaunchSweepPayload) -> tuple[dict[str, Any], JobRecord]:
        path = assert_local_config_path(payload.config_path)
        validation = validate_experiment_config(path, payload.profile)
        entity = payload.entity or self.settings.default_entity
        project = payload.project or self.settings.default_project
        job = JobRecord(
            job_id=new_id("job", payload.job_name),
            name=payload.job_name,
            status=JobStatus.validating,
            entity=entity,
            project=project,
            config_path=str(path),
            remote_host=payload.remote_host,
            remote_cwd=payload.remote_cwd,
            conda_env=payload.conda_env,
            monitor={"validation": validation},
        )
        self.store.upsert_job(job)
        sweep = self.wandb.create_sweep(path, entity=entity, project=project)
        job.sweep_id = sweep["sweep_id"]
        gpu_probe = self.ssh.probe_gpus(payload.remote_host)
        eligible = [gpu for gpu in gpu_probe["gpus"] if gpu["eligible"]]
        if payload.max_agents is not None:
            eligible = eligible[:payload.max_agents]
        launches = []
        sweep_path = f"{entity}/{project}/{job.sweep_id}"
        for gpu in eligible:
            launches.append(self.ssh.launch_agent(
                host=payload.remote_host,
                remote_cwd=payload.remote_cwd,
                sweep_path=sweep_path,
                gpu_index=gpu["index"],
                conda_env=payload.conda_env,
                conda_sh=payload.conda_sh,
            ))
        job.agent_pids = [item["pid"] for item in launches if item.get("pid")]
        job.status = JobStatus.running if launches else JobStatus.attention
        job.monitor.update({"gpu_probe": gpu_probe, "agent_launches": launches, "sweep_path": sweep_path})
        self.store.upsert_job(job)
        return {"validation": validation, "sweep": sweep, "gpu_probe": gpu_probe, "agent_launches": launches}, job

    def _status(self, payload: StatusQueryPayload) -> dict[str, Any]:
        job = self.store.get_job(payload.job_id) if payload.job_id else None
        entity = (job.entity if job else None) or self.settings.default_entity
        project = (job.project if job else None) or self.settings.default_project
        sweep_id = payload.sweep_id or (job.sweep_id if job else None)
        sweep = None
        degraded = None
        if sweep_id:
            try:
                sweep = self.wandb.get_sweep_state(entity, project, sweep_id)
            except Exception as exc:
                degraded = str(exc)
        if job and sweep:
            next_status = map_wandb_state_to_job_status(sweep.get("state"))
            try:
                job = self.store.update_job_status(job.job_id, next_status, {"last_wandb_status": sweep})
            except Exception:
                job.monitor["last_wandb_status"] = sweep
                self.store.upsert_job(job)
        return {"job": job.model_dump(mode="json") if job else None, "sweep": sweep, "degraded": degraded, "generated_at": utc_now()}

    def _stop(self, payload: StopJobPayload) -> tuple[dict[str, Any], JobRecord]:
        job = self._require_job(payload.job_id)
        if not job.sweep_id or not job.entity or not job.project or not job.remote_host:
            raise ValueError("job lacks sweep/remote metadata required for stop")
        result = {"cancel_wandb_requested": payload.cancel_wandb, "cancel_wandb_implemented": False}
        if payload.kill_agents:
            result["stop_agents"] = self.ssh.stop_agents(host=job.remote_host, sweep_path=f"{job.entity}/{job.project}/{job.sweep_id}")
        job = self.store.update_job_status(job.job_id, JobStatus.cancelled, {"stop_result": result})
        return result, job

    def _recover(self, payload: RecoverAgentsPayload) -> tuple[dict[str, Any], JobRecord]:
        job = self._require_job(payload.job_id)
        if not job.sweep_id or not job.entity or not job.project or not job.remote_host or not job.remote_cwd:
            raise ValueError("job lacks sweep/remote metadata required for recover")
        gpu_probe = self.ssh.probe_gpus(job.remote_host)
        eligible = [gpu for gpu in gpu_probe["gpus"] if gpu["eligible"]]
        if payload.max_agents is not None:
            eligible = eligible[:payload.max_agents]
        sweep_path = f"{job.entity}/{job.project}/{job.sweep_id}"
        launches = [
            self.ssh.launch_agent(
                host=job.remote_host,
                remote_cwd=job.remote_cwd,
                sweep_path=sweep_path,
                gpu_index=gpu["index"],
                conda_env=job.conda_env,
                conda_sh="/opt/anaconda3/etc/profile.d/conda.sh",
            )
            for gpu in eligible
        ]
        job.agent_pids = sorted(set(job.agent_pids + [item["pid"] for item in launches if item.get("pid")]))
        job.status = JobStatus.running if launches else JobStatus.attention
        job.monitor.update({"recover_gpu_probe": gpu_probe, "recover_agent_launches": launches})
        self.store.upsert_job(job)
        return {"gpu_probe": gpu_probe, "agent_launches": launches, "created_new_sweep": False}, job

    def overview(self) -> dict[str, Any]:
        jobs = self.store.list_jobs()
        sweeps = []
        degraded = None
        try:
            sweeps = self.wandb.discover_sweeps(self.settings.default_entity, self.settings.default_project)
            self.settings.sweeps_cache_path.write_text(json.dumps(sweeps, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            degraded = str(exc)
            if self.settings.sweeps_cache_path.exists():
                sweeps = json.loads(self.settings.sweeps_cache_path.read_text(encoding="utf-8"))
        return {
            "status": "degraded" if degraded else "ok",
            "degraded": degraded,
            "job_counts": count_jobs(jobs),
            "jobs": [job.model_dump(mode="json") for job in jobs[:20]],
            "sweeps": sweeps[:50],
            "active_sweeps": sum(1 for sweep in sweeps if sweep.get("state") == "RUNNING"),
            "finished_sweeps": sum(1 for sweep in sweeps if sweep.get("state") == "FINISHED"),
            "total_runs": sum(int(sweep.get("runCount") or 0) for sweep in sweeps),
            "generated_at": utc_now(),
        }

    def list_jobs(self) -> list[JobRecord]:
        return self.store.list_jobs()

    def events(self, limit: int = 100) -> list[AuditEvent]:
        return self.store.read_audit(limit)

    def probe_gpus(self, host: str) -> dict[str, Any]:
        return self.ssh.probe_gpus(host)

    def discover_sweeps(self, entity: str | None = None, project: str | None = None, days: int = 7) -> list[dict[str, Any]]:
        return self.wandb.discover_sweeps(entity or self.settings.default_entity, project, days)

    def _require_intent(self, intent_id: str) -> IntentRecord:
        intent = self.store.get_intent(intent_id)
        if not intent:
            raise KeyError(f"intent not found: {intent_id}")
        return intent

    def _require_job(self, job_id: str) -> JobRecord:
        job = self.store.get_job(job_id)
        if not job:
            raise KeyError(f"job not found: {job_id}")
        return job


def map_wandb_state_to_job_status(state: str | None) -> JobStatus:
    normalized = (state or "").lower()
    if normalized == "finished":
        return JobStatus.finished
    if normalized in {"failed", "crashed", "killed"}:
        return JobStatus.failed
    if normalized in {"running", "pending"}:
        return JobStatus.running
    return JobStatus.unknown


def count_jobs(jobs: list[JobRecord]) -> dict[str, int]:
    counts = {status.value: 0 for status in JobStatus}
    for job in jobs:
        counts[job.status.value] = counts.get(job.status.value, 0) + 1
    return counts
