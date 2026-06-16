from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .command import CommandFailed
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
    OperationStatus,
    TERMINAL_JOB_STATUSES,
    AuthCheckPayload,
    CancelSweepPayload,
    LaunchSweepPayload,
    PreflightPayload,
    PullResultsPayload,
    RepairWatchdogPayload,
    RecoverAgentsPayload,
    RegisterExistingSweepPayload,
    RunnerCommandResponse,
    ScheduleMonitorPayload,
    StatusQueryPayload,
    StopJobPayload,
    UnscheduleMonitorPayload,
    ValidateConfigPayload,
    WatchdogOncePayload,
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


MUTATING_INTENTS = {
    IntentType.launch_sweep,
    IntentType.register_existing_sweep,
    IntentType.stop_job,
    IntentType.cancel_sweep,
    IntentType.recover_agents,
    IntentType.repair_watchdog,
    IntentType.schedule_monitor,
    IntentType.unschedule_monitor,
    IntentType.watchdog_once,
    IntentType.pull_results,
}


REPLAY_SAFE_INTENTS = {
    IntentType.launch_sweep,
    IntentType.register_existing_sweep,
    IntentType.stop_job,
    IntentType.cancel_sweep,
    IntentType.recover_agents,
    IntentType.repair_watchdog,
    IntentType.schedule_monitor,
    IntentType.unschedule_monitor,
}


OPEN_OPERATION_STATUSES = {OperationStatus.accepted.value, OperationStatus.executing.value}


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

    def _operation_identity(self, intent: IntentType, payload: dict[str, Any]) -> tuple[str, str]:
        raw_key = payload.get("idempotency_key")
        if not raw_key:
            payload_for_key = {key: value for key, value in payload.items() if key != "idempotency_key"}
            serialized = json.dumps(
                {"intent": intent.value, "payload": payload_for_key},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            if intent in MUTATING_INTENTS and intent not in REPLAY_SAFE_INTENTS:
                nonce = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                raw_key = f"auto-volatile-{digest}-{nonce}"
            else:
                raw_key = f"auto-{digest}"
        idem_key = str(raw_key)
        op_seed = hashlib.sha256(f"{intent.value}:{idem_key}".encode("utf-8")).hexdigest()[:12]
        return f"op_{intent.value}_{op_seed}", idem_key

    def _current_operation(self, job: JobRecord | None) -> dict[str, Any] | None:
        if not job:
            return None
        if job.operation_log:
            return job.operation_log[-1]
        operation = job.monitor.get("operation")
        return operation if isinstance(operation, dict) else None

    def _operation_history(self, job: JobRecord | None, limit: int = 10) -> list[dict[str, Any]]:
        if not job:
            return []
        return list(job.operation_log or [])[-limit:]

    def _record_operation(
        self,
        job: JobRecord,
        *,
        operation_id: str,
        intent: IntentType,
        requested_by: str,
        idempotency_key: str,
        stage: str,
        classification: str,
        status: str,
        result: dict[str, Any] | None = None,
        retry_after_seconds: int | None = None,
    ) -> JobRecord:
        stamp = utc_now()
        previous = None
        for entry in job.operation_log or []:
            if entry.get("operation_id") == operation_id:
                previous = entry
                break
        stamp = utc_now()
        transitions = list(previous.get("transitions") or []) if isinstance(previous, dict) else []
        transitions.append({
            "stage": stage,
            "classification": classification,
            "status": status,
            "result": redact_value(result) if result is not None else None,
            "updated_at": stamp,
        })
        operation = {
            "operation_id": operation_id,
            "intent": intent.value,
            "requested_by": requested_by,
            "idempotency_key": idempotency_key,
            "stage": stage,
            "classification": classification,
            "status": status,
            "result": redact_value(result) if result is not None else None,
            "retry_after_seconds": retry_after_seconds,
            "created_at": previous.get("created_at") if isinstance(previous, dict) and previous.get("created_at") else stamp,
            "updated_at": stamp,
            "transitions": transitions[-20:],
        }
        log = [entry for entry in (job.operation_log or []) if entry.get("operation_id") != operation_id]
        log.append(operation)
        job.operation_log = log[-20:]
        job.operation_id = operation_id
        job.idempotency_key = idempotency_key
        job.monitor["operation"] = operation
        job.monitor["operation_log"] = job.operation_log[-10:]
        job.monitor["stage"] = stage
        job.monitor["classification"] = classification
        return self.store.upsert_job(job)

    def _find_operation_replay(self, intent: IntentType, idempotency_key: str) -> tuple[JobRecord, dict[str, Any]] | None:
        for job in self.store.list_jobs():
            for op in reversed(job.operation_log or []):
                if op.get("intent") != intent.value:
                    continue
                if op.get("idempotency_key") != idempotency_key:
                    continue
                return job, op
            op = self._current_operation(job)
            if op and op.get("intent") == intent.value and op.get("idempotency_key") == idempotency_key:
                return job, op
        return None

    def _mark_operation_failed(
        self,
        intent: IntentType,
        idempotency_key: str,
        operation_id: str,
        requested_by: str,
        exc: Exception,
    ) -> None:
        replay = self._find_operation_replay(intent, idempotency_key)
        if not replay:
            return
        job, op = replay
        self._record_operation(
            job,
            operation_id=str(op.get("operation_id") or operation_id),
            intent=intent,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="failed",
            classification="control_plane_error",
            status=OperationStatus.failed.value,
            result={
                "stage": "failed",
                "classification": "control_plane_error",
                "error": compact_error(exc),
                "next_actions": ["使用 status 查询 ledger；修复环境后用相同 idempotency_key 重试会回放该 operation。"],
            },
        )

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
        if isinstance(payload, RegisterExistingSweepPayload):
            return self._register_existing_sweep(payload)
        if isinstance(payload, StatusQueryPayload):
            return self._status(payload), None
        if isinstance(payload, StopJobPayload):
            return self._stop(payload)
        if isinstance(payload, CancelSweepPayload):
            return self._cancel_sweep(payload), None
        if isinstance(payload, RecoverAgentsPayload):
            return self._recover(payload)
        if isinstance(payload, RepairWatchdogPayload):
            return self._repair_watchdog(payload)
        if isinstance(payload, ScheduleMonitorPayload):
            return self._schedule_monitor(payload)
        if isinstance(payload, UnscheduleMonitorPayload):
            return self._unschedule_monitor(payload)
        if isinstance(payload, WatchdogOncePayload):
            return self._watchdog_once(payload), None
        if isinstance(payload, AuthCheckPayload):
            return self._auth_check(payload), None
        if isinstance(payload, PreflightPayload):
            return self._preflight(payload), None
        if isinstance(payload, PullResultsPayload):
            return self._pull_results(payload), None
        raise ValueError(f"unsupported intent: {intent.intent}")

    def runner_command(
        self,
        intent: IntentType,
        payload: dict[str, Any],
        *,
        requested_by: str = "experiment-runner",
    ) -> RunnerCommandResponse:
        parsed = parse_payload(intent, payload)
        parsed_payload = parsed.model_dump(mode="json")
        operation_id, idempotency_key = self._operation_identity(intent, parsed_payload)
        explicit_idempotency = bool(parsed_payload.get("idempotency_key"))
        if intent in MUTATING_INTENTS and (intent in REPLAY_SAFE_INTENTS or explicit_idempotency):
            replay_job = self._find_operation_replay(intent, idempotency_key)
            if replay_job:
                replay_job, op = replay_job
                replay_result = op.get("result") if isinstance(op.get("result"), dict) else {}
                return RunnerCommandResponse(
                    command=intent,
                    requested_by=requested_by,
                    result=redact_value(replay_result or {}),
                    operation_id=str(op.get("operation_id") or operation_id),
                    idempotency_key=idempotency_key,
                    job_id=replay_job.job_id,
                    job=replay_job,
                    stage=str(op.get("stage") or replay_result.get("stage") or "done"),
                    classification=str(op.get("classification") or replay_result.get("classification") or "ok"),
                    next_actions=list(replay_result.get("next_actions") or []),
                    provenance={
                        "source": "experiment_console",
                        "generated_at": utc_now(),
                        "replayed": True,
                        "operation_id": str(op.get("operation_id") or operation_id),
                    },
                    retry_after_seconds=op.get("retry_after_seconds"),
                    accepted=str(op.get("status") or "") in OPEN_OPERATION_STATUSES,
                )
        stage = "done"
        classification = "ok"
        next_actions: list[str] = []
        job = None
        result: dict[str, Any] = {}
        try:
            result, job = self._execute_intent_from_payload(intent, payload, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key)
            stage = str(result.get("stage") or stage) if isinstance(result, dict) else stage
            classification = str(result.get("classification") or classification) if isinstance(result, dict) else classification
            next_actions = list(result.get("next_actions") or []) if isinstance(result, dict) else []
        except Exception as exc:
            self._mark_operation_failed(intent, idempotency_key, operation_id, requested_by, exc)
            self.store.write_audit(AuditEvent(
                event_type="runner_command_failed",
                message=f"Runner command failed: {intent.value}.",
                detail={"intent": intent.value, "requested_by": requested_by, "payload": payload, "error": str(exc)},
            ))
            raise
        self.store.write_audit(AuditEvent(
            event_type="runner_command_executed",
            job_id=job.job_id if job else None,
            message=f"Runner command executed: {intent.value}.",
            detail={"intent": intent.value, "requested_by": requested_by, "payload": payload, "result": result},
        ))
        operation = self._current_operation(job) if job else None
        return RunnerCommandResponse(
            command=intent,
            requested_by=requested_by,
            result=redact_value(result),
            operation_id=str(operation.get("operation_id") if operation else operation_id),
            idempotency_key=idempotency_key,
            job_id=job.job_id if job else None,
            job=job,
            stage=stage,
            classification=classification,
            next_actions=next_actions,
            provenance={"source": "experiment_console", "generated_at": utc_now(), "operation_id": operation.get("operation_id") if operation else operation_id},
            retry_after_seconds=operation.get("retry_after_seconds") if operation else None,
            accepted=str(operation.get("status") if operation else "") in OPEN_OPERATION_STATUSES,
        )

    def _execute_intent_from_payload(self, intent: IntentType, payload: dict[str, Any], *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> tuple[dict[str, Any], JobRecord | None]:
        parsed = parse_payload(intent, payload)
        if isinstance(parsed, ValidateConfigPayload):
            path = assert_local_config_path(parsed.config_path)
            return validate_experiment_config(path, parsed.profile), None
        if isinstance(parsed, LaunchSweepPayload):
            return self._launch_sweep(parsed, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key)
        if isinstance(parsed, RegisterExistingSweepPayload):
            return self._register_existing_sweep(parsed, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key)
        if isinstance(parsed, StatusQueryPayload):
            return self._status(parsed), None
        if isinstance(parsed, StopJobPayload):
            return self._stop(parsed, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key)
        if isinstance(parsed, CancelSweepPayload):
            return self._cancel_sweep(parsed, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key), None
        if isinstance(parsed, RecoverAgentsPayload):
            return self._recover(parsed, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key)
        if isinstance(parsed, RepairWatchdogPayload):
            return self._repair_watchdog(parsed, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key)
        if isinstance(parsed, ScheduleMonitorPayload):
            return self._schedule_monitor(parsed, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key)
        if isinstance(parsed, UnscheduleMonitorPayload):
            return self._unschedule_monitor(parsed, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key)
        if isinstance(parsed, WatchdogOncePayload):
            return self._watchdog_once(parsed, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key), None
        if isinstance(parsed, AuthCheckPayload):
            return self._auth_check(parsed), None
        if isinstance(parsed, PreflightPayload):
            return self._preflight(parsed), None
        if isinstance(parsed, PullResultsPayload):
            return self._pull_results(parsed, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key), None
        raise ValueError(f"unsupported intent: {intent}")

    def _launch_sweep(self, payload: LaunchSweepPayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> tuple[dict[str, Any], JobRecord]:
        entity = payload.entity or self.settings.default_entity
        project = payload.project or self.settings.default_project
        operation_id = operation_id or self._operation_identity(IntentType.launch_sweep, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.launch_sweep, payload.model_dump(mode="json"))[1]
        path = Path(payload.config_path).expanduser()
        conda_env = payload.conda_env or self.settings.default_conda_env
        existing = self.store.find_job_by_launch_identity(
            name=payload.job_name,
            config_path=str(path),
            remote_host=payload.remote_host,
            remote_cwd=payload.remote_cwd,
        )
        if existing:
            if not self._current_operation(existing):
                self._record_operation(
                    existing,
                    operation_id=operation_id,
                    intent=IntentType.launch_sweep,
                    requested_by=requested_by,
                    idempotency_key=idempotency_key,
                    stage="idempotent_replay",
                    classification="existing_sweep_reused",
                    status=OperationStatus.succeeded.value,
                    result={
                        "stage": "idempotent_replay",
                        "classification": "existing_sweep_reused",
                        "created_new_sweep": False,
                        "next_actions": ["使用 recover-agents 为既有 sweep 补 agent，而不是重新创建 sweep。"],
                    },
                )
            return {
                "stage": "idempotent_replay",
                "classification": "existing_sweep_reused",
                "created_new_sweep": False,
                "job": existing.model_dump(mode="json"),
                "next_actions": ["使用 recover-agents 为既有 sweep 补 agent，而不是重新创建 sweep。"],
            }, existing

        job = JobRecord(
            job_id=new_id("job", payload.job_name),
            name=payload.job_name,
            status=JobStatus.validating,
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            entity=entity,
            project=project,
            config_path=str(path),
            remote_host=payload.remote_host,
            remote_cwd=payload.remote_cwd,
            conda_env=conda_env,
            monitor={"stage": "validating"},
        )
        self.store.upsert_job(job)
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.launch_sweep,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="accepted",
            classification="accepted",
            status=OperationStatus.accepted.value,
            result={
                "stage": "accepted",
                "classification": "accepted",
                "created_new_sweep": False,
                "next_actions": ["Console 正在执行 launch-sweep，稍后再用 status 查询进度。"],
            },
            retry_after_seconds=2,
        )
        try:
            validation = validate_experiment_config(assert_local_config_path(payload.config_path), payload.profile)
        except Exception as exc:
            job.status = JobStatus.failed
            job.monitor.update({
                "stage": "validation_failed",
                "validation_error": compact_error(exc),
            })
            self.store.upsert_job(job)
            raise
        job.monitor.update({"validation": validation, "stage": "validating"})
        self.store.upsert_job(job)
        preflight = self.ssh.preflight(
            host=payload.remote_host,
            remote_cwd=payload.remote_cwd,
            conda_env=conda_env,
            conda_sh=payload.conda_sh,
            config_path=None,
        )
        job.monitor.update({"preflight": preflight, "stage": "creating_sweep"})
        self.store.upsert_job(job)
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.launch_sweep,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="preflight",
            classification="accepted",
            status=OperationStatus.executing.value,
            result={
                "stage": "preflight",
                "classification": "accepted",
                "preflight": preflight,
                "next_actions": ["Console 正在创建 sweep。"],
            },
            retry_after_seconds=2,
        )
        sweep = self._create_sweep(path, entity=entity, project=project, payload=payload)
        job.sweep_id = sweep["sweep_id"]
        existing_sweep_job = self.store.find_job_by_sweep(entity, project, job.sweep_id)
        if existing_sweep_job and existing_sweep_job.job_id != job.job_id:
            self._record_operation(
                existing_sweep_job,
                operation_id=operation_id,
                intent=IntentType.launch_sweep,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="idempotent_replay",
                classification="existing_sweep_reused",
                status=OperationStatus.succeeded.value,
                result={
                    "stage": "idempotent_replay",
                    "classification": "existing_sweep_reused",
                    "created_new_sweep": False,
                    "sweep": sweep,
                    "next_actions": ["既有 job 已登记该 sweep；如需 agent 请 recover-agents。"],
                },
            )
            return {
                "stage": "idempotent_replay",
                "classification": "existing_sweep_reused",
                "created_new_sweep": False,
                "sweep": sweep,
                "next_actions": ["既有 job 已登记该 sweep；如需 agent 请 recover-agents。"],
            }, existing_sweep_job
        gpu_probe = self.ssh.probe_gpus(payload.remote_host)
        eligible = [gpu for gpu in gpu_probe["gpus"] if gpu["eligible"]]
        if payload.max_agents is not None:
            eligible = eligible[:payload.max_agents]
        launches = []
        sweep_path = f"{entity}/{project}/{job.sweep_id}"
        wandb_api_key = self._wandb_api_key()
        auth = self.ssh.auth_check(
            host=payload.remote_host,
            remote_cwd=payload.remote_cwd,
            sweep_path=sweep_path,
            wandb_api_key=wandb_api_key,
        )
        for gpu in eligible:
            launches.append(self.ssh.launch_agent(
                host=payload.remote_host,
                remote_cwd=payload.remote_cwd,
                sweep_path=sweep_path,
                gpu_index=gpu["index"],
                conda_env=conda_env,
                conda_sh=payload.conda_sh,
                wandb_api_key=wandb_api_key,
            ))
        job.agent_pids = [item["pid"] for item in launches if item.get("pid")]
        job.status = JobStatus.running if launches else JobStatus.attention
        classification = classify_launch(auth, launches)
        job.monitor.update({"gpu_probe": gpu_probe, "auth": auth, "agent_launches": launches, "sweep_path": sweep_path, "stage": "done", "classification": classification})
        self.store.upsert_job(job)
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.launch_sweep,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="done",
            classification=classification,
            status=OperationStatus.succeeded.value if classification == "agents_running" else OperationStatus.failed.value if classification in {"wandb_auth_missing", "agents_failed_wandb_auth"} else OperationStatus.executing.value,
            result={
                "stage": "done",
                "classification": classification,
                "created_new_sweep": True,
                "validation": validation,
                "preflight": preflight,
                "auth": auth,
                "sweep": sweep,
                "gpu_probe": gpu_probe,
                "agent_launches": launches,
                "next_actions": next_actions_for_classification(classification),
            },
            retry_after_seconds=2,
        )
        return {
            "stage": "done",
            "classification": classification,
            "created_new_sweep": True,
            "validation": validation,
            "preflight": preflight,
            "auth": auth,
            "sweep": sweep,
            "gpu_probe": gpu_probe,
            "agent_launches": launches,
            "next_actions": next_actions_for_classification(classification),
        }, job

    def _create_sweep(self, path: Path, *, entity: str, project: str, payload: LaunchSweepPayload) -> dict[str, Any]:
        try:
            return self.wandb.create_sweep(path, entity=entity, project=project)
        except FileNotFoundError:
            pass
        except CommandFailed as exc:
            argv0 = exc.result.argv[0] if exc.result.argv else ""
            if argv0 != "wandb":
                raise
        return self.ssh.create_sweep(
            host=payload.remote_host,
            remote_cwd=payload.remote_cwd,
            config_path=path,
            entity=entity,
            project=project,
            wandb_api_key=self._wandb_api_key(),
        )

    def _register_existing_sweep(self, payload: RegisterExistingSweepPayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> tuple[dict[str, Any], JobRecord]:
        entity = payload.entity or self.settings.default_entity
        project = payload.project or self.settings.default_project
        operation_id = operation_id or self._operation_identity(IntentType.register_existing_sweep, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.register_existing_sweep, payload.model_dump(mode="json"))[1]
        conda_env = payload.conda_env or self.settings.default_conda_env
        existing = self.store.find_job_by_sweep(entity, project, payload.sweep_id)
        if existing:
            if not self._current_operation(existing):
                self._record_operation(
                    existing,
                    operation_id=operation_id,
                    intent=IntentType.register_existing_sweep,
                    requested_by=requested_by,
                    idempotency_key=idempotency_key,
                    stage="idempotent_replay",
                    classification="existing_job_reused",
                    status=OperationStatus.succeeded.value,
                    result={
                        "stage": "idempotent_replay",
                        "classification": "existing_job_reused",
                        "created_new_sweep": False,
                    },
                )
            return {
                "stage": "idempotent_replay",
                "classification": "existing_job_reused",
                "created_new_sweep": False,
                "job": existing.model_dump(mode="json"),
            }, existing
        job = JobRecord(
            job_id=new_id("job", payload.job_name),
            name=payload.job_name,
            status=JobStatus.attention,
            entity=entity,
            project=project,
            sweep_id=payload.sweep_id,
            config_path=payload.config_path,
            remote_host=payload.remote_host,
            remote_cwd=payload.remote_cwd,
            conda_env=conda_env,
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            monitor={
                "stage": "registered_existing_sweep",
                "expected_total": payload.expected_total,
                "created_new_sweep": False,
            },
        )
        self.store.upsert_job(job)
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.register_existing_sweep,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="accepted",
            classification="accepted",
            status=OperationStatus.accepted.value,
            result={
                "stage": "accepted",
                "classification": "accepted",
                "created_new_sweep": False,
            },
            retry_after_seconds=2,
        )
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.register_existing_sweep,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="registered",
            classification="existing_sweep_registered",
            status=OperationStatus.succeeded.value,
            result={"stage": "registered", "classification": "existing_sweep_registered", "created_new_sweep": False},
        )
        return {"stage": "registered", "classification": "existing_sweep_registered", "created_new_sweep": False}, job

    def _status(self, payload: StatusQueryPayload) -> dict[str, Any]:
        job = self.store.get_job(payload.job_id) if payload.job_id else None
        missing_requested_job = bool(payload.job_id and not job)
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
            cached_status = compact_sweep_status(sweep, include_runs=True)
            try:
                job = self.store.update_job_status(job.job_id, next_status, {"last_wandb_status": cached_status})
            except Exception:
                job.monitor["last_wandb_status"] = cached_status
                self.store.upsert_job(job)
        current_operation = self._current_operation(job)
        operation_history = self._operation_history(job)
        agent_launches = []
        if job:
            agent_launches = list(job.monitor.get("agent_launches") or job.monitor.get("recover_agent_launches") or [])
        if not agent_launches and job and job.agent_pids:
            agent_launches = [{"pid": pid, "state": "unknown"} for pid in job.agent_pids]
        agent_health = "unknown"
        if job and job.status == JobStatus.running:
            agent_health = "running" if agent_launches else "missing"
        elif job and job.status in TERMINAL_JOB_STATUSES:
            agent_health = "terminal"
        result_pull = job.monitor.get("last_result_pull") if job else None
        result_readiness = "unknown"
        if isinstance(result_pull, dict):
            if result_pull.get("valid_results", 0):
                result_readiness = "ready"
            elif result_pull.get("partial"):
                result_readiness = "partial"
            else:
                result_readiness = "empty"
        classification = "ok"
        next_actions: list[str] = []
        if missing_requested_job:
            classification = "job_not_found"
            next_actions = [
                "使用 register-existing-sweep 将已有 W&B sweep 登记到 Console ledger，或确认 job_id 是否来自当前 state_dir。",
            ]
        elif degraded:
            classification = "degraded"
            next_actions = ["稍后重试 status；如持续 degraded，检查 W&B/API 网络与本地 WANDB_API_KEY。"]
        return {
            "stage": "done",
            "classification": classification,
            "job": compact_job_record(job) if job else None,
            "sweep": compact_sweep_status(sweep, include_runs=True) if sweep else None,
            "operation": compact_operation(current_operation) if current_operation else None,
            "operation_history": [compact_operation(entry) for entry in operation_history],
            "agent": {
                "health": agent_health,
                "launches": agent_launches,
                "pids": list(job.agent_pids) if job else [],
            },
            "results": {
                "readiness": result_readiness,
                "last_pull": result_pull,
            },
            "state": {
                "job_status": job.status.value if job else None,
                "wandb_sweep_status": sweep.get("state") if sweep else None,
                "agent_health": agent_health,
                "result_readiness": result_readiness,
            },
            "degraded": degraded,
            "next_actions": next_actions,
            "generated_at": utc_now(),
        }

    def _stop(self, payload: StopJobPayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> tuple[dict[str, Any], JobRecord]:
        job = self._require_job(payload.job_id)
        if not job.sweep_id or not job.entity or not job.project or not job.remote_host:
            raise ValueError("job lacks sweep/remote metadata required for stop")
        operation_id = operation_id or self._operation_identity(IntentType.stop_job, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.stop_job, payload.model_dump(mode="json"))[1]
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.stop_job,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="accepted",
            classification="accepted",
            status=OperationStatus.accepted.value,
            result={"stage": "accepted", "classification": "accepted", "job_id": job.job_id},
            retry_after_seconds=2,
        )
        result = {"cancel_wandb_requested": payload.cancel_wandb, "cancel_wandb_implemented": False}
        if payload.kill_agents:
            result["stop_agents"] = self.ssh.stop_agents(host=job.remote_host, sweep_path=f"{job.entity}/{job.project}/{job.sweep_id}")
        job = self.store.update_job_status(job.job_id, JobStatus.cancelled, {"stop_result": result})
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.stop_job,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="done",
            classification="job_cancelled",
            status=OperationStatus.succeeded.value,
            result={"stage": "done", "classification": "job_cancelled", "job_id": job.job_id, **result},
        )
        return result, job

    def _cancel_sweep(self, payload: CancelSweepPayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
        entity = payload.entity or self.settings.default_entity
        project = payload.project or self.settings.default_project
        sweep_path = f"{entity}/{project}/{payload.sweep_id}"
        operation_id = operation_id or self._operation_identity(IntentType.cancel_sweep, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.cancel_sweep, payload.model_dump(mode="json"))[1]
        job = self.store.find_job_by_sweep(entity, project, payload.sweep_id)
        if not job:
            job = JobRecord(
                job_id=new_id("job", f"cancel_{payload.sweep_id}"),
                name=f"cancel_{payload.sweep_id}",
                status=JobStatus.attention,
                entity=entity,
                project=project,
                sweep_id=payload.sweep_id,
                remote_host=payload.remote_host,
                remote_cwd=payload.remote_cwd,
                monitor={"stage": "cancel_requested", "created_new_sweep": False},
            )
            self.store.upsert_job(job)
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.cancel_sweep,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="accepted",
            classification="accepted",
            status=OperationStatus.accepted.value,
            result={
                "stage": "accepted",
                "classification": "accepted",
                "sweep_id": payload.sweep_id,
                "mode": payload.mode,
            },
            retry_after_seconds=2,
        )
        result = self.ssh.cancel_sweep(
            host=payload.remote_host,
            remote_cwd=payload.remote_cwd,
            sweep_path=sweep_path,
            mode=payload.mode,
            wandb_api_key=self._wandb_api_key(),
        )
        if job:
            self._record_operation(
                job,
                operation_id=operation_id,
                intent=IntentType.cancel_sweep,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="done",
                classification=result["classification"],
                status=OperationStatus.succeeded.value if result.get("classification") not in {"wandb_cancel_failed"} else OperationStatus.failed.value,
                result={"stage": "done", "classification": result["classification"], "sweep_id": payload.sweep_id, "mode": payload.mode, "cancel": result},
            )
        return {
            "stage": "done",
            "classification": result["classification"],
            "sweep_id": payload.sweep_id,
            "entity": entity,
            "project": project,
            "sweep_path": sweep_path,
            "remote": {"host": payload.remote_host, "cwd": payload.remote_cwd},
            "mode": payload.mode,
            "cancel": result,
            "next_actions": ["刷新 Console overview 确认 W&B sweep lifecycle 已更新。"],
        }

    def _recover(self, payload: RecoverAgentsPayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> tuple[dict[str, Any], JobRecord]:
        job = self._require_job(payload.job_id)
        if not job.sweep_id or not job.entity or not job.project or not job.remote_host or not job.remote_cwd:
            raise ValueError("job lacks sweep/remote metadata required for recover")
        operation_id = operation_id or self._operation_identity(IntentType.recover_agents, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.recover_agents, payload.model_dump(mode="json"))[1]
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.recover_agents,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="accepted",
            classification="accepted",
            status=OperationStatus.accepted.value,
            result={"stage": "accepted", "classification": "accepted", "job_id": job.job_id},
            retry_after_seconds=2,
        )
        gpu_probe = self.ssh.probe_gpus(job.remote_host)
        eligible = [gpu for gpu in gpu_probe["gpus"] if gpu["eligible"]]
        if payload.max_agents is not None:
            eligible = eligible[:payload.max_agents]
        sweep_path = f"{job.entity}/{job.project}/{job.sweep_id}"
        wandb_api_key = self._wandb_api_key()
        conda_env = job.conda_env or self.settings.default_conda_env
        if conda_env and not job.conda_env:
            job.conda_env = conda_env
        auth = self.ssh.auth_check(host=job.remote_host, remote_cwd=job.remote_cwd, sweep_path=sweep_path, wandb_api_key=wandb_api_key)
        launches = [
            self.ssh.launch_agent(
                host=job.remote_host,
                remote_cwd=job.remote_cwd,
                sweep_path=sweep_path,
                gpu_index=gpu["index"],
                conda_env=conda_env,
                conda_sh=payload.conda_sh,
                wandb_api_key=wandb_api_key,
            )
            for gpu in eligible
        ]
        job.agent_pids = sorted(set(job.agent_pids + [item["pid"] for item in launches if item.get("pid")]))
        job.status = JobStatus.running if launches else JobStatus.attention
        classification = classify_launch(auth, launches)
        job.monitor.update({"recover_gpu_probe": gpu_probe, "recover_auth": auth, "recover_agent_launches": launches, "classification": classification})
        self.store.upsert_job(job)
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.recover_agents,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="done",
            classification=classification,
            status=OperationStatus.succeeded.value if classification == "agents_running" else OperationStatus.failed.value if classification in {"wandb_auth_missing", "agents_failed_wandb_auth"} else OperationStatus.executing.value,
            result={
                "stage": "done",
                "classification": classification,
                "auth": auth,
                "gpu_probe": gpu_probe,
                "agent_launches": launches,
                "created_new_sweep": False,
                "next_actions": next_actions_for_classification(classification),
            },
        )
        return {
            "stage": "done",
            "classification": classification,
            "auth": auth,
            "gpu_probe": gpu_probe,
            "agent_launches": launches,
            "created_new_sweep": False,
            "next_actions": next_actions_for_classification(classification),
        }, job

    def _repair_watchdog(self, payload: RepairWatchdogPayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> tuple[dict[str, Any], JobRecord]:
        job = self._require_job(payload.job_id)
        operation_id = operation_id or self._operation_identity(IntentType.repair_watchdog, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.repair_watchdog, payload.model_dump(mode="json"))[1]
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.repair_watchdog,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="accepted",
            classification="accepted",
            status=OperationStatus.accepted.value,
            result={"stage": "accepted", "classification": "accepted", "job_id": job.job_id},
            retry_after_seconds=2,
        )
        before = {
            "remote_host": job.remote_host,
            "remote_cwd": job.remote_cwd,
            "conda_env": job.conda_env,
            "watchdog": job.monitor.get("watchdog"),
        }
        job.remote_cwd = payload.remote_cwd
        if payload.conda_env:
            job.conda_env = payload.conda_env
        watchdog = dict(job.monitor.get("watchdog") or {})
        watchdog.update({
            "remote_cwd": payload.remote_cwd,
            "remote_log_dir": payload.remote_log_dir,
            "remote_tmp_dir": payload.remote_tmp_dir,
            "conda_sh": payload.conda_sh,
            "conda_env": payload.conda_env or job.conda_env,
            "stage": "metadata_repaired",
            "classification": "watchdog_metadata_repaired",
            "updated_at": utc_now(),
        })
        job.monitor["watchdog"] = {k: v for k, v in watchdog.items() if v is not None}
        job.monitor["stage"] = "watchdog_metadata_repaired"
        job = self.store.upsert_job(job)
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.repair_watchdog,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="done",
            classification="watchdog_metadata_repaired",
            status=OperationStatus.succeeded.value,
            result={
                "stage": "done",
                "classification": "watchdog_metadata_repaired",
                "job_id": job.job_id,
                "remote": {
                    "host": job.remote_host,
                    "cwd": job.remote_cwd,
                    "log_dir": payload.remote_log_dir,
                    "tmp_dir": payload.remote_tmp_dir,
                    "conda_sh": payload.conda_sh,
                    "conda_env": job.conda_env,
                },
                "before": before,
                "next_actions": ["后续 watchdog-once 会通过 Console status 查询 job，不需要修补 runner 本地 job store。"],
            },
        )
        return {
            "stage": "done",
            "classification": "watchdog_metadata_repaired",
            "job_id": job.job_id,
            "remote": {
                "host": job.remote_host,
                "cwd": job.remote_cwd,
                "log_dir": payload.remote_log_dir,
                "tmp_dir": payload.remote_tmp_dir,
                "conda_sh": payload.conda_sh,
                "conda_env": job.conda_env,
            },
            "before": before,
            "next_actions": ["后续 watchdog-once 会通过 Console status 查询 job，不需要修补 runner 本地 job store。"],
        }, job

    def _schedule_monitor(self, payload: ScheduleMonitorPayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> tuple[dict[str, Any], JobRecord]:
        job = self._require_job(payload.job_id)
        operation_id = operation_id or self._operation_identity(IntentType.schedule_monitor, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.schedule_monitor, payload.model_dump(mode="json"))[1]
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.schedule_monitor,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="accepted",
            classification="accepted",
            status=OperationStatus.accepted.value,
            result={"stage": "accepted", "classification": "accepted", "job_id": job.job_id},
            retry_after_seconds=2,
        )
        notify = dict(job.monitor.get("notify") or {})
        if payload.notify_channel:
            notify["channel"] = payload.notify_channel
        if payload.notify_target:
            notify["target"] = payload.notify_target
        cron = dict(job.monitor.get("cron") or {})
        cron_id = cron.get("cron_id") or f"exp-watchdog-{job.job_id}"
        script = cron.get("script") or f"experiment-runner-watchdog-{job.job_id}.sh"
        cron.update({
            "cron_id": cron_id,
            "script": script,
            "every": payload.every,
            "timeout_seconds": payload.timeout_seconds,
            "command": f"experiment.py watchdog-once --job-id {job.job_id}",
            "owner": "experiment_console",
            "active": True,
            "updated_at": utc_now(),
        })
        job.monitor["cron"] = cron
        if notify:
            job.monitor["notify"] = notify
        job.monitor["stage"] = "monitor_scheduled"
        job = self.store.upsert_job(job)
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.schedule_monitor,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="done",
            classification="monitor_scheduled",
            status=OperationStatus.succeeded.value,
            result={
                "stage": "done",
                "classification": "monitor_scheduled",
                "success": True,
                "job_id": job.job_id,
                "cron": cron,
                "notify": notify,
                "next_actions": ["Hermes/cron 调度由 Console metadata 驱动；runner 不再创建本地 job store cron。"],
            },
        )
        return {
            "stage": "done",
            "classification": "monitor_scheduled",
            "success": True,
            "job_id": job.job_id,
            "cron": cron,
            "notify": notify,
            "next_actions": ["Hermes/cron 调度由 Console metadata 驱动；runner 不再创建本地 job store cron。"],
        }, job

    def _unschedule_monitor(self, payload: UnscheduleMonitorPayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> tuple[dict[str, Any], JobRecord]:
        job = self._require_job(payload.job_id)
        operation_id = operation_id or self._operation_identity(IntentType.unschedule_monitor, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.unschedule_monitor, payload.model_dump(mode="json"))[1]
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.unschedule_monitor,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="accepted",
            classification="accepted",
            status=OperationStatus.accepted.value,
            result={"stage": "accepted", "classification": "accepted", "job_id": job.job_id},
            retry_after_seconds=2,
        )
        cron = dict(job.monitor.get("cron") or {})
        had_cron = bool(cron)
        cron.update({
            "active": False,
            "unscheduled_at": utc_now(),
            "owner": cron.get("owner") or "experiment_console",
        })
        job.monitor["cron"] = cron
        job.monitor["stage"] = "monitor_unscheduled"
        job = self.store.upsert_job(job)
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.unschedule_monitor,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="done",
            classification="monitor_unscheduled" if had_cron else "monitor_not_scheduled",
            status=OperationStatus.succeeded.value,
            result={
                "stage": "done",
                "classification": "monitor_unscheduled" if had_cron else "monitor_not_scheduled",
                "success": True,
                "job_id": job.job_id,
                "cron": cron,
                "idempotent": not had_cron,
            },
        )
        return {
            "stage": "done",
            "classification": "monitor_unscheduled" if had_cron else "monitor_not_scheduled",
            "success": True,
            "job_id": job.job_id,
            "cron": cron,
            "idempotent": not had_cron,
        }, job

    def _watchdog_once(self, payload: WatchdogOncePayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
        status_result = self._status(StatusQueryPayload(job_id=payload.job_id))
        job = status_result.get("job") or {}
        sweep = status_result.get("sweep") or {}
        degraded = status_result.get("degraded")
        state = status_result.get("state") or {}
        job_status = str(job.get("status") or "unknown").lower()
        sweep_state = str(sweep.get("state") or "").lower()
        agent_health = str(state.get("agent_health") or "unknown").lower()
        run_count = sweep.get("runCount")
        expected = sweep.get("expectedRunCount")
        is_terminal = job_status in {"finished", "failed", "cancelled"} or sweep_state in {"finished", "failed", "cancelled"}
        is_attention = bool(degraded) or job_status in {"attention", "failed"} or sweep_state in {"failed", "crashed", "killed"} or agent_health in {"missing", "degraded"}
        is_running = job_status == "running" or sweep_state == "running"
        classification = "healthy_running"
        silent = True
        event = "healthy"
        message = ""
        if is_terminal:
            classification = "terminal"
            event = "terminal"
            silent = False
            message = f"job {payload.job_id} terminal: job={job_status}, sweep={sweep_state or '-'}"
        elif is_attention:
            classification = "attention"
            event = "attention"
            silent = False
            message = f"job {payload.job_id} needs attention: job={job_status}, sweep={sweep_state or '-'}, degraded={degraded or '-'}"
        elif not is_running:
            classification = "unknown"
            event = "unknown"
            silent = True
        result = {
            "stage": "done",
            "classification": classification,
            "success": not is_attention,
            "silent": silent,
            "event": event,
            "message": message,
            "job_id": payload.job_id,
            "status": job_status,
            "sweep_state": sweep.get("state"),
            "run_count": run_count,
            "expected_run_count": expected,
            "agent_health": agent_health,
            "terminal_disable_requested": payload.terminal_disable,
            "operation": status_result.get("operation"),
            "operation_history": status_result.get("operation_history"),
            "status_result": status_result,
        }
        job_record = self.store.get_job(payload.job_id)
        if job_record:
            operation_id = operation_id or self._operation_identity(IntentType.watchdog_once, payload.model_dump(mode="json"))[0]
            idempotency_key = idempotency_key or self._operation_identity(IntentType.watchdog_once, payload.model_dump(mode="json"))[1]
            job_record.monitor["last_watchdog"] = {
                "classification": classification,
                "silent": silent,
                "event": event,
                "generated_at": utc_now(),
            }
            self._record_operation(
                job_record,
                operation_id=operation_id,
                intent=IntentType.watchdog_once,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="done",
                classification=classification,
                status=OperationStatus.succeeded.value if not is_attention else OperationStatus.failed.value,
                result=result,
            )
            self.store.upsert_job(job_record)
        return result

    def _preflight(self, payload: PreflightPayload) -> dict[str, Any]:
        conda_env = payload.conda_env or self.settings.default_conda_env
        result = self.ssh.preflight(
            host=payload.remote_host,
            remote_cwd=payload.remote_cwd,
            conda_env=conda_env,
            conda_sh=payload.conda_sh,
            config_path=payload.config_path,
        )
        return {**result, "provenance": {"source": "ssh_remote_preflight"}}

    def _auth_check(self, payload: AuthCheckPayload) -> dict[str, Any]:
        job = self.store.get_job(payload.job_id) if payload.job_id else None
        entity = payload.entity or (job.entity if job else None) or self.settings.default_entity
        project = payload.project or (job.project if job else None) or self.settings.default_project
        sweep_id = payload.sweep_id or (job.sweep_id if job else None)
        remote_host = payload.remote_host or (job.remote_host if job else None)
        remote_cwd = payload.remote_cwd or (job.remote_cwd if job else None)
        if not remote_host:
            raise ValueError("remote_host is required for auth-check")
        sweep_path = f"{entity}/{project}/{sweep_id}" if sweep_id else None
        return self.ssh.auth_check(
            host=remote_host,
            remote_cwd=remote_cwd,
            sweep_path=sweep_path,
            wandb_api_key=self._wandb_api_key(),
        )

    def _pull_results(self, payload: PullResultsPayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
        job = self.store.get_job(payload.job_id) if payload.job_id else None
        entity = payload.entity or (job.entity if job else None) or self.settings.default_entity
        project = payload.project or (job.project if job else None) or self.settings.default_project
        sweep_id = payload.sweep_id or (job.sweep_id if job else None)
        remote_host = payload.remote_host or (job.remote_host if job else None)
        remote_cwd = payload.remote_cwd or (job.remote_cwd if job else None)
        if not sweep_id:
            raise ValueError("sweep_id is required for pull-results")
        operation_id = operation_id or self._operation_identity(IntentType.pull_results, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.pull_results, payload.model_dump(mode="json"))[1]
        if job:
            self._record_operation(
                job,
                operation_id=operation_id,
                intent=IntentType.pull_results,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="accepted",
                classification="accepted",
                status=OperationStatus.accepted.value,
                result={
                    "stage": "accepted",
                    "classification": "accepted",
                    "entity": entity,
                    "project": project,
                    "sweep_id": sweep_id,
                },
                retry_after_seconds=2,
            )
        if not remote_host or not remote_cwd:
            if payload.allow_partial:
                result = self._pull_results_from_wandb(entity, project, sweep_id, payload)
                if job:
                    self._record_operation(
                        job,
                        operation_id=operation_id,
                        intent=IntentType.pull_results,
                        requested_by=requested_by,
                        idempotency_key=idempotency_key,
                        stage=result.get("stage", "done"),
                        classification=result.get("classification", "wandb_partial_pull"),
                        status=OperationStatus.succeeded.value,
                        result=result,
                    )
                return result
            raise ValueError("remote_host and remote_cwd are required for remote result pullback")
        try:
            try:
                run_ids = self._sweep_run_ids(entity, project, sweep_id, payload.max_runs)
                run_id_source = "wandb_api"
            except Exception:
                run_ids = cached_run_ids(job, payload.max_runs) if job else []
                run_id_source = "cached_wandb_status"
            result = self.ssh.pull_results(
                host=remote_host,
                remote_cwd=remote_cwd,
                sweep_id=sweep_id,
                run_ids=run_ids,
                budget_seconds=payload.budget_seconds,
                max_runs=payload.max_runs,
                metric_keys=payload.metric_keys,
                group_keys=payload.group_keys,
            )
            result["run_id_source"] = run_id_source
            if job:
                job.monitor["last_result_pull"] = summarize_result_pull(result)
                self._record_operation(
                    job,
                    operation_id=operation_id,
                    intent=IntentType.pull_results,
                    requested_by=requested_by,
                    idempotency_key=idempotency_key,
                    stage="done",
                    classification="ok" if result.get("valid_results", 0) else "no_scientific_results_yet",
                    status=OperationStatus.succeeded.value,
                    result={
                        "stage": "done",
                        "classification": "ok" if result.get("valid_results", 0) else "no_scientific_results_yet",
                        "entity": entity,
                        "project": project,
                        **result,
                    },
                )
            return {
                "stage": "done",
                "classification": "ok" if result.get("valid_results", 0) else "no_scientific_results_yet",
                "entity": entity,
                "project": project,
                **result,
            }
        except Exception as exc:
            if not payload.allow_partial:
                raise
            try:
                fallback = self._pull_results_from_wandb(entity, project, sweep_id, payload)
                fallback["remote_error"] = compact_error(exc)
                fallback["classification"] = "remote_pull_failed_wandb_fallback"
                if job:
                    self._record_operation(
                        job,
                        operation_id=operation_id,
                        intent=IntentType.pull_results,
                        requested_by=requested_by,
                        idempotency_key=idempotency_key,
                        stage=fallback.get("stage", "degraded"),
                        classification=fallback.get("classification", "remote_pull_failed_wandb_fallback"),
                        status=OperationStatus.succeeded.value,
                        result=fallback,
                    )
                return fallback
            except Exception as fallback_exc:
                if job:
                    self._record_operation(
                        job,
                        operation_id=operation_id,
                        intent=IntentType.pull_results,
                        requested_by=requested_by,
                        idempotency_key=idempotency_key,
                        stage="degraded",
                        classification="result_sources_unavailable",
                        status=OperationStatus.failed.value,
                        result={
                            "stage": "degraded",
                            "classification": "result_sources_unavailable",
                            "source": "degraded_empty_partial",
                            "entity": entity,
                            "project": project,
                            "sweep_id": sweep_id,
                            "rows": [],
                            "groups": {},
                            "valid_results": 0,
                            "missing_results": 0,
                            "failed_results": 0,
                            "partial": True,
                            "remote_error": compact_error(exc),
                            "wandb_error": compact_error(fallback_exc),
                        },
                    )
                return {
                    "stage": "degraded",
                    "classification": "result_sources_unavailable",
                    "source": "degraded_empty_partial",
                    "entity": entity,
                    "project": project,
                    "sweep_id": sweep_id,
                    "rows": [],
                    "groups": {},
                    "valid_results": 0,
                    "missing_results": 0,
                    "failed_results": 0,
                    "partial": True,
                    "remote_error": compact_error(exc),
                    "wandb_error": compact_error(fallback_exc),
                    "next_actions": ["确认远端 host/cwd 可达，或配置 WANDB_API_KEY 后重试。"],
                }

    def _pull_results_from_wandb(self, entity: str, project: str, sweep_id: str, payload: PullResultsPayload) -> dict[str, Any]:
        sweep = self.wandb.get_sweep_state(entity, project, sweep_id)
        rows = []
        for run in (sweep.get("runs") or [])[:payload.max_runs]:
            summary = parse_wandb_json_field(run.get("summary_metrics"))
            config = parse_wandb_json_field(run.get("config"))
            rows.append({
                "run_id": run.get("name"),
                "state": run.get("state"),
                "config": {key: config.get(key) for key in payload.group_keys} if payload.group_keys else config,
                "metrics": {key: summary.get(key) for key in payload.metric_keys} if payload.metric_keys else summary,
                "has_scientific_result": bool(summary) and (not payload.metric_keys or any(key in summary for key in payload.metric_keys)),
            })
        return {
            "stage": "done",
            "classification": "wandb_partial_pull",
            "source": "wandb_api_fallback",
            "entity": entity,
            "project": project,
            "sweep_id": sweep_id,
            "rows": rows,
            "valid_results": sum(1 for row in rows if row["has_scientific_result"]),
            "missing_results": sum(1 for row in rows if not row["has_scientific_result"]),
            "partial": True,
        }

    def _sweep_run_ids(self, entity: str, project: str, sweep_id: str, max_runs: int) -> list[str]:
        sweep = self.wandb.get_sweep_state(entity, project, sweep_id)
        run_ids = []
        for run in (sweep.get("runs") or [])[:max_runs]:
            name = run.get("name")
            if name:
                run_ids.append(str(name))
        return run_ids

    def _wandb_api_key(self) -> str | None:
        return os.environ.get(self.settings.wandb_api_key_env)

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
            "stalled_sweeps": sum(1 for sweep in sweeps if sweep.get("state") == "STALLED"),
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


def classify_launch(auth: dict[str, Any], launches: list[dict[str, Any]]) -> str:
    if not auth.get("ok"):
        return str(auth.get("classification") or "wandb_auth_missing")
    if not launches:
        return "no_eligible_gpus"
    classifications = {str(item.get("classification") or "") for item in launches}
    if "wandb_auth_missing" in classifications:
        return "agents_failed_wandb_auth"
    if any(item.get("pid") for item in launches):
        return "agents_running"
    return "agents_started_unverified"


def next_actions_for_classification(classification: str) -> list[str]:
    if classification in {"wandb_auth_missing", "agents_failed_wandb_auth"}:
        return ["刷新 WANDB_API_KEY 或 Bitwarden session 后，对同一个 job 执行 recover-agents；不要重新创建 sweep。"]
    if classification == "no_eligible_gpus":
        return ["等待 GPU 空闲后执行 recover-agents，或显式设置更保守的 max_agents。"]
    if classification == "agents_started_unverified":
        return ["检查远端 agent log；如 agent 不存在，对既有 job 执行 recover-agents。"]
    return []


def summarize_result_pull(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": result.get("source"),
        "sweep_id": result.get("sweep_id"),
        "valid_results": result.get("valid_results"),
        "missing_results": result.get("missing_results"),
        "failed_results": result.get("failed_results"),
        "partial": result.get("partial"),
        "generated_at": utc_now(),
    }


def compact_job_record(job: JobRecord) -> dict[str, Any]:
    monitor = job.monitor or {}
    compact_monitor: dict[str, Any] = {}
    for key in [
        "stage",
        "classification",
        "expected_total",
        "created_new_sweep",
        "sweep_path",
        "last_result_pull",
        "last_watchdog",
        "cron",
        "watchdog",
    ]:
        if key in monitor:
            compact_monitor[key] = monitor[key]
    last_status = monitor.get("last_wandb_status")
    if isinstance(last_status, dict):
        compact_monitor["last_wandb_status"] = compact_sweep_status(last_status, include_runs=True)
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status.value,
        "operation_id": job.operation_id,
        "idempotency_key": job.idempotency_key,
        "entity": job.entity,
        "project": job.project,
        "sweep_id": job.sweep_id,
        "config_path": job.config_path,
        "remote_host": job.remote_host,
        "remote_cwd": job.remote_cwd,
        "conda_env": job.conda_env,
        "agent_pids": list(job.agent_pids),
        "monitor": compact_monitor,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def compact_operation(operation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(operation, dict):
        return None
    result = operation.get("result") if isinstance(operation.get("result"), dict) else {}
    compact_result: dict[str, Any] = {}
    for key in [
        "stage",
        "classification",
        "created_new_sweep",
        "job_id",
        "sweep_id",
        "entity",
        "project",
        "source",
        "valid_results",
        "missing_results",
        "failed_results",
        "partial",
        "run_count",
        "expected_run_count",
        "agent_health",
        "silent",
        "event",
        "message",
        "success",
    ]:
        if key in result:
            compact_result[key] = result[key]
    if "sweep" in result and isinstance(result["sweep"], dict):
        compact_result["sweep"] = compact_sweep_status(result["sweep"], include_runs=True)
    if "auth" in result and isinstance(result["auth"], dict):
        compact_result["auth"] = {
            key: result["auth"].get(key)
            for key in ["ok", "classification", "has_key", "target_accessible", "sweep_state"]
            if key in result["auth"]
        }
    if "next_actions" in result:
        compact_result["next_actions"] = result["next_actions"]
    return {
        "operation_id": operation.get("operation_id"),
        "intent": operation.get("intent"),
        "requested_by": operation.get("requested_by"),
        "idempotency_key": operation.get("idempotency_key"),
        "stage": operation.get("stage"),
        "classification": operation.get("classification"),
        "status": operation.get("status"),
        "result": compact_result,
        "retry_after_seconds": operation.get("retry_after_seconds"),
        "created_at": operation.get("created_at"),
        "updated_at": operation.get("updated_at"),
    }


def cached_run_ids(job: JobRecord, max_runs: int) -> list[str]:
    last_status = job.monitor.get("last_wandb_status")
    if not isinstance(last_status, dict):
        return []
    run_ids = []
    for run in (last_status.get("runs") or [])[:max_runs]:
        if isinstance(run, dict) and run.get("name"):
            run_ids.append(str(run["name"]))
    return run_ids


def compact_sweep_status(sweep: dict[str, Any], *, include_runs: bool = False) -> dict[str, Any]:
    compact = {
        "id": sweep.get("id"),
        "entity": sweep.get("entity"),
        "project": sweep.get("project"),
        "name": sweep.get("name"),
        "state": sweep.get("state"),
        "createdAt": sweep.get("createdAt"),
        "runCount": sweep.get("runCount"),
        "expectedRunCount": sweep.get("expectedRunCount"),
    }
    if include_runs:
        runs = []
        for run in sweep.get("runs") or []:
            if not isinstance(run, dict):
                continue
            runs.append({
                "name": run.get("name"),
                "state": run.get("state"),
                "created_at": run.get("created_at"),
                "heartbeat_at": run.get("heartbeat_at"),
            })
        compact["runs"] = runs
    return compact


def parse_wandb_json_field(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def compact_error(exc: Exception, max_chars: int = 500) -> str:
    result = getattr(exc, "result", None)
    if result is not None:
        stderr = str(getattr(result, "stderr", "") or "").strip()
        stdout = str(getattr(result, "stdout", "") or "").strip()
        detail = stderr or stdout or exc.__class__.__name__
        if len(detail) > max_chars:
            detail = detail[:max_chars].rstrip() + "...<truncated>"
        return f"command failed ({getattr(result, 'returncode', 'unknown')}): {detail}"
    text = str(exc)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "...<truncated>"
