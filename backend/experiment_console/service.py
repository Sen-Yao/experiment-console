from __future__ import annotations

import json
import hashlib
import os
import statistics
from datetime import datetime, timezone
from typing import Any

from .command import CommandFailed
from .config import Settings
from .models import (
    AuditEvent,
    AdvanceQueuePayload,
    ConfirmRequest,
    ExecuteResponse,
    IntentPreviewRequest,
    IntentRecord,
    IntentStatus,
    IntentType,
    JobRecord,
    JobStatus,
    LaunchRunPayload,
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
from .store import ConsoleStore, infer_job_kind
from .sweep_telemetry import (
    enrich_sweeps_with_telemetry,
    load_telemetry_cache,
    save_telemetry_cache,
    strip_runs,
)
from .validation import build_single_run_command, load_yaml_text, validate_experiment_config_text, validate_experiment_config
from .wandb_client import WandBClient, WandBUnavailable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


MUTATING_INTENTS = {
    IntentType.launch_sweep,
    IntentType.launch_run,
    IntentType.register_existing_sweep,
    IntentType.stop_job,
    IntentType.cancel_sweep,
    IntentType.recover_agents,
    IntentType.repair_watchdog,
    IntentType.schedule_monitor,
    IntentType.unschedule_monitor,
    IntentType.watchdog_once,
    IntentType.pull_results,
    IntentType.advance_queue,
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


SINGLE_RUN_BOUNDARY_NEXT_ACTIONS = [
    "launch-run 只接受所有参数均为单值的 single-run 配置；这个配置会展开成多次运行，请改用 launch-sweep。",
]


TERMINAL_SWEEP_STATES = {"finished", "failed", "crashed", "killed", "canceled", "cancelled"}


def result_pull_classification(result: dict[str, Any]) -> str:
    if result.get("classification") == "result_sources_unavailable":
        return "result_sources_unavailable"
    if result.get("truncated"):
        return "truncated_results"
    if to_non_negative_int(result.get("valid_results")) <= 0:
        return "no_scientific_results_yet"
    if not result.get("complete"):
        return "partial_results"
    return "ok"


def classify_result_readiness(result: dict[str, Any] | None, sweep_state: str | None = None) -> str:
    if not isinstance(result, dict):
        return "unknown"
    if result.get("readiness") in {"unknown", "none", "partial", "truncated", "complete", "complete_with_failures"}:
        return str(result["readiness"])
    if result.get("truncated"):
        return "truncated"
    valid = to_non_negative_int(result.get("valid_runs", result.get("valid_results")))
    if valid <= 0:
        return "none"
    missing = to_non_negative_int(result.get("missing_runs", result.get("missing_results")))
    failed = to_non_negative_int(result.get("failed_runs", result.get("failed_results")))
    complete = bool(result.get("complete"))
    if complete:
        return "complete_with_failures" if (missing or failed) else "complete"
    if sweep_state and str(sweep_state).lower() not in TERMINAL_SWEEP_STATES:
        return "partial"
    return "partial"


def numeric_metric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def ordered_numeric_metric_keys(rows: list[dict[str, Any]], metric_keys: list[str] | None = None) -> list[str]:
    seen: list[str] = []
    for metric in metric_keys or []:
        if metric not in seen:
            seen.append(metric)
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        for key, value in metrics.items():
            if key not in seen and numeric_metric(value) is not None:
                seen.append(key)
    return seen


def build_result_groups(rows: list[dict[str, Any]], group_keys: list[str], metric_keys: list[str]) -> dict[str, Any]:
    if not group_keys:
        return {"groups": [], "top_groups": []}
    metrics = ordered_numeric_metric_keys(rows, metric_keys)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        config = row.get("config") if isinstance(row.get("config"), dict) else {}
        key = tuple(config.get(group_key) for group_key in group_keys)
        grouped.setdefault(key, []).append(row)
    groups: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        item: dict[str, Any] = {
            "config": dict(zip(group_keys, key)),
            "count": len(group_rows),
            "valid_count": sum(1 for row in group_rows if row.get("has_scientific_result")),
            "run_ids": [row.get("run_id") for row in group_rows if row.get("run_id")],
            "metrics": {},
        }
        for metric in metrics:
            values = [
                numeric
                for row in group_rows
                if (numeric := numeric_metric((row.get("metrics") or {}).get(metric))) is not None
            ]
            if values:
                item["metrics"][metric] = {
                    "mean": sum(values) / len(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
        groups.append(item)
    if metrics:
        first_metric = metrics[0]
        top_groups = sorted(
            [group for group in groups if first_metric in group.get("metrics", {})],
            key=lambda group: group["metrics"][first_metric]["mean"],
            reverse=True,
        )
    else:
        top_groups = []
    return {"groups": groups, "top_groups": top_groups[:20]}


def build_metric_summaries(rows: list[dict[str, Any]], metric_keys: list[str] | None = None) -> dict[str, Any]:
    wanted = ordered_numeric_metric_keys(rows, metric_keys)
    summaries: dict[str, Any] = {}
    for metric in wanted:
        values = [
            numeric
            for row in rows
            if (numeric := numeric_metric((row.get("metrics") or {}).get(metric))) is not None
        ]
        if values:
            summaries[metric] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
    return summaries


def selector_label(selector: str) -> str:
    text = str(selector or "")
    if "=" in text:
        alias, _ = text.split("=", 1)
        if alias:
            return alias
    return text


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

    def _enrich_sweeps_with_telemetry(self, sweeps: list[dict[str, Any]], *, observed_at: str | None = None) -> list[dict[str, Any]]:
        cache = load_telemetry_cache(self.settings.sweep_telemetry_cache_path)
        enriched, updated_cache = enrich_sweeps_with_telemetry(sweeps, cache=cache, observed_at=observed_at or utc_now())
        save_telemetry_cache(self.settings.sweep_telemetry_cache_path, updated_cache)
        return enriched

    def _enrich_single_sweep_with_telemetry(self, sweep: dict[str, Any]) -> dict[str, Any]:
        enriched = self._enrich_sweeps_with_telemetry([sweep])
        return enriched[0] if enriched else sweep

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

    def _materialize_result_snapshot(
        self,
        *,
        job: JobRecord | None,
        entity: str | None,
        project: str | None,
        sweep_id: str | None,
        result: dict[str, Any],
        group_keys: list[str],
        metric_keys: list[str],
        expected_runs: int | None,
        discovered_runs: int | None,
        requested_limit: int | None,
        run_id_source: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rows = result.get("rows") if isinstance(result.get("rows"), list) else []
        fetched_runs = len(rows)
        valid_runs = to_non_negative_int(result.get("valid_results"))
        if "valid_results" not in result:
            valid_runs = sum(1 for row in rows if isinstance(row, dict) and row.get("has_scientific_result"))
        missing_runs = to_non_negative_int(result.get("missing_results"))
        failed_runs = to_non_negative_int(result.get("failed_results"))
        coverage_target = expected_runs or discovered_runs
        truncated = bool(result.get("truncated"))
        if requested_limit is not None and coverage_target and requested_limit < coverage_target:
            truncated = True
        if discovered_runs is not None and fetched_runs < discovered_runs and requested_limit is not None:
            truncated = True
        complete = bool(coverage_target and fetched_runs >= coverage_target and not truncated)
        groups = build_result_groups(rows, group_keys, metric_keys)
        metric_summaries = build_metric_summaries(rows, metric_keys)
        enriched = {
            **result,
            "expected_runs": expected_runs,
            "discovered_runs": discovered_runs,
            "requested_limit": requested_limit,
            "fetched_runs": fetched_runs,
            "valid_runs": valid_runs,
            "missing_runs": missing_runs,
            "failed_runs": failed_runs,
            "truncated": truncated,
            "complete": complete,
            "groups": groups["groups"],
            "top_groups": groups["top_groups"],
            "metric_summaries": metric_summaries,
        }
        readiness = classify_result_readiness(enriched)
        classification = result_pull_classification(enriched)
        enriched["classification"] = classification
        enriched["readiness"] = readiness
        snapshot_id = new_id("result_snapshot", job.job_id if job else (sweep_id or "unknown"))
        snapshot_key = job.job_id if job else (sweep_id or "unknown")
        safe_key = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in snapshot_key)
        snapshot_dir = self.settings.results_dir / safe_key
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"{snapshot_id}.json"
        provenance = {
            "run_id_source": run_id_source or result.get("run_id_source"),
            "config_sources": result.get("config_sources", {}),
            "metric_sources": result.get("metric_sources", {}),
            "missing_config_keys": result.get("missing_config_keys", {}),
            "missing_metric_paths": result.get("missing_metric_paths", {}),
            "discovery_sources": result.get("discovery_sources", {}),
        }
        for key_name in ["remote_error", "wandb_error"]:
            if result.get(key_name):
                provenance[key_name] = result[key_name]
        snapshot = {
            "identity": {
                "snapshot_id": snapshot_id,
                "job_id": job.job_id if job else None,
                "entity": entity,
                "project": project,
                "sweep_id": sweep_id,
                "source": result.get("source"),
                "generated_at": utc_now(),
            },
            "completeness": {
                "expected_runs": expected_runs,
                "discovered_runs": discovered_runs,
                "requested_limit": requested_limit,
                "fetched_runs": fetched_runs,
                "valid_runs": valid_runs,
                "missing_runs": missing_runs,
                "failed_runs": failed_runs,
                "truncated": truncated,
                "complete": complete,
            },
            "rows": rows,
            "groups": groups["groups"],
            "top_groups": groups["top_groups"],
            "metric_summaries": metric_summaries,
            "provenance": provenance,
            "classification": classification,
            "readiness": readiness,
        }
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = summarize_result_pull({
            **enriched,
            "snapshot_id": snapshot_id,
            "snapshot_path": str(snapshot_path),
        })
        summary["path"] = str(snapshot_path)
        enriched["snapshot"] = summary
        return enriched, summary

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

    def _fail_launch_job(
        self,
        job: JobRecord,
        *,
        intent: IntentType,
        operation_id: str,
        requested_by: str,
        idempotency_key: str,
        stage: str,
        classification: str,
        exc: Exception,
        next_actions: list[str],
        status: JobStatus = JobStatus.failed,
        extra_result: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], JobRecord]:
        error = compact_error(exc)
        job.status = status
        job.monitor.update({
            "stage": stage,
            "classification": classification,
            "error": error,
        })
        self.store.upsert_job(job)
        result = {
            "stage": stage,
            "classification": classification,
            "created_new_sweep": False,
            "job_id": job.job_id,
            "error": error,
            "next_actions": next_actions,
        }
        if extra_result:
            result.update(extra_result)
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=intent,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage=stage,
            classification=classification,
            status=OperationStatus.failed.value,
            result=result,
        )
        return result, job

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
        if isinstance(payload, LaunchRunPayload):
            return self._launch_run(payload)
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
        if isinstance(parsed, LaunchRunPayload):
            return self._launch_run(parsed, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key)
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
        if isinstance(parsed, AdvanceQueuePayload):
            return self._advance_queue(parsed, requested_by=requested_by, operation_id=operation_id, idempotency_key=idempotency_key), None
        raise ValueError(f"unsupported intent: {intent}")

    def _queue_group_for(self, *, remote_host: str, remote_cwd: str, payload: LaunchSweepPayload) -> str:
        return payload.queue_group or f"{remote_host}:{remote_cwd}"

    def _queue_position(self, job: JobRecord) -> dict[str, Any] | None:
        queue = job.monitor.get("queue") if isinstance(job.monitor.get("queue"), dict) else {}
        queue_group = queue.get("queue_group")
        if not queue_group:
            return None
        queued_jobs = self.store.list_queued_jobs(str(queue_group))
        blocker = self.store.active_queue_blocker(str(queue_group), exclude_job_id=job.job_id)
        blocker_assessment = self._assess_queue_blocker(blocker) if blocker else None
        position = None
        for index, queued in enumerate(queued_jobs, start=1):
            if queued.job_id == job.job_id:
                position = index
                break
        blocked_by = queue.get("blocked_by_job_id") or (blocker.job_id if blocker else None)
        result = {
            "queue_group": queue_group,
            "queue_position": position,
            "blocked_by_job_id": blocked_by,
            "queued_count": len(queued_jobs),
        }
        if blocker_assessment:
            result.update({
                "blocked_by_classification": blocker_assessment["classification"],
                "blocker_classification": blocker_assessment["classification"],
                "blocker_unblockable": blocker_assessment["unblockable"],
                "recommended_action": blocker_assessment["recommended_action"],
            })
        return result

    def _assess_queue_blocker(self, job: JobRecord | None) -> dict[str, Any]:
        if job is None:
            return {
                "classification": "no_blocker",
                "unblockable": False,
                "recommended_action": None,
                "evidence": {},
            }
        if job.status in TERMINAL_JOB_STATUSES:
            return {
                "classification": "remote_terminal_reconciled",
                "unblockable": False,
                "recommended_action": None,
                "evidence": {"job_status": job.status.value},
            }
        missing: list[str] = []
        if infer_job_kind(job) == "sweep":
            for key, value in {
                "sweep_id": job.sweep_id,
                "entity": job.entity,
                "project": job.project,
                "remote_host": job.remote_host,
                "remote_cwd": job.remote_cwd,
            }.items():
                if not value:
                    missing.append(key)
        if missing:
            return {
                "classification": "metadata_corrupt_blocker",
                "unblockable": True,
                "recommended_action": "Run advance-queue with auto_unblock_stale=true, or stop-job with ledger_only=true.",
                "evidence": {
                    "job_id": job.job_id,
                    "job_status": job.status.value,
                    "missing": missing,
                    "kind": infer_job_kind(job),
                },
            }
        if job.status == JobStatus.unknown:
            return {
                "classification": "ambiguous_blocker",
                "unblockable": False,
                "recommended_action": "Run status/watchdog-once before deciding whether to ledger-only cancel this job.",
                "evidence": {"job_id": job.job_id, "job_status": job.status.value},
            }
        return {
            "classification": "active_real_blocker",
            "unblockable": False,
            "recommended_action": "Wait for the blocker to finish, or use status/watchdog-once to diagnose it.",
            "evidence": {"job_id": job.job_id, "job_status": job.status.value, "kind": infer_job_kind(job)},
        }

    def _ledger_only_cancel_job(
        self,
        job: JobRecord,
        *,
        requested_by: str,
        reason: str | None,
        classification: str,
        evidence: dict[str, Any] | None = None,
    ) -> tuple[JobRecord, dict[str, Any]]:
        previous_status = job.status.value
        stop_result = {
            "ledger_only": True,
            "reason": reason or "Ledger-only cancellation requested.",
            "remote_side_effects": False,
        }
        queue_hygiene = {
            "unblocked_at": utc_now(),
            "unblocked_by": requested_by,
            "reason": reason or classification,
            "previous_status": previous_status,
            "evidence": evidence or {},
        }
        patch = {
            "classification": classification,
            "stop_result": stop_result,
            "queue_hygiene": queue_hygiene,
        }
        updated = self.store.update_job_status(job.job_id, JobStatus.cancelled, patch)
        result = {
            "job_id": updated.job_id,
            "previous_status": previous_status,
            "status": updated.status.value,
            "classification": classification,
            "ledger_only": True,
            "reason": stop_result["reason"],
            "evidence": evidence or {},
        }
        self.store.write_audit(AuditEvent(
            event_type="job_ledger_only_cancelled",
            job_id=updated.job_id,
            message="Job was cancelled in Console ledger without remote side effects.",
            detail=result,
        ))
        return updated, result

    def _queued_replay_result(self, job: JobRecord) -> dict[str, Any]:
        return {
            "stage": "queued",
            "classification": "queued",
            "created_new_sweep": False,
            "job": job.model_dump(mode="json"),
            "queue": self._queue_position(job),
            "next_actions": [
                "Queued only: W&B sweep has not been created. Wait for the blocker to finish, then run advance-queue.",
            ],
        }

    def _queue_sweep(
        self,
        payload: LaunchSweepPayload,
        *,
        entity: str,
        project: str,
        remote_host: str,
        remote_cwd: str,
        conda_env: str,
        queue_group: str,
        blocker: JobRecord,
        operation_id: str,
        idempotency_key: str,
        requested_by: str,
        conflicting_jobs: list[JobRecord],
        preflight: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], JobRecord]:
        job = JobRecord(
            job_id=new_id("job", payload.job_name),
            name=payload.job_name,
            status=JobStatus.queued,
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            entity=entity,
            project=project,
            config_path=str(payload.config_path),
            remote_host=remote_host,
            remote_cwd=remote_cwd,
            conda_env=conda_env,
            monitor={
                "kind": "sweep",
                "stage": "queued",
                "classification": "queued",
                "created_new_sweep": False,
                "launch_identity_conflicts": [
                    {
                        "job_id": conflict.job_id,
                        "kind": infer_job_kind(conflict),
                        "status": conflict.status.value,
                    }
                    for conflict in conflicting_jobs[:5]
                ],
                "preflight": preflight,
                "validation": validation,
                "queue": {
                    "queue_group": queue_group,
                    "queue_policy": payload.queue_policy,
                    "queue_after_job_id": payload.queue_after_job_id,
                    "blocked_by_job_id": blocker.job_id,
                    "payload": payload.model_dump(mode="json"),
                    "queued_at": utc_now(),
                },
            },
        )
        self.store.upsert_job(job)
        queue_state = self._queue_position(job) or {"queue_group": queue_group, "blocked_by_job_id": blocker.job_id}
        result = {
            "stage": "queued",
            "classification": "queued",
            "created_new_sweep": False,
            "job_id": job.job_id,
            "queue": queue_state,
            "launch_identity_conflicts": job.monitor.get("launch_identity_conflicts") or [],
            "next_actions": [
                "Queued only: W&B sweep has not been created and no agents were launched.",
                "When the blocking job reaches a terminal state, run advance-queue or let watchdog-once advance it.",
            ],
        }
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.launch_sweep,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="queued",
            classification="queued",
            status=OperationStatus.accepted.value,
            result=result,
            retry_after_seconds=30,
        )
        return result, job

    def _sweep_entrypoint_probe(
        self,
        *,
        remote_config_text: str,
        remote_host: str,
        remote_cwd: str,
        conda_env: str | None,
        conda_sh: str | None,
    ) -> dict[str, Any]:
        try:
            config = load_yaml_text(remote_config_text)
            program = config.get("program")
            if not program:
                return {
                    "classification": "argv_probe_unavailable",
                    "returncode": None,
                    "stdout_tail": "",
                    "stderr_tail": "config has no program field",
                    "timed_out": False,
                    "probe_argv": [],
                }
            return self.ssh.probe_argv_compat(
                host=remote_host,
                remote_cwd=remote_cwd,
                argv=["python", str(program)],
                conda_env=conda_env,
                conda_sh=conda_sh,
            )
        except Exception as exc:
            return {
                "classification": "argv_probe_unavailable",
                "returncode": None,
                "stdout_tail": "",
                "stderr_tail": compact_error(exc),
                "timed_out": False,
                "probe_argv": [],
            }

    def _sweep_launch_preflight(
        self,
        *,
        remote_host: str,
        remote_cwd: str,
        remote_config_path: str,
        conda_env: str | None,
        conda_sh: str | None,
        profile: str,
    ) -> dict[str, Any]:
        remote_snapshot = self.ssh.read_remote_file(host=remote_host, remote_path=remote_config_path)
        validation = validate_experiment_config_text(remote_snapshot["text"], profile, path_label=remote_config_path)
        entrypoint_probe = self._sweep_entrypoint_probe(
            remote_config_text=remote_snapshot["text"],
            remote_host=remote_host,
            remote_cwd=remote_cwd,
            conda_env=conda_env,
            conda_sh=conda_sh,
        )
        validation["entrypoint_probe"] = entrypoint_probe
        classification = "preflight_ok" if entrypoint_probe.get("classification") == "argv_compatible" else "entrypoint_probe_failed"
        return {
            "classification": classification,
            "ok": classification == "preflight_ok",
            "remote_config_snapshot": remote_snapshot,
            "validation": validation,
            "entrypoint_probe": entrypoint_probe,
            "preflight": {
                "entrypoint_probe": entrypoint_probe,
                "classification": classification,
                "ok": classification == "preflight_ok",
            },
        }

    def _launch_sweep(self, payload: LaunchSweepPayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> tuple[dict[str, Any], JobRecord]:
        entity = payload.entity or self.settings.default_entity
        project = payload.project or self.settings.default_project
        remote_host = payload.remote_host or self.settings.default_remote_host
        remote_cwd = payload.remote_cwd or self.settings.default_remote_cwd
        conda_env = payload.conda_env or self.settings.default_conda_env
        remote_config_path = payload.config_path
        operation_id = operation_id or self._operation_identity(IntentType.launch_sweep, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.launch_sweep, payload.model_dump(mode="json"))[1]
        matching_jobs = self.store.find_jobs_by_launch_identity(
            name=payload.job_name,
            config_path=str(remote_config_path),
            remote_host=remote_host,
            remote_cwd=remote_cwd,
        )
        conflicting_jobs = [job for job in matching_jobs if infer_job_kind(job) not in {"sweep"}]
        existing = next((job for job in matching_jobs if infer_job_kind(job) == "sweep"), None)
        if existing:
            if existing.status == JobStatus.queued:
                return self._queued_replay_result(existing), existing
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
        queue_group = self._queue_group_for(remote_host=remote_host, remote_cwd=remote_cwd, payload=payload)
        blocker = self.store.get_job(payload.queue_after_job_id) if payload.queue_after_job_id else None
        if payload.queue_after_job_id and blocker is None:
            raise ValueError(f"queue_after_job_id not found: {payload.queue_after_job_id}")
        if blocker:
            blocker_queue = blocker.monitor.get("queue") if isinstance(blocker.monitor.get("queue"), dict) else {}
            blocker_group = blocker_queue.get("queue_group") or (
                f"{blocker.remote_host}:{blocker.remote_cwd}" if blocker.remote_host and blocker.remote_cwd else None
            )
            if blocker_group and blocker_group != queue_group:
                raise ValueError(
                    f"queue_after_job_id belongs to queue_group {blocker_group}, not {queue_group}"
                )
        if blocker and blocker.status in TERMINAL_JOB_STATUSES:
            blocker = None
        if payload.queue_policy == "sequential" and blocker is None:
            blocker = self.store.active_queue_blocker(queue_group)
        if payload.queue_policy == "sequential" and blocker is not None:
            try:
                launch_preflight = self._sweep_launch_preflight(
                    remote_host=remote_host,
                    remote_cwd=remote_cwd,
                    remote_config_path=remote_config_path,
                    conda_env=conda_env,
                    conda_sh=payload.conda_sh or self.settings.default_conda_sh,
                    profile=payload.profile,
                )
            except Exception as exc:
                job = JobRecord(
                    job_id=new_id("job", payload.job_name),
                    name=payload.job_name,
                    status=JobStatus.failed,
                    operation_id=operation_id,
                    idempotency_key=idempotency_key,
                    entity=entity,
                    project=project,
                    config_path=str(remote_config_path),
                    remote_host=remote_host,
                    remote_cwd=remote_cwd,
                    conda_env=conda_env,
                )
                return self._fail_launch_job(
                    job,
                    intent=IntentType.launch_sweep,
                    operation_id=operation_id,
                    requested_by=requested_by,
                    idempotency_key=idempotency_key,
                    stage="validation_failed",
                    classification="validation_failed",
                    exc=exc,
                    next_actions=["修复 sweep 配置或远端 config 路径后，再重新 launch-sweep。"],
                )
            if launch_preflight["classification"] != "preflight_ok":
                job = JobRecord(
                    job_id=new_id("job", payload.job_name),
                    name=payload.job_name,
                    status=JobStatus.attention,
                    operation_id=operation_id,
                    idempotency_key=idempotency_key,
                    entity=entity,
                    project=project,
                    config_path=str(remote_config_path),
                    remote_host=remote_host,
                    remote_cwd=remote_cwd,
                    conda_env=conda_env,
                    monitor={
                        "kind": "sweep",
                        "validation": launch_preflight["validation"],
                        "preflight": launch_preflight["preflight"],
                        "stage": "preflight_failed",
                        "classification": "entrypoint_probe_failed",
                    },
                )
                self.store.upsert_job(job)
                result = {
                    "stage": "preflight_failed",
                    "classification": "entrypoint_probe_failed",
                    "created_new_sweep": False,
                    "job_id": job.job_id,
                    "validation": launch_preflight["validation"],
                    "entrypoint_probe": launch_preflight["entrypoint_probe"],
                    "preflight": launch_preflight["preflight"],
                    "next_actions": ["修复 sweep program 入口、导入路径或 --help CLI 后，再重新 launch-sweep。"],
                }
                self._record_operation(
                    job,
                    operation_id=operation_id,
                    intent=IntentType.launch_sweep,
                    requested_by=requested_by,
                    idempotency_key=idempotency_key,
                    stage="preflight_failed",
                    classification="entrypoint_probe_failed",
                    status=OperationStatus.failed.value,
                    result=result,
                )
                return result, job
            return self._queue_sweep(
                payload,
                entity=entity,
                project=project,
                remote_host=remote_host,
                remote_cwd=remote_cwd,
                conda_env=conda_env,
                queue_group=queue_group,
                blocker=blocker,
                operation_id=operation_id,
                idempotency_key=idempotency_key,
                requested_by=requested_by,
                conflicting_jobs=conflicting_jobs,
                preflight=launch_preflight["preflight"],
                validation=launch_preflight["validation"],
            )
        return self._launch_sweep_now(
            payload,
            requested_by=requested_by,
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            queue_group=queue_group,
            conflicting_jobs=conflicting_jobs,
        )

    def _launch_sweep_now(
        self,
        payload: LaunchSweepPayload,
        *,
        requested_by: str = "experiment-runner",
        operation_id: str | None = None,
        idempotency_key: str | None = None,
        queue_group: str | None = None,
        queued_job: JobRecord | None = None,
        conflicting_jobs: list[JobRecord] | None = None,
    ) -> tuple[dict[str, Any], JobRecord]:
        entity = payload.entity or self.settings.default_entity
        project = payload.project or self.settings.default_project
        remote_host = payload.remote_host or self.settings.default_remote_host
        remote_cwd = payload.remote_cwd or self.settings.default_remote_cwd
        conda_env = payload.conda_env or self.settings.default_conda_env
        conda_sh = payload.conda_sh or self.settings.default_conda_sh
        remote_config_path = payload.config_path
        operation_id = operation_id or self._operation_identity(IntentType.launch_sweep, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.launch_sweep, payload.model_dump(mode="json"))[1]
        if conflicting_jobs is None:
            matching_jobs = self.store.find_jobs_by_launch_identity(
                name=payload.job_name,
                config_path=str(remote_config_path),
                remote_host=remote_host,
                remote_cwd=remote_cwd,
            )
            conflicting_jobs = [job for job in matching_jobs if infer_job_kind(job) not in {"sweep"}]
        queue_group = queue_group or self._queue_group_for(remote_host=remote_host, remote_cwd=remote_cwd, payload=payload)
        job = JobRecord(
            job_id=new_id("job", payload.job_name),
            name=payload.job_name,
            status=JobStatus.validating,
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            entity=entity,
            project=project,
            config_path=str(remote_config_path),
            remote_host=remote_host,
            remote_cwd=remote_cwd,
            conda_env=conda_env,
            monitor={
                "kind": "sweep",
                "stage": "validating",
                "queue": {
                    "queue_group": queue_group,
                    "queue_policy": payload.queue_policy,
                    "queue_after_job_id": payload.queue_after_job_id,
                    "started_from_queue": bool(queued_job),
                    "payload": payload.model_dump(mode="json"),
                },
                "launch_identity_conflicts": [
                    {
                        "job_id": conflict.job_id,
                        "kind": infer_job_kind(conflict),
                        "status": conflict.status.value,
                    }
                    for conflict in conflicting_jobs[:5]
                ],
            },
        )
        if queued_job:
            job = queued_job
            job.status = JobStatus.validating
            job.operation_id = operation_id
            job.idempotency_key = idempotency_key
            job.entity = entity
            job.project = project
            job.config_path = str(remote_config_path)
            job.remote_host = remote_host
            job.remote_cwd = remote_cwd
            job.conda_env = conda_env
            job.monitor.update({
                "kind": "sweep",
                "stage": "validating",
                "classification": "accepted",
                "queue": {
                    **(job.monitor.get("queue") if isinstance(job.monitor.get("queue"), dict) else {}),
                    "queue_group": queue_group,
                    "queue_policy": payload.queue_policy,
                    "queue_after_job_id": payload.queue_after_job_id,
                    "started_from_queue": True,
                    "started_at": utc_now(),
                    "payload": payload.model_dump(mode="json"),
                },
                "launch_identity_conflicts": [
                    {
                        "job_id": conflict.job_id,
                        "kind": infer_job_kind(conflict),
                        "status": conflict.status.value,
                    }
                    for conflict in conflicting_jobs[:5]
                ],
            })
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
                "launch_identity_conflicts": job.monitor.get("launch_identity_conflicts") or [],
                "next_actions": ["Console 正在执行 launch-sweep，稍后再用 status 查询进度。"],
            },
            retry_after_seconds=2,
        )
        try:
            launch_preflight = self._sweep_launch_preflight(
                remote_host=remote_host,
                remote_cwd=remote_cwd,
                remote_config_path=remote_config_path,
                conda_env=conda_env,
                conda_sh=conda_sh,
                profile=payload.profile,
            )
            validation = launch_preflight["validation"]
            entrypoint_probe = launch_preflight["entrypoint_probe"]
        except Exception as exc:
            return self._fail_launch_job(
                job,
                intent=IntentType.launch_sweep,
                operation_id=operation_id,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="validation_failed",
                classification="validation_failed",
                exc=exc,
                next_actions=["修复 sweep 配置或远端 config 路径后，再重新 launch-sweep。"],
            )
        job.monitor.update({"validation": validation, "stage": "validating"})
        self.store.upsert_job(job)
        if launch_preflight["classification"] != "preflight_ok":
            classification = "entrypoint_probe_failed"
            job.status = JobStatus.attention
            job.monitor.update({
                "validation": validation,
                "preflight": launch_preflight["preflight"],
                "stage": "preflight_failed",
                "classification": classification,
            })
            self.store.upsert_job(job)
            result = {
                "stage": "preflight_failed",
                "classification": classification,
                "created_new_sweep": False,
                "job_id": job.job_id,
                "validation": validation,
                "entrypoint_probe": entrypoint_probe,
                "preflight": job.monitor["preflight"],
                "next_actions": ["修复 sweep program 入口、导入路径或 --help CLI 后，再重新 launch-sweep。"],
            }
            self._record_operation(
                job,
                operation_id=operation_id,
                intent=IntentType.launch_sweep,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="preflight_failed",
                classification=classification,
                status=OperationStatus.failed.value,
                result=result,
            )
            return result, job
        try:
            preflight = self.ssh.preflight(
                host=remote_host,
                remote_cwd=remote_cwd,
                conda_env=conda_env,
                conda_sh=conda_sh,
                config_path=remote_config_path,
            )
            preflight["entrypoint_probe"] = entrypoint_probe
        except Exception as exc:
            return self._fail_launch_job(
                job,
                intent=IntentType.launch_sweep,
                operation_id=operation_id,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="preflight_failed",
                classification="preflight_incomplete",
                exc=exc,
                next_actions=["修复远端环境、路径或依赖后，再重新 launch-sweep。"],
                extra_result={"validation": validation},
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
        try:
            sweep = self._create_sweep(remote_config_path, entity=entity, project=project, payload=payload)
        except Exception as exc:
            return self._fail_launch_job(
                job,
                intent=IntentType.launch_sweep,
                operation_id=operation_id,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="create_sweep_failed",
                classification="control_plane_error",
                exc=exc,
                next_actions=["检查 W&B sweep 创建错误和远端 config 后，再重新 launch-sweep。"],
                extra_result={"validation": validation, "preflight": preflight},
            )
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
        try:
            gpu_probe = self.ssh.probe_gpus(remote_host)
            eligible = [gpu for gpu in gpu_probe["gpus"] if gpu["eligible"]]
            if payload.max_agents is not None:
                eligible = eligible[:payload.max_agents]
            launches = []
            sweep_path = f"{entity}/{project}/{job.sweep_id}"
            wandb_api_key = self._wandb_api_key()
            auth = self.ssh.auth_check(
                host=remote_host,
                remote_cwd=remote_cwd,
                sweep_path=sweep_path,
                wandb_api_key=wandb_api_key,
            )
            for gpu in eligible:
                launches.append(self.ssh.launch_agent(
                    host=remote_host,
                    remote_cwd=remote_cwd,
                    sweep_path=sweep_path,
                    gpu_index=gpu["index"],
                    conda_env=conda_env,
                    conda_sh=conda_sh,
                    wandb_api_key=wandb_api_key,
                ))
        except Exception as exc:
            return self._fail_launch_job(
                job,
                intent=IntentType.launch_sweep,
                operation_id=operation_id,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="agent_launch_failed",
                classification="control_plane_error",
                exc=exc,
                next_actions=["检查 GPU 探测、W&B auth 或 agent 启动错误；修复后使用 recover-agents 或重新 launch-sweep。"],
                status=JobStatus.attention,
                extra_result={"validation": validation, "preflight": preflight, "sweep": sweep},
            )
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
                "launch_identity_conflicts": job.monitor.get("launch_identity_conflicts") or [],
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
            "config_path": remote_config_path,
            "launch_identity_conflicts": job.monitor.get("launch_identity_conflicts") or [],
            "next_actions": next_actions_for_classification(classification),
        }, job

    def _advance_queue(
        self,
        payload: AdvanceQueuePayload,
        *,
        requested_by: str = "experiment-runner",
        operation_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        groups = [payload.queue_group] if payload.queue_group else self.store.queue_groups()
        advanced: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        unblocked: list[dict[str, Any]] = []
        idle: list[str] = []
        for queue_group in [str(group) for group in groups if group]:
            while True:
                blocker = self.store.active_queue_blocker(queue_group)
                if blocker:
                    assessment = self._assess_queue_blocker(blocker)
                    if payload.auto_unblock_stale and assessment["unblockable"]:
                        _, hygiene = self._ledger_only_cancel_job(
                            blocker,
                            requested_by=requested_by,
                            reason=f"Auto-unblocked stale queue blocker for {queue_group}.",
                            classification="metadata_corrupt_cancelled",
                            evidence=assessment["evidence"],
                        )
                        unblocked.append({"queue_group": queue_group, **hygiene})
                        continue
                    blocked.append({
                        "queue_group": queue_group,
                        "blocked_by_job_id": blocker.job_id,
                        "blocked_by_status": blocker.status.value,
                        "blocked_by_classification": assessment["classification"],
                        "blocker_classification": assessment["classification"],
                        "unblockable": assessment["unblockable"],
                        "recommended_action": assessment["recommended_action"],
                        "evidence": assessment["evidence"],
                    })
                    break
                queued = self.store.next_queued_job(queue_group)
                if not queued:
                    idle.append(queue_group)
                    break
                queue_meta = queued.monitor.get("queue") if isinstance(queued.monitor.get("queue"), dict) else {}
                raw_payload = queue_meta.get("payload")
                if not isinstance(raw_payload, dict):
                    queued.monitor.update({
                        "stage": "queue_failed",
                        "classification": "queued_payload_missing",
                        "error": "queued launch payload is missing",
                        "queue_hygiene": {
                            "unblocked_at": utc_now(),
                            "unblocked_by": requested_by,
                            "reason": "queued launch payload is missing",
                            "previous_status": queued.status.value,
                            "evidence": {"job_id": queued.job_id, "queue_group": queue_group},
                        },
                    })
                    queued.status = JobStatus.failed
                    self.store.upsert_job(queued)
                    unblocked.append({
                        "queue_group": queue_group,
                        "job_id": queued.job_id,
                        "previous_status": JobStatus.queued.value,
                        "status": JobStatus.failed.value,
                        "classification": "queued_payload_missing",
                        "ledger_only": True,
                        "reason": "queued launch payload is missing",
                        "evidence": {"job_id": queued.job_id, "queue_group": queue_group},
                    })
                    continue
                launch_payload = LaunchSweepPayload.model_validate({
                    **raw_payload,
                    "queue_group": queue_group,
                })
                launch_operation_id = operation_id or self._operation_identity(IntentType.launch_sweep, launch_payload.model_dump(mode="json"))[0]
                launch_idempotency_key = idempotency_key or self._operation_identity(IntentType.launch_sweep, launch_payload.model_dump(mode="json"))[1]
                result, started_job = self._launch_sweep_now(
                    launch_payload,
                    requested_by=requested_by,
                    operation_id=launch_operation_id,
                    idempotency_key=launch_idempotency_key,
                    queue_group=queue_group,
                    queued_job=queued,
                )
                advanced.append({
                    "queue_group": queue_group,
                    "job_id": started_job.job_id,
                    "classification": result.get("classification"),
                    "stage": result.get("stage"),
                    "created_new_sweep": result.get("created_new_sweep"),
                })
                break
        classification = "advanced" if advanced else "blocked" if blocked else "unblocked" if unblocked else "idle"
        return {
            "stage": "done",
            "classification": classification,
            "advanced": advanced,
            "blocked": blocked,
            "unblocked": unblocked,
            "idle": idle,
            "next_actions": [
                "Monitor any advanced job with status/watchdog-once.",
            ] if advanced else [
                "No queued sweep was started; inspect queue for blockers.",
            ],
        }

    def _create_sweep(self, remote_config_path: str, *, entity: str, project: str, payload: LaunchSweepPayload) -> dict[str, Any]:
        remote_host = payload.remote_host or self.settings.default_remote_host
        remote_cwd = payload.remote_cwd or self.settings.default_remote_cwd
        return self.ssh.create_sweep(
            host=remote_host,
            remote_cwd=remote_cwd,
            remote_config=remote_config_path,
            entity=entity,
            project=project,
            wandb_api_key=self._wandb_api_key(),
        )

    def _launch_run(self, payload: LaunchRunPayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> tuple[dict[str, Any], JobRecord]:
        entity = payload.entity or self.settings.default_entity
        project = payload.project or self.settings.default_project
        remote_host = payload.remote_host or self.settings.default_remote_host
        remote_cwd = payload.remote_cwd or self.settings.default_remote_cwd
        conda_env = payload.conda_env or self.settings.default_conda_env
        conda_sh = payload.conda_sh or self.settings.default_conda_sh
        remote_config_path = payload.config_path
        operation_id = operation_id or self._operation_identity(IntentType.launch_run, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.launch_run, payload.model_dump(mode="json"))[1]
        job = JobRecord(
            job_id=new_id("job", payload.job_name),
            name=payload.job_name,
            status=JobStatus.validating,
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            entity=entity,
            project=project,
            config_path=str(remote_config_path),
            remote_host=remote_host,
            remote_cwd=remote_cwd,
            conda_env=conda_env,
            monitor={"kind": "single_run", "stage": "validating"},
        )
        self.store.upsert_job(job)
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.launch_run,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="accepted",
            classification="accepted",
            status=OperationStatus.accepted.value,
            result={"stage": "accepted", "classification": "accepted", "created_new_sweep": False, "job_id": job.job_id},
            retry_after_seconds=2,
        )
        try:
            remote_snapshot = self.ssh.read_remote_file(host=remote_host, remote_path=remote_config_path)
            validation = validate_experiment_config_text(remote_snapshot["text"], "single-run", path_label=remote_config_path)
            command_spec = build_single_run_command(load_yaml_text(remote_snapshot["text"]))
        except Exception as exc:
            message = str(exc)
            classification = "run_sweep_boundary_error" if "single-run parameter" in message else "validation_failed"
            next_actions = SINGLE_RUN_BOUNDARY_NEXT_ACTIONS if classification == "run_sweep_boundary_error" else ["修复 single-run 配置或远端 config 路径后，再重新 launch-run。"]
            return self._fail_launch_job(
                job,
                intent=IntentType.launch_run,
                operation_id=operation_id,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="validation_failed",
                classification=classification,
                exc=exc,
                next_actions=next_actions,
            )
        job.monitor.update({"validation": validation, "run_command": command_spec, "stage": "validating"})
        self.store.upsert_job(job)
        try:
            preflight = self._single_run_preflight(
                remote_host=remote_host,
                remote_cwd=remote_cwd,
                remote_config_path=remote_config_path,
                conda_env=conda_env,
                conda_sh=conda_sh,
                profile="single-run",
                argv_probe=True,
                command_spec=command_spec,
                remote_snapshot=remote_snapshot,
            )
        except Exception as exc:
            return self._fail_launch_job(
                job,
                intent=IntentType.launch_run,
                operation_id=operation_id,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="preflight_failed",
                classification="preflight_incomplete",
                exc=exc,
                next_actions=["修复 single-run 远端环境、路径或依赖后，再重新 launch-run。"],
                extra_result={"validation": validation},
            )
        if preflight.get("classification") == "argv_incompatible":
            classification = "argv_incompatible"
            next_actions = ["修复 single-run 配置生成的 argv 或远端训练脚本 CLI 后，再重新 launch-run。"]
            job.status = JobStatus.attention
            job.monitor.update({
                "kind": "single_run",
                "validation": validation,
                "preflight": preflight,
                "stage": "preflight_failed",
                "classification": classification,
            })
            self.store.upsert_job(job)
            self._record_operation(
                job,
                operation_id=operation_id,
                intent=IntentType.launch_run,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="failed",
                classification=classification,
                status=OperationStatus.failed.value,
                result={
                    "stage": "failed",
                    "classification": classification,
                    "created_new_sweep": False,
                    "job_id": job.job_id,
                    "validation": validation,
                    "preflight": preflight,
                    "next_actions": next_actions,
                },
            )
            return {
                "stage": "failed",
                "classification": classification,
                "created_new_sweep": False,
                "job_id": job.job_id,
                "validation": validation,
                "preflight": preflight,
                "next_actions": next_actions,
            }, job
        try:
            gpu_probe = self.ssh.probe_gpus(remote_host)
            eligible = [gpu for gpu in gpu_probe.get("gpus", []) if gpu.get("eligible")]
            selected_gpu = None
            if payload.gpu_index is not None:
                selected_gpu = next((gpu for gpu in gpu_probe.get("gpus", []) if int(gpu.get("index")) == payload.gpu_index), None)
                if selected_gpu is None:
                    raise ValueError(f"gpu_index {payload.gpu_index} not found on {remote_host}")
                if payload.gpu_mode == "strict" and not selected_gpu.get("eligible"):
                    raise ValueError(f"gpu_index {payload.gpu_index} is not eligible")
            elif eligible:
                selected_gpu = eligible[0]
            elif payload.gpu_mode == "strict":
                raise ValueError("no eligible GPU for single-run launch")
            else:
                raise ValueError("no eligible GPU for single-run launch")
            auth = self.ssh.auth_check(
                host=remote_host,
                remote_cwd=remote_cwd,
                sweep_path=None,
                wandb_api_key=self._wandb_api_key(),
            )
        except Exception as exc:
            return self._fail_launch_job(
                job,
                intent=IntentType.launch_run,
                operation_id=operation_id,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="preflight_failed",
                classification="control_plane_error",
                exc=exc,
                next_actions=["检查 GPU 探测、W&B auth 或远端环境；修复后重新 launch-run。"],
                status=JobStatus.attention,
                extra_result={"validation": validation, "preflight": preflight},
            )
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.launch_run,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="preflight",
            classification="accepted",
            status=OperationStatus.executing.value,
            result={
                "stage": "preflight",
                "classification": "accepted",
                "validation": validation,
                "preflight": preflight,
                "gpu_probe": gpu_probe,
                "selected_gpu": selected_gpu.get("index"),
                "created_new_sweep": False,
            },
            retry_after_seconds=2,
        )
        try:
            launch = self.ssh.launch_run(
                host=remote_host,
                remote_cwd=remote_cwd,
                job_id=job.job_id,
                argv=command_spec["argv"],
                gpu_index=int(selected_gpu["index"]),
                conda_env=conda_env,
                conda_sh=conda_sh,
                wandb_api_key=self._wandb_api_key(),
                result_path=payload.result_path,
            )
        except Exception as exc:
            return self._fail_launch_job(
                job,
                intent=IntentType.launch_run,
                operation_id=operation_id,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="launch_failed",
                classification="control_plane_error",
                exc=exc,
                next_actions=["检查 single-run 远端启动日志和训练脚本 CLI；修复后重新 launch-run。"],
                status=JobStatus.attention,
                extra_result={"validation": validation, "preflight": preflight, "auth": auth, "gpu_probe": gpu_probe},
            )
        job.agent_pids = [launch["pid"]] if launch.get("pid") else []
        launch_status = single_run_status_from_launch(launch, job_id=job.job_id, alive_pids=job.agent_pids)
        single_run_state = classify_single_run_state(launch_status)
        job.status = single_run_state["job_status"]
        classification = single_run_state["classification"]
        job.monitor.update({
            "kind": "single_run",
            "validation": validation,
            "preflight": preflight,
            "auth": auth,
            "gpu_probe": gpu_probe,
            "run": launch,
            "last_run_status": compact_single_run_status(launch_status),
            "stage": "done",
            "classification": classification,
        })
        self.store.upsert_job(job)
        self._record_operation(
            job,
            operation_id=operation_id,
            intent=IntentType.launch_run,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            stage="done",
            classification=classification,
            status=OperationStatus.failed.value if job.status == JobStatus.failed else OperationStatus.succeeded.value if job.agent_pids else OperationStatus.executing.value,
            result={
                "stage": "done",
                "classification": classification,
                "created_new_sweep": False,
                "job_id": job.job_id,
                "validation": validation,
                "preflight": preflight,
                "auth": auth,
                "gpu_probe": gpu_probe,
                "run": launch,
                "next_actions": ["使用 status/watchdog-once 监控单次运行；完成后用 pull-results 拉取结果。"],
            },
            retry_after_seconds=2,
        )
        return {
            "stage": "done",
            "classification": classification,
            "created_new_sweep": False,
            "job_id": job.job_id,
            "validation": validation,
            "preflight": preflight,
            "auth": auth,
            "gpu_probe": gpu_probe,
            "run": launch,
            "next_actions": ["使用 status/watchdog-once 监控单次运行；完成后用 pull-results 拉取结果。"],
        }, job

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
        result_snapshot = job.monitor.get("last_result_snapshot") if job else None
        result_pull = job.monitor.get("last_result_pull") if job else None
        result_for_readiness = result_snapshot or result_pull
        queue_state = self._queue_position(job) if job else None
        run_status = None
        if job and job.monitor.get("kind") == "single_run":
            run_meta = job.monitor.get("run") if isinstance(job.monitor.get("run"), dict) else {}
            status_path = run_meta.get("status_path")
            if status_path and job.remote_host:
                try:
                    run_status = self.ssh.check_run_status(host=job.remote_host, status_path=status_path, pids=job.agent_pids)
                    single_run_state = classify_single_run_state(run_status)
                    next_status = single_run_state["job_status"]
                    try:
                        job = self.store.update_job_status(job.job_id, next_status, {"last_run_status": compact_single_run_status(run_status), "classification": single_run_state["classification"]})
                    except Exception:
                        job.monitor["last_run_status"] = compact_single_run_status(run_status)
                        job.monitor["classification"] = single_run_state["classification"]
                        self.store.upsert_job(job)
                except Exception as exc:
                    run_status = {"classification": "run_status_unavailable", "error": compact_error(exc)}
        sweep = None
        degraded = None
        if sweep_id:
            try:
                sweep = self.wandb.get_sweep_state(entity, project, sweep_id)
            except Exception as exc:
                degraded = str(exc)
        if job and sweep:
            next_status = map_wandb_state_to_job_status(sweep.get("state"))
            sweep = self._enrich_single_sweep_with_telemetry(sweep)
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
            if job.monitor.get("kind") == "single_run" and isinstance(job.monitor.get("run"), dict):
                agent_launches = [job.monitor["run"]]
        if not agent_launches and job and job.agent_pids:
            agent_launches = [{"pid": pid, "state": "unknown"} for pid in job.agent_pids]
        sweep_path = None
        if job and job.entity and job.project and job.sweep_id:
            sweep_path = str(job.monitor.get("sweep_path") or f"{job.entity}/{job.project}/{job.sweep_id}")
        agent_probe = None
        if job and job.status != JobStatus.queued and job.monitor.get("kind") != "single_run" and job.remote_host and (job.agent_pids or sweep_path):
            try:
                agent_probe = self.ssh.check_agent_processes(
                    host=job.remote_host,
                    sweep_path=sweep_path,
                    pids=job.agent_pids,
                )
                job.monitor["last_agent_probe"] = compact_agent_probe(agent_probe)
                self.store.upsert_job(job)
            except Exception as exc:
                agent_probe = {"classification": "agent_probe_unavailable", "error": compact_error(exc)}
        agent_health = "unknown"
        if job and job.monitor.get("kind") == "single_run" and isinstance(run_status, dict):
            alive = run_status.get("alive_pids") or []
            if run_status.get("exit_code") == 0:
                agent_health = "terminal"
            elif run_status.get("exit_code") is not None:
                agent_health = "failed"
            elif alive:
                agent_health = "running"
            elif run_status.get("missing"):
                agent_health = "missing"
            else:
                agent_health = "unknown"
        elif job and isinstance(agent_probe, dict) and not agent_probe.get("classification"):
            if agent_probe.get("alive_pids") or agent_probe.get("pgrep"):
                agent_health = "running"
            elif job.status == JobStatus.running and agent_launches:
                agent_health = "missing"
            elif job.status == JobStatus.running and sweep and str(sweep.get("state") or "").lower() == "running":
                agent_health = "running"
            elif job.status in TERMINAL_JOB_STATUSES:
                agent_health = "terminal"
            else:
                agent_health = "unknown"
        elif job and job.status == JobStatus.running:
            agent_health = "running" if agent_launches or (sweep and str(sweep.get("state") or "").lower() == "running") else "missing"
        elif job and job.status in TERMINAL_JOB_STATUSES:
            agent_health = "terminal"
        elif job and job.status == JobStatus.queued:
            agent_health = "queued"
        sweep_attention = False
        sweep_attention_reasons: list[str] = []
        if job and job.monitor.get("kind") != "single_run" and sweep:
            sweep_attention, sweep_attention_reasons = sweep_requires_attention(sweep, result_for_readiness)
            if sweep_attention and job.status == JobStatus.running:
                try:
                    job = self.store.update_job_status(job.job_id, JobStatus.attention, {"sweep_attention_reasons": sweep_attention_reasons})
                except Exception:
                    job.status = JobStatus.attention
                    job.monitor["sweep_attention_reasons"] = sweep_attention_reasons
                    self.store.upsert_job(job)
        failure_diagnostics = None
        if should_collect_failure_diagnostics(job, sweep, agent_health, degraded, sweep_attention=sweep_attention):
            failure_diagnostics = self._collect_failure_diagnostics(
                job=job,
                agent_launches=agent_launches,
                sweep_path=sweep_path,
            )
            if failure_diagnostics:
                job.monitor["last_failure_diagnostics"] = compact_failure_diagnostics(failure_diagnostics)
                self.store.upsert_job(job)
        elif job and isinstance(job.monitor.get("last_failure_diagnostics"), dict):
            failure_diagnostics = job.monitor["last_failure_diagnostics"]
        result_readiness = classify_result_readiness(result_for_readiness, sweep.get("state") if sweep else None)
        classification = "ok"
        next_actions: list[str] = []
        if missing_requested_job:
            classification = "job_not_found"
            next_actions = [
                "使用 register-existing-sweep 将已有 W&B sweep 登记到 Console ledger，或确认 job_id 是否来自当前 state_dir。",
            ]
        elif job and job.status == JobStatus.queued:
            classification = "queued"
            blocker_classification = queue_state.get("blocker_classification") if isinstance(queue_state, dict) else None
            if blocker_classification == "metadata_corrupt_blocker":
                next_actions = [
                    "Queued only: W&B sweep has not been created. The blocker has corrupt ledger metadata; run advance-queue with auto_unblock_stale=true or stop-job with ledger_only=true.",
                ]
            elif blocker_classification == "active_real_blocker":
                next_actions = [
                    "Queued only: W&B sweep has not been created. Wait for the active blocker to finish, or monitor it with status/watchdog-once.",
                ]
            elif blocker_classification == "ambiguous_blocker":
                next_actions = [
                    "Queued only: W&B sweep has not been created. Run status/watchdog-once for the blocker before deciding whether ledger-only cancellation is safe.",
                ]
            else:
                next_actions = ["Queued only: W&B sweep has not been created. Wait for the blocker to finish, then run advance-queue."]
        elif degraded:
            classification = "degraded"
            next_actions = ["稍后重试 status；如持续 degraded，检查 W&B/API 网络与本地 WANDB_API_KEY。"]
        elif job and (job.status in {JobStatus.attention, JobStatus.failed} or agent_health in {"failed", "missing"}):
            classification = "attention"
            if job.monitor.get("kind") == "single_run":
                next_actions = ["检查 single-run 的 last_run_status、failure_diagnostics 和远端日志；修复配置 argv 或训练脚本后重新 launch-run。"]
            else:
                next_actions = ["检查 agent 诊断；修复远端代码/路径后重新 launch 或 recover-agents。"]
        elif sweep_attention:
            classification = "attention"
            next_actions = sweep_attention_reasons or ["检查 run 级失败诊断；如需要，修复远端代码/路径后重新 launch 或 recover-agents。"]
        consistency_warnings = sweep_consistency_warnings(sweep) if sweep else []
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
                "probe": compact_agent_probe(agent_probe) if isinstance(agent_probe, dict) else None,
            },
            "results": {
                "readiness": result_readiness,
                "last_pull": result_pull,
                "last_snapshot": result_snapshot,
            },
            "queue": queue_state,
            "failure_diagnostics": failure_diagnostics,
            "state": {
                "job_status": job.status.value if job else None,
                "wandb_sweep_status": sweep.get("state") if sweep else None,
                "run_status": run_status,
                "agent_health": agent_health,
                "result_readiness": result_readiness,
                "sweep_attention": sweep_attention,
                "sweep_attention_reasons": sweep_attention_reasons,
                "consistency_warnings": consistency_warnings,
                "queue": queue_state,
            },
            "degraded": degraded,
            "next_actions": next_actions,
            "generated_at": utc_now(),
        }

    def _stop(self, payload: StopJobPayload, *, requested_by: str = "experiment-runner", operation_id: str | None = None, idempotency_key: str | None = None) -> tuple[dict[str, Any], JobRecord]:
        job = self._require_job(payload.job_id)
        is_single_run = job.monitor.get("kind") == "single_run"
        is_queued = job.status == JobStatus.queued
        metadata_missing = False
        metadata_evidence: dict[str, Any] = {"job_id": job.job_id, "job_status": job.status.value, "kind": infer_job_kind(job), "missing": []}
        if not is_single_run and not is_queued:
            missing = [
                key
                for key, value in {
                    "sweep_id": job.sweep_id,
                    "entity": job.entity,
                    "project": job.project,
                    "remote_host": job.remote_host,
                }.items()
                if not value
            ]
            metadata_missing = bool(missing)
            metadata_evidence["missing"] = missing
        if is_single_run and not job.remote_host:
            metadata_missing = True
            metadata_evidence["missing"] = ["remote_host"]
        if metadata_missing and not payload.ledger_only:
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
        if payload.ledger_only and not is_queued:
            classification = "metadata_corrupt_cancelled" if metadata_missing else "ledger_stale_cancelled"
            job, hygiene = self._ledger_only_cancel_job(
                job,
                requested_by=requested_by,
                reason=payload.reason,
                classification=classification,
                evidence=metadata_evidence,
            )
            result = {
                "stage": "done",
                "classification": classification,
                "job_id": job.job_id,
                "cancel_wandb_requested": payload.cancel_wandb,
                "cancel_wandb_implemented": False,
                "ledger_only": True,
                "remote_side_effects": False,
                "queue_hygiene": job.monitor.get("queue_hygiene"),
                **hygiene,
            }
            self._record_operation(
                job,
                operation_id=operation_id,
                intent=IntentType.stop_job,
                requested_by=requested_by,
                idempotency_key=idempotency_key,
                stage="done",
                classification=classification,
                status=OperationStatus.succeeded.value,
                result=result,
            )
            return result, job
        result = {"cancel_wandb_requested": payload.cancel_wandb, "cancel_wandb_implemented": False}
        if is_queued:
            result["queued_cancelled"] = True
            result["note"] = "Queued job had no W&B sweep or agents."
        elif payload.kill_agents:
            if is_single_run:
                result["stop_run"] = self.ssh.stop_pids(host=job.remote_host, pids=job.agent_pids)
            else:
                result["stop_agents"] = self.ssh.stop_agents(host=job.remote_host, sweep_path=f"{job.entity}/{job.project}/{job.sweep_id}", pids=job.agent_pids)
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
        return {"stage": "done", "classification": "job_cancelled", "job_id": job.job_id, **result}, job

    def _collect_failure_diagnostics(
        self,
        *,
        job: JobRecord | None,
        agent_launches: list[dict[str, Any]],
        sweep_path: str | None,
    ) -> dict[str, Any] | None:
        if not job or not job.remote_host or not job.remote_cwd:
            return None
        if not agent_launches and not job.agent_pids:
            return None
        try:
            diagnostics = self.ssh.diagnose_agent_failure(
                host=job.remote_host,
                remote_cwd=job.remote_cwd,
                launches=agent_launches,
                pids=job.agent_pids,
                sweep_path=sweep_path,
                tail_lines=200,
            )
            diagnostics["collected_at"] = utc_now()
            return diagnostics
        except Exception as exc:
            return {
                "classification": "diagnostics_unavailable",
                "summary": "Failed to collect remote agent diagnostics.",
                "error": compact_error(exc),
                "collected_at": utc_now(),
                "next_actions": ["确认远端 host/cwd 可达后重试 status/watchdog-once。"],
            }

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
        sweep_attention = bool(state.get("sweep_attention"))
        sweep_attention_reasons = state.get("sweep_attention_reasons") or []
        run_count = sweep.get("runCount")
        expected = sweep.get("expectedRunCount")
        is_terminal = job_status in {"finished", "failed", "cancelled"} or sweep_state in {"finished", "failed", "cancelled"}
        is_attention = bool(degraded) or sweep_attention or job_status in {"attention", "failed"} or sweep_state in {"failed", "crashed", "killed"} or agent_health in {"missing", "degraded"}
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
            reason_text = "; ".join(str(item) for item in sweep_attention_reasons) if sweep_attention_reasons else degraded or "-"
            message = f"job {payload.job_id} needs attention: job={job_status}, sweep={sweep_state or '-'}, degraded={reason_text}"
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
            "sweep_attention": sweep_attention,
            "sweep_attention_reasons": sweep_attention_reasons,
            "terminal_disable_requested": payload.terminal_disable,
            "operation": status_result.get("operation"),
            "operation_history": status_result.get("operation_history"),
            "failure_diagnostics": status_result.get("failure_diagnostics"),
            "status_result": status_result,
        }
        if is_terminal:
            queue_group = ((status_result.get("queue") or {}).get("queue_group") if isinstance(status_result.get("queue"), dict) else None)
            if queue_group:
                result["queue_advance"] = self._advance_queue(
                    AdvanceQueuePayload(queue_group=queue_group),
                    requested_by=requested_by,
                )
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
        conda_sh = payload.conda_sh or self.settings.default_conda_sh
        if payload.profile == "single-run":
            return self._single_run_preflight(
                remote_host=payload.remote_host,
                remote_cwd=payload.remote_cwd,
                remote_config_path=payload.config_path,
                conda_env=conda_env,
                conda_sh=conda_sh,
                profile=payload.profile,
                argv_probe=payload.argv_probe,
            )
        result = self.ssh.preflight(
            host=payload.remote_host,
            remote_cwd=payload.remote_cwd,
            conda_env=conda_env,
            conda_sh=conda_sh,
            config_path=payload.config_path,
        )
        if payload.config_path:
            try:
                snapshot = self.ssh.read_remote_file(host=payload.remote_host, remote_path=payload.config_path)
                result["remote_config_snapshot"] = snapshot
                entrypoint_probe = self._sweep_entrypoint_probe(
                    remote_config_text=snapshot["text"],
                    remote_host=payload.remote_host,
                    remote_cwd=payload.remote_cwd,
                    conda_env=conda_env,
                    conda_sh=conda_sh,
                )
                result["entrypoint_probe"] = entrypoint_probe
                if entrypoint_probe.get("classification") != "argv_compatible":
                    result["classification"] = "entrypoint_probe_failed"
                    result["ok"] = False
            except Exception as exc:
                result["remote_config_snapshot_error"] = compact_error(exc)
        return {**result, "provenance": {"source": "ssh_remote_preflight"}}

    def _auth_check(self, payload: AuthCheckPayload) -> dict[str, Any]:
        job = self.store.get_job(payload.job_id) if payload.job_id else None
        entity = payload.entity or (job.entity if job else None) or self.settings.default_entity
        project = payload.project or (job.project if job else None) or self.settings.default_project
        sweep_id = payload.sweep_id or (job.sweep_id if job else None)
        remote_host = payload.remote_host or (job.remote_host if job else None) or self.settings.default_remote_host
        remote_cwd = payload.remote_cwd or (job.remote_cwd if job else None) or self.settings.default_remote_cwd
        if not remote_host:
            remote_host = self.settings.default_remote_host
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
        remote_host = payload.remote_host or (job.remote_host if job else None) or self.settings.default_remote_host
        remote_cwd = payload.remote_cwd or (job.remote_cwd if job else None) or self.settings.default_remote_cwd
        is_single_run = bool(job and job.monitor.get("kind") == "single_run")
        run_meta = job.monitor.get("run") if job and isinstance(job.monitor.get("run"), dict) else {}
        if not sweep_id:
            if not is_single_run:
                raise ValueError("sweep_id is required for pull-results")
        operation_id = operation_id or self._operation_identity(IntentType.pull_results, payload.model_dump(mode="json"))[0]
        idempotency_key = idempotency_key or self._operation_identity(IntentType.pull_results, payload.model_dump(mode="json"))[1]
        metric_labels = [*payload.metric_keys, *[selector_label(item) for item in payload.metric_paths]]
        group_labels = [*payload.group_keys, *[selector_label(item) for item in payload.group_paths]]
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

        def finalize_result(
            result: dict[str, Any],
            *,
            expected_runs: int | None,
            discovered_runs: int | None,
            requested_limit: int | None,
            run_id_source: str | None = None,
            operation_status: str = OperationStatus.succeeded.value,
        ) -> dict[str, Any]:
            enriched, snapshot_summary = self._materialize_result_snapshot(
                job=job,
                entity=entity,
                project=project,
                sweep_id=sweep_id,
                result={"stage": result.get("stage", "done"), "entity": entity, "project": project, "sweep_id": sweep_id, **result},
                group_keys=group_labels,
                metric_keys=metric_labels,
                expected_runs=expected_runs,
                discovered_runs=discovered_runs,
                requested_limit=requested_limit,
                run_id_source=run_id_source,
            )
            if job:
                job.monitor["last_result_pull"] = summarize_result_pull(enriched)
                job.monitor["last_result_snapshot"] = snapshot_summary
                self._record_operation(
                    job,
                    operation_id=operation_id,
                    intent=IntentType.pull_results,
                    requested_by=requested_by,
                    idempotency_key=idempotency_key,
                    stage=enriched.get("stage", "done"),
                    classification=enriched["classification"],
                    status=operation_status,
                    result={
                        "stage": enriched.get("stage", "done"),
                        "classification": enriched["classification"],
                        "entity": entity,
                        "project": project,
                        "sweep_id": sweep_id,
                        "snapshot": snapshot_summary,
                        "expected_runs": enriched.get("expected_runs"),
                        "discovered_runs": enriched.get("discovered_runs"),
                        "requested_limit": enriched.get("requested_limit"),
                        "fetched_runs": enriched.get("fetched_runs"),
                        "valid_runs": enriched.get("valid_runs"),
                        "missing_runs": enriched.get("missing_runs"),
                        "failed_runs": enriched.get("failed_runs"),
                        "truncated": enriched.get("truncated"),
                        "complete": enriched.get("complete"),
                        "readiness": enriched.get("readiness"),
                    },
                )
                self.store.upsert_job(job)
            return enriched

        if is_single_run:
            if not remote_host:
                raise ValueError("remote_host is required for single-run pull-results")
            status_path = run_meta.get("status_path")
            result_path = run_meta.get("result_path")
            recovered_paths = {}
            if not status_path:
                if job and job.remote_cwd and job.job_id:
                    recovered_paths = single_run_default_paths(job.remote_cwd, job.job_id)
                    status_path = recovered_paths["status_path"]
                    result_path = result_path or recovered_paths["result_path"]
                else:
                    raise ValueError("single-run status_path is missing")
            result = self.ssh.pull_single_run_result(
                host=remote_host,
                status_path=status_path,
                result_path=result_path,
                metric_keys=payload.metric_keys,
                group_keys=payload.group_keys,
            )
            if job:
                if recovered_paths:
                    run_meta = dict(run_meta)
                    run_meta.update({
                        "host": remote_host,
                        "remote_cwd": job.remote_cwd,
                        "job_id": job.job_id,
                        "log": recovered_paths["log_path"],
                        "status_path": status_path,
                        "result_path": result_path,
                    })
                    job.monitor["run"] = run_meta
            return finalize_result(result, expected_runs=1, discovered_runs=1, requested_limit=payload.max_runs)
        if not remote_host or not remote_cwd:
            if payload.allow_partial:
                result = self._pull_results_from_wandb(entity, project, sweep_id, payload)
                return finalize_result(
                    result,
                    expected_runs=result.get("expected_runs"),
                    discovered_runs=result.get("discovered_runs"),
                    requested_limit=payload.max_runs,
                    run_id_source="wandb_api_fallback",
                )
            raise ValueError("remote_host and remote_cwd are required for remote result pullback")
        try:
            try:
                sweep = self.wandb.get_sweep_state(entity, project, sweep_id)
                all_run_ids = [
                    str(run.get("name"))
                    for run in (sweep.get("runs") or [])
                    if run.get("name")
                ]
                expected_runs = to_optional_positive_int(sweep.get("expectedRunCount")) or to_optional_positive_int(sweep.get("runCount"))
                discovered_runs = len(all_run_ids)
                run_limit = payload.max_runs if payload.max_runs is not None else len(all_run_ids)
                run_ids = all_run_ids[:run_limit] if run_limit else all_run_ids
                run_id_source = "wandb_api"
            except Exception:
                run_ids = cached_run_ids(job, payload.max_runs) if job else []
                expected_runs = None
                discovered_runs = len(run_ids) if run_ids else None
                run_id_source = "cached_wandb_status"
            ssh_max_runs = payload.max_runs if payload.max_runs is not None else max(len(run_ids), 1)
            result = self.ssh.pull_results(
                host=remote_host,
                remote_cwd=remote_cwd,
                sweep_id=sweep_id,
                run_ids=run_ids,
                budget_seconds=payload.budget_seconds,
                max_runs=ssh_max_runs,
                metric_keys=payload.metric_keys,
                group_keys=payload.group_keys,
                metric_paths=payload.metric_paths,
                group_paths=payload.group_paths,
                output_globs=payload.output_globs,
            )
            result["run_id_source"] = run_id_source
            return finalize_result(
                result,
                expected_runs=expected_runs,
                discovered_runs=discovered_runs,
                requested_limit=payload.max_runs,
                run_id_source=run_id_source,
            )
        except Exception as exc:
            if not payload.allow_partial:
                raise
            try:
                fallback = self._pull_results_from_wandb(entity, project, sweep_id, payload)
                fallback["remote_error"] = compact_error(exc)
                fallback["classification"] = "remote_pull_failed_wandb_fallback"
                return finalize_result(
                    fallback,
                    expected_runs=fallback.get("expected_runs"),
                    discovered_runs=fallback.get("discovered_runs"),
                    requested_limit=payload.max_runs,
                    run_id_source="wandb_api_fallback",
                )
            except Exception as fallback_exc:
                degraded = {
                    "stage": "degraded",
                    "classification": "result_sources_unavailable",
                    "source": "degraded_empty_partial",
                    "entity": entity,
                    "project": project,
                    "sweep_id": sweep_id,
                    "rows": [],
                    "valid_results": 0,
                    "missing_results": 0,
                    "failed_results": 0,
                    "partial": True,
                    "remote_error": compact_error(exc),
                    "wandb_error": compact_error(fallback_exc),
                    "next_actions": ["确认远端 host/cwd 可达，或配置 WANDB_API_KEY 后重试。"],
                }
                if job:
                    return finalize_result(
                        degraded,
                        expected_runs=None,
                        discovered_runs=None,
                        requested_limit=payload.max_runs,
                        operation_status=OperationStatus.failed.value,
                    )
                return finalize_result(
                    degraded,
                    expected_runs=None,
                    discovered_runs=None,
                    requested_limit=payload.max_runs,
                    operation_status=OperationStatus.failed.value,
                )

    def _pull_results_from_wandb(self, entity: str, project: str, sweep_id: str, payload: PullResultsPayload) -> dict[str, Any]:
        sweep = self.wandb.get_sweep_state(entity, project, sweep_id)
        runs = sweep.get("runs") or []
        if payload.max_runs is not None:
            runs = runs[:payload.max_runs]
        rows = []
        for run in runs:
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
            "failed_results": sum(1 for row in rows if str(row.get("state") or "").lower() in {"failed", "crashed", "killed"}),
            "partial": True,
            "expected_runs": to_optional_positive_int(sweep.get("expectedRunCount")) or to_optional_positive_int(sweep.get("runCount")),
            "discovered_runs": len(sweep.get("runs") or []),
            "truncated": bool(payload.max_runs is not None and len(sweep.get("runs") or []) > payload.max_runs),
        }

    def _sweep_run_ids(self, entity: str, project: str, sweep_id: str, max_runs: int | None) -> list[str]:
        sweep = self.wandb.get_sweep_state(entity, project, sweep_id)
        run_ids = []
        runs = sweep.get("runs") or []
        if max_runs is not None:
            runs = runs[:max_runs]
        for run in runs:
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
            sweeps = self.wandb.discover_sweeps(self.settings.default_entity, self.settings.default_project, include_runs=True)
            self.settings.sweeps_cache_path.write_text(json.dumps(sweeps, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            sweeps = self._enrich_sweeps_with_telemetry(sweeps)
        except Exception as exc:
            degraded = str(exc)
            if self.settings.sweeps_cache_path.exists():
                sweeps = json.loads(self.settings.sweeps_cache_path.read_text(encoding="utf-8"))
                sweeps = self._enrich_sweeps_with_telemetry(sweeps)
        return {
            "status": "degraded" if degraded else "ok",
            "degraded": degraded,
            "job_counts": count_jobs(jobs),
            "jobs": [job.model_dump(mode="json") for job in jobs[:20]],
            "sweeps": [strip_runs(sweep) for sweep in sweeps[:50]],
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

    def _single_run_preflight(
        self,
        *,
        remote_host: str,
        remote_cwd: str,
        remote_config_path: str | None,
        conda_env: str | None,
        conda_sh: str | None,
        profile: str | None = None,
        argv_probe: bool = True,
        command_spec: dict[str, Any] | None = None,
        remote_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self.ssh.preflight(
            host=remote_host,
            remote_cwd=remote_cwd,
            conda_env=conda_env,
            conda_sh=conda_sh,
            config_path=remote_config_path,
        )
        snapshot = remote_snapshot
        if remote_config_path and snapshot is None:
            try:
                snapshot = self.ssh.read_remote_file(host=remote_host, remote_path=remote_config_path)
                result["remote_config_snapshot"] = snapshot
            except Exception as exc:
                result["remote_config_snapshot_error"] = compact_error(exc)
        elif snapshot is not None:
            result["remote_config_snapshot"] = snapshot
        if profile == "single-run" and argv_probe and remote_config_path:
            try:
                spec = command_spec or build_single_run_command(load_yaml_text((snapshot or {}).get("text", "")))
                probe = self.ssh.probe_argv_compat(
                    host=remote_host,
                    remote_cwd=remote_cwd,
                    argv=spec["argv"],
                    conda_env=conda_env,
                    conda_sh=conda_sh,
                )
                result["argv_probe"] = probe
                if probe.get("classification") == "argv_incompatible":
                    result["classification"] = "argv_incompatible"
                    result["ok"] = False
                elif probe.get("classification") == "argv_probe_unavailable":
                    result.setdefault("warnings", []).append("argv probe unavailable; launch is not blocked by this soft signal")
            except Exception as exc:
                result["argv_probe"] = {
                    "classification": "argv_probe_unavailable",
                    "error": compact_error(exc),
                }
                result.setdefault("warnings", []).append("argv probe unavailable; launch is not blocked by this soft signal")
        result["provenance"] = {"source": "ssh_remote_preflight"}
        return result


def map_wandb_state_to_job_status(state: str | None) -> JobStatus:
    normalized = (state or "").lower()
    if normalized == "finished":
        return JobStatus.finished
    if normalized in {"failed", "crashed", "killed"}:
        return JobStatus.failed
    if normalized in {"running", "pending"}:
        return JobStatus.running
    return JobStatus.unknown


def map_single_run_state_to_job_status(status: dict[str, Any] | None) -> JobStatus:
    return classify_single_run_state(status)["job_status"]


def single_run_default_paths(remote_cwd: str, job_id: str) -> dict[str, str]:
    runs_dir = f"{remote_cwd.rstrip('/')}/.experiment_console/runs"
    return {
        "log_path": f"{runs_dir}/{job_id}.log",
        "status_path": f"{runs_dir}/{job_id}.status.json",
        "result_path": f"{runs_dir}/{job_id}.result.json",
    }


def single_run_status_from_launch(launch: dict[str, Any], *, job_id: str, alive_pids: list[str] | None = None) -> dict[str, Any]:
    status = dict(launch.get("launcher") or {})
    status.setdefault("job_id", launch.get("job_id") or job_id)
    if launch.get("pid"):
        status.setdefault("child_pid", launch.get("pid"))
    if launch.get("status_path"):
        status.setdefault("status_path", launch.get("status_path"))
    if launch.get("result_path"):
        status.setdefault("result_path", launch.get("result_path"))
    if launch.get("log"):
        status.setdefault("log_path", launch.get("log"))
    if launch.get("launcher_pid") and not status.get("launcher_pid"):
        status["launcher_pid"] = launch.get("launcher_pid")
    status["alive_pids"] = list(alive_pids or [])
    return status


def classify_single_run_state(status: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(status, dict):
        return {"job_status": JobStatus.unknown, "classification": "run_status_unknown"}
    if status.get("exit_code") == 0:
        return {"job_status": JobStatus.finished, "classification": "run_finished"}
    if status.get("exit_code") is not None:
        return {"job_status": JobStatus.failed, "classification": "run_failed"}
    if status.get("alive_pids"):
        return {"job_status": JobStatus.running, "classification": "run_running"}
    if status.get("missing"):
        return {"job_status": JobStatus.attention, "classification": "run_status_missing"}
    if status.get("child_pid") or status.get("pid"):
        return {"job_status": JobStatus.attention, "classification": "run_started_unverified"}
    if status.get("launcher_pid") or status.get("status_path") or status.get("result_path"):
        return {"job_status": JobStatus.attention, "classification": "run_started_unverified"}
    return {"job_status": JobStatus.unknown, "classification": "run_status_unknown"}


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
        "run_id": result.get("status", {}).get("job_id") if isinstance(result.get("status"), dict) else None,
        "snapshot_id": result.get("snapshot_id"),
        "snapshot_path": result.get("snapshot_path"),
        "classification": result.get("classification"),
        "readiness": result.get("readiness"),
        "expected_runs": result.get("expected_runs"),
        "discovered_runs": result.get("discovered_runs"),
        "requested_limit": result.get("requested_limit"),
        "fetched_runs": result.get("fetched_runs"),
        "valid_runs": result.get("valid_runs"),
        "missing_runs": result.get("missing_runs"),
        "failed_runs": result.get("failed_runs"),
        "valid_results": result.get("valid_results"),
        "missing_results": result.get("missing_results"),
        "failed_results": result.get("failed_results"),
        "complete": result.get("complete"),
        "truncated": result.get("truncated"),
        "partial": result.get("partial"),
        "metric_summaries": result.get("metric_summaries"),
        "generated_at": utc_now(),
    }


def compact_single_run_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        key: status.get(key)
        for key in ["job_id", "pid", "child_pid", "launcher_pid", "ok", "timed_out", "started_at", "finished_at", "exit_code", "log_path", "status_path", "result_path", "alive_pids", "missing"]
        if key in status
    }


def should_collect_failure_diagnostics(
    job: JobRecord | None,
    sweep: dict[str, Any] | None,
    agent_health: str,
    degraded: str | None,
    *,
    sweep_attention: bool = False,
) -> bool:
    if not job:
        return False
    if degraded:
        return False
    if sweep_attention:
        return True
    if job.monitor.get("kind") == "single_run":
        return agent_health in {"failed", "missing"}
    sweep_state = str((sweep or {}).get("state") or "").lower()
    if job.status in {JobStatus.attention, JobStatus.failed}:
        return True
    if sweep_state in {"failed", "crashed", "killed"}:
        return True
    return agent_health in {"missing", "failed", "degraded"}


def sweep_requires_attention(sweep: dict[str, Any], result_pull: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    state = str(sweep.get("state") or "").lower()
    finished = to_non_negative_int(sweep.get("finished_runs"))
    failed = to_non_negative_int(sweep.get("failed_runs"))
    running = to_non_negative_int(sweep.get("running_runs"))
    expected = to_non_negative_int(sweep.get("expectedRunCount"))
    if state not in {"running", "pending"}:
        return False, []
    if failed <= 0:
        return False, []
    if running <= 0 and failed >= max(1, finished):
        return True, [
            "W&B sweep 仍显示 running，但 run 级失败已占主导；检查远端训练脚本/导入路径。",
            "先修复 VecGAD/依赖/路径问题，再重新 launch 或 recover-agents。",
        ]
    if result_pull and result_pull.get("valid_results", 0) == 0 and failed >= 1:
        return True, [
            "W&B sweep 仍显示 running，但当前没有任何有效结果而且已有失败 run。",
            "修复远端代码/环境后再继续。",
        ]
    if expected > 0 and failed >= max(3, expected // 2) and result_pull and result_pull.get("valid_results", 0) == 0:
        return True, [
            "W&B sweep 连续失败且没有有效结果，建议停止当前 sweep。",
        ]
    return False, []


def to_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except Exception:
        return 0


def to_optional_positive_int(value: Any) -> int | None:
    parsed = to_non_negative_int(value)
    return parsed or None


def compact_agent_probe(probe: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(probe, dict):
        return None
    return {
        key: probe.get(key)
        for key in ["host", "sweep_path", "tracked_pids", "alive_pids", "pgrep", "classification", "error"]
        if key in probe
    }


def compact_failure_diagnostics(diagnostics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(diagnostics, dict):
        return None
    log_tails = []
    for item in diagnostics.get("log_tails") or []:
        if not isinstance(item, dict):
            continue
        tail = str(item.get("tail") or "")
        log_tails.append({
            "gpu_index": item.get("gpu_index"),
            "pid": item.get("pid"),
            "path": item.get("path"),
            "exists": item.get("exists"),
            "tail": tail[-12000:],
        })
    return {
        "classification": diagnostics.get("classification"),
        "summary": diagnostics.get("summary"),
        "host": diagnostics.get("host"),
        "remote_cwd": diagnostics.get("remote_cwd"),
        "sweep_path": diagnostics.get("sweep_path"),
        "sources": list(diagnostics.get("sources") or []),
        "pid_state": diagnostics.get("pid_state"),
        "error_signals": list(diagnostics.get("error_signals") or [])[:20],
        "log_tails": log_tails[:8],
        "collected_at": diagnostics.get("collected_at"),
        "error": diagnostics.get("error"),
        "next_actions": list(diagnostics.get("next_actions") or []),
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
        "last_result_snapshot",
        "last_watchdog",
        "cron",
        "watchdog",
        "kind",
        "run",
        "last_run_status",
        "last_agent_probe",
        "last_failure_diagnostics",
        "error",
        "validation_error",
        "preflight",
        "launch_identity_conflicts",
    ]:
        if key in monitor:
            if key == "last_failure_diagnostics":
                compact_monitor[key] = compact_failure_diagnostics(monitor[key])
            elif key == "last_agent_probe":
                compact_monitor[key] = compact_agent_probe(monitor[key])
            elif key == "preflight" and isinstance(monitor[key], dict):
                compact_monitor[key] = {
                    preflight_key: monitor[key].get(preflight_key)
                    for preflight_key in ["ok", "classification", "checks", "argv_probe", "warnings", "remote_config_snapshot_error"]
                    if preflight_key in monitor[key]
                }
            else:
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
        "run",
        "status",
        "silent",
        "event",
        "message",
        "success",
        "failure_diagnostics",
        "launch_identity_conflicts",
    ]:
        if key in result:
            compact_result[key] = compact_failure_diagnostics(result[key]) if key == "failure_diagnostics" else result[key]
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


def cached_run_ids(job: JobRecord, max_runs: int | None) -> list[str]:
    last_status = job.monitor.get("last_wandb_status")
    if not isinstance(last_status, dict):
        return []
    run_ids = []
    runs = last_status.get("runs") or []
    if max_runs is not None:
        runs = runs[:max_runs]
    for run in runs:
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
        "finished_runs": sweep.get("finished_runs"),
        "running_runs": sweep.get("running_runs"),
        "failed_runs": sweep.get("failed_runs"),
        "raw_run_state_counts": sweep.get("raw_run_state_counts"),
        "run_state_counts_source": sweep.get("run_state_counts_source"),
        "run_state_counts_consistency": sweep.get("run_state_counts_consistency"),
        "last_sync_at": sweep.get("last_sync_at"),
        "speed_per_hour": sweep.get("speed_per_hour"),
        "eta_seconds": sweep.get("eta_seconds"),
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


def sweep_consistency_warnings(sweep: dict[str, Any]) -> list[str]:
    consistency = sweep.get("run_state_counts_consistency")
    if consistency == "terminal_run_edges_stale":
        return ["W&B sweep 已终态，但 run edge 状态仍滞后；Console 已使用终态规范化计数。"]
    return []


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
