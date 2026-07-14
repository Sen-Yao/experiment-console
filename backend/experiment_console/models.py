from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
    return "_".join(part for part in slug.split("_") if part)[:56] or "experiment"


def new_id(prefix: str, name: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tail = f"_{safe_slug(name)}" if name else ""
    return f"{prefix}_{stamp}{tail}_{uuid4().hex[:8]}"


class IntentType(str, Enum):
    validate_config = "validate_config"
    launch_sweep = "launch_sweep"
    launch_run = "launch_run"
    register_existing_sweep = "register_existing_sweep"
    status_query = "status_query"
    stop_job = "stop_job"
    cancel_sweep = "cancel_sweep"
    recover_agents = "recover_agents"
    repair_watchdog = "repair_watchdog"
    schedule_monitor = "schedule_monitor"
    unschedule_monitor = "unschedule_monitor"
    watchdog_once = "watchdog_once"
    auth_check = "auth_check"
    preflight = "preflight"
    pull_results = "pull_results"
    advance_queue = "advance_queue"


class IntentStatus(str, Enum):
    previewed = "previewed"
    confirmed = "confirmed"
    executing = "executing"
    succeeded = "succeeded"
    failed = "failed"
    expired = "expired"


class OperationStatus(str, Enum):
    accepted = "accepted"
    executing = "executing"
    succeeded = "succeeded"
    failed = "failed"


class JobStatus(str, Enum):
    planned = "planned"
    queued = "queued"
    validating = "validating"
    running = "running"
    finalizing = "finalizing"
    attention = "attention"
    finished = "finished"
    failed = "failed"
    cancelled = "cancelled"
    unknown = "unknown"


TERMINAL_JOB_STATUSES = {JobStatus.finished, JobStatus.failed, JobStatus.cancelled}


class CommandPreview(BaseModel):
    label: str
    argv: list[str]
    host: str | None = None
    reason: str
    side_effect: bool = False


class ExecutionPlan(BaseModel):
    summary: str
    risk_level: Literal["read_only", "writes_local_state", "remote_side_effect", "destructive"]
    commands: list[CommandPreview] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    expected_side_effects: list[str] = Field(default_factory=list)


class ValidateConfigPayload(BaseModel):
    profile: Literal["sweep", "mini", "probe", "single-run"] = "sweep"
    config_path: str


class LaunchSweepPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_name: str
    config_path: str
    entity: str | None = None
    project: str | None = None
    remote_host: str | None = None
    remote_cwd: str | None = None
    conda_env: str | None = None
    conda_sh: str | None = None
    gpu_mode: Literal["auto", "strict"] = "auto"
    max_agents: int | None = None
    profile: Literal["sweep", "mini"] = "sweep"
    queue_policy: Literal["sequential", "immediate"] = "sequential"
    queue_group: str | None = None
    queue_after_job_id: str | None = None
    result_contract: "ResultContract | None" = None
    thread_id: str | None = None
    monitor_every: str = "10m"
    monitor_timeout_seconds: int = 300
    idempotency_key: str | None = None

    @field_validator("max_agents")
    @classmethod
    def positive_max_agents(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_agents must be positive")
        return value


class LaunchRunPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_name: str
    config_path: str
    entity: str | None = None
    project: str | None = None
    remote_host: str | None = None
    remote_cwd: str | None = None
    conda_env: str | None = None
    conda_sh: str | None = None
    gpu_mode: Literal["auto", "strict"] = "auto"
    gpu_index: int | None = None
    profile: Literal["single-run"] = "single-run"
    result_path: str | None = None
    idempotency_key: str | None = None

    @field_validator("gpu_index")
    @classmethod
    def non_negative_gpu_index(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("gpu_index must be non-negative")
        return value


class StatusQueryPayload(BaseModel):
    job_id: str | None = None
    sweep_id: str | None = None

    @model_validator(mode="after")
    def require_identifier(self) -> "StatusQueryPayload":
        if not self.job_id and not self.sweep_id:
            raise ValueError("job_id or sweep_id is required")
        return self


class RegisterExistingSweepPayload(BaseModel):
    job_name: str
    sweep_id: str
    config_path: str | None = None
    entity: str | None = None
    project: str | None = None
    remote_host: str
    remote_cwd: str
    conda_env: str | None = None
    expected_total: int | None = None
    result_contract: "ResultContract | None" = None
    thread_id: str | None = None
    monitor_every: str = "10m"
    monitor_timeout_seconds: int = 300
    idempotency_key: str | None = None


class StopJobPayload(BaseModel):
    job_id: str
    kill_agents: bool = True
    cancel_wandb: bool = False
    ledger_only: bool = False
    reason: str | None = None
    idempotency_key: str | None = None


class CancelSweepPayload(BaseModel):
    sweep_id: str
    entity: str | None = None
    project: str | None = None
    remote_host: str
    remote_cwd: str
    mode: Literal["stop", "cancel"] = "cancel"
    idempotency_key: str | None = None


class RecoverAgentsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    idempotency_key: str | None = None


class RepairWatchdogPayload(BaseModel):
    job_id: str
    remote_cwd: str
    remote_log_dir: str | None = None
    remote_tmp_dir: str | None = None
    conda_sh: str = "/opt/anaconda3/etc/profile.d/conda.sh"
    conda_env: str | None = None
    idempotency_key: str | None = None


class ResultContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    expected_runs: int
    max_runs: int | None = None
    metric_keys: list[str] = Field(default_factory=list)
    group_keys: list[str] = Field(default_factory=list)
    metric_paths: list[str] = Field(default_factory=list)
    group_paths: list[str] = Field(default_factory=list)
    output_globs: list[str] = Field(default_factory=list)
    comparison_paths: list[str] = Field(default_factory=list)
    matrix_by: list[str] = Field(default_factory=list)
    allow_partial: bool = False
    export_artifacts: bool = True
    discovery_mode: Literal["run_id_output_globs_v1", "wandb_config_result_path_v1"] = "run_id_output_globs_v1"

    @model_validator(mode="after")
    def validate_contract(self) -> "ResultContract":
        if self.expected_runs <= 0:
            raise ValueError("result_contract.expected_runs must be positive")
        if self.max_runs is None:
            self.max_runs = self.expected_runs
        if self.max_runs is not None and self.max_runs != self.expected_runs:
            raise ValueError("result_contract.max_runs must equal expected_runs")
        if self.allow_partial:
            raise ValueError("result_contract.allow_partial must be false for production monitoring")
        if not self.export_artifacts:
            raise ValueError("result_contract.export_artifacts must be true for atomic result sync")
        if self.discovery_mode == "run_id_output_globs_v1":
            globs = [str(item).strip() for item in self.output_globs if str(item).strip()]
            if not globs:
                raise ValueError("result_contract.output_globs must be non-empty for run-id discovery")
            tokens = ["{run_id}", "${wandb.run.id}", "${wandb_run_id}", "{wandb.run.id}"]
            if any(sum(pattern.count(token) for token in tokens) != 1 for pattern in globs):
                raise ValueError("each result_contract output_glob must contain exactly one run-id token")
        return self


class ScheduleMonitorPayload(BaseModel):
    job_id: str
    every: str = "10m"
    timeout_seconds: int = 300
    notify_channel: str | None = None
    notify_target: str | None = None
    thread_id: str | None = None
    result_contract: ResultContract | None = None
    idempotency_key: str | None = None

    @field_validator("timeout_seconds")
    @classmethod
    def positive_timeout(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("timeout_seconds must be positive")
        return value


class UnscheduleMonitorPayload(BaseModel):
    job_id: str
    idempotency_key: str | None = None


class AckWakeEventPayload(BaseModel):
    consumer_id: str
    expected_ledger_id: str
    lease_token: str


class WatchdogOncePayload(BaseModel):
    job_id: str
    terminal_disable: bool = True
    idempotency_key: str | None = None


class AdvanceQueuePayload(BaseModel):
    queue_group: str | None = None
    auto_unblock_stale: bool = True
    idempotency_key: str | None = None


class AuthCheckPayload(BaseModel):
    job_id: str | None = None
    sweep_id: str | None = None
    entity: str | None = None
    project: str | None = None
    remote_host: str | None = None
    remote_cwd: str | None = None


class PreflightPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    remote_host: str
    remote_cwd: str
    config_path: str | None = None
    conda_env: str | None = None
    conda_sh: str | None = None
    profile: Literal["sweep", "single-run"] | None = None
    argv_probe: bool = True


class PullResultsPayload(BaseModel):
    job_id: str | None = None
    sweep_id: str | None = None
    entity: str | None = None
    project: str | None = None
    remote_host: str | None = None
    remote_cwd: str | None = None
    budget_seconds: int = 90
    max_runs: int | None = None
    metric_keys: list[str] = Field(default_factory=list)
    group_keys: list[str] = Field(default_factory=list)
    metric_paths: list[str] = Field(default_factory=list)
    group_paths: list[str] = Field(default_factory=list)
    output_globs: list[str] = Field(default_factory=list)
    discovery_mode: Literal["legacy_auto_v1", "run_id_output_globs_v1", "wandb_config_result_path_v1"] = "legacy_auto_v1"
    comparison_paths: list[str] = Field(default_factory=list)
    matrix_by: list[str] = Field(default_factory=list)
    export_artifacts: bool = False
    artifact_dir: str | None = None
    allow_partial: bool = True
    idempotency_key: str | None = None

    @model_validator(mode="after")
    def require_job_or_sweep(self) -> "PullResultsPayload":
        if not self.job_id and not self.sweep_id:
            raise ValueError("job_id or sweep_id is required")
        if self.budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive")
        if self.max_runs is not None and self.max_runs <= 0:
            raise ValueError("max_runs must be positive")
        return self


class IntentPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: IntentType
    payload: dict[str, Any]
    idempotency_key: str | None = None
    requested_by: str = "local-user"


class IntentRecord(BaseModel):
    intent_id: str
    intent: IntentType
    status: IntentStatus = IntentStatus.previewed
    payload: dict[str, Any]
    requested_by: str = "local-user"
    idempotency_key: str | None = None
    confirmation_phrase: str
    confirmed_at: str | None = None
    executed_at: str | None = None
    plan: ExecutionPlan
    result: dict[str, Any] | None = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class JobRecord(BaseModel):
    job_id: str
    name: str
    status: JobStatus = JobStatus.planned
    operation_id: str | None = None
    idempotency_key: str | None = None
    entity: str | None = None
    project: str | None = None
    sweep_id: str | None = None
    config_path: str | None = None
    remote_host: str | None = None
    remote_cwd: str | None = None
    conda_env: str | None = None
    agent_pids: list[str] = Field(default_factory=list)
    operation_log: list[dict[str, Any]] = Field(default_factory=list)
    monitor: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event_type: str
    message: str
    intent_id: str | None = None
    job_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)


class ConfirmRequest(BaseModel):
    confirmation_phrase: str


class ExecuteResponse(BaseModel):
    intent: IntentRecord
    job: JobRecord | None = None


class RunnerCommandResponse(BaseModel):
    status: Literal["ok"] = "ok"
    command: IntentType
    requested_by: str
    result: dict[str, Any]
    operation_id: str | None = None
    idempotency_key: str | None = None
    job_id: str | None = None
    job: JobRecord | None = None
    stage: str = "done"
    classification: str = "ok"
    next_actions: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    retry_after_seconds: int | None = None
    accepted: bool = False
    generated_at: str = Field(default_factory=now_iso)


def parse_payload(intent: IntentType, payload: dict[str, Any]) -> BaseModel:
    if intent is IntentType.validate_config:
        return ValidateConfigPayload.model_validate(payload)
    if intent is IntentType.launch_sweep:
        return LaunchSweepPayload.model_validate(payload)
    if intent is IntentType.launch_run:
        return LaunchRunPayload.model_validate(payload)
    if intent is IntentType.register_existing_sweep:
        return RegisterExistingSweepPayload.model_validate(payload)
    if intent is IntentType.status_query:
        return StatusQueryPayload.model_validate(payload)
    if intent is IntentType.stop_job:
        return StopJobPayload.model_validate(payload)
    if intent is IntentType.cancel_sweep:
        return CancelSweepPayload.model_validate(payload)
    if intent is IntentType.recover_agents:
        return RecoverAgentsPayload.model_validate(payload)
    if intent is IntentType.repair_watchdog:
        return RepairWatchdogPayload.model_validate(payload)
    if intent is IntentType.schedule_monitor:
        return ScheduleMonitorPayload.model_validate(payload)
    if intent is IntentType.unschedule_monitor:
        return UnscheduleMonitorPayload.model_validate(payload)
    if intent is IntentType.watchdog_once:
        return WatchdogOncePayload.model_validate(payload)
    if intent is IntentType.auth_check:
        return AuthCheckPayload.model_validate(payload)
    if intent is IntentType.preflight:
        return PreflightPayload.model_validate(payload)
    if intent is IntentType.pull_results:
        return PullResultsPayload.model_validate(payload)
    if intent is IntentType.advance_queue:
        return AdvanceQueuePayload.model_validate(payload)
    raise ValueError(f"unsupported intent: {intent}")


def assert_local_config_path(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"config_path does not exist: {path}")
    if not p.is_file():
        raise ValueError(f"config_path is not a file: {path}")
    return p
