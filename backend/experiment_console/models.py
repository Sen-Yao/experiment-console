from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
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
    validating = "validating"
    running = "running"
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
    job_name: str
    config_path: str
    entity: str | None = None
    project: str | None = None
    remote_host: str
    remote_cwd: str
    conda_env: str | None = None
    conda_sh: str = "/opt/anaconda3/etc/profile.d/conda.sh"
    gpu_mode: Literal["auto", "strict"] = "auto"
    max_agents: int | None = None
    profile: Literal["sweep", "mini"] = "sweep"
    idempotency_key: str | None = None

    @field_validator("max_agents")
    @classmethod
    def positive_max_agents(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_agents must be positive")
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
    idempotency_key: str | None = None


class StopJobPayload(BaseModel):
    job_id: str
    kill_agents: bool = True
    cancel_wandb: bool = False
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
    job_id: str
    gpu_mode: Literal["auto", "strict"] = "auto"
    max_agents: int | None = None
    conda_sh: str = "/opt/anaconda3/etc/profile.d/conda.sh"
    idempotency_key: str | None = None


class RepairWatchdogPayload(BaseModel):
    job_id: str
    remote_cwd: str
    remote_log_dir: str | None = None
    remote_tmp_dir: str | None = None
    conda_sh: str = "/opt/anaconda3/etc/profile.d/conda.sh"
    conda_env: str | None = None
    idempotency_key: str | None = None


class ScheduleMonitorPayload(BaseModel):
    job_id: str
    every: str = "10m"
    timeout_seconds: int = 300
    notify_channel: str | None = None
    notify_target: str | None = None
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


class WatchdogOncePayload(BaseModel):
    job_id: str
    terminal_disable: bool = True
    idempotency_key: str | None = None


class AuthCheckPayload(BaseModel):
    job_id: str | None = None
    sweep_id: str | None = None
    entity: str | None = None
    project: str | None = None
    remote_host: str | None = None
    remote_cwd: str | None = None


class PreflightPayload(BaseModel):
    remote_host: str
    remote_cwd: str
    config_path: str | None = None
    conda_env: str | None = None
    conda_sh: str = "/opt/anaconda3/etc/profile.d/conda.sh"


class PullResultsPayload(BaseModel):
    job_id: str | None = None
    sweep_id: str | None = None
    entity: str | None = None
    project: str | None = None
    remote_host: str | None = None
    remote_cwd: str | None = None
    budget_seconds: int = 90
    max_runs: int = 200
    metric_keys: list[str] = Field(default_factory=list)
    group_keys: list[str] = Field(default_factory=list)
    allow_partial: bool = True
    idempotency_key: str | None = None

    @model_validator(mode="after")
    def require_job_or_sweep(self) -> "PullResultsPayload":
        if not self.job_id and not self.sweep_id:
            raise ValueError("job_id or sweep_id is required")
        if self.budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive")
        if self.max_runs <= 0:
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
    raise ValueError(f"unsupported intent: {intent}")


def assert_local_config_path(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"config_path does not exist: {path}")
    if not p.is_file():
        raise ValueError(f"config_path is not a file: {path}")
    return p
