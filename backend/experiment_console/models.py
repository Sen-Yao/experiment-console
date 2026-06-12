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
    status_query = "status_query"
    stop_job = "stop_job"
    recover_agents = "recover_agents"


class IntentStatus(str, Enum):
    previewed = "previewed"
    confirmed = "confirmed"
    executing = "executing"
    succeeded = "succeeded"
    failed = "failed"
    expired = "expired"


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


class StopJobPayload(BaseModel):
    job_id: str
    kill_agents: bool = True
    cancel_wandb: bool = False


class RecoverAgentsPayload(BaseModel):
    job_id: str
    gpu_mode: Literal["auto", "strict"] = "auto"
    max_agents: int | None = None


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
    entity: str | None = None
    project: str | None = None
    sweep_id: str | None = None
    config_path: str | None = None
    remote_host: str | None = None
    remote_cwd: str | None = None
    conda_env: str | None = None
    agent_pids: list[str] = Field(default_factory=list)
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


def parse_payload(intent: IntentType, payload: dict[str, Any]) -> BaseModel:
    if intent is IntentType.validate_config:
        return ValidateConfigPayload.model_validate(payload)
    if intent is IntentType.launch_sweep:
        return LaunchSweepPayload.model_validate(payload)
    if intent is IntentType.status_query:
        return StatusQueryPayload.model_validate(payload)
    if intent is IntentType.stop_job:
        return StopJobPayload.model_validate(payload)
    if intent is IntentType.recover_agents:
        return RecoverAgentsPayload.model_validate(payload)
    raise ValueError(f"unsupported intent: {intent}")


def assert_local_config_path(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"config_path does not exist: {path}")
    if not p.is_file():
        raise ValueError(f"config_path is not a file: {path}")
    return p

