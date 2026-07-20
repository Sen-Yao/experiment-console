from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStatus(str, Enum):
    starting = "starting"
    running = "running"
    cancelling = "cancelling"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    unknown = "unknown"


ACTIVE_JOB_STATUSES = {
    JobStatus.starting,
    JobStatus.running,
    JobStatus.cancelling,
    JobStatus.unknown,
}
TERMINAL_JOB_STATUSES = {JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled}


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=8, max_length=128)
    task_id: str = Field(min_length=1, max_length=256)
    profile: str = Field(min_length=1, max_length=64)
    cwd: str = Field(min_length=1, max_length=4096)
    argv: list[str] = Field(min_length=1, max_length=512)
    env: dict[str, str] = Field(default_factory=dict)
    gpu_indices: list[int] = Field(default_factory=list, max_length=64)
    total_runs: int | None = Field(default=None, ge=1)
    name: str | None = Field(default=None, max_length=128)

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", value):
            raise ValueError("request_id contains unsupported characters")
        return value

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: list[str]) -> list[str]:
        if any(not item or "\x00" in item or len(item) > 16_384 for item in value):
            raise ValueError("argv entries must be non-empty strings without NUL bytes")
        return value

    @field_validator("env")
    @classmethod
    def validate_env(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 128:
            raise ValueError("env contains too many entries")
        for key, item in value.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"invalid environment key: {key!r}")
            if "\x00" in item or len(item) > 16_384:
                raise ValueError(f"environment value is invalid: {key}")
            if re.search(r"(?:SECRET|TOKEN|PASSWORD|API_KEY|PRIVATE_KEY)", key, re.IGNORECASE):
                raise ValueError(
                    f"secret-like environment key {key!r} is not accepted; provision it in the server profile"
                )
        return value

    @model_validator(mode="after")
    def validate_gpus(self) -> "RunRequest":
        if len(set(self.gpu_indices)) != len(self.gpu_indices) or any(
            index < 0 for index in self.gpu_indices
        ):
            raise ValueError("gpu_indices must be unique non-negative integers")
        return self


class JobRecord(BaseModel):
    job_id: str
    request_id: str
    request_hash: str
    task_id: str
    profile: str
    cwd: str
    argv: list[str]
    env: dict[str, str]
    gpu_indices: list[int]
    total_runs: int | None = None
    completed_runs: int = 0
    name: str | None = None
    status: JobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    remote_pid: int | None = None
    remote_pgid: int | None = None
    remote_start_ticks: str | None = None
    exit_code: int | None = None
    progress_message: str | None = None
    last_observed_at: str | None = None
    last_error: str | None = None
    cancel_requested_at: str | None = None


class CancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str | None = Field(default=None, max_length=500)


class OutboxClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    consumer_id: str = Field(min_length=1, max_length=128)
    limit: int = Field(default=20, ge=1, le=100)
    lease_seconds: int = Field(default=120, ge=5, le=3600)


class OutboxAckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    consumer_id: str = Field(min_length=1, max_length=128)
    lease_token: str = Field(min_length=1, max_length=128)


class RemoteObservation(BaseModel):
    state: Literal["running", "succeeded", "failed", "cancelled", "lost"]
    observed_at: str
    pid: int | None = None
    pgid: int | None = None
    start_ticks: str | None = None
    exit_code: int | None = None
    completed_runs: int | None = None
    total_runs: int | None = None
    progress_message: str | None = None


class LogChunk(BaseModel):
    stream: Literal["stdout", "stderr"]
    offset: int
    next_offset: int
    eof: bool
    text: str


class FileChunk(BaseModel):
    path: str
    offset: int
    next_offset: int
    eof: bool
    size: int
    data_base64: str
