from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field, field_validator, model_validator


API_VERSION = "3"
REPO_ROOT = Path(__file__).resolve().parents[2]
_SAFE_REMOTE_EXECUTABLE = re.compile(r"^[A-Za-z0-9_./+-]+$")


class ServerProfile(BaseModel):
    ssh_target: str
    allowed_roots: list[str] = Field(min_length=1)
    state_root: str
    bootstrap_argv: list[str] = Field(default_factory=list)
    remote_python: str = "python3"
    gpu_query_argv: list[str] = Field(
        default_factory=lambda: [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_min_free_mb: int = Field(default=2048, ge=0)
    gpu_max_utilization: int = Field(default=85, ge=0, le=100)

    @field_validator("ssh_target")
    @classmethod
    def validate_ssh_target(cls, value: str) -> str:
        value = value.strip()
        if not value or value.startswith("-") or any(ch.isspace() for ch in value):
            raise ValueError("ssh_target must be a non-empty SSH alias without whitespace")
        return value

    @field_validator("allowed_roots", "state_root")
    @classmethod
    def validate_remote_paths(cls, value):
        values = value if isinstance(value, list) else [value]
        for item in values:
            path = PurePosixPath(str(item))
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("remote paths must be absolute and cannot contain '..'")
        return value

    @field_validator("remote_python")
    @classmethod
    def validate_remote_python(cls, value: str) -> str:
        if not _SAFE_REMOTE_EXECUTABLE.fullmatch(value):
            raise ValueError("remote_python contains unsupported characters")
        return value

    @field_validator("bootstrap_argv", "gpu_query_argv")
    @classmethod
    def validate_argv(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or not item or "\x00" in item for item in value):
            raise ValueError("profile argv entries must be non-empty strings")
        return value

    @model_validator(mode="after")
    def validate_gpu_query(self) -> "ServerProfile":
        if not self.gpu_query_argv:
            raise ValueError("gpu_query_argv cannot be empty")
        return self


class Settings(BaseModel):
    state_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("EXPERIMENT_CONSOLE_STATE_DIR", str(REPO_ROOT / ".state-v3"))
        )
    )
    profiles_path: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "EXPERIMENT_CONSOLE_SERVER_PROFILES",
                str(REPO_ROOT / "config" / "server-profiles.json"),
            )
        )
    )
    instance_id: str = Field(
        default_factory=lambda: os.environ.get(
            "EXPERIMENT_CONSOLE_INSTANCE_ID", "local-experiment-console-v3"
        )
    )
    console_api_token_file: Path | None = Field(
        default_factory=lambda: (
            Path(os.environ["EXPERIMENT_CONSOLE_API_TOKEN_FILE"])
            if os.environ.get("EXPERIMENT_CONSOLE_API_TOKEN_FILE")
            else None
        )
    )
    console_api_token_env: str = "EXPERIMENT_CONSOLE_API_TOKEN"
    require_api_token: bool = Field(
        default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_REQUIRE_API_TOKEN", "0").lower()
        in {"1", "true", "yes"}
    )
    ssh_path: str = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_SSH_PATH", "ssh"))
    ssh_config_file: Path | None = Field(
        default_factory=lambda: Path(os.environ["EXPERIMENT_CONSOLE_SSH_CONFIG_FILE"])
        if os.environ.get("EXPERIMENT_CONSOLE_SSH_CONFIG_FILE")
        else None
    )
    ssh_key_file: Path | None = Field(
        default_factory=lambda: Path(os.environ["EXPERIMENT_CONSOLE_SSH_KEY_FILE"])
        if os.environ.get("EXPERIMENT_CONSOLE_SSH_KEY_FILE")
        else None
    )
    known_hosts_file: Path | None = Field(
        default_factory=lambda: Path(os.environ["EXPERIMENT_CONSOLE_KNOWN_HOSTS_FILE"])
        if os.environ.get("EXPERIMENT_CONSOLE_KNOWN_HOSTS_FILE")
        else None
    )
    ssh_timeout_seconds: int = Field(
        default_factory=lambda: int(os.environ.get("EXPERIMENT_CONSOLE_SSH_TIMEOUT", "20")), ge=1
    )
    command_timeout_seconds: int = Field(
        default_factory=lambda: int(os.environ.get("EXPERIMENT_CONSOLE_COMMAND_TIMEOUT", "120")), ge=1
    )
    monitor_poll_seconds: float = Field(
        default_factory=lambda: float(os.environ.get("EXPERIMENT_CONSOLE_MONITOR_POLL_SECONDS", "15")),
        gt=0,
    )
    monitor_enabled: bool = Field(
        default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_MONITOR_ENABLED", "1").lower()
        not in {"0", "false", "no"}
    )
    cancel_grace_seconds: float = Field(
        default_factory=lambda: float(os.environ.get("EXPERIMENT_CONSOLE_CANCEL_GRACE_SECONDS", "10")),
        gt=0,
    )
    max_log_chunk_bytes: int = Field(default=256 * 1024, ge=1024, le=4 * 1024 * 1024)
    max_fetch_chunk_bytes: int = Field(default=4 * 1024 * 1024, ge=1024, le=16 * 1024 * 1024)
    outbox_lease_seconds: int = Field(default=120, ge=5, le=3600)

    @property
    def sqlite_path(self) -> Path:
        return self.state_dir / "console-v3.sqlite3"

    def console_api_token(self) -> str | None:
        if self.console_api_token_file:
            try:
                value = self.console_api_token_file.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            return value or None
        value = os.environ.get(self.console_api_token_env, "").strip()
        return value or None

    def load_profiles(self) -> dict[str, ServerProfile]:
        try:
            raw = json.loads(self.profiles_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(f"server profile file does not exist: {self.profiles_path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read server profiles {self.profiles_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise RuntimeError("server profile file must contain a JSON object")
        profiles_raw = raw.get("profiles", raw)
        if not isinstance(profiles_raw, dict) or not profiles_raw:
            raise RuntimeError("server profile file must define at least one profile")
        profiles: dict[str, ServerProfile] = {}
        for name, payload in profiles_raw.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
                raise RuntimeError(f"invalid server profile name: {name!r}")
            profiles[name] = ServerProfile.model_validate(payload)
        return profiles
