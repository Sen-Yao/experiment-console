from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    state_dir: Path = Field(default_factory=lambda: Path(os.environ.get(
        "EXPERIMENT_CONSOLE_STATE_DIR",
        str(REPO_ROOT / ".state"),
    )))
    default_entity: str = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_DEFAULT_ENTITY", "HCCS"))
    default_project: str = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_DEFAULT_PROJECT", "DualRefGAD"))
    wandb_api_key_env: str = "WANDB_API_KEY"
    wandb_api_key_file: Path | None = Field(default_factory=lambda: Path(os.environ["WANDB_API_KEY_FILE"]) if os.environ.get("WANDB_API_KEY_FILE") else None)
    console_api_token_env: str = "EXPERIMENT_CONSOLE_API_TOKEN"
    console_api_token_file: Path | None = Field(default_factory=lambda: Path(os.environ["EXPERIMENT_CONSOLE_API_TOKEN_FILE"]) if os.environ.get("EXPERIMENT_CONSOLE_API_TOKEN_FILE") else None)
    authority_role: str = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_AUTHORITY_ROLE", "local-development"))
    instance_id: str = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_INSTANCE_ID", "local-experiment-console"))
    ssh_timeout_seconds: int = int(os.environ.get("EXPERIMENT_CONSOLE_SSH_TIMEOUT", "20"))
    command_timeout_seconds: int = int(os.environ.get("EXPERIMENT_CONSOLE_COMMAND_TIMEOUT", "120"))
    gpu_min_free_gb: float = float(os.environ.get("EXPERIMENT_CONSOLE_GPU_MIN_FREE_GB", "2.0"))
    gpu_max_util: int = int(os.environ.get("EXPERIMENT_CONSOLE_GPU_MAX_UTIL", "85"))
    default_remote_host: str = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_DEFAULT_REMOTE_HOST", "HCCS-25"))
    default_remote_cwd: str = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_DEFAULT_REMOTE_CWD", "/home/linziyao/DualRefGAD"))
    default_conda_env: str | None = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_DEFAULT_CONDA_ENV", "DualRefGAD"))
    default_conda_sh: str = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_DEFAULT_CONDA_SH", "/opt/anaconda3/etc/profile.d/conda.sh"))
    contract_version: str = "runner_console_agent_v1"
    monitor_worker_enabled: bool = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_MONITOR_WORKER", "1").lower() not in {"0", "false", "no"})
    monitor_worker_poll_seconds: int = int(os.environ.get("EXPERIMENT_CONSOLE_MONITOR_POLL_SECONDS", "5"))
    monitor_lease_seconds: int = int(os.environ.get("EXPERIMENT_CONSOLE_MONITOR_LEASE_SECONDS", "30"))
    observation_fresh_seconds: int = int(os.environ.get("EXPERIMENT_CONSOLE_OBSERVATION_FRESH_SECONDS", "900"))
    sync_error_consecutive_threshold: int = int(os.environ.get("EXPERIMENT_CONSOLE_SYNC_ERROR_CONSECUTIVE", "3"))
    sync_error_grace_seconds: int = int(os.environ.get("EXPERIMENT_CONSOLE_SYNC_ERROR_GRACE_SECONDS", "900"))
    artifact_sync_error_consecutive_threshold: int = int(os.environ.get("EXPERIMENT_CONSOLE_ARTIFACT_SYNC_ERROR_CONSECUTIVE", "3"))
    artifact_sync_error_grace_seconds: int = int(os.environ.get("EXPERIMENT_CONSOLE_ARTIFACT_SYNC_ERROR_GRACE_SECONDS", "900"))
    monitor_external_error_consecutive_threshold: int = int(os.environ.get("EXPERIMENT_CONSOLE_MONITOR_EXTERNAL_ERROR_CONSECUTIVE", "3"))
    monitor_external_error_grace_seconds: int = int(os.environ.get("EXPERIMENT_CONSOLE_MONITOR_EXTERNAL_ERROR_GRACE_SECONDS", "900"))
    audit_max_bytes: int = int(os.environ.get("EXPERIMENT_CONSOLE_AUDIT_MAX_BYTES", str(10 * 1024 * 1024)))
    audit_backup_count: int = int(os.environ.get("EXPERIMENT_CONSOLE_AUDIT_BACKUPS", "5"))

    @property
    def sqlite_path(self) -> Path:
        return self.state_dir / "console.sqlite3"

    @property
    def audit_path(self) -> Path:
        return self.state_dir / "audit.jsonl"

    @property
    def sweeps_cache_path(self) -> Path:
        return self.state_dir / "sweeps_cache.json"

    @property
    def sweep_telemetry_cache_path(self) -> Path:
        return self.state_dir / "sweep_telemetry_cache.json"

    @property
    def results_dir(self) -> Path:
        return self.state_dir / "results"

    def wandb_api_key(self) -> str | None:
        if self.wandb_api_key_file:
            try:
                value = self.wandb_api_key_file.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            return value or None
        value = os.environ.get(self.wandb_api_key_env, "").strip()
        return value or None

    def console_api_token(self) -> str | None:
        if self.console_api_token_file:
            try:
                value = self.console_api_token_file.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            return value or None
        value = os.environ.get(self.console_api_token_env, "").strip()
        return value or None
