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
    default_entity: str = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_DEFAULT_ENTITY", "my-team"))
    default_project: str = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_DEFAULT_PROJECT", "my-project"))
    wandb_api_key_env: str = "WANDB_API_KEY"
    ssh_timeout_seconds: int = int(os.environ.get("EXPERIMENT_CONSOLE_SSH_TIMEOUT", "20"))
    command_timeout_seconds: int = int(os.environ.get("EXPERIMENT_CONSOLE_COMMAND_TIMEOUT", "120"))
    gpu_min_free_gb: float = float(os.environ.get("EXPERIMENT_CONSOLE_GPU_MIN_FREE_GB", "2.0"))
    gpu_max_util: int = int(os.environ.get("EXPERIMENT_CONSOLE_GPU_MAX_UTIL", "85"))
    default_conda_env: str | None = Field(default_factory=lambda: os.environ.get("EXPERIMENT_CONSOLE_DEFAULT_CONDA_ENV"))

    @property
    def sqlite_path(self) -> Path:
        return self.state_dir / "console.sqlite3"

    @property
    def audit_path(self) -> Path:
        return self.state_dir / "audit.jsonl"

    @property
    def sweeps_cache_path(self) -> Path:
        return self.state_dir / "sweeps_cache.json"
