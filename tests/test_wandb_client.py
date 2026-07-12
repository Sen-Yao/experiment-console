from __future__ import annotations

import pytest

from experiment_console.config import Settings
from experiment_console.wandb_client import WandBClient, WandBUnavailable


def test_wandb_client_reads_api_key_from_secret_file(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    secret = tmp_path / "wandb_api_key"
    secret.write_text("file-backed-key\n", encoding="utf-8")
    client = WandBClient(Settings(state_dir=tmp_path / "state", wandb_api_key_file=secret))

    assert client._headers()["Authorization"] == "Bearer file-backed-key"


def test_wandb_client_rejects_missing_secret_file_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    client = WandBClient(Settings(
        state_dir=tmp_path / "state",
        wandb_api_key_file=tmp_path / "missing",
    ))

    with pytest.raises(WandBUnavailable, match="WANDB_API_KEY is not set"):
        client._headers()
