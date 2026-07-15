from __future__ import annotations

import pytest

from experiment_console.config import Settings
from experiment_console.wandb_client import WandBAuthRequired, WandBClient, WandBUnavailable


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

    with pytest.raises(WandBAuthRequired, match="WANDB_API_KEY is not set") as raised:
        client._headers()

    assert raised.value.transient is False
    assert raised.value.classification == "auth_required"


def test_wandb_client_classifies_explicit_401_as_auth_required(tmp_path, monkeypatch):
    secret = tmp_path / "wandb_api_key"
    secret.write_text("invalid-key\n", encoding="utf-8")
    client = WandBClient(Settings(state_dir=tmp_path / "state", wandb_api_key_file=secret))

    class UnauthorizedResponse:
        status_code = 401

        def raise_for_status(self):
            import requests

            response = requests.Response()
            response.status_code = 401
            raise requests.HTTPError("unauthorized", response=response)

    monkeypatch.setattr("experiment_console.wandb_client.requests.post", lambda *args, **kwargs: UnauthorizedResponse())

    with pytest.raises(WandBAuthRequired) as raised:
        client.graphql("query { viewer { id } }", {})

    assert raised.value.status_code == 401
    assert raised.value.transient is False
