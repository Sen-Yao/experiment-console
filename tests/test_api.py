from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from experiment_console.api import create_app


ROOT = Path(__file__).resolve().parents[1]


def run_payload():
    return {
        "request_id": "req_api_123456",
        "task_id": "task-api",
        "profile": "test",
        "cwd": "/work/project",
        "argv": ["python", "train.py"],
        "gpu_indices": [0],
        "total_runs": 5,
    }


def test_v3_api_surface(service, settings):
    app = create_app(settings, service=service, start_monitor=False)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["api_version"] == "3"

        resources = client.get("/api/resources?profile=test")
        assert resources.status_code == 200
        assert resources.json()["profiles"][0]["gpus"][0]["available"] is True

        created = client.post("/api/jobs", json=run_payload())
        assert created.status_code == 201
        job_id = created.json()["job"]["job_id"]

        replayed = client.post("/api/jobs", json=run_payload())
        assert replayed.status_code == 200
        assert replayed.json()["replayed"] is True

        assert client.get(f"/api/jobs/{job_id}").status_code == 200
        assert client.get(f"/api/jobs/{job_id}/logs").json()["text"]
        fetched = client.get(
            f"/api/jobs/{job_id}/files", params={"path": "result.json"}
        )
        assert fetched.content == b"result-data"
        assert fetched.headers["x-end-of-file"] == "1"

        cancelled = client.post(f"/api/jobs/{job_id}/cancel", json={})
        assert cancelled.json()["job"]["status"] == "cancelled"

        claimed = client.post(
            "/api/outbox/claim",
            json={"consumer_id": "bridge", "limit": 10, "lease_seconds": 60},
        ).json()["events"]
        assert len(claimed) == 1
        acked = client.post(
            f"/api/outbox/{claimed[0]['event_id']}/ack",
            json={
                "consumer_id": "bridge",
                "lease_token": claimed[0]["lease_token"],
            },
        )
        assert acked.status_code == 200

        paths = {route.path for route in app.routes if route.path.startswith("/api/")}
        assert paths == {
            "/api/resources",
            "/api/jobs",
            "/api/jobs/{job_id}",
            "/api/jobs/{job_id}/logs",
            "/api/jobs/{job_id}/files",
            "/api/jobs/{job_id}/cancel",
            "/api/outbox/claim",
            "/api/outbox/{event_id}/ack",
        }


def test_required_api_token_disappearing_fails_closed(service, settings, tmp_path):
    token_file = tmp_path / "console.token"
    token = "a" * 32
    token_file.write_text(token)
    secured = settings.model_copy(
        update={"console_api_token_file": token_file, "require_api_token": True}
    )
    app = create_app(secured, service=service, start_monitor=False)

    with TestClient(app) as client:
        assert (
            client.get(
                "/api/resources", headers={"Authorization": f"Bearer {token}"}
            ).status_code
            == 200
        )
        token_file.unlink()
        response = client.get(
            "/api/resources", headers={"Authorization": f"Bearer {token}"}
        )
        token_file.write_text("short")
        short_response = client.get(
            "/api/resources", headers={"Authorization": "Bearer short"}
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Console API token is unavailable"
    assert short_response.status_code == 503
    assert short_response.json()["detail"] == "Console API token is unavailable"


def test_importing_api_does_not_create_a_ledger(tmp_path):
    state_dir = tmp_path / "state"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "backend")
    environment["EXPERIMENT_CONSOLE_STATE_DIR"] = str(state_dir)
    result = subprocess.run(
        [sys.executable, "-c", "import experiment_console.api"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not state_dir.exists()
