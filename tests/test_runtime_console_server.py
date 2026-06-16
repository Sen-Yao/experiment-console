from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runtime_console_server.py"
spec = importlib.util.spec_from_file_location("runtime_console_server_test", SCRIPT)
runtime = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


def runner_response(command: str, classification: str = "ok"):
    return SimpleNamespace(
        model_dump=lambda *args, **kwargs: {
            "status": "ok",
            "command": command,
            "stage": "done",
            "classification": classification,
            "result": {"ok": True},
            "provenance": {"source": "test"},
            "next_actions": [],
        }
    )


def test_runtime_health_exposes_console_contract():
    client = TestClient(runtime.app)
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["runtime"] == "experiment_console_runtime"
    assert data["contract"] == "runner_console_agent_v1"


def test_runtime_mounts_runner_contract(monkeypatch):
    calls = []

    def fake_runner_command(intent, payload, requested_by="experiment-runner"):
        calls.append((intent.value, payload, requested_by))
        return runner_response(intent.value)

    monkeypatch.setattr(runtime.service, "runner_command", fake_runner_command)
    client = TestClient(runtime.app)
    endpoints = [
        ("/api/runner/status", {"job_id": "job1"}),
        ("/api/runner/preflight", {"remote_host": "HCCS-25", "remote_cwd": "/project"}),
        ("/api/runner/auth-check", {"job_id": "job1"}),
        ("/api/runner/pull-results", {"job_id": "job1"}),
        ("/api/runner/watchdog-once", {"job_id": "job1"}),
        ("/api/runner/recover-agents", {"job_id": "job1"}),
        ("/api/runner/stop-job", {"job_id": "job1"}),
        ("/api/runner/repair-watchdog", {"job_id": "job1", "remote_cwd": "/project"}),
        ("/api/runner/schedule-monitor", {"job_id": "job1"}),
        ("/api/runner/unschedule-monitor", {"job_id": "job1"}),
    ]
    for path, payload in endpoints:
        response = client.post(path, json=payload)
        assert response.status_code != 404, path
        assert response.json()["status"] == "ok"
    assert [call[0] for call in calls] == [
        "status_query",
        "preflight",
        "auth_check",
        "pull_results",
        "watchdog_once",
        "recover_agents",
        "stop_job",
        "repair_watchdog",
        "schedule_monitor",
        "unschedule_monitor",
    ]
