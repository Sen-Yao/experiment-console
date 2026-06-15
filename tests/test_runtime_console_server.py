from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runtime_console_server.py"
spec = importlib.util.spec_from_file_location("runtime_console_server_test", SCRIPT)
runtime = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


JOB = {
    "job_id": "job1",
    "name": "demo",
    "status": "running",
    "entity": "HCCS",
    "project": "DualRefGAD",
    "sweep_id": "sw1",
    "remote_host": "HCCS-25",
    "remote_cwd": "/project",
    "config_path": "/tmp/sweep_2880.yaml",
    "remote_config_path": "/project/sweep_2880.yaml",
    "agent_pids": [],
    "monitor": {},
    "created_at": "2026-06-15T00:00:00Z",
    "updated_at": "2026-06-15T00:00:00Z",
}


def client(monkeypatch):
    monkeypatch.setattr(runtime, "load_jobs", lambda: [dict(JOB)])
    monkeypatch.setattr(runtime, "upsert_job", lambda job: job)
    monkeypatch.setattr(runtime, "cached_sweep_summary", lambda job: {
        "id": "sw1",
        "name": "sw1",
        "state": "RUNNING",
        "runCount": 5,
        "expectedRunCount": 2880,
        "finished_runs": 4,
        "running_runs": 1,
        "failed_runs": 0,
        "speed_per_hour": 12.0,
        "eta_seconds": 120.0,
        "last_sync_at": "2026-06-15T00:00:00Z",
        "source": "fake",
    })
    monkeypatch.setattr(runtime, "agent_health", lambda job: {
        "available": True,
        "active_count": 1,
        "active_processes": [{"pid": "123", "command": "wandb agent HCCS/DualRefGAD/sw1"}],
        "recent_logs": ["console_wandb_agent_HCCS_DualRefGAD_sw1_gpu0.log"],
        "recent_run_ids": ["run1"],
    })
    return TestClient(runtime.app)


def completed(stdout: str = "{}\n") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["ssh"], 0, stdout=stdout, stderr="")


def test_status_returns_sweep_counts_and_agent_health(monkeypatch):
    c = client(monkeypatch)
    data = c.post("/api/runner/status", json={"job_id": "job1"}).json()
    assert data["classification"] == "ok"
    assert data["result"]["sweep"]["runCount"] == 5
    assert data["result"]["sweep"]["expectedRunCount"] == 2880
    assert data["result"]["agent_health"]["active_count"] == 1


def test_runner_contract_endpoints_exist(monkeypatch):
    c = client(monkeypatch)
    monkeypatch.setattr(runtime, "launch_agents_for_job", lambda *args, **kwargs: (
        {"host": "HCCS-25", "gpus": [], "eligible_count": 0},
        [{"gpu_index": 0, "pid": "123", "log": "/project/agent.log"}],
    ))
    monkeypatch.setattr(runtime, "stop_agents_for_job", lambda job: {"stopped_pids": ["123"]})
    monkeypatch.setattr(runtime, "pull_results_for_job", lambda job, payload: {
        "classification": "results_available",
        "source": "fake",
        "valid_results": 1,
        "missing_results": 0,
        "failed_results": 0,
        "groups": [],
        "rows": [{"run_id": "run1", "metrics": {"AUC": 0.9}, "config": {}}],
    })
    monkeypatch.setattr(runtime, "ssh_with_key", lambda host, remote, timeout=180: completed('{"ok": true, "checks": {"remote_cwd_exists": true, "config_exists": true, "wandb_cli": true, "python": true}}\n'))

    expectations = [
        ("post", "/api/runner/recover-agents", {"job_id": "job1"}, "agents_running"),
        ("post", "/api/runner/stop-job", {"job_id": "job1"}, "job_cancelled"),
        ("post", "/api/runner/repair-watchdog", {"job_id": "job1", "remote_cwd": "/project"}, "watchdog_metadata_repaired"),
        ("post", "/api/runner/schedule-monitor", {"job_id": "job1"}, "monitor_scheduled"),
        ("post", "/api/runner/unschedule-monitor", {"job_id": "job1"}, "monitor_not_scheduled"),
        ("post", "/api/runner/watchdog-once", {"job_id": "job1"}, "healthy_running"),
        ("post", "/api/runner/preflight", {"remote_host": "HCCS-25", "remote_cwd": "/project"}, "ok"),
        ("post", "/api/runner/pull-results", {"job_id": "job1", "metric_keys": ["AUC"]}, "results_available"),
    ]

    for method, path, payload, classification in expectations:
        response = getattr(c, method)(path, json=payload)
        assert response.status_code != 404, path
        assert response.json()["classification"] == classification


def test_cancel_sweep_and_auth_check(monkeypatch):
    c = client(monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", "dummy_key_for_tests")
    calls = []

    def fake_ssh(host, remote, timeout=180):
        calls.append(remote)
        if "wandb sweep" in remote:
            return completed("wandb: Done.\n")
        return completed('{"has_key": true, "target_accessible": true, "sweep_state": "RUNNING"}\n')

    monkeypatch.setattr(runtime, "ssh_with_key", fake_ssh)
    cancel = c.post("/api/runner/cancel-sweep", json={"sweep_id": "sw1", "remote_host": "HCCS-25", "remote_cwd": "/project"}).json()
    auth = c.post("/api/runner/auth-check", json={"job_id": "job1"}).json()
    assert cancel["classification"] == "sweep_cancelled"
    assert auth["classification"] == "ok"
    assert calls
