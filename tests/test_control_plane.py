from __future__ import annotations

import pytest

from experiment_console.config import Settings
from experiment_console.models import AuditEvent, ConfirmRequest, IntentPreviewRequest, IntentType, JobRecord, JobStatus, PullResultsPayload
from experiment_console.redaction import redact_value
from experiment_console.service import ConsoleService
from experiment_console.state import InvalidTransition, validate_job_transition
from experiment_console.store import ConsoleStore


class FakeWandB:
    def create_sweep(self, config_path, *, entity, project):
        return {"sweep_id": "abc123", "entity": entity, "project": project, "command": {"stdout": "ok"}}

    def get_sweep_state(self, entity, project, sweep_id):
        return {
            "id": sweep_id,
            "entity": entity,
            "project": project,
            "state": "RUNNING",
            "runCount": 1,
            "expectedRunCount": 10,
            "runs": [],
        }

    def discover_sweeps(self, entity, project=None, days=7, include_runs=False):
        sweep = {"id": "abc123", "entity": entity, "project": project or "P", "state": "RUNNING", "runCount": 1, "expectedRunCount": 10}
        if include_runs:
            sweep["runs"] = [{"name": "run-a", "state": "finished"}]
        return [sweep]


class FakeSSH:
    def __init__(self):
        self.launches = []
        self.run_launches = []
        self.agent_probe = None
        self.argv_probe = {
            "classification": "argv_compatible",
            "returncode": 0,
            "stdout_tail": "usage: train.py [-h]\n",
            "stderr_tail": "",
            "timed_out": False,
            "probe_argv": ["python", "train.py", "--help"],
            "timeout_seconds": 20,
        }
        self.failure_logs = []
        self.run_status = {
            "job_id": "job_run",
            "pid": "2000",
            "child_pid": "2001",
            "exit_code": None,
            "alive_pids": ["2000", "2001"],
            "status_path": "/tmp/demo/.experiment_console/runs/job_run.status.json",
            "result_path": "/tmp/demo/.experiment_console/runs/job_run.result.json",
        }

    def probe_gpus(self, host):
        return {
            "host": host,
            "eligible_count": 2,
            "gpus": [
                {"index": 0, "eligible": True, "memory_free_mb": 12000, "utilization_gpu": 0},
                {"index": 1, "eligible": True, "memory_free_mb": 11000, "utilization_gpu": 3},
            ],
        }

    def launch_agent(self, *, host, remote_cwd, sweep_path, gpu_index, conda_env, conda_sh, wandb_api_key=None):
        self.launches.append({
            "host": host,
            "remote_cwd": remote_cwd,
            "sweep_path": sweep_path,
            "gpu_index": gpu_index,
            "conda_env": conda_env,
            "conda_sh": conda_sh,
            "wandb_api_key": wandb_api_key,
        })
        return {"host": host, "gpu_index": gpu_index, "pid": str(1000 + gpu_index), "sweep_path": sweep_path}

    def create_sweep(self, *, host, remote_cwd, remote_config, entity, project, wandb_api_key):
        return {"sweep_id": "abc123", "entity": entity, "project": project, "remote_config_path": remote_config}

    def read_remote_file(self, *, host, remote_path):
        from pathlib import Path

        local_path = Path(remote_path)
        if local_path.exists():
            return {
                "host": host,
                "remote_path": remote_path,
                "text": local_path.read_text(encoding="utf-8"),
            }
        return {
            "host": host,
            "remote_path": remote_path,
            "text": (
                "name: single\n"
                "program: train.py\n"
                "parameters:\n"
                "  dataset:\n"
                "    value: Cora\n"
                "  seed:\n"
                "    value: 0\n"
            ),
        }

    def launch_run(self, *, host, remote_cwd, job_id, argv, gpu_index, conda_env, conda_sh, wandb_api_key=None, result_path=None):
        launch = {
            "host": host,
            "remote_cwd": remote_cwd,
            "job_id": job_id,
            "argv": argv,
            "gpu_index": gpu_index,
            "conda_env": conda_env,
            "conda_sh": conda_sh,
            "wandb_api_key": wandb_api_key,
            "pid": "2000",
            "log": f"{remote_cwd}/.experiment_console/runs/{job_id}.log",
            "status_path": f"{remote_cwd}/.experiment_console/runs/{job_id}.status.json",
            "result_path": result_path or f"{remote_cwd}/.experiment_console/runs/{job_id}.result.json",
        }
        self.run_launches.append(launch)
        self.run_status = {
            **self.run_status,
            "job_id": job_id,
            "pid": "2000",
            "status_path": launch["status_path"],
            "result_path": launch["result_path"],
        }
        return launch

    def check_run_status(self, *, host, status_path, pids=None):
        return {**self.run_status, "host": host, "status_path": status_path, "alive_pids": list(self.run_status.get("alive_pids") or [])}

    def check_agent_processes(self, *, host, sweep_path=None, pids=None):
        if self.agent_probe is not None:
            return {**self.agent_probe, "host": host, "sweep_path": sweep_path}
        return {
            "host": host,
            "sweep_path": sweep_path,
            "tracked_pids": list(pids or []),
            "alive_pids": list(pids or []),
            "pgrep": [],
        }

    def diagnose_agent_failure(self, *, host, remote_cwd, launches, pids=None, sweep_path=None, tail_lines=200):
        from experiment_console.ssh import build_failure_diagnostics

        logs = self.failure_logs or [
            {
                "gpu_index": launch.get("gpu_index"),
                "pid": launch.get("pid"),
                "path": launch.get("log"),
                "exists": True,
                "tail": "Traceback (most recent call last):\n  File \"train.py\", line 7, in <module>\nRuntimeError: shape mismatch\n",
            }
            for launch in launches
        ]
        return build_failure_diagnostics(
            host=host,
            remote_cwd=remote_cwd,
            sweep_path=sweep_path,
            launches=launches,
            pid_state={
                "tracked_pids": list(pids or []),
                "alive_pids": [],
                "pgrep": [],
            },
            logs=logs,
            command={"stdout": "ok"},
        )

    def stop_pids(self, *, host, pids):
        return {"host": host, "stopped_pids": list(pids), "missing_pids": [], "still_running_pids": []}

    def stop_agents(self, *, host, sweep_path, pids=None):
        return {"host": host, "stopped_pids": list(pids or ["1000"]), "sweep_path": sweep_path}

    def auth_check(self, *, host, remote_cwd, sweep_path, wandb_api_key):
        return {"ok": bool(wandb_api_key), "classification": "auth_ok" if wandb_api_key else "wandb_auth_missing"}

    def preflight(self, *, host, remote_cwd, conda_env=None, conda_sh="/opt/anaconda3/etc/profile.d/conda.sh", config_path=None):
        return {
            "ok": True,
            "classification": "ok",
            "host": host,
            "remote_cwd": remote_cwd,
            "checks": {"remote_cwd_exists": True, "wandb_cli": True, "python": True},
        }

    def probe_argv_compat(self, *, host, remote_cwd, argv, conda_env=None, conda_sh=None, timeout_seconds=20):
        return {
            **self.argv_probe,
            "host": host,
            "remote_cwd": remote_cwd,
            "probe_argv": [*argv, "--help"],
            "timeout_seconds": timeout_seconds,
        }

    def pull_results(self, *, host, remote_cwd, sweep_id, run_ids, budget_seconds, max_runs, metric_keys, group_keys):
        assert run_ids == ["run-a", "run-b"]
        return {
            "source": "remote_local_files",
            "sweep_id": sweep_id,
            "rows": [
                {"run_id": "run-a", "metrics": {"semantic_token_dim": 256, "rwse_token_steps": 0}, "has_scientific_result": True},
                {"run_id": "run-b", "metrics": {"final_test_auc": 0.9}, "has_scientific_result": True},
            ],
            "valid_results": 2,
            "missing_results": 0,
            "failed_results": 0,
            "partial": False,
        }

    def pull_single_run_result(self, *, host, status_path, result_path, metric_keys, group_keys):
        return {
            "source": "remote_single_run_files",
            "status": {**self.run_status, "status_path": status_path, "result_path": result_path},
            "rows": [
                {"run_id": self.run_status["job_id"], "state": "finished", "metrics": {"final_test_auc": 0.91}, "config": {}, "has_scientific_result": True}
            ],
            "valid_results": 1,
            "missing_results": 0,
            "failed_results": 0,
            "partial": False,
        }


def make_service(tmp_path):
    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    store = ConsoleStore(settings.sqlite_path, settings.audit_path)
    return ConsoleService(settings=settings, store=store, wandb=FakeWandB(), ssh=FakeSSH())


def test_preview_idempotency_replays_existing_intent(tmp_path):
    service = make_service(tmp_path)
    request = IntentPreviewRequest(
        intent=IntentType.status_query,
        payload={"job_id": "job_1"},
        idempotency_key="same",
    )
    first, replay1 = service.preview(request)
    second, replay2 = service.preview(request)
    assert replay1 is False
    assert replay2 is True
    assert first.intent_id == second.intent_id


def test_confirmation_phrase_required_for_real_execution(tmp_path):
    service = make_service(tmp_path)
    intent, _ = service.preview(IntentPreviewRequest(
        intent=IntentType.stop_job,
        payload={"job_id": "missing"},
    ))
    with pytest.raises(ValueError):
        service.execute(intent.intent_id)
    with pytest.raises(ValueError):
        service.confirm(intent.intent_id, ConfirmRequest(confirmation_phrase="wrong"))


def test_launch_sweep_creates_job_after_confirmation(tmp_path):
    service = make_service(tmp_path)
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        "method: grid\nname: demo\nprogram: train.py\nparameters:\n  dataset:\n    values: [Cora]\n  seed:\n    values: [0, 1, 2, 3, 4]\n",
        encoding="utf-8",
    )
    intent, _ = service.preview(IntentPreviewRequest(
        intent=IntentType.launch_sweep,
        payload={
            "job_name": "demo",
            "config_path": str(config_path),
            "remote_host": "gpu-host-1",
            "remote_cwd": "/tmp/demo",
            "max_agents": 1,
        },
    ))
    service.confirm(intent.intent_id, ConfirmRequest(confirmation_phrase=intent.confirmation_phrase))
    response = service.execute(intent.intent_id)
    assert response.job is not None
    assert response.job.sweep_id == "abc123"
    assert response.job.status == JobStatus.running
    assert response.job.agent_pids == ["1000"]


def test_launch_sweep_uses_console_default_conda_env(tmp_path):
    settings = Settings(
        state_dir=tmp_path,
        default_entity="my-team",
        default_project="my-project",
        default_conda_env="DualRefGAD",
    )
    ssh = FakeSSH()
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        "method: grid\nname: demo\nprogram: train.py\nparameters:\n  dataset:\n    values: [Cora]\n  seed:\n    values: [0, 1, 2, 3, 4]\n",
        encoding="utf-8",
    )

    result = service.runner_command(IntentType.launch_sweep, {
        "job_name": "demo_default_env",
        "config_path": str(config_path),
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "max_agents": 1,
        "idempotency_key": "default-env-test",
    })

    assert result.job is not None
    assert result.job.conda_env == "DualRefGAD"
    assert ssh.launches[0]["conda_env"] == "DualRefGAD"


def test_launch_run_creates_managed_single_run_job(tmp_path):
    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project", default_conda_env="DualRefGAD")
    ssh = FakeSSH()
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)

    result = service.runner_command(IntentType.launch_run, {
        "job_name": "single",
        "config_path": "/tmp/demo/single.yaml",
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "idempotency_key": "single-run-test",
    })

    assert result.job is not None
    assert result.job.status == JobStatus.running
    assert result.job.monitor["kind"] == "single_run"
    assert result.job.agent_pids == ["2000"]
    assert ssh.run_launches[0]["gpu_index"] == 0
    assert ssh.run_launches[0]["argv"] == ["python", "train.py", "--dataset", "Cora", "--seed", "0"]
    assert result.result["run"]["status_path"].endswith(".status.json")
    assert result.result["preflight"]["argv_probe"]["classification"] == "argv_compatible"


def test_runner_payload_rejects_legacy_remote_config_field(tmp_path):
    service = make_service(tmp_path)
    with pytest.raises(Exception, match="remote_config"):
        service.runner_command(IntentType.launch_run, {
            "job_name": "legacy_remote_config",
            "remote_config": "/tmp/demo/single.yaml",
            "remote_host": "gpu-host-1",
            "remote_cwd": "/tmp/demo",
        })


def test_launch_run_unverified_start_preserves_recoverable_run_metadata(tmp_path):
    class UnverifiedStartSSH(FakeSSH):
        def launch_run(self, **kwargs):
            launch = super().launch_run(**kwargs)
            launch["pid"] = ""
            launch["launcher"] = {
                "ok": False,
                "timed_out": True,
                "launcher_pid": "1999",
                "status_path": launch["status_path"],
                "result_path": launch["result_path"],
                "log_path": launch["log"],
            }
            return launch

    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = UnverifiedStartSSH()
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)

    response = service.runner_command(IntentType.launch_run, {
        "job_name": "single_unverified",
        "config_path": "/tmp/demo/single.yaml",
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "idempotency_key": "single-unverified-test",
    })

    assert response.classification == "run_started_unverified"
    assert response.job is not None
    assert response.job.status == JobStatus.attention
    assert response.job.agent_pids == []
    assert response.job.operation_log[-1]["status"] == "executing"
    assert response.result["run"]["status_path"].endswith(".status.json")
    assert response.result["run"]["result_path"].endswith(".result.json")
    assert response.job.monitor["last_run_status"]["status_path"] == response.result["run"]["status_path"]


def test_single_run_preflight_detects_incompatible_argv(tmp_path):
    class WandbFlagSSH(FakeSSH):
        def read_remote_file(self, *, host, remote_path):
            return {
                "host": host,
                "remote_path": remote_path,
                "text": (
                    "name: single\n"
                    "program: main.py\n"
                    "parameters:\n"
                    "  wandb:\n"
                    "    value: true\n"
                ),
            }

    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = WandbFlagSSH()
    ssh.argv_probe = {
        "classification": "argv_incompatible",
        "returncode": 2,
        "stdout_tail": "",
        "stderr_tail": "main.py: error: argument --wandb: expected one argument\n",
        "timed_out": False,
    }
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)

    response = service.runner_command(IntentType.preflight, {
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "config_path": "/tmp/demo/single.yaml",
        "profile": "single-run",
    })

    assert response.classification == "argv_incompatible"
    assert response.result["argv_probe"]["returncode"] == 2


def test_launch_run_blocks_incompatible_argv_before_remote_launch(tmp_path):
    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = FakeSSH()
    ssh.argv_probe = {
        "classification": "argv_incompatible",
        "returncode": 2,
        "stdout_tail": "",
        "stderr_tail": "main.py: error: argument --wandb: expected one argument\n",
        "timed_out": False,
    }
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)

    response = service.runner_command(IntentType.launch_run, {
        "job_name": "single_bad_argv",
        "config_path": "/tmp/demo/single.yaml",
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "idempotency_key": "single-bad-argv-test",
    })

    assert response.classification == "argv_incompatible"
    assert response.job is not None
    assert response.job.status == JobStatus.attention
    assert response.job.monitor["classification"] == "argv_incompatible"
    assert ssh.run_launches == []
    assert response.job.operation_log[-1]["status"] == "failed"


def test_launch_run_allows_unavailable_argv_probe(tmp_path):
    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = FakeSSH()
    ssh.argv_probe = {
        "classification": "argv_probe_unavailable",
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "probe timed out",
        "timed_out": True,
    }
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)

    response = service.runner_command(IntentType.launch_run, {
        "job_name": "single_soft_probe",
        "config_path": "/tmp/demo/single.yaml",
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "idempotency_key": "single-soft-probe-test",
    })

    assert response.classification == "run_running"
    assert response.result["preflight"]["argv_probe"]["classification"] == "argv_probe_unavailable"
    assert ssh.run_launches


def test_launch_run_fast_exit_is_classified_failed(tmp_path):
    class FastExitSSH(FakeSSH):
        def launch_run(self, **kwargs):
            launch = super().launch_run(**kwargs)
            launch["launcher"] = {
                "job_id": kwargs["job_id"],
                "pid": "2000",
                "child_pid": "2001",
                "exit_code": 2,
                "finished_at": "2026-06-17T00:00:01+00:00",
                "status_path": launch["status_path"],
                "result_path": launch["result_path"],
            }
            return launch

    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = FastExitSSH()
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)

    response = service.runner_command(IntentType.launch_run, {
        "job_name": "single_fast_exit",
        "config_path": "/tmp/demo/single.yaml",
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "idempotency_key": "single-fast-exit-test",
    })

    assert response.classification == "run_failed"
    assert response.job is not None
    assert response.job.status == JobStatus.failed
    assert response.job.monitor["last_run_status"]["exit_code"] == 2


def test_launch_run_rejects_multi_value_config(tmp_path):
    class MultiValueSSH(FakeSSH):
        def read_remote_file(self, *, host, remote_path):
            return {
                "host": host,
                "remote_path": remote_path,
                "text": "program: train.py\nparameters:\n  seed:\n    values: [0, 1]\n",
            }

    service = ConsoleService(settings=Settings(state_dir=tmp_path), store=ConsoleStore(Settings(state_dir=tmp_path).sqlite_path, Settings(state_dir=tmp_path).audit_path), wandb=FakeWandB(), ssh=MultiValueSSH())
    response = service.runner_command(IntentType.launch_run, {
        "job_name": "bad_single",
        "config_path": "/tmp/demo/single.yaml",
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
    })

    assert response.classification == "run_sweep_boundary_error"
    assert response.job is not None
    assert response.job.status == JobStatus.failed
    assert response.job.monitor["kind"] == "single_run"
    assert response.job.monitor["stage"] == "validation_failed"
    assert "exactly one value" in response.result["error"]
    assert "launch-sweep" in response.next_actions[0]
    assert service.store.get_job(response.job_id).status == JobStatus.failed


def test_failed_launch_run_does_not_block_same_identity_launch_sweep(tmp_path):
    class SwitchingSSH(FakeSSH):
        def __init__(self):
            super().__init__()
            self.mode = "single"

        def read_remote_file(self, *, host, remote_path):
            if self.mode == "single":
                return {
                    "host": host,
                    "remote_path": remote_path,
                    "text": "program: train.py\nparameters:\n  seed:\n    values: [0, 1]\n",
                }
            return {
                "host": host,
                "remote_path": remote_path,
                "text": (
                    "method: grid\n"
                    "name: sweep\n"
                    "program: train.py\n"
                    "parameters:\n"
                    "  dataset:\n"
                    "    values: [Cora]\n"
                    "  seed:\n"
                    "    values: [0, 1, 2, 3, 4]\n"
                ),
            }

    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = SwitchingSSH()
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)

    bad_run = service.runner_command(IntentType.launch_run, {
        "job_name": "same_name",
        "config_path": "/tmp/demo/shared.yaml",
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
    })
    assert bad_run.classification == "run_sweep_boundary_error"

    ssh.mode = "sweep"
    sweep = service.runner_command(IntentType.launch_sweep, {
        "job_name": "same_name",
        "config_path": "/tmp/demo/shared.yaml",
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "max_agents": 1,
        "idempotency_key": "same-name-sweep-test",
    })

    assert sweep.classification in {"agents_running", "wandb_auth_missing", "agents_failed_wandb_auth"}
    assert sweep.job is not None
    assert sweep.job.job_id != bad_run.job_id
    assert sweep.job.sweep_id == "abc123"
    assert sweep.job.monitor["kind"] == "sweep"
    assert sweep.job.monitor["launch_identity_conflicts"][0]["job_id"] == bad_run.job_id


def test_launch_sweep_reuses_only_existing_sweep_job(tmp_path):
    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = FakeSSH()
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        "method: grid\nname: demo\nprogram: train.py\nparameters:\n  dataset:\n    values: [Cora]\n  seed:\n    values: [0, 1, 2, 3, 4]\n",
        encoding="utf-8",
    )

    first = service.runner_command(IntentType.launch_sweep, {
        "job_name": "replay_sweep",
        "config_path": str(config_path),
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "max_agents": 1,
        "idempotency_key": "replay-sweep-first",
    })
    replay = service.runner_command(IntentType.launch_sweep, {
        "job_name": "replay_sweep",
        "config_path": str(config_path),
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "max_agents": 1,
        "idempotency_key": "replay-sweep-second",
    })

    assert first.job_id == replay.job_id
    assert replay.classification == "existing_sweep_reused"
    assert replay.result["job"]["sweep_id"] == "abc123"


def test_single_run_status_stop_and_pull_results(tmp_path):
    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = FakeSSH()
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)
    launch = service.runner_command(IntentType.launch_run, {
        "job_name": "single_status",
        "config_path": "/tmp/demo/single.yaml",
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "idempotency_key": "single-status-test",
    })
    job_id = launch.job_id
    assert job_id

    running = service.runner_command(IntentType.status_query, {"job_id": job_id})
    assert running.result["state"]["agent_health"] == "running"

    ssh.run_status = {**ssh.run_status, "exit_code": 0, "alive_pids": [], "finished_at": "2026-06-17T00:00:00+00:00"}
    finished = service.runner_command(IntentType.status_query, {"job_id": job_id})
    assert finished.result["state"]["job_status"] == "finished"

    pulled = service.runner_command(IntentType.pull_results, {"job_id": job_id, "metric_keys": ["final_test_auc"]})
    assert pulled.result["valid_results"] == 1
    assert pulled.result["rows"][0]["metrics"]["final_test_auc"] == 0.91

    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=FakeSSH())
    stopped_launch = service.runner_command(IntentType.launch_run, {
        "job_name": "single_stop",
        "config_path": "/tmp/demo/single.yaml",
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "idempotency_key": "single-stop-test",
    })
    assert stopped_launch.job_id
    stopped = service.runner_command(IntentType.stop_job, {"job_id": stopped_launch.job_id})
    assert stopped.result["stop_run"]["stopped_pids"] == ["2000"]


def test_single_run_pull_results_recovers_default_paths_from_legacy_ledger(tmp_path):
    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = FakeSSH()
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)
    job = JobRecord(
        job_id="job_legacy_single",
        name="legacy-single",
        status=JobStatus.attention,
        entity="my-team",
        project="my-project",
        remote_host="gpu-host-1",
        remote_cwd="/tmp/demo",
        monitor={"kind": "single_run", "run": {}},
    )
    service.store.upsert_job(job)

    pulled = service.runner_command(IntentType.pull_results, {"job_id": job.job_id, "metric_keys": ["final_test_auc"]})

    assert pulled.result["valid_results"] == 1
    assert pulled.result["status"]["status_path"] == "/tmp/demo/.experiment_console/runs/job_legacy_single.status.json"
    recovered = service.store.get_job(job.job_id)
    assert recovered is not None
    assert recovered.monitor["run"]["status_path"] == "/tmp/demo/.experiment_console/runs/job_legacy_single.status.json"
    assert recovered.monitor["run"]["result_path"] == "/tmp/demo/.experiment_console/runs/job_legacy_single.result.json"


def test_unverified_single_run_status_can_transition_to_finished(tmp_path):
    class UnverifiedStartSSH(FakeSSH):
        def launch_run(self, **kwargs):
            launch = super().launch_run(**kwargs)
            launch["pid"] = ""
            launch["launcher"] = {
                "ok": False,
                "timed_out": True,
                "launcher_pid": "1999",
                "status_path": launch["status_path"],
                "result_path": launch["result_path"],
                "log_path": launch["log"],
            }
            return launch

    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = UnverifiedStartSSH()
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)
    launch = service.runner_command(IntentType.launch_run, {
        "job_name": "single_unverified_finished",
        "config_path": "/tmp/demo/single.yaml",
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "idempotency_key": "single-unverified-finished-test",
    })
    assert launch.job_id
    assert launch.classification == "run_started_unverified"

    ssh.run_status = {
        **ssh.run_status,
        "job_id": launch.job_id,
        "exit_code": 0,
        "alive_pids": [],
        "finished_at": "2026-06-17T00:00:00+00:00",
        "status_path": launch.result["run"]["status_path"],
        "result_path": launch.result["run"]["result_path"],
    }
    status = service.runner_command(IntentType.status_query, {"job_id": launch.job_id})

    assert status.result["state"]["job_status"] == "finished"
    assert status.result["state"]["agent_health"] == "terminal"
    assert status.result["job"]["monitor"]["classification"] == "run_finished"


def test_single_run_status_failed_exit_needs_attention(tmp_path):
    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = FakeSSH()
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)
    launch = service.runner_command(IntentType.launch_run, {
        "job_name": "single_failed_status",
        "config_path": "/tmp/demo/single.yaml",
        "remote_host": "gpu-host-1",
        "remote_cwd": "/tmp/demo",
        "idempotency_key": "single-failed-status-test",
    })
    assert launch.job_id

    ssh.run_status = {**ssh.run_status, "exit_code": 2, "alive_pids": [], "finished_at": "2026-06-17T00:00:00+00:00"}
    status = service.runner_command(IntentType.status_query, {"job_id": launch.job_id})

    assert status.classification == "attention"
    assert status.result["state"]["job_status"] == "failed"
    assert status.result["state"]["agent_health"] == "failed"
    assert status.result["job"]["monitor"]["classification"] == "run_failed"


def test_status_collects_failure_diagnostics_for_missing_sweep_agent(tmp_path):
    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = FakeSSH()
    ssh.agent_probe = {"tracked_pids": ["1000"], "alive_pids": [], "pgrep": []}
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)
    service.store.upsert_job(JobRecord(
        job_id="job_failed_agent",
        name="failed-agent",
        status=JobStatus.running,
        entity="my-team",
        project="my-project",
        sweep_id="abc123",
        remote_host="gpu-host-1",
        remote_cwd="/tmp/demo",
        agent_pids=["1000"],
        monitor={
            "agent_launches": [{
                "host": "gpu-host-1",
                "gpu_index": 0,
                "pid": "1000",
                "log": "/tmp/demo/console_wandb_agent_my-team_my-project_abc123_gpu0.log",
            }],
            "sweep_path": "my-team/my-project/abc123",
        },
    ))

    response = service.runner_command(IntentType.status_query, {"job_id": "job_failed_agent"})

    assert response.result["state"]["agent_health"] == "missing"
    diagnostics = response.result["failure_diagnostics"]
    assert diagnostics["classification"] == "failure_signals_found"
    assert diagnostics["error_signals"][0]["kind"] == "traceback"
    assert "shape mismatch" in diagnostics["error_signals"][1]["excerpt"]
    assert service.store.get_job("job_failed_agent").monitor["last_failure_diagnostics"]["summary"]


def test_watchdog_includes_failure_diagnostics_from_status(tmp_path):
    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = FakeSSH()
    ssh.agent_probe = {"tracked_pids": ["1000"], "alive_pids": [], "pgrep": []}
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FakeWandB(), ssh=ssh)
    service.store.upsert_job(JobRecord(
        job_id="job_watchdog_failed_agent",
        name="watchdog-failed-agent",
        status=JobStatus.running,
        entity="my-team",
        project="my-project",
        sweep_id="abc123",
        remote_host="gpu-host-1",
        remote_cwd="/tmp/demo",
        agent_pids=["1000"],
        monitor={
            "agent_launches": [{
                "host": "gpu-host-1",
                "gpu_index": 0,
                "pid": "1000",
                "log": "/tmp/demo/console_wandb_agent_my-team_my-project_abc123_gpu0.log",
            }],
            "sweep_path": "my-team/my-project/abc123",
        },
    ))

    response = service.runner_command(IntentType.watchdog_once, {"job_id": "job_watchdog_failed_agent"})

    assert response.result["classification"] == "attention"
    diagnostics = response.result["status_result"]["failure_diagnostics"]
    assert diagnostics["classification"] == "failure_signals_found"
    assert diagnostics["error_signals"]


def test_running_sweep_with_failed_runs_and_no_results_needs_attention(tmp_path):
    class FailedRunsWandB(FakeWandB):
        def get_sweep_state(self, entity, project, sweep_id):
            return {
                "id": sweep_id,
                "entity": entity,
                "project": project,
                "state": "RUNNING",
                "runCount": 15,
                "expectedRunCount": 25,
                "runs": [
                    {"name": f"run-{index}", "state": "failed", "summary_metrics": "{}", "config": "{}"}
                    for index in range(15)
                ],
            }

    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    ssh = FakeSSH()
    ssh.failure_logs = [{
        "gpu_index": 0,
        "pid": "1000",
        "path": "/tmp/demo/console_wandb_agent_my-team_my-project_abc123_gpu0.log",
        "exists": True,
        "tail": "ModuleNotFoundError: No module named 'VecGAD'\n",
    }]
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=FailedRunsWandB(), ssh=ssh)
    service.store.upsert_job(JobRecord(
        job_id="job_failed_runs",
        name="failed-runs",
        status=JobStatus.running,
        entity="my-team",
        project="my-project",
        sweep_id="abc123",
        remote_host="gpu-host-1",
        remote_cwd="/tmp/demo",
        agent_pids=["1000"],
        monitor={
            "agent_launches": [{
                "host": "gpu-host-1",
                "gpu_index": 0,
                "pid": "1000",
                "log": "/tmp/demo/console_wandb_agent_my-team_my-project_abc123_gpu0.log",
            }],
            "last_result_pull": {"valid_results": 0, "partial": True},
            "sweep_path": "my-team/my-project/abc123",
        },
    ))

    status = service.runner_command(IntentType.status_query, {"job_id": "job_failed_runs"})

    assert status.classification == "attention"
    assert status.result["state"]["sweep_attention"] is True
    assert status.result["job"]["status"] == "attention"
    assert status.result["sweep"]["failed_runs"] == 15
    diagnostics = status.result["failure_diagnostics"]
    assert diagnostics["classification"] == "failure_signals_found"
    assert diagnostics["error_signals"][0]["kind"] == "import_error"
    assert "VecGAD" in diagnostics["error_signals"][0]["excerpt"]

    watchdog = service.runner_command(IntentType.watchdog_once, {"job_id": "job_failed_runs"})
    assert watchdog.result["classification"] == "attention"
    assert watchdog.result["silent"] is False
    assert "run 级失败" in watchdog.result["message"] or "失败" in watchdog.result["message"]


def test_recover_agents_does_not_create_new_sweep(tmp_path):
    service = make_service(tmp_path)
    job = JobRecord(
        job_id="job_existing",
        name="existing",
        status=JobStatus.attention,
        entity="my-team",
        project="my-project",
        sweep_id="abc123",
        remote_host="gpu-host-1",
        remote_cwd="/tmp/demo",
    )
    service.store.upsert_job(job)
    intent, _ = service.preview(IntentPreviewRequest(
        intent=IntentType.recover_agents,
        payload={"job_id": "job_existing", "max_agents": 1},
    ))
    service.confirm(intent.intent_id, ConfirmRequest(confirmation_phrase=intent.confirmation_phrase))
    response = service.execute(intent.intent_id)
    assert response.intent.result
    assert response.intent.result["created_new_sweep"] is False
    assert response.job is not None
    assert response.job.status == JobStatus.running


def test_terminal_job_cannot_transition_to_running():
    with pytest.raises(InvalidTransition):
        validate_job_transition(JobStatus.finished, JobStatus.running)


def test_audit_redacts_secret_values(tmp_path, monkeypatch):
    dummy_key = "dummy_key_for_redaction_tests_only"
    monkeypatch.setenv("WANDB_API_KEY", dummy_key)
    service = make_service(tmp_path)
    service.store.write_audit(
        AuditEvent(
            event_type="secret_test",
            message="secret test",
            detail={"line": f"WANDB_API_KEY={dummy_key}", "api_key": dummy_key},
        )
    )
    text = service.settings.audit_path.read_text(encoding="utf-8")
    assert dummy_key not in text
    assert "<redacted>" in text


def test_redaction_keeps_scientific_token_metric_names():
    cleaned = redact_value({"semantic_token_dim": 256, "rwse_token_steps": 0, "token": "secret"})
    assert cleaned["semantic_token_dim"] == 256
    assert cleaned["rwse_token_steps"] == 0
    assert cleaned["token"] == "<redacted>"


def test_status_for_missing_job_is_not_ok(tmp_path):
    service = make_service(tmp_path)
    response = service.runner_command(IntentType.status_query, {"job_id": "missing_job"})
    assert response.classification == "job_not_found"
    assert response.result["job"] is None
    assert response.result["state"]["job_status"] is None


def test_overview_returns_sweep_telemetry_without_run_payloads(tmp_path):
    service = make_service(tmp_path)

    overview = service.overview()
    sweep = overview["sweeps"][0]

    assert "runs" not in sweep
    assert sweep["finished_runs"] == 0
    assert sweep["running_runs"] == 0
    assert sweep["failed_runs"] == 0
    assert sweep["last_sync_at"]
    assert sweep["speed_per_hour"] is None
    assert sweep["eta_seconds"] is None


def test_status_returns_sweep_telemetry(tmp_path):
    service = make_service(tmp_path)
    service.store.upsert_job(JobRecord(
        job_id="job_telemetry",
        name="telemetry",
        status=JobStatus.running,
        entity="my-team",
        project="my-project",
        sweep_id="abc123",
        remote_host="gpu-host-1",
        remote_cwd="/tmp/demo",
    ))

    response = service.runner_command(IntentType.status_query, {"job_id": "job_telemetry"})
    sweep = response.result["sweep"]

    assert sweep["finished_runs"] == 0
    assert sweep["running_runs"] == 0
    assert sweep["failed_runs"] == 0
    assert sweep["last_sync_at"]
    assert sweep["speed_per_hour"] is None
    assert sweep["eta_seconds"] is None


def test_status_returns_compact_sweep_without_run_payloads(tmp_path):
    class VerboseWandB(FakeWandB):
        def get_sweep_state(self, entity, project, sweep_id):
            return {
                "id": sweep_id,
                "entity": entity,
                "project": project,
                "state": "FINISHED",
                "runCount": 1,
                "expectedRunCount": 1,
                "runs": [{
                    "name": "run-a",
                    "state": "finished",
                    "summary_metrics": "{\"final_test_auc\": 0.9}",
                    "config": "{\"seed\": {\"value\": 0}}",
                }],
            }

    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=VerboseWandB(), ssh=FakeSSH())
    service.store.upsert_job(JobRecord(
        job_id="job_status",
        name="status",
        status=JobStatus.running,
        entity="my-team",
        project="my-project",
        sweep_id="abc123",
        remote_host="gpu-host-1",
        remote_cwd="/tmp/demo",
    ))
    response = service.runner_command(IntentType.status_query, {"job_id": "job_status"})
    run = response.result["sweep"]["runs"][0]
    cached_run = response.result["job"]["monitor"]["last_wandb_status"]["runs"][0]
    assert "summary_metrics" not in run
    assert "config" not in run
    assert "summary_metrics" not in cached_run
    assert "config" not in cached_run
    assert "operation_log" not in response.result["job"]


def test_pull_results_uses_target_sweep_run_ids(tmp_path):
    class RunWandB(FakeWandB):
        def get_sweep_state(self, entity, project, sweep_id):
            return {
                "id": sweep_id,
                "entity": entity,
                "project": project,
                "state": "FINISHED",
                "runCount": 2,
                "expectedRunCount": 2,
                "runs": [{"name": "run-a"}, {"name": "run-b"}],
            }

    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=RunWandB(), ssh=FakeSSH())
    job = JobRecord(
        job_id="job_results",
        name="results",
        status=JobStatus.finished,
        entity="my-team",
        project="my-project",
        sweep_id="abc123",
        remote_host="gpu-host-1",
        remote_cwd="/tmp/demo",
    )
    service.store.upsert_job(job)
    result = service._pull_results(PullResultsPayload(job_id="job_results", max_runs=2))
    assert result["valid_results"] == 2
    assert [row["run_id"] for row in result["rows"]] == ["run-a", "run-b"]
    assert result["rows"][0]["metrics"]["semantic_token_dim"] == 256


def test_pull_results_without_explicit_key_does_not_replay_stale_result(tmp_path):
    class RunWandB(FakeWandB):
        def get_sweep_state(self, entity, project, sweep_id):
            return {
                "id": sweep_id,
                "entity": entity,
                "project": project,
                "state": "FINISHED",
                "runCount": 2,
                "expectedRunCount": 2,
                "runs": [{"name": "run-a"}, {"name": "run-b"}],
            }

    settings = Settings(state_dir=tmp_path, default_entity="my-team", default_project="my-project")
    service = ConsoleService(settings=settings, store=ConsoleStore(settings.sqlite_path, settings.audit_path), wandb=RunWandB(), ssh=FakeSSH())
    service.store.upsert_job(JobRecord(
        job_id="job_results",
        name="results",
        status=JobStatus.finished,
        entity="my-team",
        project="my-project",
        sweep_id="abc123",
        remote_host="gpu-host-1",
        remote_cwd="/tmp/demo",
    ))
    first = service.runner_command(IntentType.pull_results, {"job_id": "job_results", "max_runs": 2})
    second = service.runner_command(IntentType.pull_results, {"job_id": "job_results", "max_runs": 2})
    assert first.operation_id != second.operation_id
    assert first.provenance.get("replayed") is None
    assert second.provenance.get("replayed") is None
