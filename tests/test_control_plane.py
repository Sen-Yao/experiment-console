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

    def discover_sweeps(self, entity, project=None, days=7):
        return [{"id": "abc123", "entity": entity, "project": project or "P", "state": "RUNNING", "runCount": 1, "expectedRunCount": 10}]


class FakeSSH:
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
        return {"host": host, "gpu_index": gpu_index, "pid": str(1000 + gpu_index), "sweep_path": sweep_path}

    def stop_agents(self, *, host, sweep_path):
        return {"host": host, "stopped_pids": ["1000"], "sweep_path": sweep_path}

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
