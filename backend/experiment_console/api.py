from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .models import ConfirmRequest, IntentPreviewRequest, JobRecord, JobStatus, LaunchSweepPayload, RecoverAgentsPayload, StatusQueryPayload, StopJobPayload, ValidateConfigPayload, assert_local_config_path, new_id
from .service import ConsoleService
from .validation import validate_experiment_config


settings = Settings()
service = ConsoleService(settings)

app = FastAPI(title="Experiment Console", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunnerPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str | None = None
    job_name: str | None = None
    name: str | None = None
    sweep_id: str | None = None
    config_path: str | None = None
    entity: str | None = None
    project: str | None = None
    remote_host: str | None = None
    remote_cwd: str | None = None
    conda_env: str | None = None
    conda_sh: str = "/opt/anaconda3/etc/profile.d/conda.sh"
    gpu_mode: str = "auto"
    max_agents: int | None = None
    profile: str = "sweep"
    kill_agents: bool = True
    cancel_wandb: bool = False
    mode: str = "cancel"
    metric_keys: list[str] = Field(default_factory=list)
    group_keys: list[str] = Field(default_factory=list)
    budget_seconds: int = 90
    max_runs: int = 200
    allow_partial: bool = True
    remote_log_dir: str | None = None
    remote_tmp_dir: str | None = None
    every: str | None = None
    timeout_seconds: int | None = None
    notify_channel: str | None = None
    notify_target: str | None = None


def runner_response(
    *,
    command: str,
    stage: str,
    classification: str,
    result: dict[str, Any] | None = None,
    job: JobRecord | dict[str, Any] | None = None,
    success: bool = True,
    next_actions: list[str] | None = None,
) -> dict[str, Any]:
    if isinstance(job, JobRecord):
        job_body = job.model_dump(mode="json")
    else:
        job_body = job
    return {
        "success": success,
        "command": command,
        "stage": stage,
        "classification": classification,
        "job": job_body,
        "result": result or {},
        "provenance": {"source": "fastapi_console_runtime"},
        "next_actions": next_actions or [],
    }


@app.get("/health")
def health():
    return {"status": "ok", "runtime": "fastapi_console_runtime", "state_dir": str(settings.state_dir), "real_execution_enabled": True, "wandb_api_key_present": bool(os.environ.get("WANDB_API_KEY"))}


@app.get("/api/overview")
def overview():
    return service.overview()


@app.get("/api/jobs")
def list_jobs():
    return [job.model_dump(mode="json") for job in service.list_jobs()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = service.store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job.model_dump(mode="json")


@app.post("/api/intents/preview")
def preview_intent(request: IntentPreviewRequest):
    try:
        intent, replay = service.preview(request)
        return {"intent": intent.model_dump(mode="json"), "idempotent_replay": replay}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/intents/{intent_id}/confirm")
def confirm_intent(intent_id: str, request: ConfirmRequest):
    try:
        return service.confirm(intent_id, request).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/intents/{intent_id}/execute")
def execute_intent(intent_id: str):
    try:
        return service.execute(intent_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/events")
def events(limit: int = Query(default=100, ge=1, le=1000)):
    return [event.model_dump(mode="json") for event in service.events(limit)]


@app.get("/api/wandb/sweeps")
def wandb_sweeps(entity: str | None = None, project: str | None = None, days: int = Query(default=7, ge=1, le=90)):
    try:
        return {"status": "ok", "sweeps": service.discover_sweeps(entity, project, days)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/api/hosts/gpus")
def host_gpus(host: str):
    try:
        return service.probe_gpus(host)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/runner/validate-config")
def runner_validate_config(payload: ValidateConfigPayload):
    try:
        path = assert_local_config_path(payload.config_path)
        result = validate_experiment_config(path, payload.profile)
        return runner_response(
            command="validate-config",
            stage="config_validated",
            classification="ok" if result.get("is_valid") else "validation_failed",
            result=result,
            success=bool(result.get("is_valid")),
            next_actions=[] if result.get("is_valid") else ["修复配置后重试 launch-sweep"],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/launch-sweep")
def runner_launch_sweep(payload: LaunchSweepPayload, requested_by: str | None = None):
    try:
        result, job = service._launch_sweep(payload)
        classification = "agents_running" if job.status is JobStatus.running else "agents_started_unverified"
        job.monitor["requested_by"] = requested_by
        service.store.upsert_job(job)
        return runner_response(
            command="launch-sweep",
            stage="agents_launched",
            classification=classification,
            result=result,
            job=job,
            next_actions=["用 runner status 或 watchdog-once 检查 agent 健康", "用 runner pull-results 拉取可读实验摘要"],
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/runner/register-existing-sweep")
def runner_register_existing_sweep(payload: RunnerPayload, requested_by: str | None = None):
    if not payload.sweep_id:
        raise HTTPException(status_code=400, detail="sweep_id is required")
    entity = payload.entity or settings.default_entity
    project = payload.project or settings.default_project
    job = JobRecord(
        job_id=new_id("job", payload.job_name or payload.name or payload.sweep_id),
        name=payload.job_name or payload.name or f"registered_{payload.sweep_id}",
        status=JobStatus.running,
        entity=entity,
        project=project,
        sweep_id=payload.sweep_id,
        config_path=payload.config_path,
        remote_host=payload.remote_host,
        remote_cwd=payload.remote_cwd,
        conda_env=payload.conda_env,
        monitor={"registered_existing": True, "requested_by": requested_by},
    )
    service.store.upsert_job(job)
    return runner_response(
        command="register-existing-sweep",
        stage="registered",
        classification="existing_sweep_registered",
        result={"job": job.model_dump(mode="json"), "sweep": {"sweep_id": payload.sweep_id, "entity": entity, "project": project}},
        job=job,
        next_actions=["用 runner status 检查 sweep 状态"],
    )


@app.post("/api/runner/status")
def runner_status(payload: StatusQueryPayload):
    try:
        result = service._status(payload)
        classification = "ok" if not result.get("degraded") else "degraded"
        return runner_response(
            command="status",
            stage="status_checked",
            classification=classification,
            result=result,
            job=result.get("job"),
            success=classification == "ok",
            next_actions=[] if classification == "ok" else ["稍后重试 status 或检查 W&B/网络"],
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/runner/recover-agents")
def runner_recover_agents(payload: RecoverAgentsPayload):
    try:
        result, job = service._recover(payload)
        launches = result.get("agent_launches") or []
        classification = "agents_running" if launches else "agents_started_unverified"
        return runner_response(
            command="recover-agents",
            stage="agents_recovered",
            classification=classification,
            result=result,
            job=job,
            next_actions=["用 runner status 或 watchdog-once 验证 agent 健康"],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/runner/stop-job")
def runner_stop_job(payload: StopJobPayload):
    try:
        result, job = service._stop(payload)
        return runner_response(
            command="stop-job",
            stage="job_stopped",
            classification="job_cancelled",
            result=result,
            job=job,
            next_actions=["用 runner status 确认 job 状态"],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/runner/cancel-sweep")
def runner_cancel_sweep(payload: RunnerPayload):
    job = service.store.get_job(payload.job_id) if payload.job_id else None
    if not job and payload.sweep_id:
        for candidate in service.store.list_jobs():
            if candidate.sweep_id == payload.sweep_id:
                job = candidate
                break
    if not job:
        return runner_response(
            command="cancel-sweep",
            stage="unsupported",
            classification="sweep_cancel_unavailable",
            result={"sweep_id": payload.sweep_id, "reason": "full FastAPI runtime has no direct W&B cancel implementation yet"},
            success=False,
            next_actions=["使用 temporary runtime 的 cancel-sweep 或先登记 job 后 stop-job"],
        )
    result, stopped_job = service._stop(StopJobPayload(job_id=job.job_id, kill_agents=True, cancel_wandb=payload.mode != "stop"))
    return runner_response(
        command="cancel-sweep",
        stage="job_stopped",
        classification="sweep_cancel_unavailable",
        result=result,
        job=stopped_job,
        success=False,
        next_actions=["full runtime 已停止本地 job/agent；W&B sweep cancel 需由 runtime 实现后确认"],
    )


@app.post("/api/runner/auth-check")
def runner_auth_check(payload: RunnerPayload):
    job = service.store.get_job(payload.job_id) if payload.job_id else None
    has_key = bool(os.environ.get("WANDB_API_KEY"))
    if not has_key:
        return runner_response(
            command="auth-check",
            stage="auth_checked",
            classification="wandb_auth_missing",
            result={"auth": {"has_key": False}},
            job=job,
            success=False,
            next_actions=["配置 WANDB_API_KEY 后重启 Console"],
        )
    status = None
    if payload.sweep_id or (job and job.sweep_id):
        status = service._status(StatusQueryPayload(job_id=payload.job_id, sweep_id=payload.sweep_id))
    return runner_response(
        command="auth-check",
        stage="auth_checked",
        classification="ok" if not (status or {}).get("degraded") else "wandb_auth_unverified",
        result={"auth": {"has_key": True, "target_accessible": not (status or {}).get("degraded")}, "status": status},
        job=job,
        success=not (status or {}).get("degraded"),
        next_actions=[] if not (status or {}).get("degraded") else ["检查 W&B API 或网络"],
    )


@app.post("/api/runner/preflight")
def runner_preflight(payload: RunnerPayload):
    checks = {
        "remote_host_present": bool(payload.remote_host),
        "remote_cwd_present": bool(payload.remote_cwd),
        "config_path_present": bool(payload.config_path),
    }
    return runner_response(
        command="preflight",
        stage="preflight_checked",
        classification="ok" if all(checks.values()) else "preflight_incomplete",
        result={"ok": all(checks.values()), "checks": checks, "note": "full FastAPI runtime compatibility preflight; temporary runtime performs remote filesystem checks"},
        success=all(checks.values()),
        next_actions=[] if all(checks.values()) else ["补齐 remote_host、remote_cwd、config_path 后重试"],
    )


@app.post("/api/runner/pull-results")
def runner_pull_results(payload: RunnerPayload):
    job = service.store.get_job(payload.job_id) if payload.job_id else None
    return runner_response(
        command="pull-results",
        stage="results_pulled",
        classification="result_sources_unavailable",
        result={
            "classification": "result_sources_unavailable",
            "valid_results": 0,
            "missing_results": 0,
            "failed_results": 0,
            "metrics": {},
            "sources": [],
            "budget_seconds": payload.budget_seconds,
            "max_runs": payload.max_runs,
            "note": "full FastAPI runtime exposes the runner contract; production result aggregation currently lives in temporary runtime",
        },
        job=job,
        success=bool(payload.allow_partial),
        next_actions=["在支持结果聚合的 Console runtime 上重试 pull-results"],
    )


@app.post("/api/runner/repair-watchdog")
def runner_repair_watchdog(payload: RunnerPayload):
    if not payload.job_id:
        raise HTTPException(status_code=400, detail="job_id is required")
    job = service.store.get_job(payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    watchdog = {
        "remote_cwd": payload.remote_cwd or job.remote_cwd,
        "remote_log_dir": payload.remote_log_dir,
        "remote_tmp_dir": payload.remote_tmp_dir,
        "conda_sh": payload.conda_sh,
        "conda_env": payload.conda_env or job.conda_env,
    }
    job.monitor["watchdog"] = watchdog
    if payload.remote_cwd:
        job.remote_cwd = payload.remote_cwd
    service.store.upsert_job(job)
    return runner_response(
        command="repair-watchdog",
        stage="watchdog_repaired",
        classification="watchdog_metadata_repaired",
        result={"job": job.model_dump(mode="json"), "watchdog": watchdog},
        job=job,
        next_actions=["用 runner watchdog-once 验证修复结果"],
    )


@app.post("/api/runner/schedule-monitor")
def runner_schedule_monitor(payload: RunnerPayload):
    if not payload.job_id:
        raise HTTPException(status_code=400, detail="job_id is required")
    job = service.store.get_job(payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    cron = {
        "cron_id": f"console_watchdog_{payload.job_id}",
        "every": payload.every or "10m",
        "timeout_seconds": payload.timeout_seconds or 300,
        "notify_channel": payload.notify_channel,
        "notify_target": payload.notify_target,
        "active": True,
    }
    job.monitor["cron"] = cron
    service.store.upsert_job(job)
    return runner_response(command="schedule-monitor", stage="monitor_scheduled", classification="monitor_scheduled", result={"cron": cron}, job=job)


@app.post("/api/runner/unschedule-monitor")
def runner_unschedule_monitor(payload: RunnerPayload):
    if not payload.job_id:
        raise HTTPException(status_code=400, detail="job_id is required")
    job = service.store.get_job(payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    cron = job.monitor.get("cron")
    classification = "monitor_unscheduled" if cron else "monitor_not_scheduled"
    job.monitor["cron"] = {**(cron or {}), "active": False}
    service.store.upsert_job(job)
    return runner_response(command="unschedule-monitor", stage="monitor_unscheduled", classification=classification, result={"cron": job.monitor["cron"]}, job=job)


@app.post("/api/runner/watchdog-once")
def runner_watchdog_once(payload: RunnerPayload):
    if not payload.job_id and not payload.sweep_id:
        raise HTTPException(status_code=400, detail="job_id or sweep_id is required")
    status = runner_status(StatusQueryPayload(job_id=payload.job_id, sweep_id=payload.sweep_id))
    body = status.get("result") or {}
    sweep = body.get("sweep") or {}
    healthy_running = sweep.get("state") == "RUNNING" and status.get("classification") == "ok"
    terminal = sweep.get("state") in {"FINISHED", "CANCELED", "CANCELLED", "FAILED"}
    result = {
        "success": True,
        "silent": healthy_running,
        "message": "" if healthy_running else ("Sweep 已进入终态" if terminal else "Sweep 需要关注"),
        "classification": "healthy_running" if healthy_running else "terminal" if terminal else "attention",
        "status_result": status,
    }
    return runner_response(
        command="watchdog-once",
        stage="watchdog_checked",
        classification=result["classification"],
        result=result,
        job=body.get("job"),
        next_actions=[] if healthy_running else ["用 runner status 或 recover-agents 处理当前任务"],
    )
