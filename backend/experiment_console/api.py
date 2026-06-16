from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .models import (
    ConfirmRequest,
    AuthCheckPayload,
    CancelSweepPayload,
    IntentPreviewRequest,
    IntentType,
    LaunchSweepPayload,
    PreflightPayload,
    PullResultsPayload,
    RepairWatchdogPayload,
    RecoverAgentsPayload,
    RegisterExistingSweepPayload,
    ScheduleMonitorPayload,
    StatusQueryPayload,
    StopJobPayload,
    UnscheduleMonitorPayload,
    ValidateConfigPayload,
    WatchdogOncePayload,
)
from .service import ConsoleService


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


@app.get("/health")
def health():
    return {"status": "ok", "state_dir": str(settings.state_dir), "real_execution_enabled": True}


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
def runner_validate_config(payload: ValidateConfigPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.validate_config, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/launch-sweep")
def runner_launch_sweep(payload: LaunchSweepPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.launch_sweep, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/register-existing-sweep")
def runner_register_existing_sweep(payload: RegisterExistingSweepPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.register_existing_sweep, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/recover-agents")
def runner_recover_agents(payload: RecoverAgentsPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.recover_agents, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/repair-watchdog")
def runner_repair_watchdog(payload: RepairWatchdogPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.repair_watchdog, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/schedule-monitor")
def runner_schedule_monitor(payload: ScheduleMonitorPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.schedule_monitor, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/unschedule-monitor")
def runner_unschedule_monitor(payload: UnscheduleMonitorPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.unschedule_monitor, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/watchdog-once")
def runner_watchdog_once(payload: WatchdogOncePayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.watchdog_once, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/stop-job")
def runner_stop_job(payload: StopJobPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.stop_job, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/cancel-sweep")
def runner_cancel_sweep(payload: CancelSweepPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.cancel_sweep, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/auth-check")
def runner_auth_check(payload: AuthCheckPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.auth_check, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/preflight")
def runner_preflight(payload: PreflightPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.preflight, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/pull-results")
def runner_pull_results(payload: PullResultsPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.pull_results, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runner/status")
def runner_status(payload: StatusQueryPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.status_query, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
