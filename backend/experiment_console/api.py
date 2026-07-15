from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
from pathlib import Path
import subprocess

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import Settings
from .models import (
    ConfirmRequest,
    AckWakeEventPayload,
    AdvanceQueuePayload,
    AuthCheckPayload,
    CancelSweepPayload,
    IntentPreviewRequest,
    IntentType,
    LaunchRunPayload,
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
from .monitor import MonitorWorker
from .store import WakeEventLedgerMismatch, WakeEventLeaseConflict


settings = Settings()
service = ConsoleService(settings)
monitor_worker = MonitorWorker(service)


def _configured_console_api_token() -> str | None:
    if settings.authority_role == "authoritative" and settings.console_api_token_file is None:
        return None
    token = settings.console_api_token()
    if settings.authority_role == "authoritative" and (not token or len(token) < 32):
        return None
    return token


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.authority_role == "authoritative" and not _configured_console_api_token():
        raise RuntimeError("authoritative Console requires EXPERIMENT_CONSOLE_API_TOKEN_FILE")
    monitor_worker.start()
    try:
        yield
    finally:
        monitor_worker.stop()


app = FastAPI(title="Experiment Console", version="0.1.0", lifespan=lifespan)
app.state.started_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _protected_api_path(path: str) -> bool:
    return (
        path == "/api/bridge/events"
        or path.startswith("/api/bridge/events/")
        or path.startswith("/api/artifacts/")
        or path == "/api/intents"
        or path.startswith("/api/intents/")
        or path == "/api/runner"
        or path.startswith("/api/runner/")
        or path == "/api/hosts"
        or path.startswith("/api/hosts/")
    )


def _public_wake_event(event: dict) -> dict:
    return {key: value for key, value in event.items() if key != "dedupe_key"}


@app.middleware("http")
async def require_console_bearer(request: Request, call_next):
    if not _protected_api_path(request.url.path):
        return await call_next(request)
    configured_token = _configured_console_api_token()
    if not configured_token:
        if settings.authority_role == "authoritative":
            return JSONResponse(status_code=503, content={"detail": "Console API token is not configured"})
        return await call_next(request)
    authorization = request.headers.get("authorization", "")
    scheme, separator, provided_token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not hmac.compare_digest(provided_token, configured_token):
        return JSONResponse(
            status_code=401,
            content={"detail": "invalid or missing bearer token"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


@app.get("/health")
def health():
    repo_root = Path(__file__).resolve().parents[2]
    try:
        git_sha = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:
        git_sha = None
    return {
        "status": "ok",
        "runtime": "experiment_console_fastapi",
        "cwd": str(repo_root),
        "repo_root": str(repo_root),
        "git_sha": git_sha,
        "state_dir": str(settings.state_dir),
        "contract": settings.contract_version,
        "contract_version": settings.contract_version,
        "authority_role": settings.authority_role,
        "instance_id": settings.instance_id,
        "ledger_id": service.store.metadata("ledger_id"),
        "ledger_schema_version": service.store.metadata("ledger_schema_version"),
        "cutover_committed_at": service.store.metadata("cutover_committed_at"),
        "monitor_worker": monitor_worker.status(),
        "wandb_api_key_present": bool(service._wandb_api_key()),
        "console_api_auth_configured": bool(_configured_console_api_token()),
        "real_execution_enabled": True,
        "started_at": getattr(app.state, "started_at", None),
    }


@app.get("/api/bridge/events")
def bridge_events(
    consumer_id: str = Query(min_length=1, max_length=128),
    limit: int = Query(default=20, ge=1, le=100),
    lease_seconds: int = Query(default=60, ge=5, le=3600),
):
    ledger_id, events = service.store.claim_wake_events_with_ledger(
        consumer_id=consumer_id,
        limit=limit,
        lease_seconds=lease_seconds,
    )
    return {
        "authority_role": settings.authority_role,
        "instance_id": settings.instance_id,
        "ledger_id": ledger_id,
        "events": [_public_wake_event(event) for event in events],
    }


@app.post("/api/bridge/events/{event_id}/ack")
def ack_bridge_event(event_id: str, payload: AckWakeEventPayload):
    try:
        event, idempotent = service.store.ack_wake_event(
            event_id,
            consumer_id=payload.consumer_id,
            expected_ledger_id=payload.expected_ledger_id,
            lease_token=payload.lease_token,
        )
        return {
            "status": "ok",
            "event_id": event_id,
            "acked": True,
            "idempotent": idempotent,
            "event": _public_wake_event(event),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except WakeEventLeaseConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except WakeEventLedgerMismatch as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/artifacts/{snapshot_id}/download")
def download_artifact_bundle(snapshot_id: str):
    try:
        path = service.artifact_download_path(snapshot_id)
        return FileResponse(path, media_type="application/zip", filename=f"{snapshot_id}.zip")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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


@app.post("/api/runner/launch-run")
def runner_launch_run(payload: LaunchRunPayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.launch_run, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
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


@app.post("/api/runner/advance-queue")
def runner_advance_queue(payload: AdvanceQueuePayload, requested_by: str = "experiment-runner"):
    try:
        return service.runner_command(IntentType.advance_queue, payload.model_dump(mode="json"), requested_by=requested_by).model_dump(mode="json", exclude_none=True)
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
