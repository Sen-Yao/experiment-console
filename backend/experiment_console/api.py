from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .models import ConfirmRequest, IntentPreviewRequest
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
