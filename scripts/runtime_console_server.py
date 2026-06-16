#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


STATE_DIR = Path(os.environ.get("EXPERIMENT_CONSOLE_STATE_DIR", "/private/tmp/experiment-console-runtime"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("EXPERIMENT_CONSOLE_STATE_DIR", str(STATE_DIR))
os.environ.setdefault("EXPERIMENT_CONSOLE_DEFAULT_ENTITY", "HCCS")
os.environ.setdefault("EXPERIMENT_CONSOLE_DEFAULT_PROJECT", "DualRefGAD")
os.environ.setdefault("EXPERIMENT_CONSOLE_DEFAULT_CONDA_ENV", "DualRefGAD")

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

from backend.experiment_console.api import app as backend_app, service  # noqa: E402
from backend.experiment_console.models import JobRecord  # noqa: E402


def migrate_legacy_state() -> None:
    legacy_jobs = STATE_DIR / "jobs.json"
    legacy_cache = STATE_DIR / "sweep_summary_cache.json"
    try:
        if legacy_cache.exists() and not service.settings.sweeps_cache_path.exists():
            service.settings.sweeps_cache_path.write_text(legacy_cache.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass
    try:
        if not legacy_jobs.exists() or service.list_jobs():
            return
        raw = json.loads(legacy_jobs.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return
        for item in raw:
            try:
                service.store.upsert_job(JobRecord.model_validate(item))
            except Exception:
                continue
    except Exception:
        return


migrate_legacy_state()


app = FastAPI(title="Experiment Console Runtime", version="runtime-2026-06-15")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5174", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "runtime": "experiment_console_runtime",
        "state_dir": str(STATE_DIR),
        "contract": "runner_console_agent_v1",
        "wandb_api_key_present": bool(os.environ.get("WANDB_API_KEY")),
    }


app.include_router(backend_app.router)

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
