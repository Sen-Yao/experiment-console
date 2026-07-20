from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from .config import API_VERSION, Settings
from .models import CancelRequest, OutboxAckRequest, OutboxClaimRequest, RunRequest
from .monitor import MonitorWorker
from .remote import RemoteError, RemoteExecutor
from .service import ConsoleService, ResourceUnavailable, UnknownProfile
from .store import IdempotencyConflict, ResourceBusy


def create_app(
    settings: Settings | None = None,
    *,
    service: ConsoleService | None = None,
    remote: RemoteExecutor | None = None,
    start_monitor: bool = True,
) -> FastAPI:
    resolved_settings = settings or Settings()
    resolved_service = service or ConsoleService(resolved_settings, remote=remote)
    monitor = MonitorWorker(resolved_service)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        token = resolved_settings.console_api_token()
        if resolved_settings.require_api_token and (not token or len(token) < 32):
            raise RuntimeError(
                "production Console requires a bearer token of at least 32 characters"
            )
        if start_monitor:
            monitor.start()
        try:
            yield
        finally:
            monitor.stop()

    app = FastAPI(title="Experiment Console", version="3.0.0", lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.service = resolved_service
    app.state.monitor = monitor

    @app.middleware("http")
    async def bearer_auth(request: Request, call_next):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        token = resolved_settings.console_api_token()
        if resolved_settings.require_api_token and (not token or len(token) < 32):
            return JSONResponse(
                status_code=503,
                content={"detail": "Console API token is unavailable"},
            )
        if not token:
            return await call_next(request)
        scheme, separator, provided = request.headers.get("authorization", "").partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not hmac.compare_digest(provided, token)
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "invalid or missing bearer token"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def key_error_handler(_: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'")})

    @app.exception_handler(UnknownProfile)
    async def profile_error_handler(_: Request, exc: UnknownProfile):
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'")})

    @app.exception_handler(IdempotencyConflict)
    async def idempotency_error_handler(_: Request, exc: IdempotencyConflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ResourceBusy)
    @app.exception_handler(ResourceUnavailable)
    async def resource_error_handler(_: Request, exc: Exception):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(RemoteError)
    async def remote_error_handler(_: Request, exc: RemoteError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "instance_id": resolved_settings.instance_id,
            "ledger_id": resolved_service.store.metadata("ledger_id"),
            "schema_version": resolved_service.store.metadata("schema_version"),
            "profiles": sorted(resolved_service.profiles),
            "monitor": monitor.status(),
            "api_auth_configured": bool(resolved_settings.console_api_token()),
        }

    @app.get("/api/resources")
    def resources(profile: str | None = None):
        return resolved_service.resources(profile)

    @app.post("/api/jobs", status_code=201)
    def run_job(payload: RunRequest, response: Response):
        job, replayed = resolved_service.run(payload)
        response.status_code = 200 if replayed else 201
        return {"replayed": replayed, "job": resolved_service.public_job(job)}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, refresh: bool = True):
        return {"job": resolved_service.status(job_id, refresh=refresh)}

    @app.get("/api/jobs/{job_id}/logs")
    def job_logs(
        job_id: str,
        stream: str = Query(default="stdout", pattern="^(stdout|stderr)$"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=65536, ge=1),
        tail: bool = True,
    ):
        return resolved_service.logs(
            job_id, stream=stream, offset=offset, limit=limit, tail=tail
        ).model_dump()

    @app.get("/api/jobs/{job_id}/files")
    def job_file(
        job_id: str,
        path: str = Query(min_length=1, max_length=4096),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=1024 * 1024, ge=1),
    ):
        data, chunk = resolved_service.fetch(
            job_id, path=path, offset=offset, limit=limit
        )
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "X-File-Size": str(chunk.size),
                "X-Next-Offset": str(chunk.next_offset),
                "X-End-Of-File": "1" if chunk.eof else "0",
            },
        )

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, payload: CancelRequest):
        return {"job": resolved_service.cancel(job_id, reason=payload.reason)}

    @app.post("/api/outbox/claim")
    def claim_outbox(payload: OutboxClaimRequest):
        return {
            "instance_id": resolved_settings.instance_id,
            "api_version": API_VERSION,
            "events": resolved_service.store.claim_outbox(
                payload.consumer_id, payload.limit, payload.lease_seconds
            ),
        }

    @app.post("/api/outbox/{event_id}/ack")
    def ack_outbox(event_id: str, payload: OutboxAckRequest):
        acked = resolved_service.store.ack_outbox(
            event_id, payload.consumer_id, payload.lease_token
        )
        if not acked:
            raise HTTPException(status_code=409, detail="event lease does not match")
        return {"event_id": event_id, "acked": True}

    return app
