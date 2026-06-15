#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


STATE_DIR = Path("/private/tmp/experiment-console-runtime")
STATE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_PATH = STATE_DIR / "jobs.json"
SWEEP_CACHE_PATH = STATE_DIR / "sweep_summary_cache.json"
SWEEP_REFRESH_LOCK = threading.Lock()
SWEEP_REFRESHING: set[str] = set()
ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
REMOTE_CONFIG_PATH = os.environ.get(
    "EXPERIMENT_CONSOLE_REMOTE_CONFIG_PATH",
    "/home/linziyao/DualRefGAD/investigations/nexus/2026-06-01-dualrefgad-matguardgt/experiments/configs/matguardgt_cleg3_v4_reference_token_gt_sweep_2880.yaml",
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_jobs() -> list[dict[str, Any]]:
    if not JOBS_PATH.exists():
        return []
    return json.loads(JOBS_PATH.read_text(encoding="utf-8"))


def load_sweep_cache() -> dict[str, Any]:
    if not SWEEP_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(SWEEP_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_sweep_cache(cache: dict[str, Any]) -> None:
    SWEEP_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_jobs(jobs: list[dict[str, Any]]) -> None:
    JOBS_PATH.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def upsert_job(job: dict[str, Any]) -> dict[str, Any]:
    jobs = [item for item in load_jobs() if item.get("job_id") != job.get("job_id")]
    jobs.insert(0, job)
    save_jobs(jobs)
    return job


def redact(text: str) -> str:
    key = os.environ.get("WANDB_API_KEY", "")
    if key:
        text = text.replace(key, "[REDACTED]")
    return re.sub(r"[A-Za-z0-9_-]{40,}", "[REDACTED_TOKEN]", text)


def ssh_with_key(host: str, remote: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    key = os.environ.get("WANDB_API_KEY", "")
    wrapped = "read -r WANDB_API_KEY; export WANDB_API_KEY; " + remote
    return subprocess.run(
        ["ssh", host, wrapped],
        input=key + "\n",
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def require_ok(proc: subprocess.CompletedProcess[str], context: str) -> None:
    if proc.returncode != 0:
        detail = redact((proc.stdout or "") + "\n" + (proc.stderr or ""))
        raise HTTPException(status_code=503, detail=f"{context} failed: {detail[-2000:]}")


def parse_sweep_id(text: str) -> str | None:
    patterns = [
        r"wandb agent ([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
        r"Created sweep with ID:\s*([A-Za-z0-9_.-]+)",
        r"View sweep at .*?/sweeps/([A-Za-z0-9_.-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).rsplit("/", 1)[-1]
    return None


def expected_from_config_path(config_path: str | None) -> int:
    if config_path and "2880" in config_path:
        return 2880
    return 0


def fetch_sweep_summary(job: dict[str, Any], cache_key: str) -> dict[str, Any]:
    remote = (
        "python3 - <<'PY'\n"
        "import json, time\n"
        "from collections import Counter\n"
        "from wandb.apis.public import Api\n"
        f"path={cache_key!r}\n"
        f"expected={expected_from_config_path(job.get('remote_config_path') or job.get('config_path'))!r}\n"
        "s=Api().sweep(path)\n"
        "runs=list(s.runs)\n"
        "counts=Counter(getattr(r,'state',None) for r in runs)\n"
        "created=[getattr(r,'created_at',None) for r in runs if getattr(r,'created_at',None)]\n"
        "created_sorted=sorted(str(x) for x in created)\n"
        "started=created_sorted[0] if created_sorted else None\n"
        "finished=int(counts.get('finished',0))\n"
        "running=int(counts.get('running',0))\n"
        "failed=int(counts.get('failed',0) + counts.get('crashed',0))\n"
        "speed=0.0\n"
        "eta=None\n"
        "if started and finished > 0:\n"
        "    import datetime\n"
        "    st=datetime.datetime.fromisoformat(started.replace('Z','+00:00'))\n"
        "    elapsed=max((datetime.datetime.now(datetime.timezone.utc)-st).total_seconds(),1)\n"
        "    speed=finished/(elapsed/3600)\n"
        "    if expected and speed > 0 and finished < expected:\n"
        "        eta=(expected-finished)/speed*3600\n"
        "print(json.dumps({\n"
        "  'id': getattr(s,'id',None) or getattr(s,'name',None) or path.rsplit('/',1)[-1],\n"
        "  'name': getattr(s,'name',None) or path.rsplit('/',1)[-1],\n"
        "  'state': getattr(s,'state',None) or 'UNKNOWN',\n"
        "  'runCount': len(runs),\n"
        "  'expectedRunCount': expected,\n"
        "  'finished_runs': finished,\n"
        "  'running_runs': running,\n"
        "  'failed_runs': failed,\n"
        "  'speed_per_hour': speed,\n"
        "  'eta_seconds': eta,\n"
        "  'last_sync_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n"
        "}))\n"
        "PY"
    )
    proc = ssh_with_key(job.get("remote_host") or "HCCS-25", remote, timeout=12)
    require_ok(proc, "wandb sweep summary")
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    summary["_cached_at"] = time.time()
    summary["source"] = "wandb_public_api_cached"
    return summary


def refresh_sweep_cache(job: dict[str, Any], cache_key: str) -> None:
    try:
        summary = fetch_sweep_summary(job, cache_key)
        cache = load_sweep_cache()
        cache[cache_key] = summary
        save_sweep_cache(cache)
    finally:
        with SWEEP_REFRESH_LOCK:
            SWEEP_REFRESHING.discard(cache_key)


def cached_sweep_summary(job: dict[str, Any], ttl_seconds: int = 60) -> dict[str, Any]:
    sweep_id = job.get("sweep_id")
    entity = job.get("entity") or "HCCS"
    project = job.get("project") or "DualRefGAD"
    if not sweep_id:
        return {}
    cache_key = f"{entity}/{project}/{sweep_id}"
    cache = load_sweep_cache()
    cached = cache.get(cache_key)
    now_epoch = time.time()
    if cached and now_epoch - float(cached.get("_cached_at", 0)) <= ttl_seconds:
        return cached
    if cached:
        with SWEEP_REFRESH_LOCK:
            should_refresh = cache_key not in SWEEP_REFRESHING
            if should_refresh:
                SWEEP_REFRESHING.add(cache_key)
        if should_refresh:
            threading.Thread(target=refresh_sweep_cache, args=(job, cache_key), daemon=True).start()
        cached["source"] = "wandb_public_api_cache_refreshing"
        return cached

    try:
        summary = fetch_sweep_summary(job, cache_key)
        cache[cache_key] = summary
        save_sweep_cache(cache)
        return summary
    except Exception as exc:
        if cached:
            cached["source"] = "wandb_public_api_cache_stale"
            cached["degraded"] = redact(str(exc))
            return cached
        return {}


class LaunchSweepPayload(BaseModel):
    job_name: str
    config_path: str | None = None
    entity: str | None = "HCCS"
    project: str | None = "DualRefGAD"
    remote_host: str = "HCCS-25"
    remote_cwd: str = "/home/linziyao/DualRefGAD"
    conda_env: str | None = "DualRefGAD"
    conda_sh: str = "/opt/anaconda3/etc/profile.d/conda.sh"
    max_agents: int | None = 1
    gpu_mode: str | None = "auto"
    profile: str | None = "sweep"


class JobPayload(BaseModel):
    job_id: str | None = None
    sweep_id: str | None = None
    entity: str | None = "HCCS"
    project: str | None = "DualRefGAD"
    remote_host: str | None = "HCCS-25"
    remote_cwd: str | None = "/home/linziyao/DualRefGAD"
    job_name: str | None = None
    config_path: str | None = None


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
        "runtime": "temporary_control_plane",
        "state_dir": str(STATE_DIR),
        "wandb_api_key_present": bool(os.environ.get("WANDB_API_KEY")),
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    jobs = load_jobs()
    sweeps = []
    for job in jobs:
        if job.get("sweep_id"):
            summary = cached_sweep_summary(job)
            expected = int(summary.get("expectedRunCount") or expected_from_config_path(job.get("remote_config_path") or job.get("config_path")) or 0)
            run_count = int(summary.get("runCount") or 0)
            finished = int(summary.get("finished_runs") or 0)
            progress_base = expected or run_count or 1
            sweeps.append(
                {
                    "id": summary.get("id") or job.get("sweep_id"),
                    "name": summary.get("name") or job.get("sweep_id"),
                    "entity": job.get("entity"),
                    "project": job.get("project"),
                    "state": summary.get("state") or ("RUNNING" if job.get("status") == "running" else str(job.get("status") or "UNKNOWN").upper()),
                    "runCount": run_count,
                    "expectedRunCount": expected,
                    "progress": min(finished / progress_base, 1),
                    "finished_runs": finished,
                    "running_runs": int(summary.get("running_runs") or 0),
                    "failed_runs": int(summary.get("failed_runs") or 0),
                    "speed_per_hour": float(summary.get("speed_per_hour") or 0),
                    "eta_seconds": summary.get("eta_seconds"),
                    "last_sync_at": summary.get("last_sync_at") or now(),
                    "createdAt": job.get("created_at"),
                    "source": summary.get("source") or "temporary_console_runtime",
                }
            )
    counts: dict[str, int] = {}
    for job in jobs:
        status = str(job.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    active_count = sum(1 for item in sweeps if item.get("state") == "RUNNING")
    return {
        "status": "ok",
        "degraded": None,
        "job_counts": counts,
        "jobs": jobs[:20],
        "sweeps": sweeps,
        "active_sweeps": active_count,
        "active_sweeps_count": active_count,
        "stalled_sweeps": [],
        "finished_sweeps": sum(1 for item in sweeps if item.get("state") == "FINISHED"),
        "total_runs": sum(int(item.get("runCount") or 0) for item in sweeps),
        "generated_at": now(),
    }


@app.get("/api/events")
def events(limit: int = Query(default=40, ge=1, le=1000)) -> list[dict[str, Any]]:
    return [
        {
            "event_type": "runtime_console_active",
            "message": "临时 Console 控制面正在管理生产 sweep。",
            "created_at": now(),
            "detail": {"jobs": len(load_jobs())},
        }
    ][:limit]


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return load_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    for job in load_jobs():
        if job.get("job_id") == job_id:
            return job
    raise HTTPException(status_code=404, detail="job not found")


@app.get("/api/hosts/gpus")
def gpus(host: str = Query(default="HCCS-25")) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "ssh",
            host,
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    require_ok(proc, "gpu probe")
    out = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        idx, name, total, used, free, util = parts
        free_mb = int(float(free))
        util_pct = int(float(util))
        out.append(
            {
                "index": int(idx),
                "name": name,
                "memory_total_mb": int(float(total)),
                "memory_used_mb": int(float(used)),
                "memory_free_mb": free_mb,
                "utilization_gpu": util_pct,
                "eligible": free_mb >= 8 * 1024 and util_pct <= 10,
            }
        )
    return {"host": host, "gpus": out, "eligible_count": sum(1 for item in out if item["eligible"])}


@app.post("/api/runner/launch-sweep")
def launch_sweep(payload: LaunchSweepPayload, requested_by: str | None = None) -> dict[str, Any]:
    if not os.environ.get("WANDB_API_KEY"):
        raise HTTPException(status_code=503, detail="WANDB_API_KEY is not set")
    entity = payload.entity or "HCCS"
    project = payload.project or "DualRefGAD"
    job_id = "job_" + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + "_" + re.sub(r"[^a-zA-Z0-9]+", "_", payload.job_name).strip("_")[:48]
    remote_config = REMOTE_CONFIG_PATH
    sweep_cmd = (
        f"cd {shlex.quote(payload.remote_cwd)} && "
        f"wandb sweep --entity {shlex.quote(entity)} --project {shlex.quote(project)} {shlex.quote(remote_config)}"
    )
    sweep_proc = ssh_with_key(payload.remote_host, sweep_cmd, timeout=240)
    require_ok(sweep_proc, "wandb sweep")
    sweep_id = parse_sweep_id(sweep_proc.stdout + "\n" + sweep_proc.stderr)
    if not sweep_id:
        raise HTTPException(status_code=503, detail=redact("failed to parse sweep id: " + sweep_proc.stdout + "\n" + sweep_proc.stderr))

    gpu_info = gpus(payload.remote_host)
    eligible = [item for item in gpu_info["gpus"] if item.get("eligible")]
    max_agents = payload.max_agents if payload.max_agents is not None else 1
    selected = eligible[:max_agents]
    launches: list[dict[str, Any]] = []
    sweep_path = f"{entity}/{project}/{sweep_id}"
    for gpu in selected:
        log_name = f"console_wandb_agent_{entity}_{project}_{sweep_id}_gpu{gpu['index']}.log"
        conda = ""
        if payload.conda_env:
            conda = f"source {shlex.quote(payload.conda_sh)} && conda activate {shlex.quote(payload.conda_env)} && "
        remote = (
            f"cd {shlex.quote(payload.remote_cwd)} && "
            f"export CUDA_VISIBLE_DEVICES={int(gpu['index'])}; "
            f"{conda}nohup wandb agent {shlex.quote(sweep_path)} </dev/null > {shlex.quote(log_name)} 2>&1 & echo $!"
        )
        try:
            proc = ssh_with_key(payload.remote_host, remote, timeout=45)
            require_ok(proc, "wandb agent launch")
            pid = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        except subprocess.TimeoutExpired:
            pid = ""
        launches.append({"gpu_index": gpu["index"], "pid": pid, "log": f"{payload.remote_cwd.rstrip('/')}/{log_name}"})

    job = {
        "job_id": job_id,
        "name": payload.job_name,
        "status": "running" if launches else "attention",
        "entity": entity,
        "project": project,
        "sweep_id": sweep_id,
        "remote_host": payload.remote_host,
        "remote_cwd": payload.remote_cwd,
        "config_path": payload.config_path,
        "remote_config_path": remote_config,
        "agent_pids": [item["pid"] for item in launches if item.get("pid")],
        "monitor": {"gpu_probe": gpu_info, "agent_launches": launches, "requested_by": requested_by},
        "created_at": now(),
        "updated_at": now(),
    }
    upsert_job(job)
    return {
        "success": True,
        "command": "launch-sweep",
        "stage": "agents_launched",
        "classification": "agents_running" if launches else "agents_started_unverified",
        "job": job,
        "result": {
            "sweep": {"sweep_id": sweep_id, "entity": entity, "project": project},
            "gpu_probe": gpu_info,
            "agent_launches": launches,
        },
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": ["用 runner status/show-job 检查任务", "查看远端 agent log 确认首个 run 启动"],
    }


@app.post("/api/runner/register-existing-sweep")
def register_existing(payload: JobPayload, requested_by: str | None = None) -> dict[str, Any]:
    if not payload.sweep_id:
        raise HTTPException(status_code=400, detail="sweep_id is required")
    entity = payload.entity or "HCCS"
    project = payload.project or "DualRefGAD"
    job_id = "job_" + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + "_" + re.sub(r"[^a-zA-Z0-9]+", "_", payload.job_name or payload.sweep_id).strip("_")[:48]
    job = {
        "job_id": job_id,
        "name": payload.job_name or f"registered_{payload.sweep_id}",
        "status": "running",
        "entity": entity,
        "project": project,
        "sweep_id": payload.sweep_id,
        "remote_host": payload.remote_host or "HCCS-25",
        "remote_cwd": payload.remote_cwd or "/home/linziyao/DualRefGAD",
        "config_path": payload.config_path,
        "remote_config_path": REMOTE_CONFIG_PATH,
        "agent_pids": [],
        "monitor": {"registered_existing": True, "requested_by": requested_by},
        "created_at": now(),
        "updated_at": now(),
    }
    upsert_job(job)
    return {
        "success": True,
        "command": "register-existing-sweep",
        "stage": "registered",
        "classification": "existing_sweep_registered",
        "job": job,
        "result": {"job": job, "sweep": {"sweep_id": payload.sweep_id, "entity": entity, "project": project}},
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": ["用 runner status 检查 sweep 状态"],
    }


@app.post("/api/runner/status")
def runner_status(payload: JobPayload) -> dict[str, Any]:
    job = None
    if payload.job_id:
        job = next((item for item in load_jobs() if item.get("job_id") == payload.job_id), None)
    sweep_id = payload.sweep_id or (job or {}).get("sweep_id")
    entity = payload.entity or (job or {}).get("entity") or "HCCS"
    project = payload.project or (job or {}).get("project") or "DualRefGAD"
    sweep = None
    if sweep_id:
        remote = (
            "python3 - <<'PY'\n"
            "import json\n"
            "from wandb.apis.public import Api\n"
            f"s=Api().sweep({(entity + '/' + project + '/' + sweep_id)!r})\n"
            "print(json.dumps({'id': getattr(s,'id',None) or getattr(s,'name',None), 'state': getattr(s,'state',None), 'name': getattr(s,'name',None)}))\n"
            "PY"
        )
        proc = ssh_with_key((job or {}).get("remote_host") or payload.remote_host or "HCCS-25", remote, timeout=60)
        require_ok(proc, "wandb status")
        sweep = json.loads(proc.stdout.strip().splitlines()[-1])
    return {
        "success": True,
        "command": "status",
        "stage": "status_checked",
        "classification": "ok",
        "job": job,
        "result": {"job": job, "sweep": sweep, "degraded": None, "generated_at": now()},
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": [],
    }


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
