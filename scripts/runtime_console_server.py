#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


STATE_DIR = Path("/private/tmp/experiment-console-runtime")
STATE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_PATH = STATE_DIR / "jobs.json"
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


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Experiment Console</title>
  <style>
    body{margin:0;background:#f7f8f5;color:#17201b;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    main{max-width:1060px;margin:0 auto;padding:28px}
    header{display:flex;justify-content:space-between;gap:16px;align-items:center;border-bottom:1px solid #d8ded6;padding-bottom:18px}
    h1{margin:0;font-size:24px;letter-spacing:0}
    .pill{border:1px solid #bad0c3;color:#245b47;background:#edf5ef;padding:6px 10px;border-radius:999px;font-weight:650}
    .panel{margin-top:22px;border:1px solid #d8ded6;background:#fff;border-radius:8px;padding:18px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
    .metric{border-left:3px solid #315d8c;padding-left:12px}
    .label{color:#5e6a61;font-size:12px}
    .value{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:22px;font-weight:750}
    pre{white-space:pre-wrap;background:#f2f4ef;border:1px solid #d8ded6;border-radius:8px;padding:12px;overflow:auto}
  </style>
</head>
<body>
  <main>
    <header><h1>Experiment Console</h1><span class="pill" id="status">读取中</span></header>
    <section class="panel">
      <div class="grid">
        <div class="metric"><div class="label">活跃 Sweep</div><div class="value" id="active">-</div></div>
        <div class="metric"><div class="label">运行作业</div><div class="value" id="running">-</div></div>
        <div class="metric"><div class="label">最近同步</div><div class="value" id="sync">-</div></div>
      </div>
    </section>
    <section class="panel"><h2>当前任务</h2><pre id="jobs">读取中...</pre></section>
  </main>
  <script>
    async function refresh(){
      const r = await fetch('/api/overview');
      const data = await r.json();
      document.getElementById('status').textContent = data.status === 'ok' ? '控制台运行正常' : '控制台降级';
      document.getElementById('active').textContent = data.active_sweeps_count ?? data.active_sweeps ?? 0;
      document.getElementById('running').textContent = data.job_counts?.running ?? 0;
      document.getElementById('sync').textContent = (data.generated_at || '').replace('T',' ').replace('Z','');
      document.getElementById('jobs').textContent = JSON.stringify(data.jobs || [], null, 2);
    }
    refresh(); setInterval(refresh, 15000);
  </script>
</body>
</html>
"""


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    jobs = load_jobs()
    sweeps = []
    for job in jobs:
        if job.get("sweep_id"):
            sweeps.append(
                {
                    "id": job.get("sweep_id"),
                    "name": job.get("sweep_id"),
                    "entity": job.get("entity"),
                    "project": job.get("project"),
                    "state": "RUNNING" if job.get("status") == "running" else str(job.get("status") or "UNKNOWN").upper(),
                    "runCount": 0,
                    "expectedRunCount": 2880,
                    "progress": 0,
                    "createdAt": job.get("created_at"),
                    "source": "temporary_console_runtime",
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
