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
from statistics import mean
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


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
    last_proc: subprocess.CompletedProcess[str] | None = None
    for attempt in range(2):
        proc = subprocess.run(
            ["ssh", host, wrapped],
            input=key + "\n",
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        last_proc = proc
        transient = any(
            marker in (proc.stderr or "")
            for marker in ("Connection closed", "Connection reset", "kex_exchange_identification", "Connection timed out")
        )
        if proc.returncode == 0 or not transient or attempt == 1:
            return proc
        time.sleep(0.8)
    assert last_proc is not None
    return last_proc


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


def find_job(job_id: str | None) -> dict[str, Any] | None:
    if not job_id:
        return None
    return next((item for item in load_jobs() if item.get("job_id") == job_id), None)


def job_sweep_path(job: dict[str, Any], sweep_id: str | None = None, entity: str | None = None, project: str | None = None) -> str:
    actual_sweep = sweep_id or job.get("sweep_id")
    actual_entity = entity or job.get("entity") or "HCCS"
    actual_project = project or job.get("project") or "DualRefGAD"
    if not actual_sweep:
        raise HTTPException(status_code=400, detail="sweep_id is required")
    return f"{actual_entity}/{actual_project}/{actual_sweep}"


def summarize_sweep_for_job(job: dict[str, Any]) -> dict[str, Any] | None:
    if not job.get("sweep_id"):
        return None
    summary = cached_sweep_summary(job)
    expected = int(summary.get("expectedRunCount") or expected_from_config_path(job.get("remote_config_path") or job.get("config_path")) or 0)
    run_count = int(summary.get("runCount") or 0)
    finished = int(summary.get("finished_runs") or 0)
    progress_base = expected or run_count or 1
    return {
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


def agent_health(job: dict[str, Any]) -> dict[str, Any]:
    if not job.get("sweep_id"):
        return {"available": False, "active_processes": [], "recent_logs": [], "recent_run_ids": []}
    sweep_path = job_sweep_path(job)
    sweep_id = job.get("sweep_id")
    remote_cwd = job.get("remote_cwd") or "/home/linziyao/DualRefGAD"
    remote = (
        f"cd {shlex.quote(remote_cwd)} && python3 - <<'PY'\n"
        "import glob, json, os, re, subprocess\n"
        f"sweep_path={sweep_path!r}\n"
        f"sweep_id={sweep_id!r}\n"
        "proc=subprocess.run(['pgrep','-af','wandb agent'], text=True, capture_output=True)\n"
        "active=[]\n"
        "for line in proc.stdout.splitlines():\n"
        "    if sweep_path not in line or 'wandb agent ' not in line:\n"
        "        continue\n"
        "    if 'pgrep -af' in line or 'python3 - <<' in line or 'bash -c read -r WANDB_API_KEY' in line:\n"
        "        continue\n"
        "    parts=line.split(None,1)\n"
        "    active.append({'pid': parts[0], 'command': parts[1] if len(parts)>1 else ''})\n"
        "patterns=[f'console_wandb_agent_*{sweep_id}*.log', f'logs/*{sweep_id}*.log', f'*{sweep_id}*.log']\n"
        "logs=[]\n"
        "for pattern in patterns:\n"
        "    logs.extend(glob.glob(pattern))\n"
        "logs=sorted(set(logs), key=lambda p: os.path.getmtime(p), reverse=True)[:5]\n"
        "run_ids=[]\n"
        "if logs:\n"
        "    try:\n"
        "        text=open(logs[0], encoding='utf-8', errors='replace').read()[-20000:]\n"
        "        run_ids=re.findall(r'(?:Agent Starting Run:|Starting Run:|wandb: Agent Starting Run:)\\s*([A-Za-z0-9_-]+)', text)[-10:]\n"
        "    except Exception:\n"
        "        pass\n"
        "print(json.dumps({'available': True, 'active_processes': active, 'active_count': len(active), 'recent_logs': logs, 'recent_run_ids': run_ids}))\n"
        "PY"
    )
    try:
        proc = ssh_with_key(job.get("remote_host") or "HCCS-25", remote, timeout=15)
        require_ok(proc, "agent health")
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "available": False,
            "active_processes": [],
            "active_count": 0,
            "recent_logs": [],
            "recent_run_ids": [],
            "error": redact(str(exc)),
        }


def select_gpus(host: str, max_agents: int | None = 1) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gpu_info = gpus(host)
    eligible = [item for item in gpu_info["gpus"] if item.get("eligible")]
    limit = max_agents if max_agents is not None else 1
    return gpu_info, eligible[:limit]


def launch_agents_for_job(job: dict[str, Any], *, max_agents: int | None = 1, conda_sh: str | None = None, conda_env: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sweep_path = job_sweep_path(job)
    host = job.get("remote_host") or "HCCS-25"
    remote_cwd = job.get("remote_cwd") or "/home/linziyao/DualRefGAD"
    gpu_info, selected = select_gpus(host, max_agents=max_agents)
    launches: list[dict[str, Any]] = []
    for gpu in selected:
        log_name = f"console_wandb_agent_{sweep_path.replace('/', '_')}_gpu{gpu['index']}.log"
        conda = ""
        env_name = conda_env or job.get("conda_env") or "DualRefGAD"
        if env_name:
            conda = f"source {shlex.quote(conda_sh or '/opt/anaconda3/etc/profile.d/conda.sh')} && conda activate {shlex.quote(env_name)} && "
        remote = (
            f"cd {shlex.quote(remote_cwd)} && "
            f"export CUDA_VISIBLE_DEVICES={int(gpu['index'])}; "
            f"{conda}nohup wandb agent {shlex.quote(sweep_path)} </dev/null > {shlex.quote(log_name)} 2>&1 & echo $!"
        )
        try:
            proc = ssh_with_key(host, remote, timeout=45)
            require_ok(proc, "wandb agent launch")
            pid = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        except subprocess.TimeoutExpired:
            pid = ""
        launches.append({"gpu_index": gpu["index"], "pid": pid, "log": f"{remote_cwd.rstrip('/')}/{log_name}"})
    return gpu_info, launches


def stop_agents_for_job(job: dict[str, Any]) -> dict[str, Any]:
    sweep_path = job_sweep_path(job)
    remote = (
        "python3 - <<'PY'\n"
        "import json, os, signal, subprocess\n"
        f"sweep_path={sweep_path!r}\n"
        "proc=subprocess.run(['pgrep','-af','wandb agent'], text=True, capture_output=True)\n"
        "stopped=[]\n"
        "for line in proc.stdout.splitlines():\n"
        "    if sweep_path not in line:\n"
        "        continue\n"
        "    pid=line.split(None,1)[0]\n"
        "    try:\n"
        "        os.kill(int(pid), signal.SIGTERM)\n"
        "        stopped.append(pid)\n"
        "    except Exception:\n"
        "        pass\n"
        "print(json.dumps({'sweep_path': sweep_path, 'stopped_pids': stopped}))\n"
        "PY"
    )
    proc = ssh_with_key(job.get("remote_host") or "HCCS-25", remote, timeout=30)
    require_ok(proc, "stop agents")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def aggregate_rows(rows: list[dict[str, Any]], metric_keys: list[str], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get("config", {}).get(group) for group in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, items in groups.items():
        metrics: dict[str, Any] = {}
        for metric in metric_keys:
            values = [item.get("metrics", {}).get(metric) for item in items if isinstance(item.get("metrics", {}).get(metric), (int, float))]
            metrics[metric] = {"count": len(values), "mean": mean(values) if values else None}
        out.append({"group": dict(zip(group_keys, key)), "runs": len(items), "metrics": metrics})
    return out


def pull_results_for_job(job: dict[str, Any], payload: "RunnerPayload") -> dict[str, Any]:
    metric_keys = payload.metric_keys or ["final_test_auc", "final_test_ap", "AUC", "AP"]
    group_keys = payload.group_keys or []
    max_runs = payload.max_runs or 200
    sweep_id = payload.sweep_id or job.get("sweep_id")
    sweep_path = job_sweep_path(job, sweep_id=sweep_id)
    remote_cwd = payload.remote_cwd or job.get("remote_cwd") or "/home/linziyao/DualRefGAD"
    remote = (
        f"cd {shlex.quote(remote_cwd)} && python3 - <<'PY'\n"
        "import glob, json, os, re\n"
        f"sweep_id={sweep_id!r}\n"
        f"sweep_path={sweep_path!r}\n"
        f"metric_keys={metric_keys!r}\n"
        f"group_keys={group_keys!r}\n"
        f"max_runs={max_runs!r}\n"
        "patterns=[f'console_wandb_agent_*{sweep_id}*.log', f'logs/*{sweep_id}*.log', f'*{sweep_id}*.log']\n"
        "logs=[]\n"
        "for pattern in patterns:\n"
        "    logs.extend(glob.glob(pattern))\n"
        "run_ids=[]\n"
        "for log in sorted(set(logs), key=lambda p: os.path.getmtime(p), reverse=True):\n"
        "    try:\n"
        "        text=open(log, encoding='utf-8', errors='replace').read()\n"
        "    except Exception:\n"
        "        continue\n"
        "    for rid in re.findall(r'(?:Agent Starting Run:|Starting Run:|wandb: Agent Starting Run:)\\s*([A-Za-z0-9_-]+)', text):\n"
        "        if rid not in run_ids:\n"
        "            run_ids.append(rid)\n"
        "    if len(run_ids) >= max_runs:\n"
        "        break\n"
        "rows=[]; missing=[]; failures=[]\n"
        "for rid in run_ids[:max_runs]:\n"
        "    dirs=glob.glob(f'wandb/run-*-{rid}/files')\n"
        "    if not dirs:\n"
        "        missing.append({'run_id': rid, 'reason': 'wandb_files_missing'})\n"
        "        continue\n"
        "    files=dirs[0]\n"
        "    metrics={}; config={}\n"
        "    try:\n"
        "        summary_path=os.path.join(files, 'wandb-summary.json')\n"
        "        if os.path.exists(summary_path):\n"
        "            summary=json.load(open(summary_path, encoding='utf-8'))\n"
        "            metrics={k: summary.get(k) for k in metric_keys if k in summary}\n"
        "        config_path=os.path.join(files, 'config.yaml')\n"
        "        if os.path.exists(config_path):\n"
        "            import yaml\n"
        "            raw=yaml.safe_load(open(config_path, encoding='utf-8')) or {}\n"
        "            for key, value in raw.items():\n"
        "                if isinstance(value, dict) and 'value' in value:\n"
        "                    config[key]=value.get('value')\n"
        "                else:\n"
        "                    config[key]=value\n"
        "    except Exception as exc:\n"
        "        failures.append({'run_id': rid, 'reason': str(exc)})\n"
        "        continue\n"
        "    rows.append({'run_id': rid, 'metrics': metrics, 'config': {k: config.get(k) for k in group_keys} if group_keys else config})\n"
        "source='remote_local_wandb_files_via_agent_logs'\n"
        "api_error=None\n"
        "if not rows:\n"
        "    try:\n"
        "        from wandb.apis.public import Api\n"
        "        sweep=Api().sweep(sweep_path)\n"
        "        source='remote_wandb_public_api'\n"
        "        for run in list(sweep.runs)[:max_runs]:\n"
        "            rid=getattr(run, 'id', None) or getattr(run, 'name', None)\n"
        "            state=getattr(run, 'state', None)\n"
        "            summary_obj=getattr(run, 'summary', {}) or {}\n"
        "            try:\n"
        "                summary=dict(summary_obj)\n"
        "            except Exception:\n"
        "                summary=getattr(summary_obj, '_json_dict', {}) or {}\n"
        "            config=dict(getattr(run, 'config', {}) or {})\n"
        "            metrics={k: summary.get(k) for k in metric_keys if k in summary}\n"
        "            projected={k: config.get(k) for k in group_keys} if group_keys else config\n"
        "            if metrics:\n"
        "                rows.append({'run_id': rid, 'state': state, 'metrics': metrics, 'config': projected})\n"
        "            else:\n"
        "                missing.append({'run_id': rid, 'state': state, 'reason': 'requested_metrics_missing'})\n"
        "    except Exception as exc:\n"
        "        api_error=str(exc)\n"
        "        failures.append({'run_id': None, 'reason': 'wandb_api_unavailable', 'error': api_error})\n"
        "print(json.dumps({'run_ids_seen': len(run_ids), 'rows': rows, 'missing': missing, 'failures': failures, 'source': source, 'api_error': api_error}))\n"
        "PY"
    )
    try:
        proc = ssh_with_key(payload.remote_host or job.get("remote_host") or "HCCS-25", remote, timeout=min(payload.budget_seconds or 90, 120))
        require_ok(proc, "pull results")
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "classification": "result_sources_unavailable",
            "source": "none",
            "error": redact(str(exc)),
            "valid_results": 0,
            "missing_results": 0,
            "failed_results": 0,
            "groups": [],
            "rows": [],
        }
    rows = data.get("rows") or []
    classification = "results_available" if rows else "partial_results" if (data.get("run_ids_seen") or data.get("missing") or data.get("failures")) else "result_sources_unavailable"
    result = {
        "classification": classification,
        "source": data.get("source"),
        "valid_results": len(rows),
        "missing_results": len(data.get("missing") or []),
        "failed_results": len(data.get("failures") or []),
        "groups": aggregate_rows(rows, metric_keys, group_keys) if group_keys else [],
        "rows": rows[:max_runs],
        "representative_failures": (data.get("failures") or data.get("missing") or [])[:5],
        "api_error": data.get("api_error"),
        "generated_at": now(),
    }
    if result["classification"] == "result_sources_unavailable" and payload.allow_partial:
        result["classification"] = "partial_results"
    return result


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


class RunnerPayload(JobPayload):
    conda_env: str | None = "DualRefGAD"
    conda_sh: str | None = "/opt/anaconda3/etc/profile.d/conda.sh"
    gpu_mode: str | None = "auto"
    max_agents: int | None = 1
    kill_agents: bool = True
    cancel_wandb: bool = False
    mode: str | None = "cancel"
    remote_log_dir: str | None = None
    remote_tmp_dir: str | None = None
    every: str | None = "10m"
    timeout_seconds: int | None = 300
    notify_channel: str | None = None
    notify_target: str | None = None
    terminal_disable: bool = True
    budget_seconds: int | None = 90
    max_runs: int | None = 200
    metric_keys: list[str] = Field(default_factory=list)
    group_keys: list[str] = Field(default_factory=list)
    allow_partial: bool = True


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
            sweep = summarize_sweep_for_job(job)
            if sweep:
                sweeps.append(sweep)
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

    job = {
        "job_id": job_id,
        "name": payload.job_name,
        "status": "attention",
        "entity": entity,
        "project": project,
        "sweep_id": sweep_id,
        "remote_host": payload.remote_host,
        "remote_cwd": payload.remote_cwd,
        "conda_env": payload.conda_env,
        "config_path": payload.config_path,
        "remote_config_path": remote_config,
        "agent_pids": [],
        "monitor": {"requested_by": requested_by},
        "created_at": now(),
        "updated_at": now(),
    }
    gpu_info, launches = launch_agents_for_job(job, max_agents=payload.max_agents, conda_sh=payload.conda_sh, conda_env=payload.conda_env)
    job["status"] = "running" if launches else "attention"
    job["agent_pids"] = [item["pid"] for item in launches if item.get("pid")]
    job["monitor"].update({"gpu_probe": gpu_info, "agent_launches": launches})
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
        "next_actions": ["用 runner status 或 watchdog-once 检查 agent 健康", "用 runner pull-results 拉取可读实验摘要"],
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
def runner_status(payload: RunnerPayload) -> dict[str, Any]:
    job = find_job(payload.job_id)
    sweep_id = payload.sweep_id or (job or {}).get("sweep_id")
    sweep = None
    if job:
        sweep = summarize_sweep_for_job(job)
    elif sweep_id:
        temp_job = {
            "sweep_id": sweep_id,
            "entity": payload.entity or "HCCS",
            "project": payload.project or "DualRefGAD",
            "remote_host": payload.remote_host or "HCCS-25",
            "remote_cwd": payload.remote_cwd or "/home/linziyao/DualRefGAD",
            "remote_config_path": payload.config_path,
            "status": "running",
        }
        sweep = summarize_sweep_for_job(temp_job)
        job = temp_job
    health = agent_health(job) if job else {"available": False, "active_count": 0, "active_processes": []}
    classification = "ok"
    if sweep and sweep.get("state") == "RUNNING" and not health.get("active_count"):
        classification = "attention"
    return {
        "success": True,
        "command": "status",
        "stage": "status_checked",
        "classification": classification,
        "job": job,
        "result": {"job": job, "sweep": sweep, "agent_health": health, "degraded": None, "generated_at": now()},
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": [] if classification == "ok" else ["用 runner recover-agents 恢复现有 sweep 的 agent"],
    }


@app.post("/api/runner/recover-agents")
def recover_agents(payload: RunnerPayload, requested_by: str | None = None) -> dict[str, Any]:
    job = find_job(payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    gpu_info, launches = launch_agents_for_job(job, max_agents=payload.max_agents, conda_sh=payload.conda_sh, conda_env=payload.conda_env)
    job["status"] = "running" if launches else "attention"
    job["agent_pids"] = [item["pid"] for item in launches if item.get("pid")]
    monitor = job.setdefault("monitor", {})
    monitor.update({"recover_gpu_probe": gpu_info, "recover_agent_launches": launches, "requested_by": requested_by, "updated_at": now()})
    job["updated_at"] = now()
    upsert_job(job)
    return {
        "success": True,
        "command": "recover-agents",
        "stage": "agents_recovered",
        "classification": "agents_running" if launches else "agents_started_unverified",
        "job": job,
        "result": {"job": job, "gpu_probe": gpu_info, "agent_launches": launches, "agent_health": agent_health(job)},
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": ["用 runner status 或 watchdog-once 验证 agent 健康"],
    }


@app.post("/api/runner/stop-job")
def stop_job(payload: RunnerPayload) -> dict[str, Any]:
    job = find_job(payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    stop_result = stop_agents_for_job(job) if payload.kill_agents else {"stopped_pids": [], "skipped": True}
    cancel_result = None
    if payload.cancel_wandb and job.get("sweep_id"):
        cancel_result = cancel_sweep(
            RunnerPayload(
                sweep_id=job.get("sweep_id"),
                entity=job.get("entity"),
                project=job.get("project"),
                remote_host=job.get("remote_host"),
                remote_cwd=job.get("remote_cwd"),
                mode="cancel",
            )
        )
    job["status"] = "cancelled"
    job["updated_at"] = now()
    job.setdefault("monitor", {}).update({"stop_agents": stop_result, "cancel_wandb": bool(payload.cancel_wandb), "stopped_at": now()})
    upsert_job(job)
    return {
        "success": True,
        "command": "stop-job",
        "stage": "job_stopped",
        "classification": "job_cancelled",
        "job": job,
        "result": {"job": job, "stop_agents": stop_result, "cancel_wandb": cancel_result},
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": ["用 runner status 确认 job 状态"],
    }


@app.post("/api/runner/cancel-sweep")
def cancel_sweep(payload: RunnerPayload) -> dict[str, Any]:
    sweep_id = payload.sweep_id
    if not sweep_id:
        raise HTTPException(status_code=400, detail="sweep_id is required")
    entity = payload.entity or "HCCS"
    project = payload.project or "DualRefGAD"
    remote_host = payload.remote_host or "HCCS-25"
    remote_cwd = payload.remote_cwd or "/home/linziyao/DualRefGAD"
    mode = "cancel" if payload.mode != "stop" else "stop"
    sweep_path = f"{entity}/{project}/{sweep_id}"
    proc = ssh_with_key(remote_host, f"cd {shlex.quote(remote_cwd)} && wandb sweep --{mode} {shlex.quote(sweep_path)}", timeout=90)
    require_ok(proc, f"wandb sweep {mode}")
    for job in load_jobs():
        if job.get("sweep_id") == sweep_id and job.get("entity") == entity and job.get("project") == project:
            job["status"] = "cancelled" if mode == "cancel" else "attention"
            job["updated_at"] = now()
            job.setdefault("monitor", {}).update({"sweep_cancel": {"mode": mode, "stdout": redact(proc.stdout), "stderr": redact(proc.stderr)}})
            upsert_job(job)
    return {
        "success": True,
        "command": "cancel-sweep",
        "stage": f"sweep_{mode}ed",
        "classification": "sweep_cancelled" if mode == "cancel" else "sweep_stopped",
        "job": None,
        "result": {"sweep": {"sweep_id": sweep_id, "entity": entity, "project": project}, "stdout": redact(proc.stdout[-1000:]), "stderr": redact(proc.stderr[-1000:])},
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": ["用 runner status 或 Console 页面确认 sweep 状态"],
    }


@app.post("/api/runner/auth-check")
def auth_check(payload: RunnerPayload) -> dict[str, Any]:
    has_key = bool(os.environ.get("WANDB_API_KEY"))
    if not has_key:
        return {
            "success": False,
            "command": "auth-check",
            "stage": "auth_checked",
            "classification": "wandb_auth_missing",
            "job": find_job(payload.job_id),
            "result": {"auth": {"has_key": False}},
            "provenance": {"source": "temporary_console_runtime"},
            "next_actions": ["配置 ~/.config/experiment-console/secrets.env 后重启 Console"],
        }
    job = find_job(payload.job_id) or {}
    target = payload.sweep_id or job.get("sweep_id")
    entity = job.get("entity") or payload.entity or "HCCS"
    project = job.get("project") or payload.project or "DualRefGAD"
    remote = (
        "python3 - <<'PY'\n"
        "import json\n"
        "from wandb.apis.public import Api\n"
        "api=Api()\n"
        f"target={target!r}\n"
        f"path={f'{entity}/{project}/{target}' if target else ''!r}\n"
        "if target:\n"
        "    s=api.sweep(path)\n"
        "    out={'has_key': True, 'target_accessible': True, 'sweep_state': getattr(s,'state',None)}\n"
        "else:\n"
        "    out={'has_key': True, 'target_accessible': True}\n"
        "print(json.dumps(out))\n"
        "PY"
    )
    try:
        proc = ssh_with_key(payload.remote_host or job.get("remote_host") or "HCCS-25", remote, timeout=45)
        require_ok(proc, "auth check")
        auth = json.loads(proc.stdout.strip().splitlines()[-1])
        classification = "ok"
    except Exception as exc:
        auth = {"has_key": True, "target_accessible": False, "error": redact(str(exc))}
        classification = "wandb_auth_unverified"
    return {
        "success": classification == "ok",
        "command": "auth-check",
        "stage": "auth_checked",
        "classification": classification,
        "job": job or None,
        "result": {"auth": auth},
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": [] if classification == "ok" else ["检查远端 Python/W&B 环境或网络"],
    }


@app.post("/api/runner/preflight")
def preflight(payload: RunnerPayload) -> dict[str, Any]:
    remote_host = payload.remote_host or "HCCS-25"
    remote_cwd = payload.remote_cwd or "/home/linziyao/DualRefGAD"
    remote_config = payload.config_path or REMOTE_CONFIG_PATH
    remote = (
        "python3 - <<'PY'\n"
        "import json, os, shutil\n"
        f"remote_cwd={remote_cwd!r}\n"
        f"remote_config={remote_config!r}\n"
        "checks={\n"
        "  'remote_cwd_exists': os.path.isdir(remote_cwd),\n"
        "  'config_exists': os.path.exists(remote_config),\n"
        "  'wandb_cli': shutil.which('wandb') is not None,\n"
        "  'python': shutil.which('python3') is not None,\n"
        "}\n"
        "print(json.dumps({'checks': checks, 'ok': all(checks.values())}))\n"
        "PY"
    )
    proc = ssh_with_key(remote_host, remote, timeout=45)
    require_ok(proc, "preflight")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    return {
        "success": bool(result.get("ok")),
        "command": "preflight",
        "stage": "preflight_checked",
        "classification": "ok" if result.get("ok") else "preflight_failed",
        "job": None,
        "result": result,
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": [] if result.get("ok") else ["修复远端路径、配置文件或 W&B CLI 后重试"],
    }


@app.post("/api/runner/pull-results")
def pull_results(payload: RunnerPayload) -> dict[str, Any]:
    job = find_job(payload.job_id)
    if not job:
        if not payload.sweep_id:
            raise HTTPException(status_code=400, detail="job_id or sweep_id is required")
        job = {
            "sweep_id": payload.sweep_id,
            "entity": payload.entity or "HCCS",
            "project": payload.project or "DualRefGAD",
            "remote_host": payload.remote_host or "HCCS-25",
            "remote_cwd": payload.remote_cwd or "/home/linziyao/DualRefGAD",
        }
    result = pull_results_for_job(job, payload)
    return {
        "success": result["classification"] != "result_sources_unavailable",
        "command": "pull-results",
        "stage": "results_pulled",
        "classification": result["classification"],
        "job": job,
        "result": result,
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": [] if result["classification"] == "results_available" else ["稍后重试 pull-results 或检查 watchdog 状态"],
    }


@app.post("/api/runner/repair-watchdog")
def repair_watchdog(payload: RunnerPayload) -> dict[str, Any]:
    job = find_job(payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    monitor = job.setdefault("monitor", {})
    monitor["watchdog"] = {
        "remote_cwd": payload.remote_cwd or job.get("remote_cwd"),
        "remote_log_dir": payload.remote_log_dir,
        "remote_tmp_dir": payload.remote_tmp_dir,
        "conda_sh": payload.conda_sh,
        "conda_env": payload.conda_env,
        "repaired_at": now(),
    }
    if payload.remote_cwd:
        job["remote_cwd"] = payload.remote_cwd
    job["updated_at"] = now()
    upsert_job(job)
    return {
        "success": True,
        "command": "repair-watchdog",
        "stage": "watchdog_repaired",
        "classification": "watchdog_metadata_repaired",
        "job": job,
        "result": {"job": job, "watchdog": monitor["watchdog"]},
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": ["用 runner watchdog-once 验证修复结果"],
    }


@app.post("/api/runner/schedule-monitor")
def schedule_monitor(payload: RunnerPayload) -> dict[str, Any]:
    job = find_job(payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    cron = {
        "cron_id": f"console_watchdog_{payload.job_id}",
        "every": payload.every or "10m",
        "timeout_seconds": payload.timeout_seconds or 300,
        "notify_channel": payload.notify_channel,
        "notify_target": payload.notify_target,
        "active": True,
        "scheduled_at": now(),
    }
    job.setdefault("monitor", {})["cron"] = cron
    job["updated_at"] = now()
    upsert_job(job)
    return {
        "success": True,
        "command": "schedule-monitor",
        "stage": "monitor_scheduled",
        "classification": "monitor_scheduled",
        "job": job,
        "result": {"job": job, "cron": cron},
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": [],
    }


@app.post("/api/runner/unschedule-monitor")
def unschedule_monitor(payload: RunnerPayload) -> dict[str, Any]:
    job = find_job(payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    cron = job.setdefault("monitor", {}).get("cron")
    if cron:
        cron["active"] = False
        cron["unscheduled_at"] = now()
        classification = "monitor_unscheduled"
    else:
        cron = {"active": False}
        classification = "monitor_not_scheduled"
    job["monitor"]["cron"] = cron
    job["updated_at"] = now()
    upsert_job(job)
    return {
        "success": True,
        "command": "unschedule-monitor",
        "stage": "monitor_unscheduled",
        "classification": classification,
        "job": job,
        "result": {"job": job, "cron": cron},
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": [],
    }


@app.post("/api/runner/watchdog-once")
def watchdog_once(payload: RunnerPayload) -> dict[str, Any]:
    status = runner_status(payload)
    body = status.get("result") or {}
    sweep = body.get("sweep") or {}
    health = body.get("agent_health") or {}
    healthy_running = sweep.get("state") == "RUNNING" and bool(health.get("active_count"))
    terminal = sweep.get("state") in {"FINISHED", "CANCELED", "CANCELLED", "FAILED"}
    result = {
        "success": True,
        "silent": healthy_running,
        "message": "" if healthy_running else ("Sweep 已进入终态" if terminal else "Sweep 需要关注：未发现活跃 agent 或状态不可确认"),
        "classification": "healthy_running" if healthy_running else "terminal" if terminal else "attention",
        "status_result": status,
        "event": {"created_at": now(), "classification": status.get("classification")},
    }
    return {
        "success": True,
        "command": "watchdog-once",
        "stage": "watchdog_checked",
        "classification": result["classification"],
        "job": body.get("job"),
        "result": result,
        "provenance": {"source": "temporary_console_runtime"},
        "next_actions": [] if healthy_running else ["用 runner status 或 recover-agents 处理当前任务"],
    }


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
