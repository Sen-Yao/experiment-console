from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from .command import CommandRunner
from .config import Settings


def quote_remote(command: str) -> str:
    return command


def parse_sweep_id(text: str) -> str | None:
    patterns = [
        r"wandb agent ([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)",
        r"Created sweep with ID:\s*([A-Za-z0-9_.\-]+)",
        r"View sweep at .*?/sweeps/([A-Za-z0-9_.\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).rsplit("/", 1)[-1]
    return None


class SSHExecutor:
    def __init__(self, settings: Settings, runner: CommandRunner | None = None):
        self.settings = settings
        self.runner = runner or CommandRunner()

    def run(self, host: str, remote_command: str, *, timeout: int | None = None, input_text: str | None = None):
        argv = ["ssh", host, quote_remote(remote_command)]
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return self.runner.run(argv, timeout=timeout or self.settings.ssh_timeout_seconds, input_text=input_text)
            except Exception as exc:
                last_exc = exc
                if attempt == 1 or not is_transient_ssh_error(exc):
                    raise
        assert last_exc is not None
        raise last_exc

    def run_with_wandb_key(self, host: str, remote_command: str, *, wandb_api_key: str | None, timeout: int | None = None):
        if not wandb_api_key:
            return self.run(host, remote_command, timeout=timeout)
        wrapped = "read -r WANDB_API_KEY; export WANDB_API_KEY; " + remote_command
        return self.run(host, wrapped, timeout=timeout, input_text=wandb_api_key + "\n")

    def preflight(
        self,
        *,
        host: str,
        remote_cwd: str,
        conda_env: str | None = None,
        conda_sh: str | None = None,
        config_path: str | None = None,
    ) -> dict[str, Any]:
        script = """
import json, os, shutil
cwd = __CWD__
config_path = __CONFIG__
conda_env = __CONDA_ENV__
conda_sh = __CONDA_SH__
checks = {
    "remote_cwd_exists": os.path.isdir(cwd),
    "python3_available": shutil.which("python3") is not None,
    "wandb_available": shutil.which("wandb") is not None,
}
if config_path:
    checks["config_path_exists"] = os.path.isfile(config_path)
if conda_env:
    checks["conda_sh_exists"] = os.path.isfile(conda_sh) if conda_sh else False
print(json.dumps({"ok": all(checks.values()), "checks": checks}))
"""
        script = (
            script.replace("__CWD__", repr(remote_cwd))
            .replace("__CONFIG__", repr(config_path))
            .replace("__CONDA_ENV__", repr(conda_env))
            .replace("__CONDA_SH__", repr(conda_sh))
        )
        result = self.run(host, "python3 -c " + shlex.quote(script), timeout=self.settings.command_timeout_seconds)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        classification = "ok" if payload.get("ok") else "preflight_incomplete"
        return {
            **payload,
            "classification": classification,
            "host": host,
            "remote_cwd": remote_cwd,
            "command": result.summary(),
        }

    def probe_gpus(self, host: str, *, min_free_gb: float | None = None, max_util: int | None = None) -> dict[str, Any]:
        query = (
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu "
            "--format=csv,noheader,nounits"
        )
        result = self.run(host, query)
        gpus = []
        threshold_gb = min_free_gb if min_free_gb is not None else self.settings.gpu_min_free_gb
        util_limit = max_util if max_util is not None else self.settings.gpu_max_util
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue
            index, name, total, used, free, util = parts
            try:
                free_gb = float(free) / 1024
                util_pct = int(float(util))
                gpus.append({
                    "index": int(index),
                    "name": name,
                    "memory_total_mb": int(float(total)),
                    "memory_used_mb": int(float(used)),
                    "memory_free_mb": int(float(free)),
                    "utilization_gpu": util_pct,
                    "eligible": free_gb >= threshold_gb and util_pct <= util_limit,
                })
            except ValueError:
                continue
        return {"host": host, "gpus": gpus, "eligible_count": sum(1 for gpu in gpus if gpu["eligible"])}

    def auth_check(
        self,
        *,
        host: str,
        remote_cwd: str | None,
        sweep_path: str | None,
        wandb_api_key: str | None,
    ) -> dict[str, Any]:
        if not wandb_api_key:
            return {"ok": False, "classification": "wandb_auth_missing", "has_key": False}
        script = """
import json
sweep_path = __SWEEP_PATH__
try:
    import wandb
    api = wandb.Api()
    sweep = api.sweep(sweep_path) if sweep_path else None
    state = getattr(sweep, "state", None) if sweep is not None else None
    print(json.dumps({"ok": True, "classification": "auth_ok", "has_key": True, "target_accessible": True, "sweep_state": state}))
except Exception as exc:
    print(json.dumps({"ok": False, "classification": "wandb_auth_failed", "has_key": True, "target_accessible": False, "error": str(exc)}))
"""
        script = script.replace("__SWEEP_PATH__", repr(sweep_path))
        prefix = f"cd {shlex.quote(remote_cwd)} && " if remote_cwd else ""
        result = self.run_with_wandb_key(
            host,
            prefix + "python3 -c " + shlex.quote(script),
            wandb_api_key=wandb_api_key,
            timeout=self.settings.command_timeout_seconds,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload.update({"host": host, "sweep_path": sweep_path, "command": result.summary()})
        return payload

    def launch_agent(
        self,
        *,
        host: str,
        remote_cwd: str,
        sweep_path: str,
        gpu_index: int,
        conda_env: str | None,
        conda_sh: str,
        wandb_api_key: str | None = None,
    ) -> dict[str, Any]:
        agent_cmd = ["wandb", "agent", sweep_path]
        prefix = ""
        if conda_env:
            agent_cmd = ["conda", "run", "-n", conda_env, "--no-capture-output", *agent_cmd]
            prefix = f"source {shlex.quote(conda_sh)} && "
        inner = (
            f"cd {shlex.quote(remote_cwd)} && "
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu_index))} && "
            f"{prefix}"
            + " ".join(shlex.quote(part) for part in agent_cmd)
        )
        log_name = f"console_wandb_agent_{sweep_path.replace('/', '_')}_gpu{gpu_index}.log"
        remote = (
            f"cd {shlex.quote(remote_cwd)} && "
            f"nohup bash -lc {shlex.quote(inner)} > {shlex.quote(log_name)} 2>&1 < /dev/null & echo $!"
        )
        result = self.run_with_wandb_key(host, remote, wandb_api_key=wandb_api_key, timeout=self.settings.command_timeout_seconds)
        pid = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        return {
            "host": host,
            "gpu_index": gpu_index,
            "pid": pid,
            "log": f"{remote_cwd.rstrip('/')}/{log_name}",
            "conda_env": conda_env,
            "command": result.summary(),
        }

    def stop_agents(self, *, host: str, sweep_path: str) -> dict[str, Any]:
        pattern = f"wandb agent {sweep_path}"
        quoted_pattern = shlex.quote(pattern)
        remote = (
            f"pattern={quoted_pattern}; "
            f"pids=$(pgrep -af -- \"$pattern\" | awk '{{print $1}}'); "
            f"if [ -n \"$pids\" ]; then kill -TERM $pids; fi; "
            f"printf '%s\\n' $pids"
        )
        result = self.run(host, remote, timeout=self.settings.command_timeout_seconds)
        pids = []
        for line in result.stdout.splitlines():
            parts = line.strip().split(None, 1)
            if parts and parts[0].isdigit():
                pids.append(parts[0])
        return {"host": host, "stopped_pids": pids, "command": result.summary()}

    def create_sweep(
        self,
        *,
        host: str,
        remote_cwd: str,
        config_path: Path,
        entity: str,
        project: str,
        wandb_api_key: str | None,
    ) -> dict[str, Any]:
        remote_config = str(config_path)
        if config_path.exists():
            remote_dir = f"{remote_cwd.rstrip('/')}/.experiment-console/configs"
            remote_config = f"{remote_dir}/{config_path.name}"
            self.run(host, f"mkdir -p {shlex.quote(remote_dir)} && cat > {shlex.quote(remote_config)}", input_text=config_path.read_text(encoding="utf-8"))
        remote = f"cd {shlex.quote(remote_cwd)} && wandb sweep --entity {shlex.quote(entity)} --project {shlex.quote(project)} {shlex.quote(remote_config)}"
        result = self.run_with_wandb_key(host, remote, wandb_api_key=wandb_api_key, timeout=self.settings.command_timeout_seconds)
        sweep_id = parse_sweep_id(result.stdout + "\n" + result.stderr)
        if not sweep_id:
            raise RuntimeError("failed to parse sweep id from remote wandb sweep output")
        return {"sweep_id": sweep_id, "entity": entity, "project": project, "command": result.summary(), "remote_config_path": remote_config}

    def cancel_sweep(
        self,
        *,
        host: str,
        remote_cwd: str,
        sweep_path: str,
        mode: str,
        wandb_api_key: str | None,
    ) -> dict[str, Any]:
        flag = "--stop" if mode == "stop" else "--cancel"
        remote = f"cd {shlex.quote(remote_cwd)} && wandb sweep {flag} {shlex.quote(sweep_path)}"
        result = self.run_with_wandb_key(host, remote, wandb_api_key=wandb_api_key, timeout=self.settings.command_timeout_seconds)
        return {
            "classification": "sweep_stopped" if mode == "stop" else "sweep_cancelled",
            "mode": mode,
            "sweep_path": sweep_path,
            "command": result.summary(),
        }

    def pull_results(
        self,
        *,
        host: str,
        remote_cwd: str,
        sweep_id: str,
        run_ids: list[str],
        budget_seconds: int,
        max_runs: int,
        metric_keys: list[str],
        group_keys: list[str],
    ) -> dict[str, Any]:
        script = """
import glob, json, os, time
cwd = __CWD__
sweep_id = __SWEEP_ID__
run_ids = __RUN_IDS__
max_runs = __MAX_RUNS__
metric_keys = __METRIC_KEYS__
group_keys = __GROUP_KEYS__
deadline = time.time() + __BUDGET_SECONDS__
if not run_ids:
    raise SystemExit("target run ids unavailable")

def load_json(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def scalar_map(data):
    out = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out

rows = []
for run_id in run_ids[:max_runs]:
    if len(rows) >= max_runs or time.time() > deadline:
        break
    summary_paths = glob.glob(os.path.join(cwd, "wandb", f"run-*{run_id}", "files", "wandb-summary.json"))
    summary_paths = sorted(set(summary_paths), key=lambda p: os.path.getmtime(p), reverse=True)
    output_patterns = [
        os.path.join(cwd, "investigations", "**", "experiments", "outputs", f"*{run_id}*.json"),
        os.path.join(cwd, "outputs", f"*{run_id}*.json"),
    ]
    output_paths = []
    for pattern in output_patterns:
        output_paths.extend(glob.glob(pattern, recursive=True))
    output_paths = sorted(set(output_paths), key=lambda p: os.path.getmtime(p), reverse=True)
    data = {}
    sources = []
    if summary_paths:
        data.update(load_json(summary_paths[0]))
        sources.append(summary_paths[0])
    for output_path in output_paths[:3]:
        data.update(load_json(output_path))
        sources.append(output_path)
    if not data:
        rows.append({"run_id": run_id, "sources": [], "config": {}, "metrics": {}, "has_scientific_result": False})
        continue
    flat = scalar_map(data)
    metrics = {key: flat.get(key) for key in metric_keys if key in flat} if metric_keys else {
        key: value for key, value in flat.items()
        if isinstance(value, (int, float, bool)) and not key.startswith("_")
    }
    config = {key: flat.get(key) for key in group_keys if key in flat} if group_keys else {}
    has_result = bool(metrics)
    rows.append({"run_id": run_id, "sources": sources, "config": config, "metrics": metrics, "has_scientific_result": has_result})
if not rows:
    raise SystemExit("no remote result files found")
print(json.dumps({
    "source": "remote_local_files",
    "sweep_id": sweep_id,
    "rows": rows,
    "valid_results": sum(1 for row in rows if row["has_scientific_result"]),
    "missing_results": sum(1 for row in rows if not row["has_scientific_result"]),
    "failed_results": 0,
    "partial": len(rows) >= max_runs,
}))
"""
        script = (
            script.replace("__CWD__", repr(remote_cwd))
            .replace("__SWEEP_ID__", repr(sweep_id))
            .replace("__RUN_IDS__", repr(run_ids[:max_runs]))
            .replace("__MAX_RUNS__", repr(max_runs))
            .replace("__METRIC_KEYS__", repr(metric_keys))
            .replace("__GROUP_KEYS__", repr(group_keys))
            .replace("__BUDGET_SECONDS__", repr(budget_seconds))
        )
        result = self.run(host, "python3 -c " + shlex.quote(script), timeout=budget_seconds + 10)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload["host"] = host
        payload["command"] = result.summary()
        return payload


def is_transient_ssh_error(exc: Exception) -> bool:
    result = getattr(exc, "result", None)
    stderr = str(getattr(result, "stderr", "") or "") if result is not None else str(exc)
    return any(
        marker in stderr
        for marker in (
            "Connection closed",
            "Connection reset",
            "kex_exchange_identification",
            "Connection timed out",
        )
    )
