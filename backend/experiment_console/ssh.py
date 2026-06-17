from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from .command import CommandRunner
from .config import Settings
from .redaction import redact_text


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


    def read_remote_file(
        self,
        *,
        host: str,
        remote_path: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        remote = f"python3 - <<'PY'\nfrom pathlib import Path\npath = Path({remote_path!r})\nprint(path.read_text(encoding=\"utf-8\"))\nPY"
        result = self.run(host, remote, timeout=timeout or self.settings.command_timeout_seconds)
        return {
            "host": host,
            "remote_path": remote_path,
            "text": result.stdout,
            "command": result.summary(),
        }

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
import json, os, shlex, shutil, subprocess
cwd = __CWD__
config_path = __CONFIG__
conda_env = __CONDA_ENV__
conda_sh = __CONDA_SH__
checks = {
    "remote_cwd_exists": os.path.isdir(cwd),
    "python3_available": shutil.which("python3") is not None,
    "wandb_available": shutil.which("wandb") is not None,
    "conda_sh_exists": os.path.isfile(conda_sh) if conda_sh else False,
    "conda_env": conda_env,
}
python = shutil.which("python3") or "python3"
if conda_sh and conda_env:
    probe = (
        "source " + shlex.quote(conda_sh)
        + " && conda activate " + shlex.quote(conda_env)
        + " && python - <<'PY'\\n"
        "import json, shutil, torch\\n"
        "print(json.dumps({'ok': True, 'python': shutil.which('python'), 'torch': getattr(torch, '__version__', None)}))\\n"
        "PY"
    )
    result = subprocess.run(['bash', '-lc', probe], capture_output=True, text=True)
    checks['conda_activate_ok'] = result.returncode == 0
    checks['torch_import_ok'] = result.returncode == 0
    if result.returncode != 0:
        checks['conda_probe_stdout'] = result.stdout[-1000:]
        checks['conda_probe_stderr'] = result.stderr[-1000:]
if config_path:
    checks["config_path_exists"] = os.path.isfile(config_path)
print(json.dumps({"ok": all(v for v in checks.values() if isinstance(v, bool)), "checks": checks}))
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
        if conda_env:
            launch_prefix = f"source {shlex.quote(conda_sh)} && conda run -n {shlex.quote(conda_env)} "
        else:
            launch_prefix = ""
        inner = (
            f"cd {shlex.quote(remote_cwd)} && "
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu_index))} && "
            f"{launch_prefix}"
            f"wandb agent {shlex.quote(sweep_path)}"
        )
        log_name = f"console_wandb_agent_{sweep_path.replace('/', '_')}_gpu{gpu_index}.log"
        log_path = f"{remote_cwd.rstrip('/')}/{log_name}"
        script = """
import json, os, subprocess
cwd = __CWD__
inner = __INNER__
log_path = __LOG_PATH__
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, "ab", buffering=0) as log:
    proc = subprocess.Popen(
        ["bash", "-lc", inner],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        start_new_session=True,
    )
print(json.dumps({"ok": True, "pid": proc.pid, "log": log_path}))
"""
        script = (
            script.replace("__CWD__", repr(remote_cwd))
            .replace("__INNER__", repr(inner))
            .replace("__LOG_PATH__", repr(log_path))
        )
        remote = f"cd {shlex.quote(remote_cwd)} && python3 -c {shlex.quote(script)}"
        result = self.run_with_wandb_key(host, remote, wandb_api_key=wandb_api_key, timeout=self.settings.command_timeout_seconds)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        pid = str(payload.get("pid") or "").strip()
        return {
            "host": host,
            "gpu_index": gpu_index,
            "pid": pid,
            "log": log_path,
            "conda_env": conda_env,
            "command": result.summary(),
            "launcher": payload,
        }

    def launch_run(
        self,
        *,
        host: str,
        remote_cwd: str,
        job_id: str,
        argv: list[str],
        gpu_index: int,
        conda_env: str | None,
        conda_sh: str,
        wandb_api_key: str | None = None,
        result_path: str | None = None,
    ) -> dict[str, Any]:
        runs_dir = f"{remote_cwd.rstrip('/')}/.experiment_console/runs"
        log_path = f"{runs_dir}/{job_id}.log"
        status_path = f"{runs_dir}/{job_id}.status.json"
        result_path = result_path or f"{runs_dir}/{job_id}.result.json"
        command_json = json.dumps([str(item) for item in argv])
        script = """
import json, os, subprocess, sys
from datetime import datetime, timezone

job_id = __JOB_ID__
cwd = __CWD__
argv = __ARGV__
log_path = __LOG_PATH__
status_path = __STATUS_PATH__
result_path = __RESULT_PATH__
gpu_index = __GPU_INDEX__
os.makedirs(os.path.dirname(status_path), exist_ok=True)
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

status = {
    "job_id": job_id,
    "pid": os.getpid(),
    "started_at": now(),
    "finished_at": None,
    "exit_code": None,
    "command": argv,
    "log_path": log_path,
    "status_path": status_path,
    "result_path": result_path,
}
with open(status_path, "w", encoding="utf-8") as handle:
    json.dump(status, handle, ensure_ascii=False)
with open(log_path, "ab", buffering=0) as log:
    proc = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    status["child_pid"] = proc.pid
    with open(status_path, "w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=False)
    exit_code = proc.wait()
status["finished_at"] = now()
status["exit_code"] = exit_code
with open(status_path, "w", encoding="utf-8") as handle:
    json.dump(status, handle, ensure_ascii=False)
"""
        script = (
            script.replace("__JOB_ID__", repr(job_id))
            .replace("__CWD__", repr(remote_cwd))
            .replace("__ARGV__", command_json)
            .replace("__LOG_PATH__", repr(log_path))
            .replace("__STATUS_PATH__", repr(status_path))
            .replace("__RESULT_PATH__", repr(result_path))
            .replace("__GPU_INDEX__", repr(gpu_index))
        )
        if conda_env:
            launch_prefix = f"source {shlex.quote(conda_sh)} && conda activate {shlex.quote(conda_env)} && "
        else:
            launch_prefix = ""
        status_probe = """
import json, os, time
status_path = __STATUS_PATH__
launcher_pid = os.environ.get("LAUNCHER_PID", "")
deadline = time.time() + 30
payload = {}
while time.time() < deadline:
    try:
        with open(status_path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict) and loaded.get("child_pid"):
            payload = loaded
            break
    except Exception:
        time.sleep(1)
payload["ok"] = bool(payload.get("child_pid"))
payload["launcher_pid"] = launcher_pid
payload.setdefault("status_path", status_path)
print(json.dumps(payload))
"""
        status_probe = (
            status_probe
            .replace("__STATUS_PATH__", repr(status_path))
        )
        remote = (
            f"mkdir -p {shlex.quote(runs_dir)} && cd {shlex.quote(remote_cwd)} && "
            f"nohup bash -lc {shlex.quote(launch_prefix + 'python -c ' + shlex.quote(script))} "
            f"> {shlex.quote(log_path)}.launcher 2>&1 < /dev/null & launcher_pid=$!; "
            f"LAUNCHER_PID=$launcher_pid python3 -c {shlex.quote(status_probe)}"
        )
        result = self.run_with_wandb_key(host, remote, wandb_api_key=wandb_api_key, timeout=self.settings.command_timeout_seconds)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        pid = str(payload.get("child_pid") or "").strip()
        return {
            "host": host,
            "gpu_index": gpu_index,
            "pid": pid,
            "log": log_path,
            "status_path": status_path,
            "result_path": result_path,
            "conda_env": conda_env,
            "argv": argv,
            "command": result.summary(),
            "launcher": payload,
        }

    def check_run_status(self, *, host: str, status_path: str, pids: list[str] | None = None) -> dict[str, Any]:
        pid_json = json.dumps([str(pid) for pid in (pids or []) if str(pid).strip()])
        script = """
import json, os, signal
status_path = __STATUS_PATH__
pids = [int(pid) for pid in json.loads(__PIDS__) if str(pid).isdigit()]
status = {}
try:
    with open(status_path, encoding="utf-8") as handle:
        loaded = json.load(handle)
        if isinstance(loaded, dict):
            status = loaded
except FileNotFoundError:
    status = {"status_path": status_path, "missing": True}
alive = []
for pid in pids:
    try:
        os.kill(pid, 0)
        alive.append(str(pid))
    except ProcessLookupError:
        pass
    except PermissionError:
        alive.append(str(pid))
child_pid = status.get("child_pid")
if child_pid:
    try:
        os.kill(int(child_pid), 0)
        alive.append(str(child_pid))
    except Exception:
        pass
status["alive_pids"] = sorted(set(alive))
print(json.dumps(status))
"""
        script = script.replace("__STATUS_PATH__", repr(status_path)).replace("__PIDS__", repr(pid_json))
        result = self.run(host, "python3 -c " + shlex.quote(script), timeout=self.settings.command_timeout_seconds)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload["host"] = host
        payload["command"] = result.summary()
        return payload

    def check_agent_processes(
        self,
        *,
        host: str,
        sweep_path: str | None = None,
        pids: list[str] | None = None,
    ) -> dict[str, Any]:
        pid_json = json.dumps([str(pid) for pid in (pids or []) if str(pid).strip()])
        script = """
import json, os, subprocess
pids = [int(pid) for pid in json.loads(__PIDS__) if str(pid).isdigit()]
sweep_path = __SWEEP_PATH__
alive_pids = []
for pid in pids:
    try:
        os.kill(pid, 0)
        alive_pids.append(str(pid))
    except ProcessLookupError:
        pass
    except PermissionError:
        alive_pids.append(str(pid))
pgrep = []
if sweep_path:
    try:
        out = subprocess.run(["pgrep", "-af", "wandb agent " + sweep_path], capture_output=True, text=True)
        for line in out.stdout.splitlines():
            parts = line.split(None, 1)
            if parts and parts[0].isdigit():
                pgrep.append({"pid": parts[0], "command": parts[1] if len(parts) > 1 else ""})
    except Exception as exc:
        pgrep.append({"error": str(exc)})
print(json.dumps({
    "tracked_pids": [str(pid) for pid in pids],
    "alive_pids": sorted(set(alive_pids)),
    "pgrep": pgrep,
}))
"""
        script = script.replace("__PIDS__", repr(pid_json)).replace("__SWEEP_PATH__", repr(sweep_path))
        result = self.run(host, "python3 -c " + shlex.quote(script), timeout=self.settings.command_timeout_seconds)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload.update({"host": host, "sweep_path": sweep_path, "command": result.summary()})
        return payload

    def diagnose_agent_failure(
        self,
        *,
        host: str,
        remote_cwd: str,
        launches: list[dict[str, Any]],
        pids: list[str] | None = None,
        sweep_path: str | None = None,
        tail_lines: int = 200,
    ) -> dict[str, Any]:
        payload = {
            "launches": launches,
            "pids": [str(pid) for pid in (pids or []) if str(pid).strip()],
            "sweep_path": sweep_path,
            "tail_lines": tail_lines,
        }
        script = """
import json, os, subprocess
payload = json.loads(__PAYLOAD__)
remote_cwd = __REMOTE_CWD__
tail_lines = int(payload.get("tail_lines") or 200)

def tail_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        return "".join(lines[-tail_lines:])
    except FileNotFoundError:
        return None
    except Exception as exc:
        return "failed to read log: " + str(exc)

def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False

pid_state = {
    "tracked_pids": payload.get("pids") or [],
    "alive_pids": [pid for pid in payload.get("pids") or [] if str(pid).isdigit() and alive(pid)],
    "pgrep": [],
}
sweep_path = payload.get("sweep_path")
if sweep_path:
    try:
        out = subprocess.run(["pgrep", "-af", "wandb agent " + sweep_path], capture_output=True, text=True)
        for line in out.stdout.splitlines():
            parts = line.split(None, 1)
            if parts and parts[0].isdigit():
                pid_state["pgrep"].append({"pid": parts[0], "command": parts[1] if len(parts) > 1 else ""})
    except Exception as exc:
        pid_state["pgrep"].append({"error": str(exc)})

logs = []
for launch in payload.get("launches") or []:
    if not isinstance(launch, dict):
        continue
    path = launch.get("log")
    if not path:
        continue
    if not os.path.isabs(path):
        path = os.path.join(remote_cwd, path)
    text = tail_text(path)
    logs.append({
        "gpu_index": launch.get("gpu_index"),
        "pid": launch.get("pid"),
        "path": path,
        "exists": text is not None,
        "tail": text or "",
    })
print(json.dumps({"pid_state": pid_state, "logs": logs}))
"""
        script = script.replace("__PAYLOAD__", repr(json.dumps(payload))).replace("__REMOTE_CWD__", repr(remote_cwd))
        result = self.run(host, "python3 -c " + shlex.quote(script), timeout=self.settings.command_timeout_seconds)
        remote_payload = json.loads(result.stdout.strip().splitlines()[-1])
        return build_failure_diagnostics(
            host=host,
            remote_cwd=remote_cwd,
            sweep_path=sweep_path,
            launches=launches,
            pid_state=remote_payload.get("pid_state") or {},
            logs=remote_payload.get("logs") or [],
            command=result.summary(),
        )

    def stop_pids(self, *, host: str, pids: list[str]) -> dict[str, Any]:
        pid_json = json.dumps([str(pid) for pid in pids if str(pid).strip()])
        script = """
import json, os, signal
target_pids = [int(pid) for pid in json.loads(__PIDS__) if str(pid).isdigit()]
stopped = []
missing = []
still_running = []
for pid in target_pids:
    try:
        os.kill(pid, signal.SIGTERM)
        stopped.append(str(pid))
    except ProcessLookupError:
        missing.append(str(pid))
    except Exception:
        still_running.append(str(pid))
print(json.dumps({"stopped_pids": sorted(set(stopped)), "missing_pids": sorted(set(missing)), "still_running_pids": sorted(set(still_running))}))
"""
        script = script.replace("__PIDS__", repr(pid_json))
        result = self.run(host, "python3 -c " + shlex.quote(script), timeout=self.settings.command_timeout_seconds)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload.update({"host": host, "command": result.summary()})
        return payload

    def stop_agents(self, *, host: str, sweep_path: str, pids: list[str] | None = None) -> dict[str, Any]:
        pid_terms = " ".join(shlex.quote(str(pid)) for pid in (pids or []) if str(pid).strip())
        pid_kill = f"kill -TERM {pid_terms} 2>/dev/null || true; " if pid_terms else ""
        remote = (
            pid_kill
            + f"pgrep -af {shlex.quote('wandb agent ' + sweep_path)} "
            + "| awk '{print $1}' | xargs -r kill -TERM"
        )
        result = self.run(host, remote, timeout=self.settings.command_timeout_seconds)
        return {"host": host, "sweep_path": sweep_path, "command": result.summary()}

    def pull_single_run_result(
        self,
        *,
        host: str,
        status_path: str,
        result_path: str | None,
        metric_keys: list[str],
        group_keys: list[str],
    ) -> dict[str, Any]:
        script = """
import json, os
status_path = __STATUS_PATH__
result_path = __RESULT_PATH__
metric_keys = __METRIC_KEYS__
group_keys = __GROUP_KEYS__

def load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

status = load(status_path)
data = load(result_path) if result_path else {}
flat = {key: value for key, value in data.items() if isinstance(value, (str, int, float, bool)) or value is None}
metrics = {key: flat.get(key) for key in metric_keys if key in flat} if metric_keys else {
    key: value for key, value in flat.items()
    if isinstance(value, (int, float, bool)) and not key.startswith("_")
}
config = {key: flat.get(key) for key in group_keys if key in flat} if group_keys else {}
row = {
    "run_id": status.get("job_id"),
    "state": "finished" if status.get("exit_code") == 0 else "failed" if status.get("exit_code") is not None else "running",
    "config": config,
    "metrics": metrics,
    "has_scientific_result": bool(metrics),
    "status": status,
}
print(json.dumps({
    "source": "remote_single_run_files",
    "status": status,
    "rows": [row],
    "valid_results": 1 if row["has_scientific_result"] else 0,
    "missing_results": 0 if row["has_scientific_result"] else 1,
    "failed_results": 1 if row["state"] == "failed" else 0,
    "partial": row["state"] == "running",
}))
"""
        script = (
            script.replace("__STATUS_PATH__", repr(status_path))
            .replace("__RESULT_PATH__", repr(result_path))
            .replace("__METRIC_KEYS__", repr(metric_keys))
            .replace("__GROUP_KEYS__", repr(group_keys))
        )
        result = self.run(host, "python3 -c " + shlex.quote(script), timeout=self.settings.command_timeout_seconds)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload["host"] = host
        payload["command"] = result.summary()
        return payload

    def create_sweep(
        self,
        *,
        host: str,
        remote_cwd: str,
        remote_config: str,
        entity: str,
        project: str,
        wandb_api_key: str | None,
    ) -> dict[str, Any]:
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


ERROR_SIGNAL_PATTERNS = [
    ("traceback", re.compile(r"Traceback \(most recent call last\):[\s\S]*?(?=\n\S|\Z)", re.MULTILINE)),
    ("cuda_oom", re.compile(r"(?:CUDA out of memory|out of memory)", re.IGNORECASE)),
    ("import_error", re.compile(r"(?:ModuleNotFoundError|ImportError):[^\n]+")),
    ("file_error", re.compile(r"(?:FileNotFoundError|No such file or directory):[^\n]+")),
    ("runtime_error", re.compile(r"RuntimeError:[^\n]+")),
    ("wandb_error", re.compile(r"(?:wandb:[^\n]*(?:ERROR|error)[^\n]*|CommError:[^\n]+|Authentication failed[^\n]*)", re.IGNORECASE)),
    ("command_error", re.compile(r"(?:command not found|No such file or directory|conda:|CondaError)[^\n]*", re.IGNORECASE)),
]


def build_failure_diagnostics(
    *,
    host: str,
    remote_cwd: str,
    sweep_path: str | None,
    launches: list[dict[str, Any]],
    pid_state: dict[str, Any],
    logs: list[dict[str, Any]],
    command: dict[str, Any] | None = None,
) -> dict[str, Any]:
    log_tails = []
    signals = []
    for item in logs:
        if not isinstance(item, dict):
            continue
        tail = str(item.get("tail") or "")
        log_tails.append({
            "gpu_index": item.get("gpu_index"),
            "pid": item.get("pid"),
            "path": item.get("path"),
            "exists": bool(item.get("exists")),
            "tail": redact_text(tail),
        })
        if not item.get("exists"):
            signals.append({
                "kind": "missing_log",
                "source": item.get("path"),
                "excerpt": "agent log was not found",
            })
            continue
        signals.extend(extract_error_signals(redact_text(tail), source=str(item.get("path") or "")))
    alive = pid_state.get("alive_pids") or []
    pgrep = pid_state.get("pgrep") or []
    if not alive and not pgrep:
        signals.append({
            "kind": "agent_process_missing",
            "source": "pid_state",
            "excerpt": "no tracked or pgrep-matched wandb agent process is alive",
        })
    classification = "failure_signals_found" if signals else "no_failure_signal_found"
    summary = summarize_failure_signals(signals)
    return {
        "classification": classification,
        "summary": summary,
        "host": host,
        "remote_cwd": remote_cwd,
        "sweep_path": sweep_path,
        "sources": [item.get("path") for item in log_tails if item.get("path")],
        "pid_state": pid_state,
        "error_signals": signals[:20],
        "log_tails": log_tails,
        "launches": launches,
        "command": command,
        "next_actions": next_actions_for_diagnostics(signals),
    }


def extract_error_signals(text: str, *, source: str = "") -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for kind, pattern in ERROR_SIGNAL_PATTERNS:
        for match in pattern.finditer(text):
            excerpt = compact_excerpt(redact_text(match.group(0)))
            if excerpt:
                signals.append({"kind": kind, "source": source, "excerpt": excerpt})
            if len(signals) >= 20:
                return signals
    return signals


def compact_excerpt(text: str, max_chars: int = 1200) -> str:
    cleaned = redact_text(text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[-max_chars:].lstrip() + "...<truncated>"


def summarize_failure_signals(signals: list[dict[str, Any]]) -> str:
    kinds = [str(item.get("kind")) for item in signals if item.get("kind")]
    if not kinds:
        return "No recognizable failure signal found in available agent log tails."
    priority = [
        "cuda_oom",
        "traceback",
        "runtime_error",
        "import_error",
        "file_error",
        "wandb_error",
        "command_error",
        "agent_process_missing",
        "missing_log",
    ]
    ordered = [kind for kind in priority if kind in kinds]
    return redact_text("Detected " + ", ".join(ordered[:4]) + ".")


def next_actions_for_diagnostics(signals: list[dict[str, Any]]) -> list[str]:
    kinds = {str(item.get("kind")) for item in signals}
    if "cuda_oom" in kinds:
        return ["降低 batch/model 尺寸或选择更空闲 GPU；修复后对既有 sweep/job 先 stop/cancel 再重新 launch。"]
    if {"traceback", "runtime_error", "import_error", "file_error"} & kinds:
        return ["根据 error_signals 中的 traceback/excerpt 在本地修复代码，走 GitHub 同步后再创建新的验证 run/sweep。"]
    if "wandb_error" in kinds:
        return ["检查 WANDB_API_KEY、entity/project/sweep 权限与网络；修复后对同一 job 执行 recover-agents。"]
    if "command_error" in kinds:
        return ["检查远端 conda/env/path/program 配置；修复环境后对同一 job 执行 recover-agents。"]
    if "agent_process_missing" in kinds:
        return ["agent 进程已消失；查看 log_tails/error_signals 后决定修复代码或 recover-agents。"]
    return ["没有提取到明确错误；必要时 SSH 到远端查看 sources 中的完整日志。"]
