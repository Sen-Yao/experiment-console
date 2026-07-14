from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .command import CommandRunner
from .config import Settings
from .redaction import redact_text


MAX_RESULT_ARTIFACT_FILE_BYTES = 8 * 1024 * 1024
MAX_RESULT_ARTIFACT_TOTAL_BYTES = 64 * 1024 * 1024
SSH_CONNECTION_OPTIONS = (
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPersist=60",
    "-o",
    "ControlPath=/tmp/experiment-console-ssh-%C",
)


def quote_remote(command: str) -> str:
    return command


def validate_ssh_host(host: str) -> str:
    value = str(host or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise ValueError("SSH host must be a configured alias or hostname, not an option or command")
    return value


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


ARGV_INCOMPATIBLE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"(?:^|\n)\s*usage:",
        r"unrecognized arguments?:",
        r"unknown (?:option|argument)",
        r"no such option",
        r"expected one argument",
        r"requires an argument",
        r"invalid choice",
        r"missing (?:option|argument|required)",
        r"too few arguments",
    ]
]


def classify_argv_probe(returncode: int | None, stdout: str, stderr: str, *, timed_out: bool = False) -> str:
    if timed_out or returncode is None:
        return "argv_probe_unavailable"
    if returncode == 0:
        return "argv_compatible"
    combined = f"{stdout}\n{stderr}"
    if any(pattern.search(combined) for pattern in ARGV_INCOMPATIBLE_PATTERNS):
        return "argv_incompatible"
    return "argv_probe_unavailable"


class SSHExecutor:
    def __init__(self, settings: Settings, runner: CommandRunner | None = None):
        self.settings = settings
        self.runner = runner or CommandRunner()

    def run(
        self,
        host: str,
        remote_command: str,
        *,
        timeout: int | None = None,
        input_text: str | None = None,
        read_only: bool = False,
    ):
        host = validate_ssh_host(host)
        argv = ["ssh", *SSH_CONNECTION_OPTIONS, "--", host, quote_remote(remote_command)]
        last_exc: Exception | None = None
        attempts = 2 if read_only else 1
        for attempt in range(attempts):
            try:
                return self.runner.run(argv, timeout=timeout or self.settings.ssh_timeout_seconds, input_text=input_text)
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= attempts or not is_transient_ssh_error(exc):
                    raise
        assert last_exc is not None
        raise last_exc

    def run_with_wandb_key(
        self,
        host: str,
        remote_command: str,
        *,
        wandb_api_key: str | None,
        timeout: int | None = None,
        read_only: bool = False,
    ):
        if not wandb_api_key:
            return self.run(host, remote_command, timeout=timeout, read_only=read_only)
        wrapped = "read -r WANDB_API_KEY; export WANDB_API_KEY; " + remote_command
        return self.run(
            host,
            wrapped,
            timeout=timeout,
            input_text=wandb_api_key + "\n",
            read_only=read_only,
        )


    def read_remote_file(
        self,
        *,
        host: str,
        remote_path: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        remote = f"python3 - <<'PY'\nfrom pathlib import Path\npath = Path({remote_path!r})\nprint(path.read_text(encoding=\"utf-8\"))\nPY"
        result = self.run(
            host,
            remote,
            timeout=timeout or self.settings.command_timeout_seconds,
            read_only=True,
        )
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
runtime = {
    "requested_conda_env": conda_env,
    "conda_sh": conda_sh,
    "base_python": shutil.which("python3"),
}
checks = {
    "remote_cwd_exists": os.path.isdir(cwd),
    "python3_available": shutil.which("python3") is not None,
    "wandb_available": shutil.which("wandb") is not None,
    "conda_sh_exists": os.path.isfile(conda_sh) if conda_sh else False,
    "conda_env": conda_env,
}
git = {}
if os.path.isdir(cwd):
    try:
        head = subprocess.run(["git", "-C", cwd, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        status = subprocess.run(["git", "-C", cwd, "status", "--short"], capture_output=True, text=True, timeout=5)
        branch = subprocess.run(["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=5)
        git = {
            "available": head.returncode == 0,
            "head": head.stdout.strip() if head.returncode == 0 else None,
            "branch": branch.stdout.strip() if branch.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "status_short": status.stdout.strip().splitlines()[:20] if status.returncode == 0 else [],
        }
    except Exception as exc:
        git = {"available": False, "error": str(exc)}
python = shutil.which("python3") or "python3"
if conda_sh and conda_env:
    runtime_probe = (
        "import importlib.metadata, json, shutil, sys\\n"
        "packages = {}\\n"
        "for name, dist in [('numpy','numpy'),('scipy','scipy'),('torch','torch'),('wandb','wandb'),('sklearn','scikit-learn')]:\\n"
        "    try:\\n"
        "        packages[name] = importlib.metadata.version(dist)\\n"
        "    except Exception:\\n"
        "        packages[name] = None\\n"
        "print(json.dumps({'ok': True, 'python': sys.executable, 'python_on_path': shutil.which('python'), 'packages': packages}))\\n"
    )
    probe = (
        "source " + shlex.quote(conda_sh)
        + " && conda activate " + shlex.quote(conda_env)
        + " && python -c " + shlex.quote(runtime_probe)
    )
    result = subprocess.run(['bash', '-lc', probe], capture_output=True, text=True)
    checks['conda_activate_ok'] = result.returncode == 0
    if result.returncode == 0:
        try:
            runtime_payload = json.loads(result.stdout.strip().splitlines()[-1])
            runtime.update(runtime_payload)
            packages = runtime_payload.get("packages") if isinstance(runtime_payload.get("packages"), dict) else {}
            checks['torch_import_ok'] = bool(packages.get("torch"))
        except Exception as exc:
            runtime["parse_error"] = str(exc)
            runtime["probe_stdout_tail"] = result.stdout[-1000:]
            checks['torch_import_ok'] = False
    else:
        checks['torch_import_ok'] = False
    if result.returncode != 0:
        checks['conda_probe_stdout'] = result.stdout[-1000:]
        checks['conda_probe_stderr'] = result.stderr[-1000:]
        runtime["probe_stdout_tail"] = result.stdout[-1000:]
        runtime["probe_stderr_tail"] = result.stderr[-1000:]
if config_path:
    checks["config_path_exists"] = os.path.isfile(config_path)
print(json.dumps({"ok": all(v for v in checks.values() if isinstance(v, bool)), "checks": checks, "git": git, "runtime": runtime}))
"""
        script = (
            script.replace("__CWD__", repr(remote_cwd))
            .replace("__CONFIG__", repr(config_path))
            .replace("__CONDA_ENV__", repr(conda_env))
            .replace("__CONDA_SH__", repr(conda_sh))
        )
        result = self.run(
            host,
            "python3 -c " + shlex.quote(script),
            timeout=self.settings.command_timeout_seconds,
            read_only=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        classification = "ok" if payload.get("ok") else "preflight_incomplete"
        return {
            **payload,
            "classification": classification,
            "host": host,
            "remote_cwd": remote_cwd,
            "command": result.summary(),
        }

    def probe_argv_compat(
        self,
        *,
        host: str,
        remote_cwd: str,
        argv: list[str],
        conda_env: str | None = None,
        conda_sh: str | None = None,
        timeout_seconds: int = 20,
    ) -> dict[str, Any]:
        probe_argv = [str(item) for item in argv] + ["--help"]
        script = """
import json, os, shlex, subprocess
cwd = __CWD__
argv = __ARGV__
conda_env = __CONDA_ENV__
conda_sh = __CONDA_SH__
timeout_seconds = __TIMEOUT__
env = os.environ.copy()
env["WANDB_MODE"] = "disabled"
env["WANDB_DISABLED"] = "true"
env["CUDA_VISIBLE_DEVICES"] = ""
cmd = " ".join(shlex.quote(str(item)) for item in argv)
if conda_env and conda_sh:
    cmd = "source " + shlex.quote(conda_sh) + " && conda activate " + shlex.quote(conda_env) + " && " + cmd
try:
    result = subprocess.run(["bash", "-lc", cmd], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    payload = {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "timed_out": False,
    }
except subprocess.TimeoutExpired as exc:
    payload = {
        "returncode": None,
        "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
        "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        "timed_out": True,
    }
print(json.dumps(payload))
"""
        script = (
            script.replace("__CWD__", repr(remote_cwd))
            .replace("__ARGV__", json.dumps(probe_argv))
            .replace("__CONDA_ENV__", repr(conda_env))
            .replace("__CONDA_SH__", repr(conda_sh))
            .replace("__TIMEOUT__", repr(timeout_seconds))
        )
        try:
            result = self.run(
                host,
                "python3 -c " + shlex.quote(script),
                timeout=timeout_seconds + 10,
                read_only=True,
            )
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            stdout_tail = redact_text(str(payload.get("stdout_tail") or ""))
            stderr_tail = redact_text(str(payload.get("stderr_tail") or ""))
            classification = classify_argv_probe(
                payload.get("returncode"),
                stdout_tail,
                stderr_tail,
                timed_out=bool(payload.get("timed_out")),
            )
            return {
                "classification": classification,
                "returncode": payload.get("returncode"),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "timed_out": bool(payload.get("timed_out")),
                "probe_argv": probe_argv,
                "timeout_seconds": timeout_seconds,
                "host": host,
                "remote_cwd": remote_cwd,
                "command": result.summary(),
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "classification": "argv_probe_unavailable",
                "returncode": None,
                "stdout_tail": "",
                "stderr_tail": f"local ssh command timed out: {exc}",
                "timed_out": True,
                "probe_argv": probe_argv,
                "timeout_seconds": timeout_seconds,
                "host": host,
                "remote_cwd": remote_cwd,
            }
        except Exception as exc:
            return {
                "classification": "argv_probe_unavailable",
                "returncode": None,
                "stdout_tail": "",
                "stderr_tail": redact_text(str(exc)),
                "timed_out": False,
                "probe_argv": probe_argv,
                "timeout_seconds": timeout_seconds,
                "host": host,
                "remote_cwd": remote_cwd,
            }

    def probe_gpus(self, host: str, *, min_free_gb: float | None = None, max_util: int | None = None) -> dict[str, Any]:
        query = (
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu "
            "--format=csv,noheader,nounits"
        )
        result = self.run(host, query, read_only=True)
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
        conda_env: str | None = None,
        conda_sh: str = "/opt/anaconda3/etc/profile.d/conda.sh",
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
        runtime = "python3"
        if conda_env:
            runtime = f"source {shlex.quote(conda_sh)} && conda run -n {shlex.quote(conda_env)} python3"
        result = self.run_with_wandb_key(
            host,
            prefix + runtime + " -c " + shlex.quote(script),
            wandb_api_key=wandb_api_key,
            timeout=self.settings.command_timeout_seconds,
            read_only=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload.update({"host": host, "sweep_path": sweep_path, "command": result.summary()})
        return payload

    def reconcile_agent_capacity(
        self,
        *,
        host: str,
        job_id: str,
        remote_cwd: str,
        sweep_path: str,
        desired_agents: int,
        eligible_gpu_indices: list[int],
        conda_env: str | None,
        conda_sh: str,
        wandb_api_key: str | None = None,
        receipt_root: str | None = None,
        proc_root: str = "/proc",
    ) -> dict[str, Any]:
        script = r'''
import fcntl, hashlib, json, os, re, shlex, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

job_id = __JOB_ID__
remote_cwd = __REMOTE_CWD__
sweep_path = __SWEEP_PATH__
desired_agents = __DESIRED_AGENTS__
eligible_gpu_indices = __ELIGIBLE_GPU_INDICES__
conda_env = __CONDA_ENV__
conda_sh = __CONDA_SH__
receipt_root_value = __RECEIPT_ROOT__
proc_root = __PROC_ROOT__

def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def safe_component(value):
    value = str(value or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", value):
        raise ValueError("unsafe receipt identity")
    return value

def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

def read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}

def read_cmdline(pid):
    try:
        raw = open(os.path.join(proc_root, str(pid), "cmdline"), "rb").read()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]

def read_environ(pid):
    try:
        raw = open(os.path.join(proc_root, str(pid), "environ"), "rb").read()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return {}
    values = {}
    for part in raw.split(b"\0"):
        if b"=" not in part:
            continue
        key, value = part.split(b"=", 1)
        values[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return values

def read_process_identity(pid):
    try:
        text = open(os.path.join(proc_root, str(pid), "stat"), encoding="utf-8").read()
        tail = text[text.rfind(")") + 2:].split()
        return {"session_id": int(tail[3]), "start_ticks": int(tail[19])}
    except Exception:
        return {"session_id": None, "start_ticks": None}

def wandb_agent_sweep(argv):
    for index in range(max(0, len(argv) - 2)):
        if os.path.basename(argv[index]) == "wandb" and argv[index + 1] == "agent":
            return argv[index + 2]
    return None

def visible_gpu(pid):
    raw = str(read_environ(pid).get("CUDA_VISIBLE_DEVICES") or "").strip()
    if re.fullmatch(r"[0-9]+", raw):
        return int(raw)
    return None

def discover_agents():
    agents = []
    try:
        names = os.listdir(proc_root)
    except OSError:
        names = []
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        argv = read_cmdline(pid)
        target = wandb_agent_sweep(argv)
        if not target:
            continue
        identity = read_process_identity(pid)
        agents.append({
            "pid": str(pid),
            "sweep_path": target,
            "gpu_index": visible_gpu(pid),
            "session_id": identity["session_id"],
            "start_ticks": identity["start_ticks"],
        })
    return agents

def pid_exists(pid):
    value = str(pid or "")
    if not value.isdigit():
        return False
    if os.path.isdir(os.path.join(proc_root, value)):
        return True
    # The production host exposes /proc.  The fallback keeps the same remote
    # script testable on macOS, where process inspection is not available.
    try:
        os.kill(int(value), 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False

def receipt_is_live(receipt, observed_agents=None):
    if not receipt:
        return False
    leader_exists = pid_exists(receipt.get("pid"))
    expected_sweep = receipt.get("sweep_path")
    expected_gpu = receipt.get("gpu_index")
    expected_session = receipt.get("session_id")
    expected_start = receipt.get("pid_start_ticks")
    identity = read_process_identity(receipt.get("pid")) if leader_exists else {"session_id": None, "start_ticks": None}
    leader_identity_matches = leader_exists
    if leader_exists and expected_start is not None and identity.get("start_ticks") is not None:
        leader_identity_matches = int(expected_start) == int(identity["start_ticks"])
    if leader_identity_matches and expected_session is not None and identity.get("session_id") is not None:
        if int(expected_session) == int(identity["session_id"]):
            return True
    for agent in observed_agents if observed_agents is not None else discover_agents():
        if agent.get("sweep_path") != expected_sweep:
            continue
        if expected_gpu is not None and agent.get("gpu_index") != int(expected_gpu):
            continue
        if expected_session is not None and agent.get("session_id") is not None:
            if int(expected_session) != int(agent["session_id"]):
                continue
        return True
    # On hosts without /proc, the leader PID is the only identity available.
    # Linux takes the stricter session/cmdline path above.
    return leader_identity_matches and (expected_session is None or identity.get("session_id") is None)

def receipt_matches_agent(receipt, agent):
    if receipt.get("sweep_path") != agent.get("sweep_path"):
        return False
    if receipt.get("gpu_index") is not None and agent.get("gpu_index") != int(receipt["gpu_index"]):
        return False
    if str(receipt.get("pid") or "") == str(agent.get("pid") or ""):
        return True
    receipt_session = receipt.get("session_id")
    agent_session = agent.get("session_id")
    return receipt_session is not None and agent_session is not None and int(receipt_session) == int(agent_session)

safe_job = safe_component(job_id)
sweep_slug = safe_component(sweep_path.replace("/", "_"))
root = Path(receipt_root_value).expanduser() if receipt_root_value else Path.home() / ".local" / "state" / "experiment-console" / "agent-receipts"
root.mkdir(parents=True, exist_ok=True)
os.chmod(root, 0o700)
job_dir = root / safe_job / sweep_slug
job_dir.mkdir(parents=True, exist_ok=True)
os.chmod(job_dir, 0o700)
log_dir = root.parent / "agent-logs" / safe_job / sweep_slug
log_dir.mkdir(parents=True, exist_ok=True)
os.chmod(log_dir, 0o700)

def receipt_paths(gpu_index):
    def generation(path):
        match = re.search(r"_generation_([0-9]+)\.json$", path.name)
        return int(match.group(1)) if match else 0
    return sorted(job_dir.glob("gpu_%d_generation_*.json" % gpu_index), key=generation)

def next_generation(gpu_index):
    generations = []
    for path in receipt_paths(gpu_index):
        match = re.search(r"_generation_([0-9]+)\.json$", path.name)
        if match:
            generations.append(int(match.group(1)))
    return max(generations, default=0) + 1

def write_receipt(gpu_index, generation, payload):
    path = job_dir / ("gpu_%d_generation_%d.json" % (gpu_index, generation))
    atomic_json(path, payload)
    return str(path)

def current_assignments(agents):
    matching = [item for item in agents if item.get("sweep_path") == sweep_path and item.get("gpu_index") is not None]
    assignments = []
    seen_process_groups = set()
    for agent in sorted(matching, key=lambda item: (int(item["gpu_index"]), int(item["pid"]))):
        gpu_index = int(agent["gpu_index"])
        process_group = (gpu_index, agent.get("session_id") or agent.get("pid"))
        if process_group in seen_process_groups:
            continue
        seen_process_groups.add(process_group)
        paths = receipt_paths(gpu_index)
        matching_path = next(
            (path for path in reversed(paths) if receipt_matches_agent(read_json(path), agent)),
            None,
        )
        receipt = read_json(matching_path) if matching_path else {}
        if not receipt_matches_agent(receipt, agent):
            generation = next_generation(gpu_index)
            receipt = {
                "version": 1,
                "job_id": job_id,
                "sweep_path": sweep_path,
                "gpu_index": gpu_index,
                "generation": generation,
                "pid": str(agent["pid"]),
                "session_id": agent.get("session_id"),
                "pid_start_ticks": agent.get("start_ticks"),
                "log": None,
                "conda_env": conda_env,
                "command": ["wandb", "agent", sweep_path],
                "classification": "discovered_existing_agent",
                "started_at": stamp(),
                "updated_at": stamp(),
            }
            receipt["receipt_path"] = write_receipt(gpu_index, generation, receipt)
        else:
            receipt = dict(receipt)
            receipt["receipt_path"] = str(matching_path)
            receipt["agent_pid"] = str(agent["pid"])
            receipt["classification"] = "live_receipt"
        assignments.append(receipt)
    receipt_gpu_indices = set()
    for path in job_dir.glob("gpu_*_generation_*.json"):
        match = re.match(r"gpu_([0-9]+)_generation_", path.name)
        if match:
            receipt_gpu_indices.add(int(match.group(1)))
    for gpu_index in sorted(receipt_gpu_indices):
        if any(int(item.get("gpu_index", -1)) == int(gpu_index) for item in assignments):
            continue
        paths = receipt_paths(int(gpu_index))
        live_path = next(
            (path for path in reversed(paths) if receipt_is_live(read_json(path), agents)),
            None,
        )
        receipt = read_json(live_path) if live_path else {}
        if receipt and receipt_is_live(receipt, agents):
            receipt = dict(receipt)
            receipt["receipt_path"] = str(live_path)
            receipt["classification"] = "agent_starting"
            assignments.append(receipt)
    return assignments

def live_receipts_for_gpu(gpu_index, agents=None):
    live = []
    pattern = "*/*/gpu_%d_generation_*.json" % gpu_index
    for path in root.glob(pattern):
        receipt = read_json(path)
        if receipt and receipt_is_live(receipt, agents):
            live.append({
                "job_id": receipt.get("job_id"),
                "sweep_path": receipt.get("sweep_path"),
                "gpu_index": receipt.get("gpu_index"),
                "pid": receipt.get("pid"),
                "receipt_path": str(path),
            })
    return live

launched = []
failed = []
skipped = []
agents = discover_agents()
assignments = current_assignments(agents)

for gpu_index in sorted({int(item) for item in eligible_gpu_indices if int(item) >= 0}):
    if len(assignments) >= desired_agents:
        break
    if any(int(item.get("gpu_index", -1)) == gpu_index for item in assignments):
        continue
    lock_path = root / ("gpu_%d.lock" % gpu_index)
    lock_descriptor = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(lock_descriptor, "r+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        agents = discover_agents()
        occupied = [item for item in agents if item.get("gpu_index") == gpu_index]
        occupied_receipts = live_receipts_for_gpu(gpu_index, agents)
        same_sweep = [item for item in occupied if item.get("sweep_path") == sweep_path]
        if same_sweep:
            assignments = current_assignments(agents)
            continue
        same_sweep_receipts = [item for item in occupied_receipts if item.get("sweep_path") == sweep_path]
        if same_sweep_receipts:
            assignments = current_assignments(agents)
            for receipt in same_sweep_receipts:
                if not any(int(item.get("gpu_index", -1)) == gpu_index for item in assignments):
                    assignments.append(read_json(Path(str(receipt["receipt_path"]))))
            continue
        if occupied or occupied_receipts:
            skipped.append({
                "gpu_index": gpu_index,
                "classification": "gpu_owned_by_other_agent",
                "sweep_paths": sorted({str(item.get("sweep_path")) for item in [*occupied, *occupied_receipts]}),
            })
            continue
        generation = next_generation(gpu_index)
        log_path = log_dir / ("gpu_%d_generation_%d.log" % (gpu_index, generation))
        wandb_executable = shutil.which("wandb") or "wandb"
        if conda_env:
            inner = "cd %s && source %s && conda run -n %s wandb agent %s" % (
                shlex.quote(remote_cwd), shlex.quote(conda_sh), shlex.quote(conda_env), shlex.quote(sweep_path),
            )
        else:
            inner = "cd %s && %s agent %s" % (shlex.quote(remote_cwd), shlex.quote(wandb_executable), shlex.quote(sweep_path))
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        try:
            with open(log_path, "ab", buffering=0) as log:
                process = subprocess.Popen(
                    ["bash", "-lc", inner],
                    cwd=remote_cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
            identity = read_process_identity(process.pid)
            receipt = {
                "version": 1,
                "job_id": job_id,
                "sweep_path": sweep_path,
                "gpu_index": gpu_index,
                "generation": generation,
                "pid": str(process.pid),
                "session_id": process.pid,
                "pid_start_ticks": identity.get("start_ticks"),
                "log": str(log_path),
                "conda_env": conda_env,
                "command": ["wandb", "agent", sweep_path],
                "command_sha256": hashlib.sha256(("wandb agent " + sweep_path).encode("utf-8")).hexdigest(),
                "classification": "agent_started",
                "started_at": stamp(),
                "updated_at": stamp(),
            }
            try:
                receipt["receipt_path"] = write_receipt(gpu_index, generation, receipt)
            except Exception:
                try:
                    os.killpg(process.pid, 15)
                except Exception:
                    pass
                raise
            launched.append(receipt)
            assignments.append(receipt)
        except Exception as exc:
            failed.append({"gpu_index": gpu_index, "classification": "agent_launch_failed", "error": type(exc).__name__ + ": " + str(exc)})

agents = discover_agents()
assignments = current_assignments(agents)
for receipt in launched:
    if not any(int(item.get("gpu_index", -1)) == int(receipt["gpu_index"]) for item in assignments) and receipt_is_live(receipt, agents):
        assignments.append(receipt)
assignments = sorted(assignments, key=lambda item: (int(item.get("gpu_index", -1)), int(item.get("generation", 0))))
occupied_gpus = sorted({
    int(gpu_index)
    for gpu_index in eligible_gpu_indices
    if any(item.get("gpu_index") is not None and int(item["gpu_index"]) == int(gpu_index) for item in agents)
    or live_receipts_for_gpu(int(gpu_index), agents)
})
if failed:
    classification = "capacity_launch_failed"
elif len(assignments) >= desired_agents:
    classification = "capacity_satisfied"
elif desired_agents <= 0:
    classification = "terminal_observed"
elif not eligible_gpu_indices:
    classification = "resource_wait"
else:
    classification = "capacity_degraded"
print(json.dumps({
    "classification": classification,
    "job_id": job_id,
    "sweep_path": sweep_path,
    "desired_agents": desired_agents,
    "live_agents": len(assignments),
    "assignments": assignments,
    "launched": launched,
    "failed": failed,
    "skipped": skipped,
    "occupied_gpus": occupied_gpus,
    "receipt_root": str(root),
    "observed_at": stamp(),
}))
'''
        script = (
            script.replace("__JOB_ID__", repr(job_id))
            .replace("__REMOTE_CWD__", repr(remote_cwd))
            .replace("__SWEEP_PATH__", repr(sweep_path))
            .replace("__DESIRED_AGENTS__", repr(max(0, int(desired_agents))))
            .replace("__ELIGIBLE_GPU_INDICES__", repr([int(item) for item in eligible_gpu_indices]))
            .replace("__CONDA_ENV__", repr(conda_env))
            .replace("__CONDA_SH__", repr(conda_sh))
            .replace("__RECEIPT_ROOT__", repr(receipt_root))
            .replace("__PROC_ROOT__", repr(proc_root))
        )
        remote = f"cd {shlex.quote(remote_cwd)} && python3 -c {shlex.quote(script)}"
        result = self.run_with_wandb_key(
            host,
            remote,
            wandb_api_key=wandb_api_key,
            timeout=self.settings.command_timeout_seconds,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload.update({"host": host, "command": result.summary()})
        return payload

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
result_path = __RESULT_PATH__
log_path = __LOG_PATH__
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
payload["timed_out"] = not payload["ok"]
payload.setdefault("status_path", status_path)
payload.setdefault("result_path", result_path)
payload.setdefault("log_path", log_path)
print(json.dumps(payload))
"""
        status_probe = (
            status_probe
            .replace("__STATUS_PATH__", repr(status_path))
            .replace("__RESULT_PATH__", repr(result_path))
            .replace("__LOG_PATH__", repr(log_path))
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

    def check_run_status(
        self,
        *,
        host: str,
        status_path: str,
        pids: list[str] | None = None,
        proc_root: str = "/proc",
    ) -> dict[str, Any]:
        pid_json = json.dumps([str(pid) for pid in (pids or []) if str(pid).strip()])
        script = """
import json, os
status_path = __STATUS_PATH__
pids = [int(pid) for pid in json.loads(__PIDS__) if str(pid).isdigit()]
proc_root = __PROC_ROOT__
status = {}
try:
    with open(status_path, encoding="utf-8") as handle:
        loaded = json.load(handle)
        if isinstance(loaded, dict):
            status = loaded
except FileNotFoundError:
    status = {"status_path": status_path, "missing": True}
expected_command = [str(item) for item in (status.get("command") or [])]

def read_cmdline(pid):
    try:
        raw = open(os.path.join(proc_root, str(pid), "cmdline"), "rb").read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (PermissionError, OSError):
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\\0") if part]

def matches_expected(argv):
    if not argv or not expected_command:
        return False
    return argv == expected_command or (len(argv) >= len(expected_command) and argv[-len(expected_command):] == expected_command)

alive = []
reused = []
for pid in sorted(set(pids + [int(status.get("child_pid"))] if str(status.get("child_pid") or "").isdigit() else pids)):
    argv = read_cmdline(pid)
    if argv is None:
        continue
    if matches_expected(argv):
        alive.append(str(pid))
    else:
        reused.append(str(pid))
status["alive_pids"] = sorted(set(alive))
status["unmatched_reused_pids"] = sorted(set(reused))
print(json.dumps(status))
"""
        script = (
            script.replace("__STATUS_PATH__", repr(status_path))
            .replace("__PIDS__", repr(pid_json))
            .replace("__PROC_ROOT__", repr(proc_root))
        )
        result = self.run(
            host,
            "python3 -c " + shlex.quote(script),
            timeout=self.settings.command_timeout_seconds,
            read_only=True,
        )
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
        proc_root: str = "/proc",
    ) -> dict[str, Any]:
        pid_json = json.dumps([str(pid) for pid in (pids or []) if str(pid).strip()])
        script = """
import glob, json, os
pids = [int(pid) for pid in json.loads(__PIDS__) if str(pid).isdigit()]
sweep_path = __SWEEP_PATH__
proc_root = __PROC_ROOT__

def read_cmdline(pid):
    try:
        raw = open(os.path.join(proc_root, str(pid), "cmdline"), "rb").read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (PermissionError, OSError):
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\\0") if part]

def matches_sweep(argv):
    if not argv or not sweep_path:
        return False
    return any(
        os.path.basename(argv[index]) == "wandb"
        and argv[index + 1:index + 3] == ["agent", sweep_path]
        for index in range(max(0, len(argv) - 2))
    )

matched = []
tracked_alive = []
reused = []
all_pids = sorted({int(path.rsplit("/", 1)[-1]) for path in glob.glob(os.path.join(proc_root, "[0-9]*"))})
for pid in all_pids:
    argv = read_cmdline(pid)
    if matches_sweep(argv):
        matched.append({"pid": str(pid)})
for pid in pids:
    argv = read_cmdline(pid)
    if argv is None:
        continue
    if matches_sweep(argv):
        tracked_alive.append(str(pid))
    else:
        reused.append(str(pid))
print(json.dumps({
    "tracked_pids": [str(pid) for pid in pids],
    "alive_pids": sorted(set(tracked_alive)),
    "pgrep": matched,
    "unmatched_reused_pids": sorted(set(reused)),
}))
"""
        script = (
            script.replace("__PIDS__", repr(pid_json))
            .replace("__SWEEP_PATH__", repr(sweep_path))
            .replace("__PROC_ROOT__", repr(proc_root))
        )
        result = self.run(
            host,
            "python3 -c " + shlex.quote(script),
            timeout=self.settings.command_timeout_seconds,
            read_only=True,
        )
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
        result = self.run(
            host,
            "python3 -c " + shlex.quote(script),
            timeout=self.settings.command_timeout_seconds,
            read_only=True,
        )
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

    def stop_pids(
        self,
        *,
        host: str,
        pids: list[str],
        status_path: str | None = None,
        expected_job_id: str | None = None,
        proc_root: str = "/proc",
    ) -> dict[str, Any]:
        pid_json = json.dumps([str(pid) for pid in pids if str(pid).strip()])
        script = """
import json, os, signal
target_pids = [int(pid) for pid in json.loads(__PIDS__) if str(pid).isdigit()]
status_path = __STATUS_PATH__
expected_job_id = __EXPECTED_JOB_ID__
proc_root = __PROC_ROOT__
status = {}
try:
    with open(status_path, encoding="utf-8") as handle:
        loaded = json.load(handle)
        if isinstance(loaded, dict):
            status = loaded
except (FileNotFoundError, OSError, json.JSONDecodeError):
    pass
expected_command = [str(item) for item in (status.get("command") or [])]

def read_cmdline(pid):
    try:
        raw = open(os.path.join(proc_root, str(pid), "cmdline"), "rb").read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (PermissionError, OSError):
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\\0") if part]

def matches_receipt(pid, argv):
    if not argv or not expected_command or status.get("job_id") != expected_job_id:
        return False
    if str(status.get("child_pid") or "") != str(pid):
        return False
    return argv == expected_command or (len(argv) >= len(expected_command) and argv[-len(expected_command):] == expected_command)

stopped = []
missing = []
still_running = []
unmatched_reused = []
for pid in target_pids:
    argv = read_cmdline(pid)
    if argv is None:
        missing.append(str(pid))
        continue
    if not matches_receipt(pid, argv):
        unmatched_reused.append(str(pid))
        continue
    try:
        os.kill(pid, signal.SIGTERM)
        stopped.append(str(pid))
    except ProcessLookupError:
        missing.append(str(pid))
    except Exception:
        still_running.append(str(pid))
print(json.dumps({
    "stopped_pids": sorted(set(stopped)),
    "missing_pids": sorted(set(missing)),
    "still_running_pids": sorted(set(still_running)),
    "unmatched_reused_pids": sorted(set(unmatched_reused)),
}))
"""
        script = (
            script.replace("__PIDS__", repr(pid_json))
            .replace("__STATUS_PATH__", repr(status_path or ""))
            .replace("__EXPECTED_JOB_ID__", repr(expected_job_id or ""))
            .replace("__PROC_ROOT__", repr(proc_root))
        )
        result = self.run(
            host,
            "python3 -c " + shlex.quote(script),
            timeout=self.settings.command_timeout_seconds,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload.update({"host": host, "command": result.summary()})
        return payload

    def stop_agents(
        self,
        *,
        host: str,
        sweep_path: str,
        pids: list[str] | None = None,
        proc_root: str = "/proc",
    ) -> dict[str, Any]:
        pid_json = json.dumps([str(pid) for pid in (pids or []) if str(pid).strip()])
        script = """
import glob, json, os, signal
tracked_pids = [int(pid) for pid in json.loads(__PIDS__) if str(pid).isdigit()]
sweep_path = __SWEEP_PATH__
proc_root = __PROC_ROOT__

def read_cmdline(pid):
    try:
        raw = open(os.path.join(proc_root, str(pid), "cmdline"), "rb").read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (PermissionError, OSError):
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\\0") if part]

def matches_sweep(argv):
    if not argv:
        return False
    return any(
        os.path.basename(argv[index]) == "wandb"
        and argv[index + 1:index + 3] == ["agent", sweep_path]
        for index in range(max(0, len(argv) - 2))
    )

matched = []
reused = []
missing = []
all_pids = sorted({int(path.rsplit("/", 1)[-1]) for path in glob.glob(os.path.join(proc_root, "[0-9]*"))})
for pid in all_pids:
    argv = read_cmdline(pid)
    if matches_sweep(argv):
        matched.append(pid)
for pid in tracked_pids:
    argv = read_cmdline(pid)
    if argv is None:
        missing.append(str(pid))
    elif not matches_sweep(argv):
        reused.append(str(pid))
stopped = []
failed = []
for pid in matched:
    try:
        os.kill(pid, signal.SIGTERM)
        stopped.append(str(pid))
    except ProcessLookupError:
        missing.append(str(pid))
    except Exception:
        failed.append(str(pid))
print(json.dumps({
    "matched_pids": sorted({str(pid) for pid in matched}),
    "stopped_pids": sorted(set(stopped)),
    "missing_pids": sorted(set(missing)),
    "still_running_pids": sorted(set(failed)),
    "unmatched_reused_pids": sorted(set(reused)),
}))
"""
        script = (
            script.replace("__PIDS__", repr(pid_json))
            .replace("__SWEEP_PATH__", repr(sweep_path))
            .replace("__PROC_ROOT__", repr(proc_root))
        )
        result = self.run(
            host,
            "python3 -c " + shlex.quote(script),
            timeout=self.settings.command_timeout_seconds,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload.update({"host": host, "sweep_path": sweep_path, "command": result.summary()})
        return payload

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
        result = self.run(
            host,
            "python3 -c " + shlex.quote(script),
            timeout=self.settings.command_timeout_seconds,
            read_only=True,
        )
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
        metric_paths: list[str] | None = None,
        group_paths: list[str] | None = None,
        output_globs: list[str] | None = None,
        discovery_mode: str = "legacy_auto_v1",
        comparison_paths: list[str] | None = None,
        include_raw_artifacts: bool = False,
    ) -> dict[str, Any]:
        script = """
import glob, json, os, re, stat, time
cwd = __CWD__
cwd_real = os.path.realpath(cwd)
sweep_id = __SWEEP_ID__
run_ids = __RUN_IDS__
max_runs = __MAX_RUNS__
metric_keys = __METRIC_KEYS__
group_keys = __GROUP_KEYS__
metric_paths = __METRIC_PATHS__
group_paths = __GROUP_PATHS__
output_globs = __OUTPUT_GLOBS__
discovery_mode = __DISCOVERY_MODE__
comparison_paths = __COMPARISON_PATHS__
include_raw_artifacts = __INCLUDE_RAW_ARTIFACTS__
artifact_file_max_bytes = __ARTIFACT_FILE_MAX_BYTES__
artifact_total_max_bytes = __ARTIFACT_TOTAL_MAX_BYTES__
artifact_bytes_read = 0
artifact_cache = {}
artifact_rejections = {}
deadline = time.time() + __BUDGET_SECONDS__
if not run_ids:
    raise SystemExit("target run ids unavailable")

def reject_artifact(run_id, path, reason, source):
    key = str(run_id or "unknown")
    item = {"path": str(path), "reason": reason, "source": source}
    bucket = artifact_rejections.setdefault(key, [])
    if item not in bucket:
        bucket.append(item)

def has_parent_reference(path):
    return ".." in str(path or "").split(os.sep)

def canonical_artifact_path(path, run_id, source, *, reject_parent=True):
    text = str(path or "").strip()
    if not text:
        return None
    if reject_parent and has_parent_reference(text):
        reject_artifact(run_id, text, "artifact_path_traversal", source)
        return None
    canonical = os.path.realpath(text)
    try:
        contained = os.path.commonpath([cwd_real, canonical]) == cwd_real
    except ValueError:
        contained = False
    if not contained:
        reject_artifact(run_id, text, "artifact_outside_remote_cwd", source)
        return None
    return canonical

def contained_existing_paths(paths, run_id, source):
    safe = []
    for path in paths:
        canonical = canonical_artifact_path(path, run_id, source, reject_parent=False)
        if canonical and os.path.isfile(canonical):
            safe.append(canonical)
    return sorted(set(safe), key=lambda path: os.path.getmtime(path), reverse=True)

def load_json(path, run_id, source):
    global artifact_bytes_read
    canonical = canonical_artifact_path(path, run_id, source, reject_parent=False)
    if not canonical:
        return {}
    if canonical in artifact_cache:
        return artifact_cache[canonical]
    descriptor = None
    try:
        descriptor = os.open(canonical, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            reject_artifact(run_id, canonical, "artifact_not_regular_file", source)
            return {}
        if info.st_size > artifact_file_max_bytes:
            reject_artifact(run_id, canonical, "artifact_file_size_limit_exceeded", source)
            return {}
        if artifact_bytes_read + info.st_size > artifact_total_max_bytes:
            reject_artifact(run_id, canonical, "artifact_total_size_limit_exceeded", source)
            return {}
        chunks = []
        remaining = artifact_file_max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > artifact_file_max_bytes:
            reject_artifact(run_id, canonical, "artifact_file_size_limit_exceeded", source)
            return {}
        if artifact_bytes_read + len(raw) > artifact_total_max_bytes:
            reject_artifact(run_id, canonical, "artifact_total_size_limit_exceeded", source)
            return {}
        artifact_bytes_read += len(raw)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            reject_artifact(run_id, canonical, "artifact_json_object_required", source)
            return {}
        artifact_cache[canonical] = data
        return data
    except (OSError, UnicodeError, json.JSONDecodeError):
        reject_artifact(run_id, canonical, "artifact_read_or_json_invalid", source)
        return {}
    finally:
        if descriptor is not None:
            os.close(descriptor)

def scalar_map(data):
    out = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out

def parse_scalar(value):
    text = str(value).strip()
    if not text:
        return ""
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if text in {"null", "None", "~"}:
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        if re.match(r"^[-+]?\\d+$", text):
            return int(text)
        return float(text)
    except Exception:
            return text

def parse_selector(selector):
    text = str(selector or "")
    if "=" in text:
        alias, path = text.split("=", 1)
        return alias or path, path
    return None, text

def extract_path_values(data, path):
    parts = [part for part in str(path or "").split(".") if part]
    out = []
    def walk(value, remaining, actual):
        if not remaining:
            out.append((".".join(actual), value))
            return
        part = remaining[0]
        rest = remaining[1:]
        if part == "*":
            if isinstance(value, dict):
                for key in sorted(value):
                    walk(value[key], rest, actual + [str(key)])
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, rest, actual + [str(index)])
            return
        if isinstance(value, dict) and part in value:
            walk(value[part], rest, actual + [part])
        elif isinstance(value, list):
            try:
                index = int(part)
            except Exception:
                return
            if 0 <= index < len(value):
                walk(value[index], rest, actual + [part])
    walk(data, parts, [])
    return out

def selector_values(data, selector):
    alias, path = parse_selector(selector)
    matches = extract_path_values(data, path)
    values = {}
    for actual_path, value in matches:
        key = alias if alias and len(matches) == 1 else actual_path
        if alias and len(matches) > 1:
            key = alias + "." + actual_path
        values[key] = value
    return values

def normalize_remote_path(path, run_id):
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    for token in ["${wandb.run.id}", "${wandb_run_id}", "{wandb.run.id}", "{run_id}"]:
        text = text.replace(token, run_id)
    if not text.endswith(".json"):
        return None
    if has_parent_reference(text):
        reject_artifact(run_id, text, "artifact_path_traversal", "wandb_config")
        return None
    if not os.path.isabs(text):
        text = os.path.join(cwd, text)
    return canonical_artifact_path(text, run_id, "wandb_config", reject_parent=False)

def is_progress_path(path):
    return "progress" in os.path.basename(str(path)).lower()

def discover_config_paths(wandb_config, run_id):
    preferred = {"out", "result_path", "output_path", "result_out", "json_out"}
    candidates = []
    progress = []
    seen = set()
    items = sorted(wandb_config.items(), key=lambda item: (0 if item[0] in preferred else 1, item[0]))
    for key, value in items:
        path = normalize_remote_path(value, run_id)
        if not path or path in seen:
            continue
        seen.add(path)
        item = {"path": path, "config_key": key, "exists": os.path.isfile(path)}
        if is_progress_path(path):
            progress.append(item)
        else:
            candidates.append(item)
    return candidates, progress

def expand_output_globs(patterns, run_id, source="output_glob"):
    paths = []
    seen = set()
    for pattern in patterns or []:
        text = str(pattern or "")
        for token in ["${wandb.run.id}", "${wandb_run_id}", "{wandb.run.id}", "{run_id}"]:
            text = text.replace(token, run_id)
        if has_parent_reference(text):
            reject_artifact(run_id, text, "artifact_path_traversal", source)
            continue
        if not os.path.isabs(text):
            text = os.path.join(cwd, text)
        canonical_pattern = canonical_artifact_path(text, run_id, source, reject_parent=False)
        if not canonical_pattern:
            continue
        for path in glob.glob(canonical_pattern, recursive=True):
            canonical = canonical_artifact_path(path, run_id, source, reject_parent=False)
            if canonical and canonical.endswith(".json") and canonical not in seen:
                seen.add(canonical)
                paths.append(canonical)
    return sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)

def load_wandb_config(path, run_id):
    canonical = canonical_artifact_path(path, run_id, "wandb_config_file", reject_parent=False)
    if not canonical:
        return {}
    try:
        with open(canonical, encoding="utf-8", errors="replace") as handle:
            text = handle.read(artifact_file_max_bytes + 1)
        if len(text.encode("utf-8")) > artifact_file_max_bytes:
            reject_artifact(run_id, canonical, "wandb_config_size_limit_exceeded", "wandb_config_file")
            return {}
    except Exception:
        return {}
    config = {}
    for key, value in re.findall(r"^\\s*-\\s+--([^=\\s]+)=(.*)$", text, flags=re.M):
        config[key] = parse_scalar(value)
    current_key = None
    for line in text.splitlines():
        top = re.match(r"^([A-Za-z_][A-Za-z0-9_.-]*):\\s*$", line)
        if top:
            current_key = top.group(1)
            continue
        direct = re.match(r"^([A-Za-z_][A-Za-z0-9_.-]*):\\s+(.+)$", line)
        if direct:
            config.setdefault(direct.group(1), parse_scalar(direct.group(2)))
            current_key = None
            continue
        nested = re.match(r"^\\s+value:\\s*(.*)$", line)
        if current_key and nested:
            config.setdefault(current_key, parse_scalar(nested.group(1)))
            current_key = None
    return config

def update_discovery_rejection(run_id):
    rejections = artifact_rejections.get(str(run_id)) or []
    if rejections and run_id in discovery_sources:
        discovery_sources[run_id]["classification"] = rejections[0]["reason"]
        discovery_sources[run_id]["artifact_rejections"] = rejections

rows = []
config_sources = {}
metric_sources = {}
comparison_sources = {}
missing_config_keys = {}
missing_metric_paths = {}
missing_comparison_paths = {}
discovery_sources = {}
raw_artifacts = []
claimed_output_paths = {}
for run_id in run_ids[:max_runs]:
    if len(rows) >= max_runs or time.time() > deadline:
        break
    summary_paths = contained_existing_paths(
        glob.glob(os.path.join(cwd, "wandb", f"run-*{run_id}", "files", "wandb-summary.json")),
        run_id,
        "wandb_summary",
    )
    config_paths = contained_existing_paths(
        glob.glob(os.path.join(cwd, "wandb", f"run-*{run_id}", "files", "config.yaml")),
        run_id,
        "wandb_config_file",
    )
    wandb_config = load_wandb_config(config_paths[0], run_id) if config_paths else {}
    config_candidates, progress_candidates = discover_config_paths(wandb_config, run_id)
    config_output_paths = []
    for item in config_candidates:
        if item["exists"]:
            config_output_paths.append(item["path"])
    contract_error = None
    if discovery_mode == "run_id_output_globs_v1":
        raw_output_paths = sorted(set(expand_output_globs(output_globs, run_id)))
        if len(raw_output_paths) != 1:
            contract_error = "expected_exactly_one_run_artifact"
            raw_output_paths = []
    elif discovery_mode == "wandb_config_result_path_v1":
        raw_output_paths = sorted(set(os.path.realpath(path) for path in config_output_paths))
        if len(raw_output_paths) != 1:
            contract_error = "expected_exactly_one_config_result_path"
            raw_output_paths = []
    else:
        output_patterns = [
            os.path.join(cwd, "investigations", "**", "experiments", "outputs", f"*{run_id}*.json"),
            os.path.join(cwd, "outputs", f"*{run_id}*.json"),
        ]
        fallback_output_paths = []
        if not config_output_paths:
            fallback_output_paths.extend(expand_output_globs(output_patterns, run_id, "default_output_glob"))
            fallback_output_paths.extend(expand_output_globs(output_globs, run_id))
        raw_output_paths = config_output_paths or fallback_output_paths
    if len(raw_output_paths) == 1:
        canonical = os.path.realpath(raw_output_paths[0])
        previous_owner = claimed_output_paths.get(canonical)
        if previous_owner and previous_owner != run_id:
            contract_error = "artifact_path_claimed_by_multiple_runs"
            raw_output_paths = []
        else:
            claimed_output_paths[canonical] = run_id
    progress_paths = [item["path"] for item in progress_candidates if item.get("exists")]
    progress_paths.extend(path for path in raw_output_paths if is_progress_path(path))
    output_paths = []
    seen_output_paths = set()
    for path in raw_output_paths:
        if is_progress_path(path) or path in seen_output_paths:
            continue
        seen_output_paths.add(path)
        output_paths.append(path)
    if not config_output_paths:
        output_paths = sorted(output_paths, key=lambda p: os.path.getmtime(p), reverse=True)
    discovery_sources[run_id] = {
        "config_paths": config_paths[:1],
        "config_candidates": config_candidates,
        "progress_paths": sorted(set(progress_paths)),
        "output_globs": list(output_globs or []),
        "selected_paths": output_paths[:3],
        "classification": contract_error or "ok",
    }
    update_discovery_rejection(run_id)
    data = {}
    sources = []
    if summary_paths:
        summary_data = load_json(summary_paths[0], run_id, "wandb_summary")
        if summary_data:
            data.update(summary_data)
            sources.append(summary_paths[0])
    for output_path in output_paths[:3]:
        output_data = load_json(output_path, run_id, "result_artifact")
        if output_data:
            data.update(output_data)
            sources.append(output_path)
    update_discovery_rejection(run_id)
    if not data:
        config = {key: wandb_config.get(key) for key in group_keys} if group_keys else {}
        for key in group_keys:
            if key in wandb_config and config_paths:
                config_sources.setdefault(run_id, {})[key] = config_paths[0]
            else:
                missing_config_keys.setdefault(run_id, []).append(key)
        for selector in metric_paths:
            missing_metric_paths.setdefault(run_id, []).append(selector)
        for selector in comparison_paths:
            missing_comparison_paths.setdefault(run_id, []).append(selector)
        rows.append({"run_id": run_id, "sources": config_paths[:1], "config": config, "metrics": {}, "comparisons": {}, "has_scientific_result": False})
        continue
    flat = scalar_map(data)
    metrics = {key: flat.get(key) for key in metric_keys if key in flat} if metric_keys else {
        key: value for key, value in flat.items()
        if isinstance(value, (int, float, bool)) and not key.startswith("_")
    }
    for selector in metric_paths:
        selected = selector_values(data, selector)
        if not selected:
            missing_metric_paths.setdefault(run_id, []).append(selector)
            continue
        selected_numeric = False
        for key, value in selected.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = value
                selected_numeric = True
        if not selected_numeric:
            missing_metric_paths.setdefault(run_id, []).append(selector)
    comparisons = {}
    for selector in comparison_paths:
        selected = selector_values(data, selector)
        if not selected:
            missing_comparison_paths.setdefault(run_id, []).append(selector)
            continue
        selected_numeric = False
        for key, value in selected.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                comparisons[key] = value
                selected_numeric = True
        if not selected_numeric:
            missing_comparison_paths.setdefault(run_id, []).append(selector)
    config = {}
    if group_keys:
        for key in group_keys:
            if key in flat:
                config[key] = flat.get(key)
                config_sources.setdefault(run_id, {})[key] = sources[0] if sources else None
            elif key in wandb_config:
                config[key] = wandb_config.get(key)
                config_sources.setdefault(run_id, {})[key] = config_paths[0] if config_paths else None
            else:
                config[key] = None
                missing_config_keys.setdefault(run_id, []).append(key)
    for selector in group_paths:
        selected = selector_values(data, selector)
        alias, path = parse_selector(selector)
        if selected:
            key, value = next(iter(selected.items()))
            config[key] = value
            config_sources.setdefault(run_id, {})[key] = sources[0] if sources else None
        else:
            config[alias or path] = None
            missing_config_keys.setdefault(run_id, []).append(selector)
    for key in metrics:
        metric_sources.setdefault(run_id, {})[key] = sources[0] if sources else None
    for key in comparisons:
        comparison_sources.setdefault(run_id, {})[key] = sources[0] if sources else None
    if include_raw_artifacts:
        for output_path in output_paths[:3]:
            raw_content = load_json(output_path, run_id, "raw_result_artifact")
            if raw_content:
                raw_artifacts.append({
                    "run_id": run_id,
                    "path": output_path,
                    "basename": os.path.basename(output_path),
                    "content": raw_content,
                    "valid_json": True,
                })
    update_discovery_rejection(run_id)
    has_result = bool(metrics)
    rows.append({"run_id": run_id, "sources": sources, "config": config, "metrics": metrics, "comparisons": comparisons, "has_scientific_result": has_result, "missing_metric_paths": missing_metric_paths.get(run_id, []), "missing_comparison_paths": missing_comparison_paths.get(run_id, [])})
if not rows:
    raise SystemExit("no remote result files found")
print(json.dumps({
    "source": "remote_local_files",
    "sweep_id": sweep_id,
    "rows": rows,
    "valid_results": sum(1 for row in rows if row["has_scientific_result"]),
    "missing_results": sum(1 for row in rows if not row["has_scientific_result"]),
    "failed_results": 0,
    "partial": len(rows) < len(run_ids),
    "config_sources": config_sources,
    "metric_sources": metric_sources,
    "comparison_sources": comparison_sources,
    "missing_config_keys": missing_config_keys,
    "missing_metric_paths": missing_metric_paths,
    "missing_comparison_paths": missing_comparison_paths,
    "discovery_sources": discovery_sources,
    "raw_artifacts": raw_artifacts,
    "discovery_mode": discovery_mode,
    "artifact_rejections": artifact_rejections,
    "artifact_bytes_read": artifact_bytes_read,
    "artifact_limits": {
        "file_bytes": artifact_file_max_bytes,
        "total_bytes": artifact_total_max_bytes,
    },
}))
"""
        script = (
            script.replace("__CWD__", repr(remote_cwd))
            .replace("__SWEEP_ID__", repr(sweep_id))
            .replace("__RUN_IDS__", repr(run_ids[:max_runs]))
            .replace("__MAX_RUNS__", repr(max_runs))
            .replace("__METRIC_KEYS__", repr(metric_keys))
            .replace("__GROUP_KEYS__", repr(group_keys))
            .replace("__METRIC_PATHS__", repr(metric_paths or []))
            .replace("__GROUP_PATHS__", repr(group_paths or []))
            .replace("__OUTPUT_GLOBS__", repr(output_globs or []))
            .replace("__DISCOVERY_MODE__", repr(discovery_mode))
            .replace("__COMPARISON_PATHS__", repr(comparison_paths or []))
            .replace("__INCLUDE_RAW_ARTIFACTS__", repr(bool(include_raw_artifacts)))
            .replace("__ARTIFACT_FILE_MAX_BYTES__", repr(MAX_RESULT_ARTIFACT_FILE_BYTES))
            .replace("__ARTIFACT_TOTAL_MAX_BYTES__", repr(MAX_RESULT_ARTIFACT_TOTAL_BYTES))
            .replace("__BUDGET_SECONDS__", repr(budget_seconds))
        )
        result = self.run(
            host,
            "python3 -c " + shlex.quote(script),
            timeout=budget_seconds + 10,
            read_only=True,
        )
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
    ("wandb_agent_killed_runs", re.compile(r"(?:Killing runs and quitting|killing run[s]?|agent .*quitting|received .*exit)", re.IGNORECASE)),
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
        "wandb_agent_killed_runs",
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
    if "wandb_agent_killed_runs" in kinds:
        return ["W&B agent 日志显示调度层中断并杀掉运行；不要解释为科学失败，先核对 agent 生命周期、max_agents/队列策略和既有 job 诊断后再决定 recover 或重跑。"]
    if "command_error" in kinds:
        return ["检查远端 conda/env/path/program 配置；修复环境后对同一 job 执行 recover-agents。"]
    if "agent_process_missing" in kinds:
        return ["agent 进程已消失；查看 log_tails/error_signals 后决定修复代码或 recover-agents。"]
    return ["没有提取到明确错误；必要时 SSH 到远端查看 sources 中的完整日志。"]
