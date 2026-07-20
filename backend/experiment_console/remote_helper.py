#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JOB_ID_PATTERN = re.compile(r"^job_[A-Za-z0-9]{16,64}$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def read_payload() -> dict[str, Any]:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def process_start_ticks(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        try:
            value = subprocess.run(
                ["ps", "-o", "lstart=", "-p", str(pid)],
                text=True,
                capture_output=True,
                timeout=2,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return f"ps:{value}" if value else None
    _, separator, suffix = raw.rpartition(")")
    if not separator:
        return None
    fields = suffix.strip().split()
    return fields[19] if len(fields) > 19 else None


def process_matches(receipt: dict[str, Any]) -> bool:
    try:
        pid = int(receipt["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    expected_ticks = str(receipt.get("start_ticks") or "")
    if expected_ticks:
        return process_start_ticks(pid) == expected_ticks
    # Linux receipts always carry /proc start ticks. On development hosts
    # without /proc or a permitted ps invocation, retain the process-group
    # identity check rather than misclassifying a live supervisor as lost.
    try:
        return os.getpgid(pid) == int(receipt["pgid"])
    except (KeyError, ProcessLookupError, PermissionError, TypeError, ValueError):
        return False


def require_job_id(value: Any) -> str:
    job_id = str(value or "")
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("invalid job id")
    return job_id


def resolved_within(path_value: str, roots: list[str]) -> Path:
    path = Path(path_value).resolve(strict=True)
    allowed = [Path(item).resolve(strict=True) for item in roots]
    if not any(path == root or path.is_relative_to(root) for root in allowed):
        raise ValueError(f"path is outside allowed roots: {path}")
    return path


def job_dir(payload: dict[str, Any]) -> Path:
    job_id = require_job_id(payload.get("job_id"))
    state_root = Path(str(payload["state_root"])).resolve()
    return state_root / "jobs" / job_id


def inspect_job(directory: Path) -> dict[str, Any]:
    request = read_json(directory / "request.json") or {}
    receipt = read_json(directory / "receipt.json") or {}
    status = read_json(directory / "status.json")
    progress = read_json(directory / "progress.json") or {}
    if status:
        state = str(status.get("state") or "failed")
    elif receipt and process_matches(receipt):
        state = "running"
    else:
        state = "lost"
    result: dict[str, Any] = {
        "state": state,
        "observed_at": now(),
        "pid": receipt.get("pid"),
        "pgid": receipt.get("pgid"),
        "start_ticks": receipt.get("start_ticks"),
        "exit_code": status.get("exit_code") if status else None,
        "completed_runs": progress.get("completed_runs"),
        "total_runs": progress.get("total_runs", request.get("total_runs")),
        "progress_message": progress.get("message"),
    }
    return result


def validate_identity(directory: Path, payload: dict[str, Any]) -> None:
    request = read_json(directory / "request.json") or {}
    if request.get("job_id") != payload.get("job_id"):
        raise ValueError("remote request job id mismatch")
    expected_digest = payload.get("command_digest")
    if expected_digest and request.get("command_digest") != expected_digest:
        raise ValueError("remote request command digest mismatch")
    receipt = read_json(directory / "receipt.json")
    if expected_digest and receipt and receipt.get("command_digest") != expected_digest:
        raise ValueError("remote receipt command digest mismatch")


def action_resources(payload: dict[str, Any]) -> None:
    argv = payload.get("gpu_query_argv")
    if not isinstance(argv, list) or not argv:
        raise ValueError("gpu_query_argv is required")
    process = subprocess.run(
        argv, text=True, capture_output=True, timeout=15, check=False
    )
    if process.returncode != 0:
        raise RuntimeError((process.stderr or process.stdout)[-2000:])
    minimum_free = int(payload.get("gpu_min_free_mb") or 0)
    maximum_util = int(payload.get("gpu_max_utilization") or 100)
    gpus = []
    for row in csv.reader(process.stdout.splitlines(), skipinitialspace=True):
        if len(row) < 6:
            continue
        index, name, total, used, free, utilization = row[:6]
        item = {
            "index": int(index.strip()),
            "name": name.strip(),
            "memory_total_mb": int(total.strip()),
            "memory_used_mb": int(used.strip()),
            "memory_free_mb": int(free.strip()),
            "utilization": int(utilization.strip()),
        }
        item["available"] = (
            item["memory_free_mb"] >= minimum_free
            and item["utilization"] <= maximum_util
        )
        gpus.append(item)
    emit({"observed_at": now(), "gpus": gpus})


def action_launch(payload: dict[str, Any]) -> None:
    directory = job_dir(payload)
    request = payload.get("request")
    if not isinstance(request, dict):
        raise ValueError("request is required")
    request["job_id"] = require_job_id(payload.get("job_id"))
    request["state_root"] = str(Path(str(payload["state_root"])).resolve())
    request_hash = str(request.get("request_hash") or "")
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        existing = read_json(directory / "request.json") or {}
        if existing.get("request_hash") != request_hash:
            raise RuntimeError("remote job id already exists with a different request")
        emit(inspect_job(directory))
        return
    resolved_within(str(request["cwd"]), list(request["allowed_roots"]))
    atomic_json(directory / "request.json", request)
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "supervise", str(directory)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    receipt_path = directory / "receipt.json"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if receipt_path.is_file() or process.poll() is not None:
            break
        time.sleep(0.05)
    emit(inspect_job(directory))


def supervise(directory_value: str) -> int:
    directory = Path(directory_value).resolve(strict=True)
    request = read_json(directory / "request.json")
    if not request:
        return 70
    cwd = resolved_within(str(request["cwd"]), list(request["allowed_roots"]))
    argv = request.get("argv")
    bootstrap = request.get("bootstrap_argv") or []
    if not isinstance(argv, list) or not argv or not isinstance(bootstrap, list):
        return 70
    command = [str(item) for item in bootstrap + argv]
    environment = os.environ.copy()
    environment.update(
        {str(key): str(value) for key, value in (request.get("env") or {}).items()}
    )
    gpu_indices = [int(item) for item in request.get("gpu_indices") or []]
    if gpu_indices:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in gpu_indices)
    environment["EXPERIMENT_CONSOLE_JOB_ID"] = str(request["job_id"])
    environment["EXPERIMENT_CONSOLE_PROGRESS_FILE"] = str(directory / "progress.json")
    pid = os.getpid()
    receipt = {
        "job_id": request["job_id"],
        "command_digest": request["command_digest"],
        "pid": pid,
        "pgid": os.getpgrp(),
        "start_ticks": process_start_ticks(pid),
        "started_at": now(),
    }
    cancel_signal: list[int] = []

    def handle_cancel(signum, _frame):
        cancel_signal.append(signum)

    signal.signal(signal.SIGTERM, handle_cancel)
    signal.signal(signal.SIGINT, handle_cancel)
    log_directory = cwd / ".experiment-console-v3" / str(request["job_id"])
    log_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    stdout_path = log_directory / "stdout.log"
    stderr_path = log_directory / "stderr.log"
    stdout_path.touch(mode=0o600, exist_ok=True)
    stderr_path.touch(mode=0o600, exist_ok=True)
    receipt["stdout_path"] = str(stdout_path)
    receipt["stderr_path"] = str(stderr_path)
    atomic_json(directory / "receipt.json", receipt)
    with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
        "ab", buffering=0
    ) as stderr:
        try:
            child = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=False,
                close_fds=True,
            )
            if cancel_signal or (directory / "cancel-requested.json").exists():
                child.terminate()
            exit_code = child.wait()
        except Exception as exc:
            stderr.write(
                f"experiment-console launch failed: {type(exc).__name__}: {exc}\n".encode()
            )
            exit_code = 127
    cancelled = bool(cancel_signal) or (directory / "cancel-requested.json").exists()
    state = "cancelled" if cancelled else "succeeded" if exit_code == 0 else "failed"
    atomic_json(
        directory / "status.json",
        {"state": state, "exit_code": exit_code, "finished_at": now()},
    )
    return exit_code


def action_inspect(payload: dict[str, Any]) -> None:
    directory = job_dir(payload)
    validate_identity(directory, payload)
    emit(inspect_job(directory))


def action_cancel(payload: dict[str, Any]) -> None:
    directory = job_dir(payload)
    validate_identity(directory, payload)
    atomic_json(directory / "cancel-requested.json", {"requested_at": now()})
    receipt = read_json(directory / "receipt.json") or {}
    if not process_matches(receipt):
        emit(inspect_job(directory))
        return
    pgid = int(receipt["pgid"])
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + float(payload.get("grace_seconds") or 10)
    while time.monotonic() < deadline and process_matches(receipt):
        time.sleep(0.1)
    if process_matches(receipt):
        os.killpg(pgid, signal.SIGKILL)
    emit(inspect_job(directory))


def action_logs(payload: dict[str, Any]) -> None:
    directory = job_dir(payload)
    validate_identity(directory, payload)
    stream = str(payload.get("stream") or "stdout")
    if stream not in {"stdout", "stderr"}:
        raise ValueError("stream must be stdout or stderr")
    receipt = read_json(directory / "receipt.json") or {}
    request = read_json(directory / "request.json") or {}
    cwd = Path(str(request["cwd"])).resolve(strict=True)
    path = Path(
        str(receipt.get(f"{stream}_path") or directory / f"{stream}.log")
    ).resolve(strict=True)
    if not (path == cwd or path.is_relative_to(cwd)):
        raise ValueError("log path is outside the job working directory")
    offset = max(0, int(payload.get("offset") or 0))
    limit = min(max(1, int(payload.get("limit") or 65536)), 4 * 1024 * 1024)
    try:
        size = path.stat().st_size
        if payload.get("tail"):
            offset = max(0, size - limit)
        with path.open("rb") as handle:
            handle.seek(min(offset, size))
            data = handle.read(limit)
    except FileNotFoundError:
        size, data = 0, b""
    next_offset = min(offset, size) + len(data)
    emit(
        {
            "stream": stream,
            "offset": offset,
            "next_offset": next_offset,
            "eof": next_offset >= size,
            "text": data.decode("utf-8", errors="replace"),
        }
    )


def action_fetch(payload: dict[str, Any]) -> None:
    directory = job_dir(payload)
    validate_identity(directory, payload)
    request = read_json(directory / "request.json") or {}
    cwd = Path(str(request["cwd"])).resolve(strict=True)
    requested = Path(str(payload.get("path") or ""))
    target = (
        requested if requested.is_absolute() else cwd / requested
    ).resolve(strict=True)
    if not (target == cwd or target.is_relative_to(cwd)) or not target.is_file():
        raise ValueError("fetch path must be a regular file inside the job working directory")
    offset = max(0, int(payload.get("offset") or 0))
    limit = min(
        max(1, int(payload.get("limit") or 1024 * 1024)), 16 * 1024 * 1024
    )
    size = target.stat().st_size
    with target.open("rb") as handle:
        handle.seek(min(offset, size))
        data = handle.read(limit)
    next_offset = min(offset, size) + len(data)
    emit(
        {
            "path": str(target),
            "offset": offset,
            "next_offset": next_offset,
            "eof": next_offset >= size,
            "size": size,
            "data_base64": base64.b64encode(data).decode("ascii"),
        }
    )


def main() -> int:
    if len(sys.argv) < 2:
        return 64
    action = sys.argv[1]
    if action == "supervise":
        return supervise(sys.argv[2])
    payload = read_payload()
    actions = {
        "resources": action_resources,
        "launch": action_launch,
        "inspect": action_inspect,
        "cancel": action_cancel,
        "logs": action_logs,
        "fetch": action_fetch,
    }
    handler = actions.get(action)
    if handler is None:
        raise ValueError(f"unsupported action: {action}")
    handler(payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
