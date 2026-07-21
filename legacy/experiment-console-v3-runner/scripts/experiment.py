#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4


API_VERSION = "3"
DEFAULT_URL = os.environ.get("EXPERIMENT_CONSOLE_URL", "http://127.0.0.1:5174").rstrip("/")
EXPECTED_INSTANCE_ID = os.environ.get(
    "EXPERIMENT_CONSOLE_EXPECTED_INSTANCE_ID", "yggdrasil-production-v3"
)
TOKEN_FILE = Path(
    os.environ.get(
        "EXPERIMENT_CONSOLE_TOKEN_FILE", "~/.config/experiment-console/console_api_token"
    )
).expanduser()
DEFAULT_TASK_ID = os.environ.get("CODEX_THREAD_ID")
DEFAULT_MAX_FETCH_BYTES = 256 * 1024 * 1024


class RunnerError(RuntimeError):
    pass


def headers(*, json_body: bool = False) -> dict[str, str]:
    result = {"Accept": "application/json"}
    try:
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = ""
    except OSError as exc:
        raise RunnerError(f"cannot read Console token file {TOKEN_FILE}: {exc}") from exc
    if token:
        result["Authorization"] = f"Bearer {token}"
    if json_body:
        result["Content-Type"] = "application/json"
    return result


def request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
    timeout: float = 120,
) -> tuple[bytes, Any]:
    url = f"{DEFAULT_URL}/{path.lstrip('/')}"
    if query:
        values = {key: value for key, value in query.items() if value is not None}
        url = f"{url}?{urllib.parse.urlencode(values)}"
    body = json.dumps(payload, separators=(",", ":")).encode() if payload is not None else None
    call = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers=headers(json_body=payload is not None),
    )
    try:
        with urllib.request.urlopen(call, timeout=timeout) as response:
            return response.read(), response.headers
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            detail = str(parsed.get("detail") or parsed)
        except json.JSONDecodeError:
            pass
        raise RunnerError(f"Console HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise RunnerError(f"Console request failed: {exc}") from exc


def request_json(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
    timeout: float = 120,
) -> dict[str, Any]:
    body, _ = request(method, path, payload=payload, query=query, timeout=timeout)
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RunnerError("Console returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise RunnerError("Console returned a non-object response")
    return decoded


def verify_console() -> dict[str, Any]:
    health = request_json("GET", "/health", timeout=5)
    if health.get("status") != "ok":
        raise RunnerError(f"Console is not healthy: {health.get('status')!r}")
    if str(health.get("api_version")) != API_VERSION:
        raise RunnerError(
            f"Console API version mismatch: expected {API_VERSION}, got {health.get('api_version')!r}"
        )
    if health.get("instance_id") != EXPECTED_INSTANCE_ID:
        raise RunnerError(
            f"Console instance mismatch: expected {EXPECTED_INSTANCE_ID!r}, "
            f"got {health.get('instance_id')!r}"
        )
    return health


def emit(value: Any, *, as_json: bool, text: str | None = None) -> None:
    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif text is not None:
        print(text)
    else:
        print(value)


def parse_env(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise RunnerError(f"--env must use KEY=VALUE: {item!r}")
        result[key] = value
    return result


def parse_gpus(items: list[str]) -> list[int]:
    values: list[int] = []
    for item in items:
        for token in item.split(","):
            try:
                values.append(int(token))
            except ValueError as exc:
                raise RunnerError(f"invalid GPU index: {token!r}") from exc
    if len(values) != len(set(values)) or any(value < 0 for value in values):
        raise RunnerError("GPU indices must be unique non-negative integers")
    return values


def format_job(job: dict[str, Any]) -> str:
    lines = [
        f"job_id: {job.get('job_id')}",
        f"status: {job.get('status')}",
        f"profile: {job.get('profile')}",
        f"cwd: {job.get('cwd')}",
        f"gpus: {','.join(map(str, job.get('gpu_indices') or [])) or '-'}",
        f"elapsed_seconds: {job.get('elapsed_seconds')}",
        f"progress: {job.get('completed_runs')}/{job.get('total_runs') or '?'}",
        f"eta_seconds: {job.get('eta_seconds') if job.get('eta_seconds') is not None else '-'}",
    ]
    if job.get("exit_code") is not None:
        lines.append(f"exit_code: {job.get('exit_code')}")
    if job.get("progress_message"):
        lines.append(f"message: {job.get('progress_message')}")
    if job.get("last_error"):
        lines.append(f"last_error: {job.get('last_error')}")
    if job.get("resource_conflicts"):
        lines.append(
            "resource_conflicts: "
            + json.dumps(job["resource_conflicts"], ensure_ascii=False, sort_keys=True)
        )
    return "\n".join(lines)


def command_resources(args) -> None:
    verify_console()
    result = request_json("GET", "/api/resources", query={"profile": args.profile})
    if args.json:
        emit(result, as_json=True)
        return
    lines = []
    for profile in result.get("profiles") or []:
        lines.append(f"profile: {profile.get('name')} ({profile.get('status')})")
        if profile.get("error"):
            lines.append(f"  error: {profile['error']}")
        for gpu in profile.get("gpus") or []:
            owner = gpu.get("locked_by_job_id") or "-"
            lines.append(
                f"  gpu {gpu.get('index')}: available={str(bool(gpu.get('available'))).lower()} "
                f"free_mb={gpu.get('memory_free_mb')} util={gpu.get('utilization')}% locked_by={owner}"
            )
    emit(result, as_json=False, text="\n".join(lines))


def command_run(args) -> None:
    verify_console()
    argv = list(args.argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise RunnerError("run requires a command after '--'")
    task_id = str(args.task_id or DEFAULT_TASK_ID or "").strip()
    if not task_id:
        raise RunnerError("run requires --task-id or CODEX_THREAD_ID")
    request_id = args.request_id or f"req_{uuid4().hex}"
    payload = {
        "request_id": request_id,
        "task_id": task_id,
        "profile": args.profile,
        "cwd": args.cwd,
        "argv": argv,
        "env": parse_env(args.env),
        "gpu_indices": parse_gpus(args.gpu),
        "total_runs": args.total_runs,
        "name": args.name,
    }
    try:
        result = request_json("POST", "/api/jobs", payload=payload, timeout=args.timeout)
    except RunnerError as exc:
        raise RunnerError(f"{exc}\nrequest_id: {request_id}") from exc
    job = result["job"]
    text = f"request_id: {request_id}\nreplayed: {str(bool(result.get('replayed'))).lower()}\n{format_job(job)}"
    emit(result, as_json=args.json, text=text)


def command_status(args) -> None:
    verify_console()
    result = request_json(
        "GET", f"/api/jobs/{urllib.parse.quote(args.job_id, safe='')}", query={"refresh": "true"}
    )
    emit(result, as_json=args.json, text=format_job(result["job"]))


def command_logs(args) -> None:
    verify_console()
    result = request_json(
        "GET",
        f"/api/jobs/{urllib.parse.quote(args.job_id, safe='')}/logs",
        query={
            "stream": args.stream,
            "offset": args.offset,
            "limit": args.limit,
            "tail": str(not args.from_start and args.offset == 0).lower(),
        },
    )
    emit(result, as_json=args.json, text=str(result.get("text") or ""))


def output_name(remote_path: str) -> str:
    candidate = remote_path.rstrip("/").rsplit("/", 1)[-1]
    if candidate in {"", ".", ".."}:
        raise RunnerError("cannot derive an output filename; pass --output")
    return candidate


def command_fetch(args) -> None:
    verify_console()
    destination = Path(args.output or output_name(args.path)).expanduser()
    if destination.exists():
        raise RunnerError(f"fetch destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary)
    offset = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            while True:
                body, response_headers = request(
                    "GET",
                    f"/api/jobs/{urllib.parse.quote(args.job_id, safe='')}/files",
                    query={"path": args.path, "offset": offset, "limit": args.chunk_bytes},
                    timeout=args.timeout,
                )
                handle.write(body)
                offset = int(response_headers["X-Next-Offset"])
                if offset > args.max_bytes:
                    raise RunnerError("fetch exceeds --max-bytes")
                if response_headers.get("X-End-Of-File") == "1":
                    break
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    result = {
        "job_id": args.job_id,
        "remote_path": args.path,
        "local_path": str(destination),
        "bytes": offset,
    }
    emit(result, as_json=args.json, text=f"fetched {offset} bytes to {destination}")


def command_cancel(args) -> None:
    verify_console()
    result = request_json(
        "POST",
        f"/api/jobs/{urllib.parse.quote(args.job_id, safe='')}/cancel",
        payload={"reason": args.reason},
        timeout=args.timeout,
    )
    emit(result, as_json=args.json, text=format_job(result["job"]))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="exp", description="Experiment Console v3 durable command runner"
    )
    sub = result.add_subparsers(dest="command", required=True)

    command = sub.add_parser("resources")
    command.add_argument("--profile")
    command.add_argument("--json", action="store_true")
    command.set_defaults(func=command_resources)

    command = sub.add_parser("run")
    command.add_argument("--profile", required=True)
    command.add_argument("--cwd", required=True)
    command.add_argument("--gpu", action="append", default=[])
    command.add_argument("--env", action="append", default=[])
    command.add_argument("--total-runs", type=int)
    command.add_argument("--name")
    command.add_argument("--request-id")
    command.add_argument("--task-id")
    command.add_argument("--timeout", type=float, default=180)
    command.add_argument("--json", action="store_true")
    command.add_argument("argv", nargs=argparse.REMAINDER)
    command.set_defaults(func=command_run)

    command = sub.add_parser("status")
    command.add_argument("job_id")
    command.add_argument("--json", action="store_true")
    command.set_defaults(func=command_status)

    command = sub.add_parser("logs")
    command.add_argument("job_id")
    command.add_argument("--stream", choices=("stdout", "stderr"), default="stdout")
    command.add_argument("--offset", type=int, default=0)
    command.add_argument("--limit", type=int, default=65536)
    command.add_argument("--from-start", action="store_true")
    command.add_argument("--json", action="store_true")
    command.set_defaults(func=command_logs)

    command = sub.add_parser("fetch")
    command.add_argument("job_id")
    command.add_argument("path")
    command.add_argument("--output")
    command.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    command.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_FETCH_BYTES)
    command.add_argument("--timeout", type=float, default=120)
    command.add_argument("--json", action="store_true")
    command.set_defaults(func=command_fetch)

    command = sub.add_parser("cancel")
    command.add_argument("job_id")
    command.add_argument("--reason")
    command.add_argument("--timeout", type=float, default=60)
    command.add_argument("--json", action="store_true")
    command.set_defaults(func=command_cancel)
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        args.func(args)
        return 0
    except RunnerError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
