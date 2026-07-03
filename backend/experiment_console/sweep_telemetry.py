from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_STATES = {"finished", "failed", "crashed", "killed", "cancelled", "canceled"}
FINISHED_STATES = {"finished"}
FAILED_STATES = {"failed", "crashed", "killed"}
RUNNING_STATES = {"running", "pending"}


def sweep_key(sweep: dict[str, Any]) -> str:
    return "/".join(str(sweep.get(key) or "") for key in ["entity", "project", "id"])


def load_telemetry_cache(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_telemetry_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def enrich_sweeps_with_telemetry(
    sweeps: list[dict[str, Any]],
    *,
    cache: dict[str, Any],
    observed_at: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stamp = observed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    updated_cache = dict(cache)
    enriched = []
    for sweep in sweeps:
        item = dict(sweep)
        telemetry, history = compute_sweep_telemetry(item, cache.get(sweep_key(item)), observed_at=stamp)
        item.update(telemetry)
        updated_cache[sweep_key(item)] = history
        enriched.append(item)
    return enriched, updated_cache


def compute_sweep_telemetry(
    sweep: dict[str, Any],
    previous: dict[str, Any] | None = None,
    *,
    observed_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stamp = observed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    runs = [run for run in sweep.get("runs") or [] if isinstance(run, dict)]
    expected = _to_int(sweep.get("expectedRunCount"))
    run_count = _to_int(sweep.get("runCount"))
    state = str(sweep.get("state") or "").lower()

    if runs:
        raw_finished = sum(1 for run in runs if _run_state(run) in FINISHED_STATES)
        raw_failed = sum(1 for run in runs if _run_state(run) in FAILED_STATES)
        raw_running = sum(1 for run in runs if _run_state(run) in RUNNING_STATES)
        source = "wandb_runs"
    else:
        if state == "finished":
            raw_finished = expected or run_count
        else:
            raw_finished = 0
        raw_failed = 0
        raw_running = 0
        source = "wandb_sweep_runCount"

    finished = raw_finished
    failed = raw_failed
    running = raw_running
    consistency = "consistent"
    if state == "finished" and source == "wandb_runs" and expected > 0 and run_count >= expected:
        normalized_finished = max(raw_finished, expected - raw_failed)
        if raw_running or normalized_finished != raw_finished:
            consistency = "terminal_run_edges_stale"
        finished = normalized_finished
        running = 0
    elif source == "wandb_sweep_runCount":
        consistency = "fallback_from_sweep_count"

    typical_run_seconds = _typical_finished_run_seconds(runs)
    speed = _duration_based_speed_per_hour(typical_run_seconds, running) or _speed_per_hour(previous, finished, stamp)
    eta = _eta_seconds(
        expected,
        finished,
        speed,
        state,
        runs=runs,
        observed_at=stamp,
        typical_run_seconds=typical_run_seconds,
        running=running,
    )
    telemetry = {
        "finished_runs": finished,
        "running_runs": running,
        "failed_runs": failed,
        "last_sync_at": stamp,
        "speed_per_hour": speed,
        "eta_seconds": eta,
        "run_state_counts_source": source,
        "raw_run_state_counts": {
            "finished": raw_finished,
            "running": raw_running,
            "failed": raw_failed,
        },
        "run_state_counts_consistency": consistency,
    }
    history = {
        "entity": sweep.get("entity"),
        "project": sweep.get("project"),
        "id": sweep.get("id"),
        "state": sweep.get("state"),
        "expectedRunCount": expected,
        "finished_runs": finished,
        "running_runs": running,
        "failed_runs": failed,
        "raw_run_state_counts": telemetry["raw_run_state_counts"],
        "run_state_counts_consistency": consistency,
        "observed_at": stamp,
    }
    return telemetry, history


def strip_runs(sweep: dict[str, Any]) -> dict[str, Any]:
    compact = dict(sweep)
    compact.pop("runs", None)
    return compact


def _run_state(run: dict[str, Any]) -> str:
    return str(run.get("state") or "").lower()


def _to_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except Exception:
        return 0


def _speed_per_hour(previous: dict[str, Any] | None, finished: int, observed_at: str) -> float | None:
    if not isinstance(previous, dict):
        return None
    previous_finished = _to_int(previous.get("finished_runs"))
    previous_at = _parse_time(previous.get("observed_at"))
    current_at = _parse_time(observed_at)
    if not previous_at or not current_at:
        return None
    delta_runs = finished - previous_finished
    delta_seconds = (current_at - previous_at).total_seconds()
    if delta_runs <= 0 or delta_seconds <= 0:
        return None
    return delta_runs * 3600.0 / delta_seconds


def _duration_based_speed_per_hour(typical_run_seconds: float | None, running: int) -> float | None:
    if not typical_run_seconds or typical_run_seconds <= 0 or running <= 0:
        return None
    return running * 3600.0 / typical_run_seconds


def _typical_finished_run_seconds(runs: list[dict[str, Any]]) -> float | None:
    durations = []
    for run in runs:
        if _run_state(run) not in FINISHED_STATES:
            continue
        started = _parse_time(run.get("created_at"))
        ended = _parse_time(run.get("heartbeat_at"))
        if not started or not ended:
            continue
        duration = (ended - started).total_seconds()
        if duration > 0:
            durations.append(duration)
    if not durations:
        return None
    durations.sort()
    middle = len(durations) // 2
    if len(durations) % 2:
        return durations[middle]
    return (durations[middle - 1] + durations[middle]) / 2


def _eta_seconds(
    expected: int,
    finished: int,
    speed_per_hour: float | None,
    state: str,
    *,
    runs: list[dict[str, Any]] | None = None,
    observed_at: str | None = None,
    typical_run_seconds: float | None = None,
    running: int = 0,
) -> int | None:
    if state in TERMINAL_STATES:
        return None
    if expected <= 0 or finished >= expected:
        return None
    if typical_run_seconds and typical_run_seconds > 0 and running > 0:
        current_at = _parse_time(observed_at)
        active_runs = [run for run in (runs or []) if _run_state(run) in RUNNING_STATES]
        active_work_seconds = 0.0
        for run in active_runs:
            started = _parse_time(run.get("created_at"))
            if started and current_at:
                age_seconds = max(0.0, (current_at - started).total_seconds())
                active_work_seconds += max(typical_run_seconds - age_seconds, min(60.0, typical_run_seconds * 0.1))
            else:
                active_work_seconds += typical_run_seconds
        queued_runs = max(0, expected - finished - len(active_runs))
        total_work_seconds = active_work_seconds + queued_runs * typical_run_seconds
        return int(round(total_work_seconds / running))
    if not speed_per_hour or speed_per_hour <= 0:
        return None
    return int(round((expected - finished) / speed_per_hour * 3600))


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None
