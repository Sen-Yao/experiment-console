from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


AGENT_RECONCILER_VERSION = 2
AGENT_RETRY_BACKOFF_SECONDS = (30, 120, 300, 600)
AGENT_STEADY_RECONCILE_SECONDS = 30
AGENT_FAILURE_ATTENTION_THRESHOLD = 3
AGENT_TERMINAL_LIFECYCLES = {"terminal", "stopped", "fatal"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def initial_agent_reconciler(
    *,
    max_agents: int | None,
    expected_runs: int,
    now: str | None = None,
) -> dict[str, Any]:
    stamp = now or utc_now()
    expected = max(1, int(expected_runs))
    ceiling = max(1, int(max_agents)) if max_agents is not None else expected
    return {
        "version": AGENT_RECONCILER_VERSION,
        "lifecycle": "pending",
        "classification": "capacity_pending",
        "max_agents": ceiling,
        "expected_runs": expected,
        "remaining_runs": expected,
        "desired_agents": min(ceiling, expected),
        "live_agents": 0,
        "assignments": [],
        "consecutive_launch_failures": 0,
        "launch_failure_episode": None,
        "current_failure": None,
        "last_resolved_failure": None,
        "hardware_eligible_gpus": [],
        "allocatable_gpus": [],
        "first_agent_at": None,
        "last_attempt_at": None,
        "last_success_at": None,
        "next_reconcile_at": stamp,
        "attention_episode": None,
        "created_at": stamp,
        "updated_at": stamp,
    }


def is_managed_agent_job(job_or_monitor: Any) -> bool:
    monitor = getattr(job_or_monitor, "monitor", job_or_monitor)
    if not isinstance(monitor, dict):
        return False
    state = monitor.get("agent_reconciler")
    return isinstance(state, dict) and state.get("version") == AGENT_RECONCILER_VERSION


def agent_reconcile_due(job_or_monitor: Any, *, now: str | None = None) -> bool:
    monitor = getattr(job_or_monitor, "monitor", job_or_monitor)
    if not isinstance(monitor, dict):
        return False
    state = monitor.get("agent_reconciler")
    if not isinstance(state, dict) or state.get("version") != AGENT_RECONCILER_VERSION:
        return False
    if str(state.get("lifecycle") or "") in AGENT_TERMINAL_LIFECYCLES:
        return False
    next_at = state.get("next_reconcile_at")
    if not next_at:
        return False
    return parse_timestamp(str(next_at)) <= parse_timestamp(now or utc_now())


def sweep_capacity(sweep: dict[str, Any] | None, *, fallback_expected: int) -> dict[str, int | bool]:
    sweep = sweep if isinstance(sweep, dict) else {}
    state = str(sweep.get("state") or "").lower()
    expected = positive_int(sweep.get("expectedRunCount")) or max(
        1,
        positive_int(fallback_expected),
        positive_int(sweep.get("runCount")),
    )
    raw = sweep.get("raw_run_state_counts") if isinstance(sweep.get("raw_run_state_counts"), dict) else {}
    finished = non_negative_int(raw.get("finished"))
    failed = non_negative_int(raw.get("failed"))
    running = non_negative_int(raw.get("running"))
    if not raw and isinstance(sweep.get("runs"), list):
        states = [str(item.get("state") or "").lower() for item in sweep["runs"] if isinstance(item, dict)]
        finished = sum(item in {"finished", "completed"} for item in states)
        failed = sum(item in {"failed", "crashed", "killed", "cancelled", "canceled"} for item in states)
        running = sum(item in {"running", "pending"} for item in states)
    terminal = state in {"finished", "failed", "crashed", "killed", "cancelled", "canceled"}
    remaining = 0 if terminal else max(0, expected - finished - failed)
    return {
        "expected": expected,
        "finished": finished,
        "failed": failed,
        "running": running,
        "remaining": remaining,
        "terminal": terminal,
    }


def desired_agent_count(*, max_agents: int, remaining_runs: int) -> int:
    return max(0, min(max(1, int(max_agents)), max(0, int(remaining_runs))))


def retry_delay_seconds(consecutive_failures: int) -> int:
    index = max(0, min(len(AGENT_RETRY_BACKOFF_SECONDS) - 1, int(consecutive_failures) - 1))
    return AGENT_RETRY_BACKOFF_SECONDS[index]


def next_reconcile_at(*, now: str, delay_seconds: int) -> str:
    return (parse_timestamp(now) + timedelta(seconds=max(1, delay_seconds))).isoformat(timespec="seconds")


def age_seconds(since: str | None, *, now: str) -> int:
    if not since:
        return 0
    return max(0, int((parse_timestamp(now) - parse_timestamp(since)).total_seconds()))


def parse_timestamp(value: str) -> datetime:
    text = str(value or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def positive_int(value: Any) -> int:
    parsed = non_negative_int(value)
    return parsed if parsed > 0 else 0


def non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
