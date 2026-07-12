from __future__ import annotations

from experiment_console.sweep_telemetry import compute_sweep_telemetry
from experiment_console.wandb_client import format_sweep


def test_first_observation_has_no_speed_or_eta():
    sweep = {
        "id": "s1",
        "entity": "e",
        "project": "p",
        "state": "RUNNING",
        "runCount": 1,
        "expectedRunCount": 5,
        "runs": [{"name": "run-a", "state": "finished"}],
    }

    telemetry, _ = compute_sweep_telemetry(sweep, observed_at="2026-06-17T00:00:00+00:00")

    assert telemetry["finished_runs"] == 1
    assert telemetry["running_runs"] == 0
    assert telemetry["speed_per_hour"] is None
    assert telemetry["eta_seconds"] is None


def test_second_observation_computes_speed_and_eta():
    previous_sweep = {
        "id": "s1",
        "entity": "e",
        "project": "p",
        "state": "RUNNING",
        "runCount": 1,
        "expectedRunCount": 5,
        "runs": [{"name": "run-a", "state": "finished"}],
    }
    _, history = compute_sweep_telemetry(previous_sweep, observed_at="2026-06-17T00:00:00+00:00")
    current_sweep = {
        **previous_sweep,
        "runCount": 3,
        "runs": [
            {"name": "run-a", "state": "finished"},
            {"name": "run-b", "state": "finished"},
            {"name": "run-c", "state": "finished"},
        ],
    }

    telemetry, _ = compute_sweep_telemetry(current_sweep, history, observed_at="2026-06-17T01:00:00+00:00")

    assert telemetry["finished_runs"] == 3
    assert telemetry["running_runs"] == 0
    assert telemetry["speed_per_hour"] == 2.0
    assert telemetry["eta_seconds"] == 3600


def test_active_run_durations_keep_speed_and_eta_stable_without_new_completions():
    previous = {
        "finished_runs": 2,
        "observed_at": "2026-06-17T00:00:00+00:00",
    }
    sweep = {
        "id": "s1",
        "entity": "e",
        "project": "p",
        "state": "RUNNING",
        "runCount": 4,
        "expectedRunCount": 6,
        "runs": [
            {
                "name": "run-a",
                "state": "finished",
                "created_at": "2026-06-17T00:00:00+00:00",
                "heartbeat_at": "2026-06-17T00:10:00+00:00",
            },
            {
                "name": "run-b",
                "state": "finished",
                "created_at": "2026-06-17T00:00:30+00:00",
                "heartbeat_at": "2026-06-17T00:10:30+00:00",
            },
            {
                "name": "run-c",
                "state": "running",
                "created_at": "2026-06-17T00:05:00+00:00",
            },
            {
                "name": "run-d",
                "state": "running",
                "created_at": "2026-06-17T00:05:00+00:00",
            },
        ],
    }

    telemetry, _ = compute_sweep_telemetry(sweep, previous, observed_at="2026-06-17T00:10:00+00:00")

    assert telemetry["finished_runs"] == 2
    assert telemetry["running_runs"] == 2
    assert telemetry["speed_per_hour"] == 12.0
    assert telemetry["eta_seconds"] == 900


def test_running_sweep_with_no_finished_runs_reports_progress_evidence():
    sweep = {
        "id": "s1",
        "entity": "e",
        "project": "p",
        "state": "RUNNING",
        "runCount": 2,
        "expectedRunCount": 5,
        "runs": [
            {
                "name": "run-a",
                "state": "running",
                "created_at": "2026-06-17T00:00:00+00:00",
                "heartbeat_at": "2026-06-17T00:29:00+00:00",
            },
            {
                "name": "run-b",
                "state": "running",
                "created_at": "2026-06-17T00:10:00+00:00",
                "heartbeat_at": "2026-06-17T00:28:00+00:00",
            },
        ],
    }

    telemetry, _ = compute_sweep_telemetry(sweep, observed_at="2026-06-17T00:30:00+00:00")

    evidence = telemetry["progress_evidence"]
    assert telemetry["finished_runs"] == 0
    assert telemetry["running_runs"] == 2
    assert evidence["classification"] == "active_run_evidence"
    assert evidence["active_run_count"] == 2
    assert evidence["oldest_active_run_age_seconds"] == 1800
    assert evidence["newest_active_run_heartbeat_lag_seconds"] == 60
    assert "oldest active for 30m" in evidence["message"]


def test_running_sweep_without_active_runs_reports_zero_running():
    sweep = {
        "id": "s1",
        "entity": "e",
        "project": "p",
        "state": "RUNNING",
        "runCount": 2,
        "expectedRunCount": 5,
        "runs": [
            {"name": "run-a", "state": "finished"},
            {"name": "run-b", "state": "finished"},
        ],
    }

    telemetry, _ = compute_sweep_telemetry(sweep, observed_at="2026-06-17T00:00:00+00:00")

    assert telemetry["running_runs"] == 0


def test_terminal_sweep_does_not_emit_eta():
    previous = {
        "finished_runs": 1,
        "observed_at": "2026-06-17T00:00:00+00:00",
    }
    sweep = {
        "id": "s1",
        "entity": "e",
        "project": "p",
        "state": "FINISHED",
        "runCount": 5,
        "expectedRunCount": 5,
        "runs": [{"name": str(index), "state": "finished"} for index in range(5)],
    }

    telemetry, _ = compute_sweep_telemetry(sweep, previous, observed_at="2026-06-17T01:00:00+00:00")

    assert telemetry["speed_per_hour"] == 4.0
    assert telemetry["eta_seconds"] is None


def test_format_sweep_progress_uses_run_count_even_when_finished():
    sweep = format_sweep(
        "e",
        "p",
        {
            "name": "s1",
            "state": "FINISHED",
            "runCount": 3,
            "config": {"parameters": {"seed": {"values": list(range(5))}}},
        },
    )

    assert sweep["expectedRunCount"] == 5
    assert sweep["progress"] == 0.6


def test_terminal_sweep_preserves_stale_raw_run_edges():
    sweep = {
        "id": "s1",
        "entity": "e",
        "project": "p",
        "state": "FINISHED",
        "runCount": 15,
        "expectedRunCount": 15,
        "runs": [
            *[{"name": f"run-{index}", "state": "finished"} for index in range(14)],
            {"name": "run-14", "state": "running"},
        ],
    }

    telemetry, _ = compute_sweep_telemetry(sweep, observed_at="2026-06-17T01:00:00+00:00")

    assert telemetry["finished_runs"] == 14
    assert telemetry["running_runs"] == 1
    assert telemetry["raw_run_state_counts"]["finished"] == 14
    assert telemetry["raw_run_state_counts"]["running"] == 1
    assert telemetry["run_state_counts_consistency"] == "terminal_run_edges_stale"


def test_non_terminal_sweep_without_runs_does_not_guess_completion():
    sweep = {
        "id": "s1",
        "entity": "e",
        "project": "p",
        "state": "RUNNING",
        "runCount": 7,
        "expectedRunCount": 10,
        "runs": [],
    }

    telemetry, _ = compute_sweep_telemetry(sweep, observed_at="2026-06-17T00:00:00+00:00")

    assert telemetry["finished_runs"] == 0
    assert telemetry["running_runs"] == 0
    assert telemetry["eta_seconds"] is None
