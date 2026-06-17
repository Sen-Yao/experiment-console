from __future__ import annotations

from experiment_console.sweep_telemetry import compute_sweep_telemetry


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
