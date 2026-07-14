from datetime import datetime, timedelta, timezone

from experiment_console.agent_reconcile import (
    agent_reconcile_due,
    desired_agent_count,
    initial_agent_reconciler,
    next_reconcile_at,
    retry_delay_seconds,
    sweep_capacity,
)


def test_agent_reconciler_initializes_fixed_capacity_contract():
    state = initial_agent_reconciler(max_agents=4, expected_runs=5, now="2026-07-14T00:00:00+00:00")

    assert state["max_agents"] == 4
    assert state["desired_agents"] == 4
    assert state["remaining_runs"] == 5
    assert agent_reconcile_due({"agent_reconciler": state}, now="2026-07-14T00:00:00+00:00") is True


def test_agent_retry_backoff_is_bounded_to_agreed_sequence():
    assert [retry_delay_seconds(index) for index in range(1, 7)] == [30, 120, 300, 600, 600, 600]


def test_agent_target_is_bounded_by_remaining_runs_without_forcing_scale_down():
    assert desired_agent_count(max_agents=5, remaining_runs=8) == 5
    assert desired_agent_count(max_agents=5, remaining_runs=2) == 2
    assert desired_agent_count(max_agents=5, remaining_runs=0) == 0


def test_sweep_capacity_uses_run_edges_and_stops_at_terminal_state():
    running = sweep_capacity({
        "state": "RUNNING",
        "expectedRunCount": 5,
        "raw_run_state_counts": {"finished": 2, "running": 2, "failed": 0},
    }, fallback_expected=5)
    finished = sweep_capacity({
        "state": "FINISHED",
        "expectedRunCount": 5,
        "raw_run_state_counts": {"finished": 5, "running": 0, "failed": 0},
    }, fallback_expected=5)

    assert running["remaining"] == 3
    assert running["terminal"] is False
    assert finished["remaining"] == 0
    assert finished["terminal"] is True


def test_sweep_capacity_keeps_launch_contract_when_wandb_expected_count_is_temporarily_missing():
    capacity = sweep_capacity({
        "state": "RUNNING",
        "runCount": 1,
        "expectedRunCount": 0,
        "raw_run_state_counts": {"finished": 0, "failed": 0, "running": 1},
    }, fallback_expected=5)

    assert capacity["expected"] == 5
    assert capacity["remaining"] == 5


def test_next_reconcile_timestamp_uses_exact_delay():
    start = "2026-07-14T00:00:00+00:00"
    observed = datetime.fromisoformat(next_reconcile_at(now=start, delay_seconds=120))

    assert observed == datetime.fromisoformat(start) + timedelta(seconds=120)
    assert observed.tzinfo == timezone.utc
