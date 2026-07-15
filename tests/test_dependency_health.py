from experiment_console.dependency_health import advance_dependency_episode, dependency_retry_due
from experiment_console.wandb_client import WandBAuthRequired, WandBUnavailable


def test_external_episode_requires_time_and_attempt_threshold():
    first = advance_dependency_episode(
        None,
        source="wandb",
        scope="api.wandb.ai|credential:test",
        operation_stage="get_sweep_state",
        classification="wandb_unavailable_reconciling",
        exc=WandBUnavailable("TLS EOF"),
        retry_active=True,
        now="2026-07-15T00:00:00+00:00",
    )
    second = advance_dependency_episode(
        first,
        source="wandb",
        scope="api.wandb.ai|credential:test",
        operation_stage="get_sweep_state",
        classification="wandb_unavailable_reconciling",
        exc=WandBUnavailable("connection reset"),
        retry_active=True,
        now="2026-07-15T00:10:00+00:00",
    )
    third = advance_dependency_episode(
        second,
        source="wandb",
        scope="api.wandb.ai|credential:test",
        operation_stage="get_sweep_state",
        classification="wandb_unavailable_reconciling",
        exc=WandBUnavailable("timeout"),
        retry_active=True,
        now="2026-07-15T00:15:00+00:00",
    )

    assert first["episode_id"] == second["episode_id"] == third["episode_id"]
    assert second["action_required"] is False
    assert third["action_required"] is True
    assert third["attempts"] == 3
    assert len(third["error_fingerprints"]) == 3
    assert third["next_retry_at"] == "2026-07-15T00:25:00+00:00"
    assert dependency_retry_due(third, now="2026-07-15T00:24:59+00:00") is False
    assert dependency_retry_due(third, now="2026-07-15T00:25:00+00:00") is True


def test_deterministic_auth_failure_is_immediately_actionable_and_paused():
    episode = advance_dependency_episode(
        None,
        source="wandb",
        scope="api.wandb.ai|credential:missing",
        operation_stage="remote_auth_check",
        classification="auth_required",
        exc=WandBAuthRequired("WANDB_API_KEY is not set"),
        retry_active=False,
        action_required_immediately=True,
        now="2026-07-15T00:00:00+00:00",
    )

    assert episode["action_required"] is True
    assert episode["auto_retry_active"] is False
    assert episode["lifecycle"] == "attention"
    assert episode["next_retry_at"] is None
    assert dependency_retry_due(episode, now="2026-07-16T00:00:00+00:00") is False
