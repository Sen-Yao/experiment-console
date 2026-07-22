from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_console_packaging_remains_available_for_rollback():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yggdrasil.yaml").read_text(encoding="utf-8")
    local_start = (ROOT / "scripts" / "start_local_console.sh").read_text(
        encoding="utf-8"
    )
    assert "frontend" not in dockerfile.lower()
    assert "WANDB" not in compose
    assert "EXPERIMENT_CONSOLE_RESULTS" not in compose
    assert "runtime_console_server:create_app" in dockerfile
    assert "--factory" in dockerfile
    assert "experiment_console.api:create_app" in local_start
    assert "--factory" in local_start
    assert (ROOT / "config" / "server-profiles.json").is_file()


def test_ordinary_heartbeat_contract_stays_read_only_and_bounded():
    agent_notes = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    runner = (ROOT / "skill" / "experiment-runner" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    heartbeat = (
        ROOT
        / "skill"
        / "experiment-runner"
        / "references"
        / "dynamic-heartbeat.md"
    ).read_text(encoding="utf-8")

    for text in (agent_notes, runner, heartbeat):
        assert "read-only fast path" in text
        assert "research-investigation" in text
        assert "fresh" in text

    assert "one compact structured probe" in heartbeat
    assert "do not capture pane logs" in heartbeat
    assert "reread skills" in heartbeat
    assert "ceil_to_minute(fresh_now + next_delay)" in heartbeat
