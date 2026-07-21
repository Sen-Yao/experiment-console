from __future__ import annotations

import subprocess

from desktop_bridge.config import BridgeConfig
from desktop_bridge.models import PaneSnapshot, SessionSnapshot
from desktop_bridge.tmux import TmuxClient, classify_event


def session(panes, *, attention_after=200):
    return SessionSnapshot(
        session_id="$1",
        session_name="validation",
        thread_id="019f8223-7124-7722-aea2-05ee78f79ef4",
        generation="generation-1",
        investigation_id="2026-07-21-control-plane-validation",
        started_at=100,
        expected_seconds=60,
        attention_after=attention_after,
        panes=tuple(panes),
    )


def pane(index, *, dead=False, exit_status=None):
    return PaneSnapshot(
        pane_id=f"%{index + 1}",
        pane_index=index,
        dead=dead,
        exit_status=exit_status,
        current_command="python",
        created_at=100,
    )


def test_parallel_failures_coalesce_into_one_attention_event():
    value = session(
        [
            pane(0, dead=True, exit_status=1),
            pane(1, dead=True, exit_status=2),
            pane(2),
        ]
    )
    event = classify_event(value, 150)
    assert event is not None
    assert event.event_id == "tmux:generation-1:attention"
    assert event.reason == "pane_failed"


def test_attention_deadline_only_wakes_and_keeps_live_pane():
    value = session([pane(0)], attention_after=120)
    event = classify_event(value, 121)
    assert event is not None
    assert event.event_type == "attention"
    assert value.panes[0].dead is False


def test_healthy_partial_completion_does_not_wake():
    value = session([pane(0, dead=True, exit_status=0), pane(1)])
    assert classify_event(value, 150) is None


def test_all_dead_produces_terminal_event_after_attention_generation():
    value = session(
        [pane(0, dead=True, exit_status=1), pane(1, dead=True, exit_status=0)]
    )
    event = classify_event(value, 150)
    assert event is not None
    assert event.event_id == "tmux:generation-1:terminal"
    assert event.reason == "all_panes_terminal_with_failures"


def test_tmux_parser_reads_session_options_and_panes(tmp_path):
    line1 = "\t".join(
        [
            "$1",
            "validation",
            "%1",
            "0",
            "0",
            "",
            "python",
            "1",
            "019f8223-7124-7722-aea2-05ee78f79ef4",
            "generation-1",
            "2026-07-21-control-plane-validation",
            "100",
            "60",
            "180",
        ]
    )
    line2 = line1.replace("%1\t0\t0", "%2\t1\t0")

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout=f"{line1}\n{line2}\n", stderr="")

    config = BridgeConfig(
        ssh_target="HCCS-25",
        status_file=tmp_path / "status",
        event_file=tmp_path / "events",
        lock_file=tmp_path / "lock",
    )
    sessions = TmuxClient(config, runner=runner).sessions()
    assert len(sessions) == 1
    assert [item.pane_id for item in sessions[0].panes] == ["%1", "%2"]
    assert sessions[0].attention_after == 180


def test_tmux_no_server_is_empty(tmp_path):
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 1, stdout="", stderr="error connecting to /tmp/tmux/default"
        )

    config = BridgeConfig(
        ssh_target="HCCS-25",
        status_file=tmp_path / "status",
        event_file=tmp_path / "events",
        lock_file=tmp_path / "lock",
    )
    assert TmuxClient(config, runner=runner).sessions() == []
