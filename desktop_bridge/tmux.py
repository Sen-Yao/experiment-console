from __future__ import annotations

import re
import shlex
import subprocess
from collections import defaultdict
from typing import Callable

from .config import BridgeConfig
from .models import PaneSnapshot, SessionSnapshot, WakeEvent


class TmuxError(RuntimeError):
    pass


Runner = Callable[..., subprocess.CompletedProcess[str]]

_PANE_ID = re.compile(r"%[0-9]+")
_THREAD_ID = re.compile(r"[A-Za-z0-9-]{8,128}")
_GENERATION = re.compile(r"[A-Za-z0-9._-]{1,128}")

_FORMAT_FIELDS = (
    "#{session_id}",
    "#{session_name}",
    "#{pane_id}",
    "#{pane_index}",
    "#{pane_dead}",
    "#{pane_dead_status}",
    "#{pane_current_command}",
    "#{@codex_watch}",
    "#{@codex_thread_id}",
    "#{@codex_generation}",
    "#{@codex_investigation_id}",
    "#{@codex_started_at}",
    "#{@codex_expected_seconds}",
    "#{@codex_attention_after}",
)


class TmuxClient:
    def __init__(
        self,
        config: BridgeConfig,
        *,
        runner: Runner = subprocess.run,
    ) -> None:
        self.config = config
        self.runner = runner
        self.invalid_sessions = 0

    def sessions(self) -> list[SessionSnapshot]:
        output = self._run_remote(
            ["tmux", "list-panes", "-a", "-F", "\t".join(_FORMAT_FIELDS)],
            no_server_is_empty=True,
        )
        grouped: dict[str, list[list[str]]] = defaultdict(list)
        for line in output.splitlines():
            fields = line.split("\t")
            if len(fields) != len(_FORMAT_FIELDS):
                continue
            grouped[fields[0]].append(fields)

        self.invalid_sessions = 0
        sessions: list[SessionSnapshot] = []
        for rows in grouped.values():
            try:
                session = self._parse_session(rows)
            except (TypeError, ValueError):
                if rows and rows[0][7] == "1":
                    self.invalid_sessions += 1
                continue
            if session is not None:
                sessions.append(session)
        return sorted(sessions, key=lambda item: (item.started_at, item.session_id))

    def capture_pane(self, pane_id: str) -> str:
        if not _PANE_ID.fullmatch(pane_id):
            raise TmuxError(f"invalid tmux pane id: {pane_id!r}")
        output = self._run_remote(
            [
                "tmux",
                "capture-pane",
                "-p",
                "-t",
                pane_id,
                "-S",
                f"-{self.config.capture_lines}",
            ]
        )
        return output[-self.config.capture_max_chars :]

    def _run_remote(
        self, argv: list[str], *, no_server_is_empty: bool = False
    ) -> str:
        remote_command = shlex.join(argv)
        try:
            completed = self.runner(
                self.config.ssh_command(remote_command),
                text=True,
                capture_output=True,
                timeout=self.config.command_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TmuxError(f"cannot inspect remote tmux: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            if no_server_is_empty and (
                "no server running" in detail or "error connecting to" in detail
            ):
                return ""
            raise TmuxError(
                f"remote tmux command failed with {completed.returncode}: {detail[-2000:]}"
            )
        return completed.stdout

    @staticmethod
    def _parse_session(rows: list[list[str]]) -> SessionSnapshot | None:
        first = rows[0]
        if first[7] != "1":
            return None
        common = [row[0:2] + row[7:14] for row in rows]
        if any(value != common[0] for value in common[1:]):
            raise ValueError("tmux session metadata drifted across panes")
        (
            session_id,
            session_name,
            _watch,
            thread_id,
            generation,
            investigation_id,
            started_at,
            expected_seconds,
            attention_after,
        ) = common[0]
        if not _THREAD_ID.fullmatch(thread_id):
            raise ValueError("invalid Codex thread id")
        if not _GENERATION.fullmatch(generation):
            raise ValueError("invalid watch generation")
        if not investigation_id or len(investigation_id) > 256:
            raise ValueError("invalid investigation id")
        started = int(started_at)
        expected = int(expected_seconds)
        attention = int(attention_after)
        if min(started, expected, attention) <= 0 or attention < started:
            raise ValueError("invalid watch timing contract")

        panes = []
        for row in rows:
            pane_id = row[2]
            if not _PANE_ID.fullmatch(pane_id):
                raise ValueError("invalid pane id")
            dead = row[4] == "1"
            panes.append(
                PaneSnapshot(
                    pane_id=pane_id,
                    pane_index=int(row[3]),
                    dead=dead,
                    exit_status=int(row[5]) if dead and row[5] else None,
                    current_command=row[6],
                    # tmux 3.0a does not expose pane_created.
                    created_at=started,
                )
            )
        if not panes:
            raise ValueError("watched session has no panes")
        return SessionSnapshot(
            session_id=session_id,
            session_name=session_name,
            thread_id=thread_id,
            generation=generation,
            investigation_id=investigation_id,
            started_at=started,
            expected_seconds=expected,
            attention_after=attention,
            panes=tuple(sorted(panes, key=lambda item: item.pane_index)),
        )


def classify_event(session: SessionSnapshot, observed_at: int) -> WakeEvent | None:
    live = [pane for pane in session.panes if not pane.dead]
    failed = [
        pane
        for pane in session.panes
        if pane.dead and pane.exit_status not in (None, 0)
    ]
    if not live:
        reason = "all_panes_exited" if not failed else "all_panes_terminal_with_failures"
        event_type = "terminal"
    elif failed:
        reason = "pane_failed"
        event_type = "attention"
    elif observed_at >= session.attention_after:
        reason = "attention_deadline_reached"
        event_type = "attention"
    else:
        return None
    return WakeEvent(
        event_id=f"tmux:{session.generation}:{event_type}",
        event_type=event_type,
        reason=reason,
        observed_at=observed_at,
        session=session,
    )
