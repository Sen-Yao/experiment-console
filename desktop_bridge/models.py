from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PaneSnapshot:
    pane_id: str
    pane_index: int
    dead: bool
    exit_status: int | None
    current_command: str
    created_at: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PaneSnapshot":
        return cls(
            pane_id=str(raw["pane_id"]),
            pane_index=int(raw["pane_index"]),
            dead=bool(raw["dead"]),
            exit_status=(
                int(raw["exit_status"])
                if raw.get("exit_status") is not None
                else None
            ),
            current_command=str(raw.get("current_command") or ""),
            created_at=int(raw["created_at"]),
        )


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    session_name: str
    thread_id: str
    generation: str
    investigation_id: str
    started_at: int
    expected_seconds: int
    attention_after: int
    panes: tuple[PaneSnapshot, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SessionSnapshot":
        panes = raw.get("panes")
        if not isinstance(panes, list):
            raise ValueError("session panes must be an array")
        return cls(
            session_id=str(raw["session_id"]),
            session_name=str(raw["session_name"]),
            thread_id=str(raw["thread_id"]),
            generation=str(raw["generation"]),
            investigation_id=str(raw["investigation_id"]),
            started_at=int(raw["started_at"]),
            expected_seconds=int(raw["expected_seconds"]),
            attention_after=int(raw["attention_after"]),
            panes=tuple(PaneSnapshot.from_mapping(item) for item in panes),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WakeEvent:
    event_id: str
    event_type: str
    reason: str
    observed_at: int
    session: SessionSnapshot

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WakeEvent":
        session = raw.get("session")
        if not isinstance(session, dict):
            raise ValueError("event session must be an object")
        return cls(
            event_id=str(raw["event_id"]),
            event_type=str(raw["event_type"]),
            reason=str(raw["reason"]),
            observed_at=int(raw["observed_at"]),
            session=SessionSnapshot.from_mapping(session),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "reason": self.reason,
            "observed_at": self.observed_at,
            "session": self.session.to_mapping(),
        }
