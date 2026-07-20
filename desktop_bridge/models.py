from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class OutboxEvent:
    event_id: str
    job_id: str
    task_id: str
    event_type: str
    payload: dict[str, Any]
    lease_token: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OutboxEvent":
        required = ("event_id", "job_id", "task_id", "event_type", "lease_token")
        missing = [
            key for key in required if not isinstance(raw.get(key), str) or not raw[key]
        ]
        if missing or not isinstance(raw.get("payload"), dict):
            raise ValueError(
                f"invalid outbox event fields: {', '.join(missing) or 'payload'}"
            )
        return cls(
            event_id=str(raw["event_id"]),
            job_id=str(raw["job_id"]),
            task_id=str(raw["task_id"]),
            event_type=str(raw["event_type"]),
            payload=dict(raw["payload"]),
            lease_token=str(raw["lease_token"]),
        )

    def message(self) -> str:
        status = self.payload.get("status") or "unknown"
        exit_code = self.payload.get("exit_code")
        lines = [
            "Experiment Console job reached a terminal state.",
            f"job_id: {self.job_id}",
            f"status: {status}",
        ]
        if exit_code is not None:
            lines.append(f"exit_code: {exit_code}")
        lines.append(f"event_id: {self.event_id}")
        lines.append(f"Verify with: ./scripts/exp status {self.job_id}")
        return "\n".join(lines)
