from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class EventContractError(ValueError):
    """Raised when the Console returns an invalid outbox event."""


@dataclass(frozen=True)
class OutboxEvent:
    event_id: str
    thread_id: str
    kind: str
    summary: str
    actionable: bool
    payload: Mapping[str, Any]
    lease_token: str
    created_at: str | None = None
    job_id: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OutboxEvent":
        event_id = raw.get("event_id")
        thread_id = raw.get("thread_id")
        kind = raw.get("kind", raw.get("event_type"))
        summary = raw.get("summary", "")
        payload = raw.get("payload", {})
        if not isinstance(event_id, str) or not event_id:
            raise EventContractError("outbox event is missing event_id")
        if not isinstance(thread_id, str) or not thread_id:
            raise EventContractError(f"outbox event {event_id} is missing thread_id")
        if not isinstance(kind, str) or not kind:
            raise EventContractError(f"outbox event {event_id} is missing kind")
        if not isinstance(summary, str):
            raise EventContractError(f"outbox event {event_id} summary must be a string")
        if not isinstance(payload, dict):
            raise EventContractError(f"outbox event {event_id} payload must be an object")
        actionable = raw.get("actionable")
        if not isinstance(actionable, bool):
            raise EventContractError(f"outbox event {event_id} actionable must be a boolean")
        created_at = raw.get("created_at")
        if created_at is not None and not isinstance(created_at, str):
            raise EventContractError(f"outbox event {event_id} created_at must be a string or null")
        job_id = raw.get("job_id", payload.get("job_id"))
        if job_id is not None and not isinstance(job_id, str):
            raise EventContractError(f"outbox event {event_id} job_id must be a string or null")
        lease = raw.get("lease")
        lease_token = lease.get("token") if isinstance(lease, dict) else None
        if not isinstance(lease_token, str) or not lease_token:
            raise EventContractError(f"outbox event {event_id} is missing an opaque lease token")
        return cls(
            event_id=event_id,
            thread_id=thread_id,
            kind=kind,
            summary=summary,
            actionable=actionable,
            payload=payload,
            lease_token=lease_token,
            created_at=created_at,
            job_id=job_id,
        )
