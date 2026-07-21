from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .models import WakeEvent


class AlreadyRunning(RuntimeError):
    pass


class EventStoreFull(RuntimeError):
    pass


class InstanceLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise AlreadyRunning("Experiment Console bridge is already running") from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class StatusStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        value = {**payload, "updated_at": time.time()}
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        temporary_path = Path(temporary)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, sort_keys=True)
                handle.write("\n")
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}


class EventStore:
    def __init__(self, path: Path, *, max_records: int = 256) -> None:
        self.path = path
        self.max_records = max_records

    def reconcile(self, active_event_ids: set[str]) -> int:
        """Drop closed records after their source tmux generation disappears."""
        payload = self._read()
        events = payload.get("events", {})
        removed = 0
        for event_id, record in list(events.items()):
            if (
                isinstance(record, dict)
                and record.get("status") != "pending"
                and event_id not in active_event_ids
            ):
                del events[event_id]
                removed += 1
        if removed:
            self._write(payload)
        return removed

    def add(self, event: WakeEvent) -> bool:
        payload = self._read()
        events = payload.setdefault("events", {})
        existing = events.get(event.event_id)
        if isinstance(existing, dict):
            if existing.get("status") == "pending":
                existing["event"] = event.to_mapping()
                self._write(payload)
            return False
        if len(events) >= self.max_records:
            raise EventStoreFull(
                f"event outbox reached its {self.max_records}-record limit"
            )
        events[event.event_id] = {
            "status": "pending",
            "event": event.to_mapping(),
            "created_at": time.time(),
            "last_error": None,
        }
        self._write(payload)
        return True

    def pending(self, limit: int) -> list[WakeEvent]:
        records = self._read().get("events", {})
        pending = []
        if not isinstance(records, dict):
            return pending
        ordered = sorted(
            records.values(),
            key=lambda item: float(item.get("created_at") or 0)
            if isinstance(item, dict)
            else 0,
        )
        for record in ordered:
            if not isinstance(record, dict) or record.get("status") != "pending":
                continue
            raw = record.get("event")
            if not isinstance(raw, dict):
                continue
            try:
                pending.append(WakeEvent.from_mapping(raw))
            except (KeyError, TypeError, ValueError):
                continue
            if len(pending) >= limit:
                break
        return pending

    def mark_delivered(self, event_id: str, turn_id: str) -> None:
        self._update(
            event_id,
            status="delivered",
            delivered_at=time.time(),
            turn_id=turn_id,
            last_error=None,
        )

    def mark_orphaned(self, event_id: str, reason: str) -> None:
        self._update(
            event_id,
            status="orphaned",
            orphaned_at=time.time(),
            last_error=reason,
        )

    def mark_error(self, event_id: str, error: str) -> None:
        self._update(event_id, last_error=error, last_attempt_at=time.time())

    def _update(self, event_id: str, **values: Any) -> None:
        payload = self._read()
        record = payload.get("events", {}).get(event_id)
        if not isinstance(record, dict):
            return
        record.update(values)
        self._write(payload)

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "events": {}}
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return {"version": 1, "events": {}}
        if not isinstance(payload.get("events"), dict):
            payload["events"] = {}
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        temporary_path = Path(temporary)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)
