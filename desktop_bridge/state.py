from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import OutboxEvent


class AlreadyRunning(RuntimeError):
    """Another bridge process holds the singleton lock."""


class AuthorityPinMismatch(RuntimeError):
    """The Console identity differs from the locally pinned authority."""


class InstanceLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self) -> "InstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise AlreadyRunning(f"another bridge process holds {self.path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        self.descriptor = descriptor
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.descriptor is None:
            return
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)
        self.descriptor = None


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


class DeliveryState:
    """Persistent event-id ledger used to avoid duplicate Codex turns."""

    VERSION = 1

    def __init__(
        self,
        path: Path,
        *,
        acked_retention_seconds: int,
        max_acked_event_ids: int,
        wall_clock=time.time,
    ) -> None:
        self.path = path
        self.acked_retention_seconds = acked_retention_seconds
        self.max_acked_event_ids = max_acked_event_ids
        self.wall_clock = wall_clock
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": self.VERSION, "events": {}}
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read bridge delivery state {self.path}: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("version") != self.VERSION or not isinstance(raw.get("events"), dict):
            raise RuntimeError(f"unsupported bridge delivery state format: {self.path}")
        return raw

    def status(self, event_id: str) -> str | None:
        record = self._data["events"].get(event_id)
        return record.get("status") if isinstance(record, dict) else None

    def record(self, event_id: str) -> dict[str, Any] | None:
        record = self._data["events"].get(event_id)
        return dict(record) if isinstance(record, dict) else None

    def pin_authority(self, *, authority_role: str, instance_id: str, ledger_id: str) -> None:
        observed = {
            "authority_role": authority_role,
            "instance_id": instance_id,
            "ledger_id": ledger_id,
        }
        pinned = self._data.get("authority")
        if pinned is None:
            self._data["authority"] = observed
            self._save()
            return
        if pinned != observed:
            raise AuthorityPinMismatch(
                "Console authority pin mismatch: "
                f"pinned={json.dumps(pinned, sort_keys=True)} "
                f"observed={json.dumps(observed, sort_keys=True)}"
            )

    def repin_authority(self, *, authority_role: str, instance_id: str, ledger_id: str) -> None:
        if not ledger_id:
            raise ValueError("ledger_id cannot be empty")
        self._data["authority"] = {
            "authority_role": authority_role,
            "instance_id": instance_id,
            "ledger_id": ledger_id,
        }
        self._save()

    def authority_pin(self) -> dict[str, str] | None:
        pinned = self._data.get("authority")
        return dict(pinned) if isinstance(pinned, dict) else None

    def summary(self) -> dict[str, Any]:
        records = [record for record in self._data["events"].values() if isinstance(record, dict)]
        return {
            "authority": self.authority_pin(),
            "delivered_unacked": sum(record.get("status") == "delivered" for record in records),
            "inflight": sum(record.get("status") == "inflight" for record in records),
            "uncertain": sum(record.get("status") == "uncertain" for record in records),
            "acked_ids_retained": sum(record.get("status") == "acked" for record in records),
        }

    def mark_inflight(self, events: Iterable[OutboxEvent], *, client_message_id: str) -> None:
        events = list(events)
        group_event_ids = sorted(event.event_id for event in events)
        attempted_at = self.wall_clock()
        for event in events:
            previous = self._data["events"].get(event.event_id)
            previous = previous if isinstance(previous, dict) else {}
            same_attempt = previous.get("client_message_id") == client_message_id
            self._data["events"][event.event_id] = {
                **previous,
                "status": "inflight",
                "thread_id": event.thread_id,
                "client_message_id": client_message_id,
                "group_event_ids": group_event_ids,
                "first_attempt_at": previous.get("first_attempt_at", attempted_at) if same_attempt else attempted_at,
                "last_attempt_at": attempted_at,
                "attempts": int(previous.get("attempts", 0)) + 1 if same_attempt else 1,
                "ambiguous": False,
            }
        self._save()

    def mark_uncertain(
        self,
        events: Iterable[OutboxEvent],
        *,
        client_message_id: str,
        reason: str,
        ambiguous: bool = False,
    ) -> None:
        events = list(events)
        uncertain_at = self.wall_clock()
        for event in events:
            previous = self._data["events"].get(event.event_id)
            previous = previous if isinstance(previous, dict) else {}
            self._data["events"][event.event_id] = {
                **previous,
                "status": "uncertain",
                "thread_id": event.thread_id,
                "client_message_id": client_message_id,
                "first_attempt_at": previous.get("first_attempt_at", uncertain_at),
                "uncertain_at": uncertain_at,
                "uncertain_reason": reason,
                "ambiguous": bool(ambiguous),
            }
        self._save()

    def clear_attempt(self, events: Iterable[OutboxEvent]) -> None:
        changed = False
        for event in events:
            if self.status(event.event_id) in {"inflight", "uncertain"}:
                self._data["events"].pop(event.event_id, None)
                changed = True
        if changed:
            self._save()

    def attempt_age_seconds(self, event_id: str) -> float | None:
        record = self.record(event_id)
        if not record or not isinstance(record.get("first_attempt_at"), (int, float)):
            return None
        return max(0.0, self.wall_clock() - float(record["first_attempt_at"]))

    def mark_delivered(self, events: Iterable[OutboxEvent], *, client_message_id: str) -> None:
        delivered_at = self.wall_clock()
        for event in events:
            previous = self._data["events"].get(event.event_id)
            previous = previous if isinstance(previous, dict) else {}
            self._data["events"][event.event_id] = {
                **previous,
                "status": "delivered",
                "thread_id": event.thread_id,
                "client_message_id": client_message_id,
                "delivered_at": delivered_at,
            }
        self._prune()
        self._save()

    def mark_acked(self, event_id: str) -> None:
        now = self.wall_clock()
        record = self._data["events"].setdefault(event_id, {})
        record["status"] = "acked"
        record["acked_at"] = now
        self._prune()
        self._save()

    def _prune(self) -> None:
        now = self.wall_clock()
        events = self._data["events"]
        acked: list[tuple[str, float]] = []
        for event_id, record in list(events.items()):
            if not isinstance(record, dict):
                del events[event_id]
                continue
            if record.get("status") == "acked":
                acked_at = float(record.get("acked_at", 0.0))
                if acked_at and now - acked_at > self.acked_retention_seconds:
                    del events[event_id]
                else:
                    acked.append((event_id, acked_at))
        if len(acked) > self.max_acked_event_ids:
            for event_id, _ in sorted(acked, key=lambda item: item[1])[: len(acked) - self.max_acked_event_ids]:
                events.pop(event_id, None)

    def _save(self) -> None:
        atomic_write_json(self.path, self._data)


class StatusStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, status: Mapping[str, Any]) -> None:
        atomic_write_json(self.path, status)

    def read(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"status": "not_started", "healthy": False}
        except (OSError, json.JSONDecodeError) as exc:
            return {"status": "invalid", "healthy": False, "error": str(exc)}
        return raw if isinstance(raw, dict) else {"status": "invalid", "healthy": False}
