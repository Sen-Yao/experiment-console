from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .config import BridgeConfig
from .models import EventContractError, OutboxEvent
from .state import AuthorityPinMismatch


class ConsoleUnavailable(RuntimeError):
    """Raised when the local tunnel cannot reach Experiment Console."""

    def __init__(self, message: str, *, transport_ok: bool = False) -> None:
        super().__init__(message)
        self.transport_ok = transport_ok


class ConsoleContractError(RuntimeError):
    """Raised when the Console bridge API violates its contract."""


class AuthorityState(Protocol):
    def pin_authority(self, *, authority_role: str, instance_id: str, ledger_id: str) -> None: ...

    def authority_pin(self) -> dict[str, str] | None: ...


class ConsoleClient:
    def __init__(self, config: BridgeConfig, *, authority_state: AuthorityState | None = None) -> None:
        self.config = config
        self.authority_state = authority_state
        self.last_health_error: str | None = None
        self.last_transport_ok = False
        self._validated_ledger_id: str | None = None

    def _validate_authority(self, payload: dict[str, Any], *, operation: str) -> None:
        authority_role = payload.get("authority_role")
        instance_id = payload.get("instance_id")
        ledger_id = payload.get("ledger_id")
        if authority_role != self.config.expected_authority_role:
            raise ConsoleContractError(
                f"Console {operation} authority_role {authority_role!r} does not match "
                f"{self.config.expected_authority_role!r}"
            )
        if instance_id != self.config.expected_instance_id:
            raise ConsoleContractError(
                f"Console {operation} instance_id {instance_id!r} does not match "
                f"{self.config.expected_instance_id!r}"
            )
        if not isinstance(ledger_id, str) or not ledger_id:
            raise ConsoleContractError(f"Console {operation} is missing a non-empty ledger_id")
        if self.authority_state is not None:
            try:
                self.authority_state.pin_authority(
                    authority_role=authority_role,
                    instance_id=instance_id,
                    ledger_id=ledger_id,
                )
            except AuthorityPinMismatch as exc:
                raise ConsoleContractError(str(exc)) from exc
        self._validated_ledger_id = ledger_id

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.config.console_token_file:
            path = Path(self.config.console_token_file).expanduser()
            try:
                token = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise ConsoleUnavailable(f"cannot read Console token file {path}: {exc}") from exc
            if not token:
                raise ConsoleUnavailable(f"Console token file is empty: {path}")
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _request_json(self, request: Request) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=self.config.http_timeout_seconds) as response:
                body = response.read()
        except HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", "replace")
            raise ConsoleUnavailable(f"Console HTTP {exc.code}: {detail}", transport_ok=True) from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise ConsoleUnavailable(f"Console request failed: {exc}") from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ConsoleContractError("Console returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise ConsoleContractError("Console response must be a JSON object")
        return decoded

    def health(self) -> bool:
        try:
            request = Request(f"{self.config.console_url.rstrip('/')}/health", headers=self._headers())
            payload = self._request_json(request)
            self.last_transport_ok = True
            if payload.get("status") not in {"ok", "healthy"}:
                raise ConsoleContractError(f"Console health status is {payload.get('status')!r}")
            self._validate_authority(payload, operation="health")
        except ConsoleUnavailable as exc:
            self.last_transport_ok = exc.transport_ok
            self.last_health_error = str(exc)
            return False
        except (ConsoleContractError, AuthorityPinMismatch) as exc:
            self.last_transport_ok = True
            self.last_health_error = str(exc)
            return False
        self.last_health_error = None
        return True

    def claim_events(self) -> list[OutboxEvent]:
        query = urlencode(
            {
                "consumer_id": self.config.consumer_id,
                "limit": self.config.poll_limit,
                "lease_seconds": self.config.lease_seconds,
            }
        )
        url = f"{self.config.console_url.rstrip('/')}/api/bridge/events?{query}"
        payload = self._request_json(Request(url, headers=self._headers()))
        self._validate_authority(payload, operation="claim")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ConsoleContractError("Console outbox response is missing an events array")
        events: list[OutboxEvent] = []
        seen: set[str] = set()
        for raw in raw_events:
            if not isinstance(raw, dict):
                raise ConsoleContractError("Console outbox events must be JSON objects")
            lease = raw.get("lease")
            if not isinstance(lease, dict) or lease.get("consumer_id") != self.config.consumer_id:
                raise ConsoleContractError("Console outbox event lease belongs to another consumer")
            if not isinstance(lease.get("expires_at"), str) or not lease["expires_at"]:
                raise ConsoleContractError("Console outbox event lease is missing expires_at")
            try:
                event = OutboxEvent.from_mapping(raw)
            except EventContractError as exc:
                raise ConsoleContractError(str(exc)) from exc
            if event.event_id in seen:
                continue
            seen.add(event.event_id)
            events.append(event)
        return events

    def ack_event(self, event: OutboxEvent) -> bool:
        if not self._validated_ledger_id:
            raise ConsoleContractError("cannot ack before validating the Console ledger")
        body = json.dumps(
            {
                "consumer_id": self.config.consumer_id,
                "expected_ledger_id": self._validated_ledger_id,
                "lease_token": event.lease_token,
            }
        ).encode("utf-8")
        url = f"{self.config.console_url.rstrip('/')}/api/bridge/events/{quote(event.event_id, safe='')}/ack"
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        request = Request(url, data=body, method="POST", headers=headers)
        payload = self._request_json(request)
        if payload.get("event_id") not in {None, event.event_id}:
            raise ConsoleContractError("Console acknowledged a different event id")
        return payload.get("acked") is True
