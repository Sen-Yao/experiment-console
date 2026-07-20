from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import BridgeConfig
from .models import OutboxEvent


class ConsoleError(RuntimeError):
    pass


class ConsoleClient:
    def __init__(self, config: BridgeConfig) -> None:
        self.config = config

    def _headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.config.console_token_file:
            path = Path(self.config.console_token_file).expanduser()
            try:
                token = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise ConsoleError(f"cannot read Console token file {path}: {exc}") from exc
            if not token:
                raise ConsoleError("Console token file is empty")
            headers["Authorization"] = f"Bearer {token}"
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _json(self, request: Request) -> dict:
        try:
            with urlopen(
                request, timeout=self.config.http_timeout_seconds
            ) as response:
                payload = json.load(response)
        except HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise ConsoleError(f"Console HTTP {exc.code}: {detail}") from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise ConsoleError(f"Console request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConsoleError("Console returned a non-object response")
        return payload

    def health(self) -> None:
        payload = self._json(
            Request(
                f"{self.config.console_url.rstrip('/')}/health",
                headers=self._headers(),
            )
        )
        if payload.get("status") != "ok" or str(payload.get("api_version")) != "3":
            raise ConsoleError("Console health contract is not v3")
        if payload.get("instance_id") != self.config.expected_instance_id:
            raise ConsoleError("Console instance id does not match bridge config")

    def claim(self) -> list[OutboxEvent]:
        body = json.dumps(
            {
                "consumer_id": self.config.consumer_id,
                "limit": self.config.poll_limit,
                "lease_seconds": self.config.lease_seconds,
            }
        ).encode()
        payload = self._json(
            Request(
                f"{self.config.console_url.rstrip('/')}/api/outbox/claim",
                data=body,
                method="POST",
                headers=self._headers(json_body=True),
            )
        )
        if (
            payload.get("instance_id") != self.config.expected_instance_id
            or str(payload.get("api_version")) != "3"
        ):
            raise ConsoleError("Console claim identity does not match bridge config")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ConsoleError("Console claim response has no events array")
        return [OutboxEvent.from_mapping(item) for item in raw_events]

    def ack(self, event: OutboxEvent) -> None:
        body = json.dumps(
            {
                "consumer_id": self.config.consumer_id,
                "lease_token": event.lease_token,
            }
        ).encode()
        payload = self._json(
            Request(
                f"{self.config.console_url.rstrip('/')}/api/outbox/"
                f"{quote(event.event_id, safe='')}/ack",
                data=body,
                method="POST",
                headers=self._headers(json_body=True),
            )
        )
        if payload.get("acked") is not True:
            raise ConsoleError(f"Console did not acknowledge {event.event_id}")
