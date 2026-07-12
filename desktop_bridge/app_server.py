from __future__ import annotations

import json
import os
import select
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .config import BridgeConfig


class AppServerError(RuntimeError):
    """Base class for local Codex app-server failures."""


class AppServerOffline(AppServerError):
    """The running Codex app-server control socket is unavailable."""


class ThreadBusy(AppServerError):
    """The target thread already has active work."""


class ActiveGoalConflict(AppServerError):
    """The target thread still has an active Goal continuation loop."""


@dataclass(frozen=True)
class DeliveryInspection:
    client_message_count: int
    thread_status: str
    ambiguous: bool = False

    @property
    def found(self) -> bool:
        return self.client_message_count == 1 and not self.ambiguous


class LineTransport(Protocol):
    def send(self, message: Mapping[str, Any]) -> None: ...

    def receive(self, timeout: float) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class SubprocessLineTransport:
    """JSONL transport through `codex app-server proxy`.

    The proxy only connects to an already running app-server control socket. It
    does not bootstrap the daemon or start Codex Desktop.
    """

    def __init__(self, command: list[str]) -> None:
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                close_fds=True,
            )
        except OSError as exc:
            raise AppServerOffline(f"cannot start Codex app-server proxy: {exc}") from exc
        if self.process.stdin is None or self.process.stdout is None:
            self.close()
            raise AppServerOffline("Codex app-server proxy did not expose stdio")
        self._read_buffer = bytearray()

    def send(self, message: Mapping[str, Any]) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise AppServerOffline("Codex app-server proxy disconnected") from exc

    def receive(self, timeout: float) -> Mapping[str, Any]:
        deadline = time.monotonic() + timeout
        assert self.process.stdout is not None
        descriptor = self.process.stdout.fileno()
        while True:
            newline = self._read_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._read_buffer[:newline])
                del self._read_buffer[: newline + 1]
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AppServerError("Codex app-server returned invalid JSONL") from exc
                if not isinstance(message, dict):
                    raise AppServerError("Codex app-server message must be an object")
                return message
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AppServerOffline("timed out waiting for Codex app-server")
            readable, _, _ = select.select([descriptor], [], [], remaining)
            if not readable:
                raise AppServerOffline("timed out waiting for Codex app-server")
            chunk = os.read(descriptor, 65536)
            if not chunk:
                code = self.process.poll()
                raise AppServerOffline(f"Codex app-server proxy closed (exit={code})")
            self._read_buffer.extend(chunk)

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)


TransportFactory = Callable[[list[str]], LineTransport]


class CodexAppServerSession:
    def __init__(
        self,
        config: BridgeConfig,
        *,
        transport_factory: TransportFactory = SubprocessLineTransport,
    ) -> None:
        self.config = config
        self.transport_factory = transport_factory
        self.transport: LineTransport | None = None
        self._request_id = 0

    def __enter__(self) -> "CodexAppServerSession":
        self.transport = self.transport_factory(self.config.app_server_command())
        try:
            self._initialize()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()
            self.transport = None

    def _initialize(self) -> None:
        self._request(
            "initialize",
            {
                "clientInfo": {
                    "name": "experiment_console_bridge",
                    "title": "Experiment Console Desktop Bridge",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "optOutNotificationMethods": [
                        "item/agentMessage/delta",
                        "item/reasoning/summaryTextDelta",
                        "thread/tokenUsage/updated",
                    ]
                },
            },
        )
        self._send({"method": "initialized", "params": {}})

    def deliver(self, thread_id: str, text: str, *, client_message_id: str) -> str:
        resume_result = self._request("thread/resume", {"threadId": thread_id})
        self._require_available_status(resume_result, operation="thread/resume")
        goal_result = self._request("thread/goal/get", {"threadId": thread_id})
        goal = goal_result.get("goal")
        goal_status = goal.get("status") if isinstance(goal, dict) else None
        if goal_status == "active":
            raise ActiveGoalConflict(
                "target Codex thread has an active Goal; pause or complete it before Console wake delivery"
            )
        read_result = self._request("thread/read", {"threadId": thread_id, "includeTurns": False})
        self._require_available_status(read_result, operation="final thread/read")
        turn_result = self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": text}],
                "clientUserMessageId": client_message_id,
            },
        )
        turn = turn_result.get("turn") if isinstance(turn_result, dict) else None
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        if not isinstance(turn_id, str) or not turn_id:
            raise AppServerError("turn/start did not return an accepted turn id")
        return turn_id

    def inspect_delivery(self, thread_id: str, *, client_message_id: str) -> DeliveryInspection:
        result = self._request("thread/read", {"threadId": thread_id, "includeTurns": True})
        thread = result.get("thread")
        if not isinstance(thread, dict):
            return DeliveryInspection(0, "unknown", ambiguous=True)
        status = thread.get("status")
        status_type = status.get("type") if isinstance(status, dict) else None
        turns = thread.get("turns")
        if status_type not in {"active", "idle", "notLoaded", "systemError"} or not isinstance(turns, list):
            return DeliveryInspection(0, str(status_type or "unknown"), ambiguous=True)
        matches = 0
        malformed = False
        for turn in turns:
            items = turn.get("items") if isinstance(turn, dict) else None
            if not isinstance(items, list):
                malformed = True
                continue
            for item in items:
                if not isinstance(item, dict):
                    malformed = True
                    continue
                if item.get("type") == "userMessage" and item.get("clientId") == client_message_id:
                    matches += 1
        return DeliveryInspection(
            client_message_count=matches,
            thread_status=status_type,
            ambiguous=malformed or matches > 1,
        )

    def _require_available_status(self, result: Mapping[str, Any], *, operation: str) -> None:
        thread = result.get("thread")
        status = thread.get("status") if isinstance(thread, dict) else None
        status_type = status.get("type") if isinstance(status, dict) else None
        if status_type == "active":
            raise ThreadBusy(f"target Codex thread is active during {operation}")
        if status_type == "systemError":
            raise AppServerError(f"target Codex thread is in systemError during {operation}")
        if status_type not in {"idle", "notLoaded"}:
            raise AppServerError(f"target Codex thread returned unknown status {status_type!r}")

    def _request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        self._request_id += 1
        request_id = self._request_id
        self._send({"method": method, "id": request_id, "params": dict(params)})
        deadline = time.monotonic() + self.config.app_server_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AppServerOffline(f"timed out waiting for {method}")
            message = self._receive(remaining)
            if message.get("id") != request_id:
                if "id" in message and "method" in message:
                    self._send(
                        {
                            "id": message["id"],
                            "error": {"code": -32601, "message": "bridge does not handle server requests"},
                        }
                    )
                continue
            error = message.get("error")
            if isinstance(error, dict):
                message_text = str(error.get("message", "app-server request failed"))
                lowered = message_text.lower()
                if "active turn" in lowered or "already active" in lowered or "already running" in lowered:
                    raise ThreadBusy(message_text)
                raise AppServerError(f"{method} failed: {message_text}")
            result = message.get("result")
            if not isinstance(result, dict):
                raise AppServerError(f"{method} returned no result object")
            return result

    def _send(self, message: Mapping[str, Any]) -> None:
        if self.transport is None:
            raise AppServerOffline("Codex app-server session is not connected")
        self.transport.send(message)

    def _receive(self, timeout: float) -> Mapping[str, Any]:
        if self.transport is None:
            raise AppServerOffline("Codex app-server session is not connected")
        return self.transport.receive(timeout)
