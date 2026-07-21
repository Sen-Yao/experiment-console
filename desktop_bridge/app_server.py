from __future__ import annotations

import json
import select
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .config import BridgeConfig


class AppServerError(RuntimeError):
    pass


class ThreadBusy(AppServerError):
    pass


@dataclass(frozen=True)
class DeliveryOutcome:
    status: str
    turn_id: str | None = None
    reason: str | None = None


class LineTransport(Protocol):
    def send(self, message: Mapping[str, Any]) -> None: ...
    def receive(self, timeout: float) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


class SubprocessLineTransport:
    def __init__(self, command: list[str]) -> None:
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        if not self.process.stdin or not self.process.stdout:
            raise AppServerError("cannot open Codex app-server pipes")

    def send(self, message: Mapping[str, Any]) -> None:
        if self.process.poll() is not None:
            raise AppServerError("Codex app-server exited")
        assert self.process.stdin
        try:
            self.process.stdin.write(
                json.dumps(dict(message), separators=(",", ":")) + "\n"
            )
            self.process.stdin.flush()
        except OSError as exc:
            raise AppServerError(f"cannot write to Codex app-server: {exc}") from exc

    def receive(self, timeout: float) -> Mapping[str, Any]:
        if self.process.poll() is not None:
            raise AppServerError("Codex app-server exited")
        assert self.process.stdout
        readable, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not readable:
            raise AppServerError("timed out waiting for Codex app-server")
        line = self.process.stdout.readline()
        if not line:
            raise AppServerError("Codex app-server closed stdout")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AppServerError("Codex app-server returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise AppServerError("Codex app-server response must be an object")
        return payload

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)


TransportFactory = Callable[[list[str]], LineTransport]


class CodexAppServerSession:
    def __init__(
        self,
        config: BridgeConfig,
        *,
        transport_factory: TransportFactory = SubprocessLineTransport,
    ) -> None:
        self.config = config
        self.transport = transport_factory(config.app_server_command())
        self.request_id = 0
        try:
            self._request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "experiment_wake_bridge",
                        "title": "Experiment Wake Bridge",
                        "version": "1.0.0",
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
            self.transport.send({"method": "initialized", "params": {}})
        except Exception:
            self.transport.close()
            raise

    def __enter__(self) -> "CodexAppServerSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.transport.close()

    def deliver(
        self, task_id: str, text: str, *, event_id: str
    ) -> DeliveryOutcome:
        thread, archived = self._find_thread(task_id)
        if thread is None:
            return DeliveryOutcome("orphaned", reason="thread_not_found")
        if archived:
            return DeliveryOutcome("orphaned", reason="thread_archived")
        status_type = self._status_type(thread)
        if status_type == "active":
            return DeliveryOutcome("deferred", reason="thread_active")
        if status_type not in {"idle", "notLoaded"}:
            raise AppServerError(f"target Codex task status is {status_type!r}")

        goal_result = self._request("thread/goal/get", {"threadId": task_id})
        goal = goal_result.get("goal")
        if not isinstance(goal, dict):
            return DeliveryOutcome("orphaned", reason="goal_missing")
        goal_status = goal.get("status")
        if goal_status == "complete":
            return DeliveryOutcome("orphaned", reason="goal_complete")
        if goal_status in {
            "active",
            "paused",
            "usageLimited",
            "budgetLimited",
        }:
            return DeliveryOutcome("deferred", reason=f"goal_{goal_status}")
        if goal_status != "blocked":
            raise AppServerError(f"target Goal status is {goal_status!r}")

        result = self._request("thread/resume", {"threadId": task_id})
        self._require_idle(result)
        turn_result = self._request(
            "turn/start",
            {
                "threadId": task_id,
                "input": [{"type": "text", "text": text}],
                "clientUserMessageId": event_id,
            },
        )
        turn = turn_result.get("turn")
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        if not isinstance(turn_id, str) or not turn_id:
            raise AppServerError("turn/start did not return a turn id")
        return DeliveryOutcome("delivered", turn_id=turn_id)

    def _find_thread(
        self, task_id: str
    ) -> tuple[Mapping[str, Any] | None, bool]:
        for archived in (False, True):
            cursor: str | None = None
            for _ in range(100):
                params: dict[str, Any] = {
                    "archived": archived,
                    "limit": 100,
                    "sourceKinds": [],
                    "useStateDbOnly": True,
                }
                if cursor:
                    params["cursor"] = cursor
                result = self._request("thread/list", params)
                data = result.get("data")
                if not isinstance(data, list):
                    raise AppServerError("thread/list returned no data array")
                for item in data:
                    if isinstance(item, dict) and item.get("id") == task_id:
                        return item, archived
                next_cursor = result.get("nextCursor")
                if not isinstance(next_cursor, str) or not next_cursor:
                    break
                cursor = next_cursor
            else:
                raise AppServerError("thread/list archive scan exceeded page limit")
        return None, False

    @staticmethod
    def _status_type(thread: Mapping[str, Any]) -> str | None:
        status = thread.get("status")
        return status.get("type") if isinstance(status, dict) else None

    @staticmethod
    def _require_idle(result: Mapping[str, Any]) -> None:
        thread = result.get("thread")
        status_type = (
            CodexAppServerSession._status_type(thread)
            if isinstance(thread, dict)
            else None
        )
        if status_type == "active":
            raise ThreadBusy("target Codex task is active")
        if status_type not in {"idle", "notLoaded"}:
            raise AppServerError(f"target Codex task status is {status_type!r}")

    def _request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        self.request_id += 1
        request_id = self.request_id
        self.transport.send(
            {"id": request_id, "method": method, "params": dict(params)}
        )
        deadline = time.monotonic() + self.config.app_server_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AppServerError(f"timed out waiting for {method}")
            message = self.transport.receive(remaining)
            if message.get("id") != request_id:
                continue
            error = message.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or "app-server request failed")
                if "active" in detail.lower():
                    raise ThreadBusy(detail)
                raise AppServerError(f"{method} failed: {detail}")
            result = message.get("result")
            if not isinstance(result, dict):
                raise AppServerError(f"{method} returned no result object")
            return result
