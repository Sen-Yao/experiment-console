from __future__ import annotations

import threading
import time
from typing import Any

from .service import ConsoleService


class MonitorWorker:
    def __init__(self, service: ConsoleService) -> None:
        self.service = service
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_tick_at: float | None = None
        self._last_result: dict[str, int] | None = None

    def start(self) -> None:
        if not self.service.settings.monitor_enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="experiment-console-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.service.settings.monitor_poll_seconds + 1))
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._last_result = self.service.monitor_once()
            self._last_tick_at = time.time()
            self._stop.wait(self.service.settings.monitor_poll_seconds)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.service.settings.monitor_enabled,
            "running": bool(self._thread and self._thread.is_alive()),
            "last_tick_at": self._last_tick_at,
            "last_result": self._last_result,
            "poll_seconds": self.service.settings.monitor_poll_seconds,
        }
