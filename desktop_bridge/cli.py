from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path

from .config import BridgeConfig, ConfigError
from .service import BridgeService
from .state import AlreadyRunning, InstanceLock, StatusStore


DEFAULT_CONFIG = Path.home() / ".config" / "experiment-console" / "bridge.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="HCCS tmux-to-Codex Goal wake bridge"
    )
    result.add_argument(
        "--config",
        default=os.environ.get("EXPERIMENT_CONSOLE_BRIDGE_CONFIG", str(DEFAULT_CONFIG)),
    )
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("run", "once", "status", "health", "dry-run"):
        sub.add_parser(name)
    return result


def status(config: BridgeConfig) -> dict:
    payload = StatusStore(config.status_file).read()
    updated_at = payload.get("updated_at")
    fresh = (
        isinstance(updated_at, (int, float))
        and time.time() - updated_at <= config.status_stale_seconds
    )
    payload["fresh"] = fresh
    payload["healthy"] = bool(payload.get("healthy")) and fresh
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = BridgeConfig.from_file(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}")
        return 2
    if args.command in {"status", "health"}:
        payload = status(config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if args.command == "status" or payload["healthy"] else 1
    if args.command == "dry-run":
        print(
            json.dumps(
                {
                    "ssh_target": config.ssh_target,
                    "ssh_command": config.ssh_command("tmux list-panes -a"),
                    "codex_command": config.app_server_command(),
                    "event_file": str(config.event_file),
                },
                indent=2,
            )
        )
        return 0
    try:
        with InstanceLock(config.lock_file):
            service = BridgeService(config)
            try:
                if args.command == "once":
                    result = service.run_once()
                    print(json.dumps(asdict(result), indent=2, sort_keys=True))
                    return 0 if result.healthy else 1
                stopping = False

                def stop(_signum, _frame):
                    nonlocal stopping
                    stopping = True

                signal.signal(signal.SIGTERM, stop)
                signal.signal(signal.SIGINT, stop)
                service.run(lambda: stopping)
                return 0
            finally:
                service.close()
    except AlreadyRunning as exc:
        print(str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
