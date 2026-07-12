from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .config import BridgeConfig, ConfigError
from .service import BridgeService
from .state import AlreadyRunning, DeliveryState, InstanceLock, StatusStore


DEFAULT_CONFIG = Path.home() / ".config" / "experiment-console" / "bridge.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment-console-bridge",
        description="Maintain the Yggdrasil tunnel and conditionally wake Codex Desktop.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("EXPERIMENT_CONSOLE_BRIDGE_CONFIG", str(DEFAULT_CONFIG)),
        help="Path to the bridge JSON config.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Run the long-lived bridge supervisor.")
    subparsers.add_parser("once", help="Run one tunnel/claim/delivery cycle, then stop.")
    subparsers.add_parser("status", help="Print the last persisted bridge status.")
    subparsers.add_parser("health", help="Exit successfully only for a fresh healthy status.")
    subparsers.add_parser("dry-run", help="Validate config and print non-secret process settings.")
    repin = subparsers.add_parser(
        "repin-authority",
        help="Explicitly replace the locally pinned production ledger identity.",
    )
    repin.add_argument("--ledger-id", required=True, help="New non-empty authoritative ledger id.")
    return parser


def _load_config(path: str) -> BridgeConfig:
    return BridgeConfig.from_file(path)


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _status(config: BridgeConfig) -> dict[str, object]:
    payload = StatusStore(config.status_file).read()
    updated_at = payload.get("updated_at")
    fresh = isinstance(updated_at, (int, float)) and time.time() - float(updated_at) <= config.status_stale_seconds
    payload["fresh"] = fresh
    payload["healthy"] = bool(payload.get("healthy")) and fresh
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = _load_config(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.command == "status":
        _print(_status(config))
        return 0
    if args.command == "health":
        status = _status(config)
        _print(status)
        return 0 if status["healthy"] else 1
    if args.command == "dry-run":
        _print(
            {
                "consumer_id": config.consumer_id,
                "console_url": config.console_url,
                "ssh_command": config.ssh_command(),
                "codex_command": config.app_server_command(),
                "state_file": str(config.state_file),
                "status_file": str(config.status_file),
                "lock_file": str(config.lock_file),
                "poll_interval_seconds": config.poll_interval_seconds,
                "lease_seconds": config.lease_seconds,
            }
        )
        return 0
    if args.command == "repin-authority":
        try:
            with InstanceLock(config.lock_file):
                state = DeliveryState(
                    config.state_file,
                    acked_retention_seconds=config.acked_event_retention_seconds,
                    max_acked_event_ids=config.max_acked_event_ids,
                )
                state.repin_authority(
                    authority_role=config.expected_authority_role,
                    instance_id=config.expected_instance_id,
                    ledger_id=args.ledger_id,
                )
                _print({"status": "ok", "authority": state.authority_pin()})
                return 0
        except (AlreadyRunning, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 3

    try:
        with InstanceLock(config.lock_file):
            service = BridgeService(config)
            if args.command == "once":
                try:
                    result = service.run_once()
                    _print(asdict(result))
                    return 0 if result.status == "ok" else 1
                finally:
                    service.close()

            stop = False

            def request_stop(signum, frame) -> None:
                nonlocal stop
                stop = True

            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)
            service.run(lambda: stop)
            return 0
    except AlreadyRunning as exc:
        print(str(exc), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
