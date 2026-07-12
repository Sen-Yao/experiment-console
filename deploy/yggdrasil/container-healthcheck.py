#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any


def worker_ready(payload: dict[str, Any], *, require_lease: bool) -> bool:
    worker = payload.get("monitor_worker") or payload.get("worker")
    if isinstance(worker, bool):
        return worker
    if not isinstance(worker, dict):
        return False
    ready = bool(worker.get("enabled")) and bool(worker.get("ready")) and bool(worker.get("running"))
    if require_lease:
        ready = ready and bool(worker.get("lease_held"))
    return ready


def main() -> int:
    url = os.environ.get("EXPERIMENT_CONSOLE_HEALTH_URL", "http://127.0.0.1:5174/health")
    require_worker = os.environ.get("EXPERIMENT_CONSOLE_HEALTH_REQUIRE_WORKER", "1") not in {"0", "false", "False"}
    require_lease = os.environ.get("EXPERIMENT_CONSOLE_HEALTH_REQUIRE_LEASE", "1") not in {"0", "false", "False"}
    expected_role = os.environ.get("EXPERIMENT_CONSOLE_HEALTH_EXPECTED_AUTHORITY_ROLE", "authoritative")
    expected_instance = os.environ.get("EXPERIMENT_CONSOLE_HEALTH_EXPECTED_INSTANCE_ID")
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.load(response)
    except Exception as exc:
        print(f"health request failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    if payload.get("status") != "ok":
        print("API health status is not ok", file=sys.stderr)
        return 1
    if payload.get("authority_role") != expected_role:
        print("API authority role does not match production", file=sys.stderr)
        return 1
    if not expected_instance or payload.get("instance_id") != expected_instance:
        print("API instance id does not match deployment", file=sys.stderr)
        return 1
    if not payload.get("ledger_id"):
        print("API ledger id is empty", file=sys.stderr)
        return 1
    if not payload.get("console_api_auth_configured"):
        print("Console API bearer auth is not configured", file=sys.stderr)
        return 1
    if require_worker and not worker_ready(payload, require_lease=require_lease):
        print("monitor worker is not ready", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
