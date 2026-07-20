#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.request


def main() -> int:
    try:
        with urllib.request.urlopen(
            os.environ.get("EXPERIMENT_CONSOLE_HEALTH_URL", "http://127.0.0.1:5174/health"),
            timeout=3,
        ) as response:
            payload = json.load(response)
    except Exception as exc:
        print(f"health request failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    monitor = payload.get("monitor") or {}
    if payload.get("status") != "ok" or payload.get("api_version") != "3":
        return 1
    if payload.get("instance_id") != os.environ.get("EXPERIMENT_CONSOLE_INSTANCE_ID"):
        return 1
    if os.environ.get("EXPERIMENT_CONSOLE_REQUIRE_API_TOKEN") == "1" and not payload.get("api_auth_configured"):
        return 1
    if os.environ.get("EXPERIMENT_CONSOLE_MONITOR_ENABLED", "1") == "1" and not monitor.get("running"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
