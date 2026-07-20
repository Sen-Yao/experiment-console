#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${EXPERIMENT_CONSOLE_HOST:-127.0.0.1}"
PORT="${EXPERIMENT_CONSOLE_PORT:-5174}"

export PYTHONPATH="$ROOT/backend:$ROOT:${PYTHONPATH:-}"
export EXPERIMENT_CONSOLE_STATE_DIR="${EXPERIMENT_CONSOLE_STATE_DIR:-$ROOT/.state-v3}"
export EXPERIMENT_CONSOLE_SERVER_PROFILES="${EXPERIMENT_CONSOLE_SERVER_PROFILES:-$ROOT/config/server-profiles.json}"
export EXPERIMENT_CONSOLE_INSTANCE_ID="${EXPERIMENT_CONSOLE_INSTANCE_ID:-local-experiment-console-v3}"
export EXPERIMENT_CONSOLE_REQUIRE_API_TOKEN="${EXPERIMENT_CONSOLE_REQUIRE_API_TOKEN:-0}"

exec "$PYTHON_BIN" -m uvicorn experiment_console.api:create_app --factory --host "$HOST" --port "$PORT"
