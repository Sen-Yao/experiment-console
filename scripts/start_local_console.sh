#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3}"
HOST="${EXPERIMENT_CONSOLE_HOST:-127.0.0.1}"
PORT="${EXPERIMENT_CONSOLE_PORT:-5174}"
SECRETS_FILE="${EXPERIMENT_CONSOLE_SECRETS_FILE:-$HOME/.config/experiment-console/secrets.env}"
DEPS_DIR="${EXPERIMENT_CONSOLE_DEPS_DIR:-$ROOT/.local_deps}"

if [[ ! -r "$SECRETS_FILE" ]]; then
  echo "Missing secrets file: $SECRETS_FILE" >&2
  echo "Create it with WANDB_API_KEY=... and chmod 600." >&2
  exit 2
fi

set -a
# shellcheck source=/dev/null
source "$SECRETS_FILE"
set +a

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is empty in $SECRETS_FILE" >&2
  exit 2
fi

export PYTHONPATH="$DEPS_DIR:/private/tmp/experiment-console-deps:$ROOT/backend:$ROOT/scripts:${PYTHONPATH:-}"
exec "$PYTHON_BIN" -m uvicorn runtime_console_server:app --host "$HOST" --port "$PORT"
