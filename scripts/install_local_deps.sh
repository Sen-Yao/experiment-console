#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3}"
DEPS_DIR="${EXPERIMENT_CONSOLE_DEPS_DIR:-$ROOT/.local_deps}"

mkdir -p "$DEPS_DIR"
exec "$PYTHON_BIN" -m pip install --target "$DEPS_DIR" fastapi uvicorn pyyaml requests pydantic
