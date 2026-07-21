#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$ROOT"

echo "==> Python"
"$PYTHON_BIN" --version
echo "==> Compile"
"$PYTHON_BIN" -m compileall -q backend desktop_bridge tests legacy/experiment-console-v3-runner/scripts
echo "==> Tests"
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" -m pytest -q legacy/experiment-console-v3-runner/tests -p no:cacheprovider
echo "==> Bridge surface"
"$PYTHON_BIN" -m desktop_bridge --config config/desktop-bridge.example.json dry-run >/dev/null
echo "==> wake bridge checks passed"
