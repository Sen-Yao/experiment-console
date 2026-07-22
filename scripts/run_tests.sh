#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$ROOT"

echo "==> Python"
"$PYTHON_BIN" --version
echo "==> Compile"
"$PYTHON_BIN" -m compileall -q backend tests legacy/experiment-console-v3-runner/scripts
echo "==> Tests"
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" -m pytest -q legacy/experiment-console-v3-runner/tests -p no:cacheprovider
echo "==> legacy rollback checks passed"
