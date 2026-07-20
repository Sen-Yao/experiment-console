#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$ROOT"

echo "==> Python"
"$PYTHON_BIN" --version
echo "==> Compile"
"$PYTHON_BIN" -m compileall -q backend desktop_bridge tests skill/experiment-runner/scripts
echo "==> Tests"
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" -m pytest -q skill/experiment-runner/tests -p no:cacheprovider
echo "==> Runner surface"
"$ROOT/scripts/exp" --help >/dev/null
echo "==> v3 checks passed"
