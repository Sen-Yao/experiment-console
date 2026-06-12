#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-"$ROOT_DIR/.conda"}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

cd "$ROOT_DIR"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$CONDA_DIR/bin/python" ]]; then
    PYTHON="$CONDA_DIR/bin/python"
  elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON="$(command -v python3 || true)"
  fi
fi

if [[ -z "${NPM:-}" ]]; then
  if [[ -x "$CONDA_DIR/bin/npm" ]]; then
    NPM="$CONDA_DIR/bin/npm"
  else
    NPM="$(command -v npm || true)"
  fi
fi

if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  echo "Missing Python executable. Set PYTHON or create .venv/.conda first." >&2
  exit 2
fi

if [[ -z "$NPM" || ! -x "$NPM" ]]; then
  echo "Missing npm executable. Set NPM or install Node.js/npm first." >&2
  exit 2
fi

echo "==> Python"
"$PYTHON" --version

echo "==> Node"
if [[ -x "$(dirname "$NPM")/node" ]]; then
  "$(dirname "$NPM")/node" --version
else
  node --version
fi
"$NPM" --version

if [[ "$INSTALL_DEPS" != "0" ]]; then
  echo "==> Installing backend dependencies"
  "$PYTHON" -m pip install --no-build-isolation -e ".[dev]"

  echo "==> Installing frontend dependencies"
  if [[ -f frontend/package-lock.json ]]; then
    (cd frontend && "$NPM" ci --prefer-offline --no-audit --no-fund --fetch-timeout=30000 --fetch-retries=2)
  else
    (cd frontend && "$NPM" install --prefer-offline --no-audit --no-fund --fetch-timeout=30000 --fetch-retries=2)
  fi
else
  echo "==> Skipping dependency installation because INSTALL_DEPS=0"
fi

echo "==> Python compile check"
"$PYTHON" -m compileall -q backend tests

echo "==> Backend tests"
"$PYTHON" -m pytest

echo "==> Frontend build"
(cd frontend && "$NPM" run build)

echo "==> All checks passed"
