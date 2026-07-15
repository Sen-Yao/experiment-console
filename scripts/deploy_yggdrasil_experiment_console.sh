#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_HOST="ssh-direct.senyao.org"
SSH_PORT="4622"
SSH_USER="root"
REMOTE_BASE="/mnt/user/appdata/experiment-console"
SEED_STATE_DIR=""
LEGACY_ARCHIVE=""
FRESH_V2_LEDGER=0
APPLY=0

usage() {
  echo "Usage: $0 [--apply] [--fresh-v2-ledger | --seed-state-dir DIR] [--legacy-archive FILE] [--host HOST] [--port PORT] [--user USER] [--remote-base PATH]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --host) SSH_HOST="$2"; shift 2 ;;
    --port) SSH_PORT="$2"; shift 2 ;;
    --user) SSH_USER="$2"; shift 2 ;;
    --remote-base) REMOTE_BASE="$2"; shift 2 ;;
    --seed-state-dir) SEED_STATE_DIR="$2"; shift 2 ;;
    --legacy-archive) LEGACY_ARCHIVE="$2"; shift 2 ;;
    --fresh-v2-ledger) FRESH_V2_LEDGER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

if [[ "$FRESH_V2_LEDGER" == "1" && -n "$SEED_STATE_DIR" ]]; then
  echo "--fresh-v2-ledger and --seed-state-dir are mutually exclusive" >&2
  exit 2
fi

[[ "$SSH_PORT" =~ ^[0-9]+$ ]] || { echo "Invalid SSH port" >&2; exit 2; }
[[ "$SSH_HOST" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "Invalid SSH host" >&2; exit 2; }
[[ "$SSH_USER" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid SSH user" >&2; exit 2; }
[[ "$REMOTE_BASE" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "Invalid remote base" >&2; exit 2; }

for file in Dockerfile compose.yggdrasil.yaml deploy/yggdrasil/activate-release.sh; do
  [[ -r "$ROOT/$file" ]] || { echo "Missing deployment file: $file" >&2; exit 2; }
done
if [[ -n "$SEED_STATE_DIR" ]]; then
  [[ -r "$SEED_STATE_DIR/console.sqlite3" && -r "$SEED_STATE_DIR/migration_manifest.json" ]] || {
    echo "Seed state must contain console.sqlite3 and migration_manifest.json" >&2
    exit 2
  }
fi
if [[ -n "$LEGACY_ARCHIVE" ]]; then
  [[ -r "$LEGACY_ARCHIVE" && -r "$LEGACY_ARCHIVE.sha256" && -r "$LEGACY_ARCHIVE.manifest.json" ]] || {
    echo "Legacy archive plus .sha256 and .manifest.json sidecars are required" >&2
    exit 2
  }
fi

git_sha="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo worktree)"
release_id="$(date -u +%Y%m%dT%H%M%SZ)-$git_sha"
remote_release="$REMOTE_BASE/releases/$release_id"
target="$SSH_USER@$SSH_HOST"

echo "route=$target:$SSH_PORT"
echo "release=$remote_release"
echo "secrets=pre-provisioned-on-Yggdrasil (never uploaded or printed)"
echo "seed_state=${SEED_STATE_DIR:-none}"
echo "legacy_archive=${LEGACY_ARCHIVE:-none}"
echo "fresh_v2_ledger=$FRESH_V2_LEDGER"
if [[ "$APPLY" != "1" ]]; then
  echo "DRY RUN: no SSH, rsync, Docker, or remote write was performed."
  exit 0
fi

command -v ssh >/dev/null || { echo "ssh is required" >&2; exit 2; }
command -v rsync >/dev/null || { echo "rsync is required" >&2; exit 2; }
ssh_opts=(-p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=15)
rsync_ssh="ssh -p $SSH_PORT -o BatchMode=yes -o ConnectTimeout=15"

ssh "${ssh_opts[@]}" "$target" "install -d -m 0750 '$REMOTE_BASE/releases' '$remote_release'"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.state/' \
  --exclude '.local_deps/' \
  --exclude '.venv/' \
  --exclude '.conda/' \
  --exclude '.codex/' \
  --exclude '.agents/' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude 'secrets/' \
  --exclude 'console_api_token' \
  --exclude 'frontend/node_modules/' \
  --exclude 'frontend/dist/' \
  --exclude '*.sqlite3' \
  --exclude '*.jsonl' \
  --exclude '*.key' \
  --exclude '*.pem' \
  -e "$rsync_ssh" "$ROOT/" "$target:$remote_release/"
if [[ -n "$SEED_STATE_DIR" ]]; then
  rsync -az --delete -e "$rsync_ssh" "$SEED_STATE_DIR/" "$target:$remote_release/migration-seed/"
fi
if [[ -n "$LEGACY_ARCHIVE" ]]; then
  archive_name="$(basename "$LEGACY_ARCHIVE")"
  ssh "${ssh_opts[@]}" "$target" "install -d -m 0700 '$REMOTE_BASE/archives'"
  rsync -az -e "$rsync_ssh" "$LEGACY_ARCHIVE" "$LEGACY_ARCHIVE.sha256" "$LEGACY_ARCHIVE.manifest.json" "$target:$REMOTE_BASE/archives/"
  ssh "${ssh_opts[@]}" "$target" "chmod 0400 '$REMOTE_BASE/archives/$archive_name' '$REMOTE_BASE/archives/$archive_name.sha256' '$REMOTE_BASE/archives/$archive_name.manifest.json'"
fi
activation_args=(--base "$REMOTE_BASE" --release "$remote_release" --apply)
if [[ "$FRESH_V2_LEDGER" == "1" ]]; then
  activation_args+=(--fresh-v2-ledger)
fi
ssh "${ssh_opts[@]}" "$target" /bin/bash "$remote_release/deploy/yggdrasil/activate-release.sh" \
  "${activation_args[@]}"
