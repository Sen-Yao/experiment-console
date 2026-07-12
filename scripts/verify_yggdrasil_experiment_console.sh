#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="ssh-direct.senyao.org"
SSH_PORT="4622"
SSH_USER="root"
REMOTE_BASE="/mnt/user/appdata/experiment-console"
EXPECTED_INSTANCE="yggdrasil-production"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) SSH_HOST="$2"; shift 2 ;;
    --port) SSH_PORT="$2"; shift 2 ;;
    --user) SSH_USER="$2"; shift 2 ;;
    --remote-base) REMOTE_BASE="$2"; shift 2 ;;
    --expected-instance) EXPECTED_INSTANCE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$SSH_PORT" =~ ^[0-9]+$ ]] || { echo "Invalid SSH port" >&2; exit 2; }
[[ "$SSH_HOST" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "Invalid SSH host" >&2; exit 2; }
[[ "$SSH_USER" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid SSH user" >&2; exit 2; }
[[ "$REMOTE_BASE" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "Invalid remote base" >&2; exit 2; }
[[ "$EXPECTED_INSTANCE" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid instance id" >&2; exit 2; }

target="$SSH_USER@$SSH_HOST"
ssh -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=15 "$target" /bin/bash \
  "$REMOTE_BASE/current/deploy/yggdrasil/verify-release.sh" \
  --base "$REMOTE_BASE" --expected-instance "$EXPECTED_INSTANCE"
