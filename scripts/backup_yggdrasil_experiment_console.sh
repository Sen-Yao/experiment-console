#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="ssh-direct.senyao.org"
SSH_PORT="4622"
SSH_USER="root"
REMOTE_BASE="/mnt/user/appdata/experiment-console"
APPLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --host) SSH_HOST="$2"; shift 2 ;;
    --port) SSH_PORT="$2"; shift 2 ;;
    --user) SSH_USER="$2"; shift 2 ;;
    --remote-base) REMOTE_BASE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$SSH_PORT" =~ ^[0-9]+$ ]] || { echo "Invalid SSH port" >&2; exit 2; }
[[ "$SSH_HOST" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "Invalid SSH host" >&2; exit 2; }
[[ "$SSH_USER" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid SSH user" >&2; exit 2; }
[[ "$REMOTE_BASE" =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "Invalid remote base" >&2; exit 2; }
echo "route=$SSH_USER@$SSH_HOST:$SSH_PORT"
echo "remote_base=$REMOTE_BASE"
if [[ "$APPLY" != "1" ]]; then
  echo "DRY RUN: no SSH or remote write was performed."
  exit 0
fi
ssh -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=15 "$SSH_USER@$SSH_HOST" /bin/bash \
  "$REMOTE_BASE/current/deploy/yggdrasil/backup-runtime.sh" --base "$REMOTE_BASE" --label manual --apply
