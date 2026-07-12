#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=release-common.sh
source "$HERE/release-common.sh"

BASE="/mnt/user/appdata/experiment-console"
RELEASE=""
LABEL="manual"
ROLLBACK_RELEASE=""
APPLY=0
SERVICE_ALREADY_STOPPED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --release) RELEASE="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --rollback-release) ROLLBACK_RELEASE="$2"; shift 2 ;;
    --service-already-stopped) SERVICE_ALREADY_STOPPED=1; shift ;;
    --apply) APPLY=1; shift ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid backup label"
RELEASE="${RELEASE:-$(current_release "$BASE")}"
[[ -n "$RELEASE" ]] || fail "no release is available for compose context"
ROLLBACK_RELEASE="${ROLLBACK_RELEASE:-$RELEASE}"
if [[ "$ROLLBACK_RELEASE" != "__none__" ]]; then
  require_safe_absolute_path "$ROLLBACK_RELEASE"
fi
load_production_context "$BASE" "$RELEASE"

BACKUP_ID="$(date -u +%Y%m%dT%H%M%SZ)-$LABEL"
BACKUP_DIR="$BASE/backups/$BACKUP_ID"

if [[ "$APPLY" != "1" ]]; then
  echo "DRY RUN: create a stopped, paired state/results backup"
  echo "base=$BASE"
  echo "release=$RELEASE"
  echo "backup=$BACKUP_DIR"
  exit 0
fi

require_command docker
require_command tar
require_command sha256sum
[[ -d "$STATE_PATH" ]] || fail "state directory is missing: $STATE_PATH"
[[ -d "$RESULTS_PATH" ]] || fail "results directory is missing: $RESULTS_PATH"
[[ ! -e "$BACKUP_DIR" ]] || fail "backup already exists: $BACKUP_DIR"

was_running=0
if [[ "$SERVICE_ALREADY_STOPPED" != "1" ]] && [[ -n "$(container_id)" ]]; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "$(container_id)")" == "true" ]]; then
    was_running=1
    "${COMPOSE[@]}" stop console
  fi
fi

restart_if_needed() {
  if [[ "$was_running" == "1" ]]; then
    "${COMPOSE[@]}" up -d console >/dev/null
  fi
}
trap restart_if_needed EXIT

TMP_DIR="$BASE/backups/.backup-$BACKUP_ID-$$"
install -d -m 0700 "$BASE/backups" "$TMP_DIR"
tar -czf "$TMP_DIR/state.tar.gz" -C "$STATE_PATH" --exclude='./results' .
tar -czf "$TMP_DIR/results.tar.gz" -C "$RESULTS_PATH" .
state_sha="$(sha256sum "$TMP_DIR/state.tar.gz" | awk '{print $1}')"
results_sha="$(sha256sum "$TMP_DIR/results.tar.gz" | awk '{print $1}')"
cat >"$TMP_DIR/backup.meta" <<EOF
backup_version=1
created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
release=$ROLLBACK_RELEASE
state_path=$STATE_PATH
results_path=$RESULTS_PATH
state_sha256=$state_sha
results_sha256=$results_sha
instance_id=$INSTANCE_ID
EOF
chmod 0400 "$TMP_DIR/state.tar.gz" "$TMP_DIR/results.tar.gz" "$TMP_DIR/backup.meta"
mv "$TMP_DIR" "$BACKUP_DIR"
chmod 0500 "$BACKUP_DIR"
echo "$BACKUP_DIR"
