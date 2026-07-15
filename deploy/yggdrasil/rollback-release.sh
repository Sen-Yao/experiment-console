#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=release-common.sh
source "$HERE/release-common.sh"

BASE="/mnt/user/appdata/experiment-console"
BACKUP_DIR=""
APPLY=0

require_canonical_path_syntax() {
  local label="$1" path="$2"
  require_safe_absolute_path "$path"
  [[ "$path" != "/" && "$path" != */ ]] || fail "$label must be a canonical non-root path"
  [[ "$path" != *"//"* && "$path" != *"/./"* && "$path" != */. ]] || fail "$label must not contain empty or dot path components"
  [[ "$path" != *"/../"* && "$path" != */.. ]] || fail "$label must not contain parent path components"
}

validate_data_path_layout() {
  local base_path="$1" state_path="$2" results_path="$3"
  require_canonical_path_syntax "base" "$base_path"
  require_canonical_path_syntax "state path" "$state_path"
  require_canonical_path_syntax "results path" "$results_path"
  [[ "$state_path" == "$base_path/"* ]] || fail "state path must be a child of base"
  [[ "$results_path" == "$base_path/"* ]] || fail "results path must be a child of base"
  if [[ "$state_path" == "$results_path" || "$state_path" == "$results_path/"* || "$results_path" == "$state_path/"* ]]; then
    fail "state and results paths must be distinct and non-overlapping"
  fi
}

validate_resolved_data_path_layout() {
  local resolved_base resolved_state resolved_results
  [[ ! -L "$STATE_PATH" && ! -L "$RESULTS_PATH" ]] || fail "state and results paths must not be symlinks"
  resolved_base="$(realpath "$BASE")"
  resolved_state="$(realpath "$STATE_PATH")"
  resolved_results="$(realpath "$RESULTS_PATH")"
  validate_data_path_layout "$resolved_base" "$resolved_state" "$resolved_results"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --backup-dir) BACKUP_DIR="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    *) fail "unknown argument: $1" ;;
  esac
done

require_safe_absolute_path "$BASE"
require_safe_absolute_path "$BACKUP_DIR"
[[ "$BACKUP_DIR" == "$BASE/backups/"* ]] || fail "backup must be under $BASE/backups"
META="$BACKUP_DIR/backup.meta"
[[ -r "$META" ]] || fail "backup metadata is missing: $META"
PREVIOUS_RELEASE="$(env_value "$META" release)"
STATE_PATH="$(env_value "$META" state_path)"
RESULTS_PATH="$(env_value "$META" results_path)"
BACKUP_STATE_PATH="$STATE_PATH"
BACKUP_RESULTS_PATH="$RESULTS_PATH"
STATE_SHA="$(env_value "$META" state_sha256)"
RESULTS_SHA="$(env_value "$META" results_sha256)"
if [[ "$PREVIOUS_RELEASE" != "__none__" ]]; then
  require_safe_absolute_path "$PREVIOUS_RELEASE"
fi
require_safe_absolute_path "$STATE_PATH"
require_safe_absolute_path "$RESULTS_PATH"
validate_data_path_layout "$BASE" "$STATE_PATH" "$RESULTS_PATH"

if [[ "$APPLY" != "1" ]]; then
  echo "DRY RUN: restore paired state/results and release"
  echo "base=$BASE"
  echo "backup=$BACKUP_DIR"
  echo "release=$PREVIOUS_RELEASE"
  exit 0
fi

require_command docker
require_command realpath
require_command sha256sum
require_command tar
[[ "$(sha256sum "$BACKUP_DIR/state.tar.gz" | awk '{print $1}')" == "$STATE_SHA" ]] || fail "state backup checksum mismatch"
[[ "$(sha256sum "$BACKUP_DIR/results.tar.gz" | awk '{print $1}')" == "$RESULTS_SHA" ]] || fail "results backup checksum mismatch"
if [[ "$PREVIOUS_RELEASE" != "__none__" ]]; then
  [[ -r "$PREVIOUS_RELEASE/compose.yggdrasil.yaml" ]] || fail "previous release is unavailable: $PREVIOUS_RELEASE"
fi
[[ -d "$STATE_PATH" && -d "$RESULTS_PATH" ]] || fail "state and results directories must exist before rollback"
validate_resolved_data_path_layout

PREVIOUS_CONTEXT_READY=0
PREVIOUS_COMPOSE=()
if [[ "$PREVIOUS_RELEASE" != "__none__" ]]; then
  load_production_context "$BASE" "$PREVIOUS_RELEASE"
  [[ "$STATE_PATH" == "$BACKUP_STATE_PATH" ]] || fail "backup state path does not match the previous release context"
  [[ "$RESULTS_PATH" == "$BACKUP_RESULTS_PATH" ]] || fail "backup results path does not match the previous release context"
  PREVIOUS_CONSOLE_UID="$CONSOLE_UID"
  PREVIOUS_CONSOLE_GID="$CONSOLE_GID"
  PREVIOUS_COMPOSE=("${COMPOSE[@]}")
  PREVIOUS_CONTEXT_READY=1
fi
STATE_PATH="$BACKUP_STATE_PATH"
RESULTS_PATH="$BACKUP_RESULTS_PATH"

CURRENT_RELEASE="$(current_release "$BASE")"
if [[ -n "$CURRENT_RELEASE" && -r "$CURRENT_RELEASE/compose.yggdrasil.yaml" ]]; then
  load_production_context "$BASE" "$CURRENT_RELEASE"
  [[ "$STATE_PATH" == "$BACKUP_STATE_PATH" ]] || fail "backup state path does not match the current release context"
  [[ "$RESULTS_PATH" == "$BACKUP_RESULTS_PATH" ]] || fail "backup results path does not match the current release context"
  cutover_commit=""
  current_cid="$(container_id)"
  if [[ -n "$current_cid" ]] && [[ "$(docker inspect -f '{{.State.Running}}' "$current_cid")" == "true" ]]; then
    cutover_commit="$(docker exec -i "$current_cid" python - <<'PY'
import sqlite3

with sqlite3.connect("/var/lib/experiment-console/state/console.sqlite3") as connection:
    row = connection.execute("SELECT value FROM metadata WHERE key = 'cutover_committed_at'").fetchone()
print(row[0] if row else "")
PY
)"
  elif [[ -e "$STATE_PATH/console.sqlite3" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      cutover_commit="$(python3 - "$STATE_PATH/console.sqlite3" <<'PY'
import sqlite3
import sys

with sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True) as connection:
    row = connection.execute("SELECT value FROM metadata WHERE key = 'cutover_committed_at'").fetchone()
print(row[0] if row else "")
PY
)"
    else
      current_image="$("${COMPOSE[@]}" config --images | sed -n '1p')"
      [[ -n "$current_image" ]] || fail "cannot verify the v2 cutover commit state"
      cutover_commit="$(docker run --rm --network none --read-only \
        --entrypoint python -v "$STATE_PATH:/ledger:ro" "$current_image" -c \
        'import sqlite3; c=sqlite3.connect("file:/ledger/console.sqlite3?mode=ro", uri=True); r=c.execute("SELECT value FROM metadata WHERE key = '\''cutover_committed_at'\''").fetchone(); print(r[0] if r else "")')"
    fi
  fi
  [[ -z "$cutover_commit" ]] || fail "rollback refused after cutover_committed_at=$cutover_commit; deploy a forward fix"
  "${COMPOSE[@]}" stop console || true
fi
STATE_PATH="$BACKUP_STATE_PATH"
RESULTS_PATH="$BACKUP_RESULTS_PATH"

DISPLACED_SUFFIX="pre-rollback-$(date -u +%Y%m%dT%H%M%SZ)"
DISPLACED_STATE="$STATE_PATH.$DISPLACED_SUFFIX"
DISPLACED_RESULTS="$RESULTS_PATH.$DISPLACED_SUFFIX"
STATE_MOVED=0
RESULTS_MOVED=0
CURRENT_LINK_TOUCHED=0
restore_displaced() {
  local original_status="${1:-1}" restore_status=0
  trap - ERR
  set +e
  if [[ "$STATE_MOVED" == "1" ]]; then
    if ! rm -rf "$STATE_PATH" || ! mv "$DISPLACED_STATE" "$STATE_PATH"; then
      echo "ERROR: failed to restore displaced state directory" >&2
      restore_status=1
    fi
  fi
  if [[ "$RESULTS_MOVED" == "1" ]]; then
    if ! rm -rf "$RESULTS_PATH" || ! mv "$DISPLACED_RESULTS" "$RESULTS_PATH"; then
      echo "ERROR: failed to restore displaced results directory" >&2
      restore_status=1
    fi
  fi
  if [[ "$CURRENT_LINK_TOUCHED" == "1" ]]; then
    if [[ -n "$CURRENT_RELEASE" ]]; then
      ln -sfn "$CURRENT_RELEASE" "$BASE/current" || restore_status=1
    else
      rm -f "$BASE/current" || restore_status=1
    fi
  fi
  set -e
  if [[ "$restore_status" != "0" ]]; then
    echo "ERROR: rollback failed and restoration of the displaced runtime was incomplete" >&2
  fi
  exit "$original_status"
}
trap 'restore_displaced $?' ERR

mv "$STATE_PATH" "$DISPLACED_STATE"
STATE_MOVED=1
mv "$RESULTS_PATH" "$DISPLACED_RESULTS"
RESULTS_MOVED=1
mkdir -p "$STATE_PATH" "$RESULTS_PATH"

tar -xzf "$BACKUP_DIR/state.tar.gz" -C "$STATE_PATH"
tar -xzf "$BACKUP_DIR/results.tar.gz" -C "$RESULTS_PATH"
if [[ "$PREVIOUS_RELEASE" == "__none__" ]]; then
  CURRENT_LINK_TOUCHED=1
  rm -f "$BASE/current"
else
  CURRENT_LINK_TOUCHED=1
  ln -sfn "$PREVIOUS_RELEASE" "$BASE/current"
  [[ "$PREVIOUS_CONTEXT_READY" == "1" ]]
  CONSOLE_UID="$PREVIOUS_CONSOLE_UID"
  CONSOLE_GID="$PREVIOUS_CONSOLE_GID"
  COMPOSE=("${PREVIOUS_COMPOSE[@]}")
  chown -R "$CONSOLE_UID:$CONSOLE_GID" "$STATE_PATH" "$RESULTS_PATH"
  "${COMPOSE[@]}" up -d console

  cid="$(container_id)"
  for _ in $(seq 1 18); do
    [[ -n "$cid" ]] && [[ "$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$cid")" == "healthy" ]] && break
    sleep 5
    cid="$(container_id)"
  done
  if [[ -z "$cid" ]] || [[ "$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$cid")" != "healthy" ]]; then
    echo "ERROR: rolled-back release did not become healthy" >&2
    false
  fi
fi
trap - ERR
echo "Rollback complete; previous_release=$PREVIOUS_RELEASE; displaced data retained at $DISPLACED_STATE and $DISPLACED_RESULTS"
