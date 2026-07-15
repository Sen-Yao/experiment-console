#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=release-common.sh
source "$HERE/release-common.sh"

BASE="/mnt/user/appdata/experiment-console"
RELEASE=""
APPLY=0
FRESH_V2_LEDGER=0
NEW_LEDGER_ID=""

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
    --release) RELEASE="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    --fresh-v2-ledger) FRESH_V2_LEDGER=1; shift ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ -n "$RELEASE" ]] || fail "--release is required"
load_production_context "$BASE" "$RELEASE"
validate_data_path_layout "$BASE" "$STATE_PATH" "$RESULTS_PATH"

if [[ "$APPLY" != "1" ]]; then
  echo "DRY RUN: validate, build, back up, and activate release"
  echo "base=$BASE"
  echo "release=$RELEASE"
  echo "image_tag=$IMAGE_TAG"
  echo "instance_id=$INSTANCE_ID"
  echo "fresh_v2_ledger=$FRESH_V2_LEDGER"
  exit 0
fi

require_command docker
require_command realpath
for file in "$WANDB_SECRET_FILE" "$CONSOLE_API_TOKEN_SECRET_FILE" "$HCCS_KEY_FILE" "$HCCS_CONFIG_FILE" "$HCCS_KNOWN_HOSTS_FILE"; do
  [[ -s "$file" ]] || fail "required secret/config file is missing or empty: $file"
done
[[ "$WANDB_SECRET_FILE" != "$HCCS_KEY_FILE" ]] || fail "W&B and HCCS credentials must be separate files"
[[ "$CONSOLE_API_TOKEN_SECRET_FILE" != "$WANDB_SECRET_FILE" && "$CONSOLE_API_TOKEN_SECRET_FILE" != "$HCCS_KEY_FILE" ]] || fail "Console API, W&B, and HCCS credentials must be separate files"

SEED_LEDGER="$RELEASE/migration-seed/console.sqlite3"
SEED_MANIFEST="$RELEASE/migration-seed/migration_manifest.json"
if [[ -e "$SEED_LEDGER" && ! -r "$SEED_MANIFEST" ]]; then
  fail "migration seed requires migration_manifest.json"
fi
if [[ "$FRESH_V2_LEDGER" == "1" ]]; then
  [[ ! -e "$SEED_LEDGER" && ! -e "$SEED_MANIFEST" ]] || fail "fresh v2 cutover cannot use a migration seed"
  [[ -e "$STATE_PATH/console.sqlite3" ]] || fail "fresh v2 cutover requires an existing authoritative ledger"
  [[ -n "$(current_release "$BASE")" ]] || fail "fresh v2 cutover requires an existing current release"
else
  if [[ -e "$STATE_PATH/console.sqlite3" && -e "$SEED_LEDGER" ]]; then
    fail "migration seed refused because the authoritative ledger already exists"
  fi
  if [[ ! -e "$STATE_PATH/console.sqlite3" && ! -e "$SEED_LEDGER" ]]; then
    fail "first authoritative deployment requires an explicit migration seed or --fresh-v2-ledger"
  fi
fi

install -d -o "$CONSOLE_UID" -g "$CONSOLE_GID" -m 0750 "$STATE_PATH" "$RESULTS_PATH"
validate_resolved_data_path_layout
chown "$CONSOLE_UID:$CONSOLE_GID" "$WANDB_SECRET_FILE" "$CONSOLE_API_TOKEN_SECRET_FILE" "$HCCS_KEY_FILE"
chmod 0400 "$WANDB_SECRET_FILE" "$CONSOLE_API_TOKEN_SECRET_FILE" "$HCCS_KEY_FILE"
chmod 0444 "$HCCS_CONFIG_FILE" "$HCCS_KNOWN_HOSTS_FILE"

"${COMPOSE[@]}" config --quiet
"${COMPOSE[@]}" build --pull console

NEW_CONSOLE_UID="$CONSOLE_UID"
NEW_CONSOLE_GID="$CONSOLE_GID"
NEW_STATE_PATH="$STATE_PATH"
NEW_RESULTS_PATH="$RESULTS_PATH"
NEW_WANDB_SECRET_FILE="$WANDB_SECRET_FILE"
NEW_CONSOLE_API_TOKEN_SECRET_FILE="$CONSOLE_API_TOKEN_SECRET_FILE"
NEW_HCCS_KEY_FILE="$HCCS_KEY_FILE"
NEW_HCCS_CONFIG_FILE="$HCCS_CONFIG_FILE"
NEW_HCCS_KNOWN_HOSTS_FILE="$HCCS_KNOWN_HOSTS_FILE"
NEW_AUTHORITY_ROLE="$AUTHORITY_ROLE"
NEW_INSTANCE_ID="$INSTANCE_ID"
NEW_ENV_FILE="$ENV_FILE"
NEW_COMPOSE_FILE="$COMPOSE_FILE"
NEW_RELEASE_NAME="$RELEASE_NAME"
NEW_RELEASE="$RELEASE"
NEW_IMAGE_TAG="$IMAGE_TAG"
NEW_COMPOSE=("${COMPOSE[@]}")

restore_new_release_context() {
  CONSOLE_UID="$NEW_CONSOLE_UID"
  CONSOLE_GID="$NEW_CONSOLE_GID"
  STATE_PATH="$NEW_STATE_PATH"
  RESULTS_PATH="$NEW_RESULTS_PATH"
  WANDB_SECRET_FILE="$NEW_WANDB_SECRET_FILE"
  CONSOLE_API_TOKEN_SECRET_FILE="$NEW_CONSOLE_API_TOKEN_SECRET_FILE"
  HCCS_KEY_FILE="$NEW_HCCS_KEY_FILE"
  HCCS_CONFIG_FILE="$NEW_HCCS_CONFIG_FILE"
  HCCS_KNOWN_HOSTS_FILE="$NEW_HCCS_KNOWN_HOSTS_FILE"
  AUTHORITY_ROLE="$NEW_AUTHORITY_ROLE"
  INSTANCE_ID="$NEW_INSTANCE_ID"
  ENV_FILE="$NEW_ENV_FILE"
  COMPOSE_FILE="$NEW_COMPOSE_FILE"
  RELEASE_NAME="$NEW_RELEASE_NAME"
  RELEASE="$NEW_RELEASE"
  IMAGE_TAG="$NEW_IMAGE_TAG"
  COMPOSE=("${NEW_COMPOSE[@]}")
}

OLD_RELEASE="$(current_release "$BASE")"
BACKUP_CONTEXT="${OLD_RELEASE:-$RELEASE}"
OLD_LEDGER_ID=""
if [[ -n "$OLD_RELEASE" && -r "$OLD_RELEASE/compose.yggdrasil.yaml" ]]; then
  load_production_context "$BASE" "$OLD_RELEASE"
  if [[ "$FRESH_V2_LEDGER" == "1" ]]; then
    old_cid="$(container_id)"
    [[ -n "$old_cid" ]] || fail "fresh v2 cutover requires the current Console container"
    [[ "$(docker inspect -f '{{.State.Running}}' "$old_cid")" == "true" ]] || fail "fresh v2 cutover requires the current Console to be running"
    source_metadata="$(docker exec -i "$old_cid" python - <<'PY'
import sqlite3

database = "/var/lib/experiment-console/state/console.sqlite3"
nonterminal = ("planned", "queued", "validating", "running", "finalizing", "attention", "unknown")
with sqlite3.connect(database) as connection:
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    placeholders = ",".join("?" for _ in nonterminal)
    active = connection.execute(
        f"SELECT count(*) FROM jobs WHERE status IN ({placeholders})",
        nonterminal,
    ).fetchone()[0]
print(metadata.get("ledger_id") or "")
print(active)
print(metadata.get("cutover_committed_at") or "__none__")
PY
)"
    OLD_LEDGER_ID="$(printf '%s\n' "$source_metadata" | sed -n '1p')"
    source_nonterminal="$(printf '%s\n' "$source_metadata" | sed -n '2p')"
    source_cutover_commit="$(printf '%s\n' "$source_metadata" | sed -n '3p')"
    [[ "$OLD_LEDGER_ID" == ledger_* ]] || fail "current authority returned an invalid ledger id"
    [[ "$source_nonterminal" =~ ^[0-9]+$ ]] || fail "could not verify current nonterminal job count"
    [[ "$source_nonterminal" == "0" ]] || fail "fresh v2 cutover refused while $source_nonterminal jobs are nonterminal"
    [[ "$source_cutover_commit" == "__none__" ]] || fail "fresh v2 cutover is one-time and the current ledger is already committed"
  fi
  "${COMPOSE[@]}" stop console
fi
BACKUP_DIR="$("$HERE/backup-runtime.sh" --base "$BASE" --release "$BACKUP_CONTEXT" \
  --rollback-release "${OLD_RELEASE:-__none__}" --label pre-deploy --service-already-stopped --apply)"

ROLLBACK_ARMED=1
ROLLBACK_ATTEMPTED=0
rollback_activation_once() {
  local original_status="${1:-1}" rollback_status
  trap - ERR
  if [[ "$ROLLBACK_ARMED" == "1" && "$ROLLBACK_ATTEMPTED" == "0" ]]; then
    ROLLBACK_ATTEMPTED=1
    set +e
    "$HERE/rollback-release.sh" --base "$BASE" --backup-dir "$BACKUP_DIR" --apply
    rollback_status=$?
    set -e
    if [[ "$rollback_status" != "0" ]]; then
      echo "ERROR: activation failed and rollback also failed with status $rollback_status" >&2
    fi
  fi
  exit "$original_status"
}
trap 'rollback_activation_once $?' ERR

restore_new_release_context

if [[ "$FRESH_V2_LEDGER" == "1" ]]; then
  rm -rf -- "$STATE_PATH" "$RESULTS_PATH"
  install -d -o "$CONSOLE_UID" -g "$CONSOLE_GID" -m 0750 "$STATE_PATH" "$RESULTS_PATH"
  validate_resolved_data_path_layout
elif [[ -r "$SEED_LEDGER" ]]; then
  cp -a "$RELEASE/migration-seed/." "$STATE_PATH/"
  rm -rf "$STATE_PATH/results"
  install -d -o "$CONSOLE_UID" -g "$CONSOLE_GID" -m 0750 "$STATE_PATH/results"
  chown -R "$CONSOLE_UID:$CONSOLE_GID" "$STATE_PATH"
fi

ln -sfn "$RELEASE" "$BASE/current"
if ! "${COMPOSE[@]}" up -d --remove-orphans console; then
  echo "ERROR: new release failed to start; rolling back" >&2
  false
fi

cid="$(container_id)"
for _ in $(seq 1 18); do
  [[ -n "$cid" ]] && [[ "$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$cid")" == "healthy" ]] && break
  sleep 5
  cid="$(container_id)"
done
if [[ -z "$cid" ]] || [[ "$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$cid")" != "healthy" ]]; then
  echo "ERROR: new release did not become healthy; rolling back" >&2
  false
fi
if [[ "$FRESH_V2_LEDGER" == "1" ]]; then
  NEW_LEDGER_ID="$(docker exec -i "$cid" python - "$OLD_LEDGER_ID" <<'PY'
import json
import sqlite3
import sys
import urllib.request

previous_ledger = sys.argv[1]
with urllib.request.urlopen("http://127.0.0.1:5174/health", timeout=4) as response:
    health = json.load(response)
assert health.get("contract") == "runner_console_agent_v2", health
assert str(health.get("ledger_schema_version")) == "2", health
assert not health.get("cutover_committed_at"), health
ledger_id = str(health.get("ledger_id") or "")
assert ledger_id.startswith("ledger_"), health
assert ledger_id != previous_ledger, (ledger_id, previous_ledger)

database = "/var/lib/experiment-console/state/console.sqlite3"
with sqlite3.connect(database) as connection:
    assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {
        "jobs", "intents", "monitor_schedules", "wake_events", "source_observations",
        "dependency_episodes", "dependency_impacts", "metadata",
    }
    assert required <= tables, required - tables
    for table in required - {"metadata"}:
        assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0, table
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    assert metadata.get("ledger_id") == ledger_id, metadata
    assert metadata.get("ledger_schema_version") == "2", metadata
    assert "cutover_committed_at" not in metadata, metadata
print(ledger_id)
PY
)"
  [[ "$NEW_LEDGER_ID" == ledger_* && "$NEW_LEDGER_ID" != "$OLD_LEDGER_ID" ]]
  install -d -m 0750 "$BASE/cutovers"
  receipt_tmp="$BASE/cutovers/.$RELEASE_NAME.meta.$$"
  cat >"$receipt_tmp" <<EOF
cutover_version=2
created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
release=$RELEASE
backup_dir=$BACKUP_DIR
previous_ledger_id=$OLD_LEDGER_ID
new_ledger_id=$NEW_LEDGER_ID
ledger_schema_version=2
contract=runner_console_agent_v2
verified_empty=1
EOF
  chmod 0400 "$receipt_tmp"
  mv "$receipt_tmp" "$BASE/cutovers/$RELEASE_NAME.meta"
fi
ROLLBACK_ARMED=0
trap - ERR
echo "Activated $RELEASE with backup $BACKUP_DIR${NEW_LEDGER_ID:+; new_ledger_id=$NEW_LEDGER_ID}"
