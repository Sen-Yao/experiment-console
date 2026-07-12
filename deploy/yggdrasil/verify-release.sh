#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=release-common.sh
source "$HERE/release-common.sh"

BASE="/mnt/user/appdata/experiment-console"
EXPECTED_INSTANCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --expected-instance) EXPECTED_INSTANCE="$2"; shift 2 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

RELEASE="$(current_release "$BASE")"
[[ -n "$RELEASE" ]] || fail "current release symlink is missing"
load_production_context "$BASE" "$RELEASE"
[[ -z "$EXPECTED_INSTANCE" || "$INSTANCE_ID" == "$EXPECTED_INSTANCE" ]] || fail "configured instance id mismatch"
require_command docker

"${COMPOSE[@]}" config --quiet
cid="$(container_id)"
[[ -n "$cid" ]] || fail "console container is absent"
[[ "$(docker inspect -f '{{.State.Running}}' "$cid")" == "true" ]] || fail "console container is not running"
[[ "$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$cid")" == "healthy" ]] || fail "console container is not healthy"

container_user="$(docker inspect -f '{{.Config.User}}' "$cid")"
[[ -n "$container_user" && "$container_user" != "0" && "$container_user" != "root" && "$container_user" != "0:0" ]] || fail "container is running as root"
[[ "$(docker inspect -f '{{.Platform}}' "$cid")" == "linux" ]] || fail "container platform is not Linux"
[[ "$(docker exec "$cid" uname -m)" == "x86_64" ]] || fail "container architecture is not x86_64"

published_ips="$(docker inspect -f '{{range $port, $bindings := .NetworkSettings.Ports}}{{range $bindings}}{{println .HostIp}}{{end}}{{end}}' "$cid" | sed '/^$/d' | sort -u)"
[[ "$published_ips" == "127.0.0.1" ]] || fail "API is not exclusively published on Yggdrasil loopback: $published_ips"

for destination in /var/lib/experiment-console/state /var/lib/experiment-console/state/results; do
  rw="$(docker inspect -f "{{range .Mounts}}{{if eq .Destination \"$destination\"}}{{.RW}}{{end}}{{end}}" "$cid")"
  [[ "$rw" == "true" ]] || fail "persistent mount is not writable: $destination"
done
for destination in /run/secrets/wandb_api_key /run/secrets/console_api_token /run/secrets/hccs_ssh_key /run/config/hccs_ssh_config /run/config/hccs_known_hosts; do
  rw="$(docker inspect -f "{{range .Mounts}}{{if eq .Destination \"$destination\"}}{{.RW}}{{end}}{{end}}" "$cid")"
  [[ "$rw" == "false" ]] || fail "credential/config mount is not read-only: $destination"
done

docker exec "$cid" python - "$INSTANCE_ID" <<'PY'
import json
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

from experiment_console.config import Settings
from experiment_console.wandb_client import WandBClient

expected_instance = sys.argv[1]
with urllib.request.urlopen("http://127.0.0.1:5174/health", timeout=4) as response:
    health = json.load(response)
assert health.get("status") == "ok", health
assert health.get("authority_role") == "authoritative", health
assert health.get("instance_id") == expected_instance, health
assert health.get("ledger_id"), health
assert health.get("console_api_auth_configured") is True, health

protected_url = "http://127.0.0.1:5174/api/artifacts/__verification_missing__/download"
try:
    urllib.request.urlopen(protected_url, timeout=4)
    raise AssertionError("protected artifact API accepted a request without bearer auth")
except urllib.error.HTTPError as exc:
    assert exc.code == 401, exc.code
token = Path("/run/secrets/console_api_token").read_text(encoding="utf-8").strip()
assert token
request = urllib.request.Request(protected_url, headers={"Authorization": f"Bearer {token}"})
try:
    urllib.request.urlopen(request, timeout=4)
    raise AssertionError("missing verification artifact unexpectedly exists")
except urllib.error.HTTPError as exc:
    assert exc.code == 404, exc.code
worker = health.get("monitor_worker") or {}
assert worker.get("enabled") is True, worker
assert worker.get("ready") is True, worker
assert worker.get("running") is True, worker
assert worker.get("lease_held") is True, worker

database = Path("/var/lib/experiment-console/state/console.sqlite3")
with sqlite3.connect(database) as connection:
    assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"jobs", "monitor_schedules", "leases", "wake_events", "source_observations", "metadata"}
    assert required <= tables, (required - tables)
    jobs = connection.execute("SELECT count(*) FROM jobs").fetchone()[0]
    schedules = connection.execute("SELECT count(*) FROM monitor_schedules").fetchone()[0]
    active_schedules = connection.execute("SELECT count(*) FROM monitor_schedules WHERE active = 1").fetchone()[0]
    sweep_job = connection.execute("""
        SELECT job_id FROM jobs
        WHERE sweep_id IS NOT NULL AND sweep_id != ''
        ORDER BY created_at DESC LIMIT 1
    """).fetchone()
    unscheduled_active = connection.execute("""
        SELECT count(*)
        FROM jobs AS j
        LEFT JOIN monitor_schedules AS s ON s.job_id = j.job_id AND s.active = 1
        WHERE j.status IN ('planned', 'queued', 'validating', 'running', 'finalizing', 'attention', 'unknown')
          AND json_type(j.monitor_json, '$.result_contract') = 'object'
          AND s.job_id IS NULL
    """).fetchone()[0]
    assert unscheduled_active == 0, unscheduled_active
    manifest_path = Path("/var/lib/experiment-console/state/migration_manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        imported = set(manifest.get("requested_job_ids") or [])
        scheduled = {row[0] for row in connection.execute("SELECT job_id FROM monitor_schedules")}
        assert imported <= scheduled, sorted(imported - scheduled)

settings = Settings()
WandBClient(settings).discover_sweeps(
    settings.default_entity,
    settings.default_project,
    days=1,
    include_runs=False,
)
assert sweep_job is not None, "production verification requires at least one registered sweep"
auth_request = urllib.request.Request(
    "http://127.0.0.1:5174/api/runner/auth-check?requested_by=production-verifier",
    data=json.dumps({"job_id": sweep_job[0]}).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(auth_request, timeout=150) as response:
    auth_payload = json.load(response)
auth_result = auth_payload.get("result") or {}
assert auth_payload.get("status") == "ok", auth_payload
assert auth_result.get("ok") is True, auth_result
assert auth_result.get("classification") == "auth_ok", auth_result
print(json.dumps({
    "status": "ok",
    "authority_role": health["authority_role"],
    "instance_id": health["instance_id"],
    "ledger_id": health["ledger_id"],
    "monitor_worker": worker,
    "jobs": jobs,
    "monitor_schedules": schedules,
    "active_monitor_schedules": active_schedules,
    "unscheduled_active_contract_jobs": unscheduled_active,
    "wandb_graphql_probe": "ok",
    "hccs_wandb_auth_probe": "ok",
}, sort_keys=True))
PY

[[ -w "$STATE_PATH" ]] || fail "host state path is not writable"
[[ -w "$RESULTS_PATH" ]] || fail "host results path is not writable"
echo "Yggdrasil Experiment Console verification passed."
