#!/usr/bin/env python3
"""Create a clean production ledger and a complete read-only legacy archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote


ACTIVE_SOURCE_STATUSES = {
    "active",
    "planned",
    "validating",
    "running",
    "unknown",
    "finalizing",
    "reconciling",
    "sync_error",
}
TERMINAL_SOURCE_STATUSES = {"finished", "failed", "cancelled"}
SUPPORTED_SOURCE_STATUSES = ACTIVE_SOURCE_STATUSES | TERMINAL_SOURCE_STATUSES | {"queued", "attention"}
REQUIRED_JOB_COLUMNS = {
    "job_id",
    "name",
    "status",
    "entity",
    "project",
    "sweep_id",
    "config_path",
    "remote_host",
    "remote_cwd",
    "conda_env",
    "agent_pids_json",
    "monitor_json",
    "created_at",
    "updated_at",
}
QUEUE_PAYLOAD_REQUIRED = {"job_name", "config_path", "remote_host", "remote_cwd"}
LEGACY_METADATA_TERMS = {"cron", "hermes", "openclaw", "heartbeat"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def readonly_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def json_object(raw: Any, *, label: str, errors: list[dict[str, Any]], job_id: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError) as exc:
        errors.append({"job_id": job_id, "code": f"invalid_{label}_json", "detail": str(exc)})
        return {}
    if not isinstance(value, dict):
        errors.append({"job_id": job_id, "code": f"invalid_{label}_shape", "detail": "expected object"})
        return {}
    return value


def json_list(raw: Any, *, label: str, errors: list[dict[str, Any]], job_id: str) -> list[Any]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError) as exc:
        errors.append({"job_id": job_id, "code": f"invalid_{label}_json", "detail": str(exc)})
        return []
    if not isinstance(value, list):
        errors.append({"job_id": job_id, "code": f"invalid_{label}_shape", "detail": "expected array"})
        return []
    return value


def contains_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(term in normalized for term in ("password", "secret", "token", "api_key", "private_key")):
                return True
            if contains_secret_key(nested):
                return True
    elif isinstance(value, list):
        return any(contains_secret_key(item) for item in value)
    return False


def load_result_contracts(specifications: list[str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    contracts: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for specification in specifications:
        if "=" not in specification:
            errors.append({"job_id": None, "code": "invalid_result_contract_argument", "detail": specification})
            continue
        job_id, raw_path = specification.split("=", 1)
        job_id = job_id.strip()
        path = Path(raw_path).expanduser()
        if not job_id or not path.is_file():
            errors.append({"job_id": job_id or None, "code": "result_contract_missing", "detail": str(path)})
            continue
        try:
            contract = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"job_id": job_id, "code": "invalid_result_contract_json", "detail": str(exc)})
            continue
        if not isinstance(contract, dict) or not contract:
            errors.append({"job_id": job_id, "code": "invalid_result_contract_shape", "detail": "expected non-empty object"})
            continue
        if contains_secret_key(contract):
            errors.append({"job_id": job_id, "code": "result_contract_contains_secret_field", "detail": str(path)})
            continue
        repository_root = Path(__file__).resolve().parents[1]
        backend_root = repository_root / "backend"
        if str(backend_root) not in sys.path:
            sys.path.insert(0, str(backend_root))
        try:
            models = importlib.import_module("experiment_console.models")
            contract = models.ResultContract.model_validate(contract).model_dump(mode="json")
        except Exception as exc:
            errors.append({"job_id": job_id, "code": "invalid_result_contract", "detail": str(exc)})
            continue
        if job_id in contracts:
            errors.append({"job_id": job_id, "code": "duplicate_result_contract", "detail": str(path)})
            continue
        contracts[job_id] = contract
    return contracts, errors


def compact_queue(queue: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(queue, dict):
        return None, None
    payload = queue.get("payload")
    original_blocker = str(queue.get("blocked_by_job_id") or queue.get("queue_after_job_id") or "") or None
    compact: dict[str, Any] = {}
    for key in ("queue_group", "queue_policy", "queued_at"):
        if queue.get(key) is not None:
            compact[key] = queue[key]
    if isinstance(payload, dict):
        clean_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"idempotency_key", "queue_after_job_id"}
            and not any(term in key.lower() for term in LEGACY_METADATA_TERMS)
        }
        clean_payload["queue_after_job_id"] = None
        clean_payload["idempotency_key"] = None
        compact["payload"] = clean_payload
        original_blocker = original_blocker or str(payload.get("queue_after_job_id") or "") or None
    return compact or None, original_blocker


def compact_run_identity(run: Any) -> dict[str, Any] | None:
    if not isinstance(run, dict):
        return None
    allowed = {
        "pid",
        "gpu_index",
        "status_path",
        "result_path",
        "log_path",
        "remote_status_path",
        "remote_result_path",
        "remote_log_path",
    }
    compact = {key: run[key] for key in allowed if run.get(key) is not None}
    return compact or None


def expected_total(monitor: dict[str, Any]) -> int | None:
    candidates = [monitor.get("expected_total")]
    for key in ("watchdog", "last_wandb_status", "last_result_snapshot"):
        nested = monitor.get(key)
        if isinstance(nested, dict):
            candidates.extend((nested.get("expected_total"), nested.get("expected_run_count")))
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    return None


def transform_job(
    item: dict[str, Any],
    *,
    migration_id: str,
    reconcile_ids: set[str],
    selection_reasons: list[str],
    thread_id: str,
    result_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = item["row"]
    monitor = item["monitor"]
    source_status = str(row["status"]).lower()
    reconcile_required = (
        source_status not in TERMINAL_SOURCE_STATUSES | {"queued"}
        or row["job_id"] in reconcile_ids
    )
    lineage_only = source_status in TERMINAL_SOURCE_STATUSES and not reconcile_required
    if source_status == "queued":
        target_status = "queued"
    elif source_status == "attention":
        target_status = "attention"
    elif reconcile_required:
        target_status = "unknown"
    else:
        target_status = source_status

    clean_monitor: dict[str, Any] = {}
    kind = monitor.get("kind")
    if isinstance(kind, str) and kind:
        clean_monitor["kind"] = kind
    elif row.get("sweep_id"):
        clean_monitor["kind"] = "sweep"
    if isinstance(monitor.get("sweep_path"), str):
        clean_monitor["sweep_path"] = monitor["sweep_path"]
    total = expected_total(monitor)
    if total is not None:
        clean_monitor["expected_total"] = total
    queue, original_blocker = compact_queue(monitor.get("queue"))
    if queue:
        clean_monitor["queue"] = queue
    run = compact_run_identity(monitor.get("run"))
    if run:
        clean_monitor["run"] = run
    clean_monitor["result_contract"] = result_contract
    clean_monitor["migration"] = {
        "migration_id": migration_id,
        "imported_at": utc_now(),
        "source_status": source_status,
        "source_updated_at": row.get("updated_at"),
        "selection_reasons": selection_reasons,
        "thread_id": thread_id,
        "reconcile_required": reconcile_required,
        "lineage_only": lineage_only,
        "authorized_queued_job": source_status == "queued",
        "legacy_scheduler_metadata_removed": True,
        "original_queue_blocker": original_blocker,
    }

    record = {
        "job_id": row["job_id"],
        "name": row["name"],
        "status": target_status,
        "operation_id": None,
        "idempotency_key": None,
        "entity": row.get("entity"),
        "project": row.get("project"),
        "sweep_id": row.get("sweep_id"),
        "config_path": row.get("config_path"),
        "remote_host": row.get("remote_host"),
        "remote_cwd": row.get("remote_cwd"),
        "conda_env": row.get("conda_env"),
        "agent_pids": item["agent_pids"] if reconcile_required else [],
        "operation_log": [],
        "monitor": clean_monitor,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    manifest_item = {
        "job_id": row["job_id"],
        "name": row["name"],
        "source_status": source_status,
        "target_status": target_status,
        "selection_reasons": selection_reasons,
        "reconcile_required": reconcile_required,
        "lineage_only": lineage_only,
        "authorized_queued_job": source_status == "queued",
        "sweep_id": row.get("sweep_id"),
        "remote_host": row.get("remote_host"),
        "queue_group": (queue or {}).get("queue_group"),
        "original_queue_blocker": original_blocker,
        "result_contract_sha256": hashlib.sha256(
            json.dumps(result_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    return record, manifest_item


def inspect_source(
    source_state_dir: Path,
    *,
    requested_job_ids: list[str],
    reconcile_ids: set[str],
    lineage_ids: set[str],
    thread_id: str,
    result_contracts: dict[str, dict[str, Any]],
    contract_errors: list[dict[str, Any]],
    migration_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    database = source_state_dir / "console.sqlite3"
    if not database.is_file():
        raise ValueError(f"source database not found: {database}")
    errors: list[dict[str, Any]] = list(contract_errors)
    selected: list[dict[str, Any]] = []
    skipped_by_status: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    with readonly_connection(database) as connection:
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise ValueError(f"source SQLite quick_check failed: {quick_check}")
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "jobs" not in tables:
            raise ValueError("source database has no jobs table")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        missing = sorted(REQUIRED_JOB_COLUMNS - columns)
        if missing:
            raise ValueError(f"source jobs table is missing columns: {', '.join(missing)}")
        query_columns = sorted(columns & (REQUIRED_JOB_COLUMNS | {"operation_id", "idempotency_key", "operation_log_json"}))
        cursor = connection.execute(f"SELECT {', '.join(query_columns)} FROM jobs ORDER BY created_at, job_id")
        for sqlite_row in cursor:
            row = dict(sqlite_row)
            job_id = str(row.get("job_id") or "")
            status = str(row.get("status") or "").lower()
            source_counts[status or "<empty>"] += 1
            if job_id not in requested_job_ids:
                skipped_by_status[status or "<empty>"] += 1
                continue
            reasons = ["explicit_job_id"]
            if status not in SUPPORTED_SOURCE_STATUSES:
                errors.append({"job_id": job_id, "code": "unsupported_source_status", "detail": status})
            monitor = json_object(row.get("monitor_json"), label="monitor", errors=errors, job_id=job_id)
            agent_pids = json_list(row.get("agent_pids_json"), label="agent_pids", errors=errors, job_id=job_id)
            item = {
                "row": row,
                "monitor": monitor,
                "agent_pids": agent_pids,
                "selection_reasons": reasons,
            }
            if status == "queued":
                queue = monitor.get("queue")
                payload = queue.get("payload") if isinstance(queue, dict) else None
                if not isinstance(payload, dict):
                    errors.append({"job_id": job_id, "code": "queued_payload_missing", "detail": "queue.payload is required"})
                else:
                    missing_payload = sorted(key for key in QUEUE_PAYLOAD_REQUIRED if not payload.get(key))
                    if missing_payload:
                        errors.append({
                            "job_id": job_id,
                            "code": "queued_payload_incomplete",
                            "detail": ", ".join(missing_payload),
                        })
            selected.append(item)

    selected_ids = {item["row"]["job_id"] for item in selected}
    for job_id in sorted(set(requested_job_ids) - selected_ids):
        errors.append({"job_id": job_id, "code": "requested_job_not_found", "detail": "job is absent from source ledger"})
    for job_id in sorted(selected_ids - set(result_contracts)):
        errors.append({"job_id": job_id, "code": "result_contract_required", "detail": "provide JOB_ID=contract.json"})
    for job_id in sorted(set(result_contracts) - selected_ids):
        errors.append({"job_id": job_id, "code": "result_contract_job_not_selected", "detail": "contract has no selected job"})
    both = sorted(reconcile_ids & lineage_ids)
    for job_id in both:
        errors.append({"job_id": job_id, "code": "conflicting_explicit_classification", "detail": "both reconcile and lineage"})
    for job_id in sorted((reconcile_ids | lineage_ids) - selected_ids):
        errors.append({"job_id": job_id, "code": "explicit_job_not_selected", "detail": "job is absent or outside selection"})

    selected_terminal = [
        item for item in selected
        if str(item["row"]["status"]).lower() in TERMINAL_SOURCE_STATUSES
    ]
    for terminal_item in selected_terminal:
        terminal_id = terminal_item["row"]["job_id"]
        if terminal_id not in reconcile_ids | lineage_ids:
            errors.append({
                "job_id": terminal_id,
                "code": "selected_terminal_requires_classification",
                "detail": "classify explicitly as reconcile or lineage",
            })

    transformed: list[dict[str, Any]] = []
    manifest_items: list[dict[str, Any]] = []
    for item in selected:
        record, manifest_item = transform_job(
            item,
            migration_id=migration_id,
            reconcile_ids=reconcile_ids,
            selection_reasons=item["selection_reasons"],
            thread_id=thread_id,
            result_contract=result_contracts.get(item["row"]["job_id"], {}),
        )
        transformed.append(record)
        manifest_items.append(manifest_item)

    manifest = {
        "manifest_version": 1,
        "migration_id": migration_id,
        "generated_at": utc_now(),
        "source_state_dir": str(source_state_dir.resolve()),
        "source_database": str(database.resolve()),
        "source_sqlite_quick_check": "ok",
        "thread_id": thread_id,
        "requested_job_ids": requested_job_ids,
        "explicit_reconcile_job_ids": sorted(reconcile_ids),
        "explicit_lineage_job_ids": sorted(lineage_ids),
        "source_job_counts": dict(sorted(source_counts.items())),
        "skipped_job_counts": dict(sorted(skipped_by_status.items())),
        "selected_count": len(transformed),
        "selected_jobs": manifest_items,
        "ambiguous_count": len(errors),
        "ambiguities": errors,
        "hot_state_policy": {
            "legacy_audit_imported": False,
            "legacy_results_imported": False,
            "legacy_intents_imported": False,
            "legacy_events_imported": False,
            "operation_payloads_imported": False,
            "legacy_scheduler_metadata_imported": False,
            "queued_blocker_ids_imported": False,
            "target_schema_initialized_by_application": True,
            "selection_is_explicit_allowlist": True,
            "monitor_schedules_seeded": True,
            "wake_events_seeded": False,
            "source_observations_seeded": False,
        },
        "reconcile_required_job_ids": sorted(
            item["job_id"] for item in manifest_items if item["reconcile_required"]
        ),
        "lineage_only_job_ids": sorted(
            item["job_id"] for item in manifest_items if item["lineage_only"]
        ),
    }
    return transformed, manifest


def file_fingerprint(path: Path) -> dict[str, Any] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_fingerprints(source_state_dir: Path) -> dict[str, Any]:
    return {
        name: file_fingerprint(source_state_dir / name)
        for name in ("console.sqlite3", "console.sqlite3-wal", "console.sqlite3-shm", "audit.jsonl")
    }


def sqlite_backup(source: Path, destination: Path) -> None:
    with readonly_connection(source) as source_connection, sqlite3.connect(destination) as target_connection:
        source_connection.backup(target_connection)
        quick_check = str(target_connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise RuntimeError(f"consistent SQLite backup failed quick_check: {quick_check}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_readonly_archive(source_state_dir: Path, archive_dir: Path, migration_id: str) -> dict[str, Any]:
    archive_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive_path = archive_dir / f"legacy-runtime-{migration_id}.tar.gz"
    before = source_fingerprints(source_state_dir)
    with tempfile.TemporaryDirectory(prefix="runtime-snapshot-", dir=archive_dir) as temporary:
        snapshot = Path(temporary) / "console.sqlite3"
        sqlite_backup(source_state_dir / "console.sqlite3", snapshot)
        with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
            archive.add(source_state_dir, arcname="runtime", recursive=True)
            archive.add(snapshot, arcname="consistent/console.sqlite3")
    after = source_fingerprints(source_state_dir)
    if before != after:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError("source runtime changed during archive; keep it frozen and retry")
    digest = sha256_file(archive_path)
    archive_manifest = {
        "archive_version": 1,
        "migration_id": migration_id,
        "created_at": utc_now(),
        "source_state_dir": str(source_state_dir.resolve()),
        "archive": str(archive_path.resolve()),
        "sha256": digest,
        "size_bytes": archive_path.stat().st_size,
        "contains_full_runtime": True,
        "contains_consistent_sqlite_backup": True,
        "source_fingerprints": before,
    }
    sidecar = archive_path.with_suffix(archive_path.suffix + ".manifest.json")
    checksum = archive_path.with_suffix(archive_path.suffix + ".sha256")
    sidecar.write_text(json.dumps(archive_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum.write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
    for path in (archive_path, sidecar, checksum):
        path.chmod(0o400)
    return archive_manifest


def initialize_target_with_application(
    target_state_dir: Path,
    records: list[dict[str, Any]],
    *,
    thread_id: str,
    interval_seconds: int,
    timeout_seconds: int,
) -> list[str]:
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root / "backend"))
    os.environ["EXPERIMENT_CONSOLE_STATE_DIR"] = str(target_state_dir)
    os.environ["EXPERIMENT_CONSOLE_MONITOR_WORKER"] = "0"
    api = importlib.import_module("experiment_console.api")
    service = api.service
    if service.settings.state_dir.resolve() != target_state_dir.resolve():
        raise RuntimeError("target application initialized against an unexpected state directory")
    models = importlib.import_module("experiment_console.models")
    for record in records:
        service.store.upsert_job(models.JobRecord.model_validate(record))
        service.store.upsert_monitor_schedule(
            job_id=record["job_id"],
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            thread_id=thread_id,
        )
    (target_state_dir / "results").mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target_state_dir / "console.sqlite3") as connection:
        now = utc_now()
        connection.execute(
            "UPDATE monitor_schedules SET next_run_at = ?, updated_at = ?",
            (now, now),
        )
        return sorted(row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ))


def install_target_atomically(
    target_state_dir: Path,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    replace_target: bool,
    thread_id: str,
    interval_seconds: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    parent = target_state_dir.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{target_state_dir.name}.migration-{uuid.uuid4().hex}"
    previous: Path | None = None
    try:
        staging.mkdir(mode=0o700)
        schema_tables = initialize_target_with_application(
            staging,
            records,
            thread_id=thread_id,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
        )
        manifest["target_schema_tables"] = schema_tables
        manifest["target_state_dir"] = str(target_state_dir.resolve())
        manifest["applied_at"] = utc_now()
        database = staging / "console.sqlite3"
        with sqlite3.connect(database) as connection:
            quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            imported_count = int(connection.execute("SELECT count(*) FROM jobs").fetchone()[0])
            intent_count = int(connection.execute("SELECT count(*) FROM intents").fetchone()[0])
            schedule_count = int(connection.execute("SELECT count(*) FROM monitor_schedules WHERE active = 1").fetchone()[0])
            wake_event_count = int(connection.execute("SELECT count(*) FROM wake_events").fetchone()[0])
            observation_count = int(connection.execute("SELECT count(*) FROM source_observations").fetchone()[0])
        if (
            quick_check != "ok"
            or imported_count != len(records)
            or schedule_count != len(records)
            or intent_count != 0
            or wake_event_count != 0
            or observation_count != 0
        ):
            raise RuntimeError("target ledger verification failed")
        manifest["target_verification"] = {
            "sqlite_quick_check": quick_check,
            "imported_job_count": imported_count,
            "active_monitor_schedule_count": schedule_count,
            "intent_count": intent_count,
            "wake_event_count": wake_event_count,
            "source_observation_count": observation_count,
        }
        (staging / "migration_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if target_state_dir.exists():
            if not replace_target:
                raise FileExistsError(f"target exists; pass --replace-target: {target_state_dir}")
            previous = parent / f"{target_state_dir.name}.pre-migration-{stamp()}"
            os.replace(target_state_dir, previous)
        os.replace(staging, target_state_dir)
        return {
            "target_state_dir": str(target_state_dir.resolve()),
            "previous_target_backup": str(previous.resolve()) if previous else None,
            "target_sqlite_quick_check": quick_check,
            "imported_job_count": imported_count,
            "active_monitor_schedule_count": schedule_count,
            "target_schema_tables": schema_tables,
        }
    except Exception:
        if previous is not None and previous.exists() and not target_state_dir.exists():
            os.replace(previous, target_state_dir)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state-dir", type=Path, required=True)
    parser.add_argument("--target-state-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--job-id", action="append", default=[], help="Exact source job id. Repeat for each imported job.")
    parser.add_argument("--thread-id", help="Codex research task that receives actionable wake events.")
    parser.add_argument(
        "--result-contract",
        action="append",
        default=[],
        metavar="JOB_ID=JSON_FILE",
        help="Per-job, non-secret result contract. Required exactly once per selected job.",
    )
    parser.add_argument("--reconcile-job-id", action="append", default=[], help="Reopen this selected job as unknown.")
    parser.add_argument("--lineage-job-id", action="append", default=[], help="Acknowledge this terminal job as lineage only.")
    parser.add_argument("--apply", action="store_true", help="Create archive and atomically install the target ledger.")
    parser.add_argument("--confirm-source-frozen", action="store_true")
    parser.add_argument("--replace-target", action="store_true")
    parser.add_argument("--monitor-interval-seconds", type=int, default=60)
    parser.add_argument("--monitor-timeout-seconds", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_resolved = args.source_state_dir.resolve()
    target_resolved = args.target_state_dir.resolve()
    archive_resolved = args.archive_dir.resolve()
    path_errors: list[dict[str, Any]] = []
    if target_resolved == source_resolved or target_resolved.is_relative_to(source_resolved):
        path_errors.append({"job_id": None, "code": "target_inside_source", "detail": str(target_resolved)})
    if archive_resolved == source_resolved or archive_resolved.is_relative_to(source_resolved):
        path_errors.append({"job_id": None, "code": "archive_inside_source", "detail": str(archive_resolved)})
    requested_job_ids = [value.strip() for value in args.job_id if value.strip()]
    preflight_errors: list[dict[str, Any]] = path_errors
    if not requested_job_ids:
        preflight_errors.append({"job_id": None, "code": "explicit_job_id_required", "detail": "pass --job-id for each hot-ledger job"})
    if len(requested_job_ids) != len(set(requested_job_ids)):
        preflight_errors.append({"job_id": None, "code": "duplicate_job_id", "detail": "each --job-id must be unique"})
    thread_id = str(args.thread_id or "").strip()
    if not thread_id:
        preflight_errors.append({"job_id": None, "code": "thread_id_required", "detail": "pass --thread-id"})
    result_contracts, contract_errors = load_result_contracts(args.result_contract)
    preflight_errors.extend(contract_errors)
    if args.monitor_interval_seconds <= 0:
        preflight_errors.append({"job_id": None, "code": "monitor_interval_invalid", "detail": "must be positive"})
    if args.monitor_timeout_seconds <= 0:
        preflight_errors.append({"job_id": None, "code": "monitor_timeout_invalid", "detail": "must be positive"})
    migration_id = f"{stamp()}-{uuid.uuid4().hex[:8]}"
    try:
        records, manifest = inspect_source(
            args.source_state_dir,
            requested_job_ids=requested_job_ids,
            reconcile_ids=set(args.reconcile_job_id),
            lineage_ids=set(args.lineage_job_id),
            thread_id=thread_id,
            result_contracts=result_contracts,
            contract_errors=preflight_errors,
            migration_id=migration_id,
        )
    except Exception as exc:
        print(json.dumps({"status": "invalid_source", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    manifest["mode"] = "apply" if args.apply else "dry_run"
    manifest["target_state_dir"] = str(args.target_state_dir.resolve())
    manifest["archive_dir"] = str(args.archive_dir.resolve())
    manifest["monitor_schedule"] = {
        "count": len(records),
        "thread_id": thread_id,
        "interval_seconds": args.monitor_interval_seconds,
        "timeout_seconds": args.monitor_timeout_seconds,
        "next_run_at": "migration_apply_time",
    }
    if manifest["ambiguities"]:
        manifest["status"] = "ambiguous"
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 3
    if not args.apply:
        manifest["status"] = "dry_run_ok"
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if not args.confirm_source_frozen:
        manifest["status"] = "refused"
        manifest["error"] = "--confirm-source-frozen is required with --apply"
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 4
    try:
        archive = create_readonly_archive(args.source_state_dir, args.archive_dir, migration_id)
        manifest["legacy_archive"] = archive
        target = install_target_atomically(
            args.target_state_dir,
            records,
            manifest,
            replace_target=args.replace_target,
            thread_id=thread_id,
            interval_seconds=args.monitor_interval_seconds,
            timeout_seconds=args.monitor_timeout_seconds,
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 5
    manifest["status"] = "applied"
    manifest["target"] = target
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
