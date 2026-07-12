from __future__ import annotations

import json
import importlib.util
import os
import sqlite3
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from experiment_console.models import JobRecord, JobStatus
from experiment_console.store import ConsoleStore

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "migrate_runtime_to_yggdrasil.py"
ELLIPTIC_CONTRACT = ROOT / "deploy" / "yggdrasil" / "result-contracts" / "cjnv-a3-elliptic-training.json"
AMAZON_CONTRACT = ROOT / "deploy" / "yggdrasil" / "result-contracts" / "cjnv-a3-amazon-training.json"
MIGRATION_SPEC = importlib.util.spec_from_file_location("yggdrasil_migration_test_module", SCRIPT)
migration = importlib.util.module_from_spec(MIGRATION_SPEC)
assert MIGRATION_SPEC and MIGRATION_SPEC.loader
MIGRATION_SPEC.loader.exec_module(migration)


@pytest.fixture
def legacy_runtime(tmp_path: Path) -> tuple[Path, str, str, str]:
    source = tmp_path / "legacy"
    store = ConsoleStore(source / "console.sqlite3", source / "audit.jsonl")
    active_id = "job_explicit_elliptic"
    queued_id = "job_explicit_amazon"
    unselected_id = "job_unselected_cjnv"
    store.upsert_job(JobRecord(
        job_id=active_id,
        name="elliptic training",
        status=JobStatus.finished,
        entity="HCCS",
        project="DualRefGAD",
        sweep_id="sweep-active",
        config_path="/remote/elliptic.yaml",
        remote_host="HCCS-25",
        remote_cwd="/remote/project",
        conda_env="DualRefGAD",
        agent_pids=["101"],
        monitor={
            "kind": "sweep",
            "cron": {"owner": "Hermes", "active": True},
            "notify": {"channel": "OpenClaw"},
            "watchdog": {"expected_total": 25, "heartbeat": "10m"},
            "queue": {"queue_group": "HCCS-25:/remote/project", "queue_policy": "immediate"},
        },
        created_at="2026-07-11T23:00:00+00:00",
        updated_at="2026-07-12T00:00:00+00:00",
    ))
    store.upsert_job(JobRecord(
        job_id=queued_id,
        name="amazon training",
        status=JobStatus.queued,
        entity="HCCS",
        project="DualRefGAD",
        config_path="/remote/amazon.yaml",
        remote_host="HCCS-25",
        remote_cwd="/remote/project",
        conda_env="DualRefGAD",
        monitor={
            "kind": "sweep",
            "queue": {
                "queue_group": "HCCS-25:/remote/project",
                "queue_policy": "sequential",
                "blocked_by_job_id": active_id,
                "queue_after_job_id": active_id,
                "queued_at": "2026-07-11T22:00:00+00:00",
                "payload": {
                    "job_name": "amazon training",
                    "config_path": "/remote/amazon.yaml",
                    "remote_host": "HCCS-25",
                    "remote_cwd": "/remote/project",
                    "entity": "HCCS",
                    "project": "DualRefGAD",
                    "profile": "sweep",
                    "queue_policy": "sequential",
                    "queue_after_job_id": active_id,
                    "idempotency_key": "legacy-operation",
                },
            },
        },
        created_at="2026-07-11T22:00:00+00:00",
        updated_at="2026-07-11T22:01:00+00:00",
    ))
    store.upsert_job(JobRecord(
        job_id=unselected_id,
        name="cjnv but not allow-listed",
        status=JobStatus.attention,
        remote_host="HCCS-25",
        remote_cwd="/remote/project",
    ))
    (source / "audit.jsonl").write_text("legacy-audit-must-not-enter-hot-state\n", encoding="utf-8")
    results = source / "results"
    results.mkdir()
    (results / "legacy-result.json").write_text('{"legacy":true}\n', encoding="utf-8")
    return source, active_id, queued_id, unselected_id


def command(source: Path, target: Path, archive: Path, active_id: str, queued_id: str) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--source-state-dir", str(source),
        "--target-state-dir", str(target),
        "--archive-dir", str(archive),
        "--thread-id", "thread-cjnv",
        "--job-id", active_id,
        "--reconcile-job-id", active_id,
        "--result-contract", f"{active_id}={ELLIPTIC_CONTRACT}",
        "--job-id", queued_id,
        "--result-contract", f"{queued_id}={AMAZON_CONTRACT}",
    ]


def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True, check=False)


def test_dry_run_is_explicit_allowlist_and_writes_nothing(legacy_runtime, tmp_path: Path):
    source, active_id, queued_id, unselected_id = legacy_runtime
    target = tmp_path / "target"
    archive = tmp_path / "archive"
    result = run(command(source, target, archive, active_id, queued_id))
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["status"] == "dry_run_ok"
    assert manifest["requested_job_ids"] == [active_id, queued_id]
    assert {item["job_id"] for item in manifest["selected_jobs"]} == {active_id, queued_id}
    assert unselected_id not in json.dumps(manifest)
    assert manifest["hot_state_policy"]["selection_is_explicit_allowlist"] is True
    assert not target.exists()
    assert not archive.exists()


def test_terminal_job_requires_explicit_reconcile_or_lineage(legacy_runtime, tmp_path: Path):
    source, active_id, queued_id, _ = legacy_runtime
    args = command(source, tmp_path / "target", tmp_path / "archive", active_id, queued_id)
    index = args.index("--reconcile-job-id")
    del args[index:index + 2]
    result = run(args)
    assert result.returncode == 3
    errors = json.loads(result.stdout)["ambiguities"]
    assert any(item["job_id"] == active_id and item["code"] == "selected_terminal_requires_classification" for item in errors)


def test_result_contract_is_required_and_validated_by_backend_model(legacy_runtime, tmp_path: Path):
    source, active_id, queued_id, _ = legacy_runtime
    args = command(source, tmp_path / "target", tmp_path / "archive", active_id, queued_id)
    contract_index = args.index(f"{queued_id}={AMAZON_CONTRACT}")
    del args[contract_index - 1:contract_index + 1]
    result = run(args)
    assert result.returncode == 3
    assert any(item["code"] == "result_contract_required" for item in json.loads(result.stdout)["ambiguities"])

    invalid = tmp_path / "invalid-contract.json"
    invalid.write_text(json.dumps({
        "version": 1,
        "expected_runs": 25,
        "max_runs": 24,
        "output_globs": ["outputs/no-run-id.json"],
        "discovery_mode": "run_id_output_globs_v1",
        "allow_partial": False,
        "export_artifacts": True,
    }), encoding="utf-8")
    args = command(source, tmp_path / "target2", tmp_path / "archive2", active_id, queued_id)
    args[args.index(f"{queued_id}={AMAZON_CONTRACT}")] = f"{queued_id}={invalid}"
    result = run(args)
    assert result.returncode == 3
    assert any(item["code"] == "invalid_result_contract" for item in json.loads(result.stdout)["ambiguities"])


def test_apply_archives_all_but_imports_clean_jobs_and_schedules(legacy_runtime, tmp_path: Path):
    source, active_id, queued_id, _ = legacy_runtime
    target = tmp_path / "target"
    archive_dir = tmp_path / "archive"
    args = command(source, target, archive_dir, active_id, queued_id) + ["--apply", "--confirm-source-frozen"]
    result = run(args)
    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout)
    assert output["target"]["imported_job_count"] == 2
    assert output["target"]["active_monitor_schedule_count"] == 2

    with sqlite3.connect(target / "console.sqlite3") as connection:
        connection.row_factory = sqlite3.Row
        rows = {row["job_id"]: row for row in connection.execute("SELECT * FROM jobs")}
        assert set(rows) == {active_id, queued_id}
        assert rows[active_id]["status"] == "unknown"
        assert rows[queued_id]["status"] == "queued"
        assert json.loads(rows[active_id]["operation_log_json"]) == []
        active_monitor = json.loads(rows[active_id]["monitor_json"])
        queued_monitor = json.loads(rows[queued_id]["monitor_json"])
        assert active_monitor["result_contract"]["expected_runs"] == 25
        assert "cron" not in json.dumps(active_monitor).lower()
        assert "hermes" not in json.dumps(active_monitor).lower()
        assert "openclaw" not in json.dumps(active_monitor).lower()
        queue = queued_monitor["queue"]
        assert queue["queue_group"] == "HCCS-25:/remote/project"
        assert queue["queue_policy"] == "sequential"
        assert queue["payload"]["job_name"] == "amazon training"
        assert queue["payload"]["queue_after_job_id"] is None
        assert "blocked_by_job_id" not in queue
        assert queued_monitor["migration"]["authorized_queued_job"] is True
        schedules = connection.execute("SELECT * FROM monitor_schedules ORDER BY job_id").fetchall()
        assert len(schedules) == 2
        assert {row["thread_id"] for row in schedules} == {"thread-cjnv"}
        assert all(row["active"] == 1 for row in schedules)
        assert connection.execute("SELECT count(*) FROM wake_events").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM source_observations").fetchone()[0] == 0

    assert not (target / "results" / "legacy-result.json").exists()
    assert not (target / "audit.jsonl").exists() or "legacy-audit" not in (target / "audit.jsonl").read_text(encoding="utf-8")
    archive_path = Path(output["legacy_archive"]["archive"])
    assert stat.S_IMODE(archive_path.stat().st_mode) == 0o400
    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert "runtime/audit.jsonl" in names
    assert "runtime/results/legacy-result.json" in names
    assert "consistent/console.sqlite3" in names
    manifest = json.loads((target / "migration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["monitor_schedule"]["count"] == 2
    assert manifest["hot_state_policy"]["legacy_audit_imported"] is False


def test_source_change_during_archive_is_rejected(legacy_runtime, tmp_path: Path, monkeypatch):
    source, *_ = legacy_runtime
    calls = 0

    def changing_fingerprint(_source):
        nonlocal calls
        calls += 1
        return {"console.sqlite3": {"size": calls, "mtime_ns": calls}}

    monkeypatch.setattr(migration, "source_fingerprints", changing_fingerprint)
    with pytest.raises(RuntimeError, match="changed during archive"):
        migration.create_readonly_archive(source, tmp_path / "archives", "change-test")
    assert not list((tmp_path / "archives").glob("*.tar.gz"))


def test_target_atomic_replace_restores_previous_on_install_failure(tmp_path: Path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("original", encoding="utf-8")

    def fake_initialize(stage: Path, records, **_kwargs):
        with sqlite3.connect(stage / "console.sqlite3") as connection:
            connection.execute("CREATE TABLE jobs(job_id TEXT)")
            connection.execute("CREATE TABLE intents(intent_id TEXT)")
            connection.execute("CREATE TABLE monitor_schedules(active INTEGER)")
            connection.execute("CREATE TABLE wake_events(event_id TEXT)")
            connection.execute("CREATE TABLE source_observations(job_id TEXT)")
            connection.execute("INSERT INTO jobs VALUES ('job')")
            connection.execute("INSERT INTO monitor_schedules VALUES (1)")
        return ["intents", "jobs", "monitor_schedules", "source_observations", "wake_events"]

    monkeypatch.setattr(migration, "initialize_target_with_application", fake_initialize)
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(migration.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected"):
        migration.install_target_atomically(
            target,
            [{"job_id": "job"}],
            {"migration_id": "atomic-test"},
            replace_target=True,
            thread_id="thread",
            interval_seconds=60,
            timeout_seconds=300,
        )
    assert marker.read_text(encoding="utf-8") == "original"
