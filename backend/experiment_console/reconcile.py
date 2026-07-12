from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any
from uuid import uuid4

from .models import JobStatus


def derive_execution_state(sweep: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(sweep, dict):
        return {
            "lifecycle": "unknown",
            "job_status": JobStatus.unknown,
            "queue_releasable": False,
            "raw_gate": _empty_raw_gate(),
            "mismatches": [],
        }
    state = str(sweep.get("state") or "").lower()
    expected = _to_int(sweep.get("expectedRunCount"))
    raw = sweep.get("raw_run_state_counts") if isinstance(sweep.get("raw_run_state_counts"), dict) else {}
    finished = _to_int(raw.get("finished"))
    running = _to_int(raw.get("running"))
    failed = _to_int(raw.get("failed"))
    source = str(sweep.get("run_state_counts_source") or "")
    queue_releasable = bool(
        source == "wandb_runs"
        and expected > 0
        and finished == expected
        and running == 0
        and failed == 0
    )
    mismatches: list[str] = []
    top_terminal = state in {"finished", "failed", "crashed", "killed", "cancelled", "canceled"}
    if top_terminal and (running > 0 or (expected > 0 and finished + failed < expected)):
        mismatches.append("top_terminal_raw_incomplete")
    if queue_releasable and state not in {"finished"}:
        mismatches.append("raw_complete_top_nonterminal")

    if state in {"failed", "crashed", "killed"} or failed > 0:
        lifecycle = "attention"
        job_status = JobStatus.failed if state in {"failed", "crashed", "killed"} else JobStatus.attention
    elif state in {"cancelled", "canceled"}:
        lifecycle = "cancelled"
        job_status = JobStatus.cancelled
    elif queue_releasable and state == "finished":
        lifecycle = "finished"
        job_status = JobStatus.finished
    elif state == "finished" or queue_releasable:
        lifecycle = "finalizing"
        job_status = JobStatus.finalizing
    elif running > 0 or state in {"running", "pending"}:
        lifecycle = "running"
        job_status = JobStatus.running
    else:
        lifecycle = "unknown"
        job_status = JobStatus.unknown
    return {
        "lifecycle": lifecycle,
        "job_status": job_status,
        "queue_releasable": queue_releasable,
        "raw_gate": {
            "expected": expected,
            "finished": finished,
            "running": running,
            "failed": failed,
            "source": source,
            "satisfied": queue_releasable,
        },
        "mismatches": mismatches,
    }


def derive_result_state(
    execution: dict[str, Any],
    result_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest = result_snapshot.get("artifact_manifest") if isinstance(result_snapshot, dict) and isinstance(result_snapshot.get("artifact_manifest"), dict) else {}
    artifact_ready = bool(manifest.get("protocol_valid"))
    return {
        "artifact_manifest_ready": artifact_ready,
        "result_ready": bool(execution.get("queue_releasable") and artifact_ready),
        "artifact_manifest": manifest or None,
    }


def update_sync_consistency(
    previous: dict[str, Any] | None,
    mismatches: list[str],
    *,
    observed_at: str,
    consecutive_threshold: int,
    grace_seconds: int,
) -> dict[str, Any]:
    signature = ",".join(sorted(set(mismatches)))
    if not signature:
        return {
            "classification": "consistent",
            "signature": None,
            "consecutive": 0,
            "first_seen_at": None,
            "episode_id": None,
            "observed_at": observed_at,
        }
    previous = previous if isinstance(previous, dict) else {}
    same = previous.get("signature") == signature
    first_seen_at = str(previous.get("first_seen_at") or observed_at) if same else observed_at
    episode_id = str(previous.get("episode_id") or f"episode_{uuid4().hex}") if same else f"episode_{uuid4().hex}"
    consecutive = _to_int(previous.get("consecutive")) + 1 if same else 1
    age_seconds = max(0, int((_parse_time(observed_at) - _parse_time(first_seen_at)).total_seconds()))
    is_error = consecutive >= max(1, consecutive_threshold) and age_seconds >= max(0, grace_seconds)
    return {
        "classification": "sync_error" if is_error else "reconciling",
        "signature": signature,
        "mismatches": sorted(set(mismatches)),
        "consecutive": consecutive,
        "first_seen_at": first_seen_at,
        "episode_id": episode_id,
        "observed_at": observed_at,
        "age_seconds": age_seconds,
    }


def build_artifact_manifest(
    *,
    source: str | None,
    expected_runs: int | None,
    rows: list[dict[str, Any]],
    raw_artifacts: list[dict[str, Any]],
    discovery_sources: dict[str, Any],
    discovery_mode: str,
    complete: bool,
    truncated: bool,
    missing_runs: int,
    failed_runs: int,
    generated_at: str,
) -> dict[str, Any]:
    run_ids = sorted({
        str(row.get("run_id"))
        for row in rows
        if isinstance(row, dict) and row.get("run_id") and row.get("has_scientific_result")
    })
    inventory: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in raw_artifacts:
        if not isinstance(artifact, dict):
            continue
        path = str(artifact.get("path") or "")
        content = artifact.get("content")
        if not path or not isinstance(content, (dict, list)):
            continue
        content_text = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        content_sha = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
        linked_run_id = None
        declared_run_id = str(artifact.get("run_id") or "")
        if discovery_mode == "run_id_output_globs_v1" and declared_run_id in run_ids:
            linked_run_id = declared_run_id
        for run_id in run_ids:
            if run_id in path:
                linked_run_id = run_id
                break
        if not linked_run_id and isinstance(content, dict):
            candidate = content.get("run_id") or content.get("wandb_run_id") or content.get("wandb.run.id")
            if candidate and str(candidate) in run_ids:
                linked_run_id = str(candidate)
        inventory[(path, content_sha)] = {
            "path": path,
            "basename": path.rsplit("/", 1)[-1],
            "sha256": content_sha,
            "linked_run_id": linked_run_id,
            "valid_json": bool(artifact.get("valid_json", isinstance(content, (dict, list)) and bool(content))),
        }
    if not inventory:
        for run_id, discovery in discovery_sources.items():
            if not isinstance(discovery, dict):
                continue
            for path_value in discovery.get("selected_paths") or []:
                path = str(path_value or "")
                if not path:
                    continue
                linked_run_id = str(run_id) if str(run_id) in path else None
                inventory[(path, "unavailable")] = {
                    "path": path,
                    "basename": path.rsplit("/", 1)[-1],
                    "sha256": None,
                    "linked_run_id": linked_run_id,
                    "valid_json": None,
                }
    inventory_rows = sorted(inventory.values(), key=lambda item: (item["path"], item.get("sha256") or ""))
    linked_run_ids = sorted({str(item["linked_run_id"]) for item in inventory_rows if item.get("linked_run_id")})
    digest = hashlib.sha256(json.dumps(inventory_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    expected = _to_int(expected_runs)
    remote_source = str(source or "") in {"remote_local_files", "remote_single_run_files"}
    protocol_valid = bool(
        remote_source
        and expected > 0
        and len(run_ids) == expected
        and len(inventory_rows) == expected
        and len(linked_run_ids) == expected
        and all(item.get("linked_run_id") for item in inventory_rows)
        and all(item.get("valid_json") is True and item.get("sha256") for item in inventory_rows)
        and complete
        and not truncated
        and _to_int(missing_runs) == 0
        and _to_int(failed_runs) == 0
    )
    return {
        "version": 1,
        "source": source,
        "expected_artifacts": expected,
        "distinct_final_artifacts": len(inventory_rows),
        "run_ids": run_ids,
        "linked_run_ids": linked_run_ids,
        "artifact_inventory": inventory_rows,
        "manifest_sha256": digest,
        "protocol_valid": protocol_valid,
        "generated_at": generated_at,
    }


def _empty_raw_gate() -> dict[str, Any]:
    return {"expected": 0, "finished": 0, "running": 0, "failed": 0, "source": None, "satisfied": False}


def _to_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except Exception:
        return 0


def _parse_time(value: str) -> datetime:
    text = str(value or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
