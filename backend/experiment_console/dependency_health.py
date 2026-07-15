from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from .agent_reconcile import next_reconcile_at, retry_delay_seconds
from .models import new_id
from .redaction import redact_text


DEPENDENCY_HEALTH_VERSION = 2
EXTERNAL_ATTENTION_MIN_ATTEMPTS = 3
EXTERNAL_ATTENTION_GRACE_SECONDS = 15 * 60
ATTENTION_RETRY_SECONDS = 10 * 60
MAX_ERROR_FINGERPRINTS = 5


def dependency_key(*, source: str, scope: str, operation_stage: str) -> str:
    identity = f"{source}\0{scope}\0{operation_stage}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"dependency_{source}_{digest}"


def error_identity(exc: Exception) -> tuple[str, str, str]:
    error_type = exc.__class__.__name__
    summary = redact_text(str(exc)).strip()[:2000] or error_type
    fingerprint = hashlib.sha256(f"{error_type}:{summary}".encode("utf-8")).hexdigest()
    return error_type, fingerprint, summary


def advance_dependency_episode(
    current: dict[str, Any] | None,
    *,
    source: str,
    scope: str,
    operation_stage: str,
    classification: str,
    exc: Exception,
    retry_active: bool,
    action_required_immediately: bool = False,
    now: str,
) -> dict[str, Any]:
    key = dependency_key(source=source, scope=scope, operation_stage=operation_stage)
    same_episode = isinstance(current, dict) and current.get("dependency_key") == key
    error_type, fingerprint, summary = error_identity(exc)
    attempts = int((current or {}).get("attempts") or 0) + 1 if same_episode else 1
    first_seen_at = str((current or {}).get("first_seen_at") or now) if same_episode else now
    fingerprints = list((current or {}).get("error_fingerprints") or []) if same_episode else []
    if fingerprint not in fingerprints:
        fingerprints.append(fingerprint)
    fingerprints = fingerprints[-MAX_ERROR_FINGERPRINTS:]
    age_seconds = _age_seconds(first_seen_at, now)
    action_required = bool(
        action_required_immediately
        or (
            retry_active
            and attempts >= EXTERNAL_ATTENTION_MIN_ATTEMPTS
            and age_seconds >= EXTERNAL_ATTENTION_GRACE_SECONDS
        )
    )
    retry_seconds = ATTENTION_RETRY_SECONDS if action_required else retry_delay_seconds(attempts)
    next_retry_at = next_reconcile_at(now=now, delay_seconds=retry_seconds) if retry_active else None
    return {
        "version": DEPENDENCY_HEALTH_VERSION,
        "episode_id": str((current or {}).get("episode_id") or new_id("dependency_episode", source)) if same_episode else new_id("dependency_episode", source),
        "dependency_key": key,
        "source": source,
        "scope": scope,
        "operation_stage": operation_stage,
        "classification": classification,
        "lifecycle": "attention" if action_required else "reconciling" if retry_active else "paused",
        "action_required": action_required,
        "auto_retry_active": retry_active,
        "attempts": attempts,
        "first_seen_at": first_seen_at,
        "last_seen_at": now,
        "age_seconds": age_seconds,
        "next_retry_at": next_retry_at,
        "error_type": error_type,
        "error_summary": summary,
        "error_fingerprint": fingerprint,
        "error_fingerprints": fingerprints,
        "updated_at": now,
    }


def dependency_retry_due(episode: dict[str, Any] | None, *, now: str) -> bool:
    if not isinstance(episode, dict):
        return True
    if not episode.get("auto_retry_active"):
        return False
    next_retry = episode.get("next_retry_at")
    if not next_retry:
        return False
    return _parse_timestamp(str(next_retry)) <= _parse_timestamp(now)


def resolved_failure_summary(episode: dict[str, Any], *, resolved_at: str) -> dict[str, Any]:
    return {
        "episode_id": episode.get("episode_id"),
        "dependency_key": episode.get("dependency_key"),
        "source": episode.get("source"),
        "scope": episode.get("scope"),
        "operation_stage": episode.get("operation_stage"),
        "classification": episode.get("classification"),
        "attempts": episode.get("attempts"),
        "first_seen_at": episode.get("first_seen_at"),
        "last_seen_at": episode.get("last_seen_at"),
        "resolved_at": resolved_at,
        "error_type": episode.get("error_type"),
        "error_fingerprint": episode.get("error_fingerprint"),
    }


def _age_seconds(first_seen_at: str, now: str) -> int:
    return max(0, int((_parse_timestamp(now) - _parse_timestamp(first_seen_at)).total_seconds()))


def _parse_timestamp(value: str) -> datetime:
    text = str(value or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
