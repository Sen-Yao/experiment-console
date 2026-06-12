from __future__ import annotations

from .models import JobStatus, TERMINAL_JOB_STATUSES


class InvalidTransition(ValueError):
    pass


ALLOWED_TRANSITIONS = {
    JobStatus.planned: {JobStatus.validating, JobStatus.running, JobStatus.failed, JobStatus.cancelled, JobStatus.unknown},
    JobStatus.validating: {JobStatus.running, JobStatus.failed, JobStatus.cancelled},
    JobStatus.running: {JobStatus.attention, JobStatus.finished, JobStatus.failed, JobStatus.cancelled, JobStatus.unknown},
    JobStatus.attention: {JobStatus.running, JobStatus.finished, JobStatus.failed, JobStatus.cancelled, JobStatus.unknown},
    JobStatus.unknown: {JobStatus.running, JobStatus.attention, JobStatus.finished, JobStatus.failed, JobStatus.cancelled},
    JobStatus.finished: set(),
    JobStatus.failed: set(),
    JobStatus.cancelled: set(),
}


def validate_job_transition(current: JobStatus, new: JobStatus) -> None:
    if current == new:
        return
    if current in TERMINAL_JOB_STATUSES:
        raise InvalidTransition(f"terminal job cannot transition from {current.value} to {new.value}")
    if new not in ALLOWED_TRANSITIONS.get(current, set()):
        raise InvalidTransition(f"invalid job transition from {current.value} to {new.value}")

