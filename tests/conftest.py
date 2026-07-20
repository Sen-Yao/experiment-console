from __future__ import annotations

import base64
from pathlib import Path

import pytest

from experiment_console.config import ServerProfile, Settings
from experiment_console.models import FileChunk, LogChunk, RemoteObservation, utc_now
from experiment_console.service import ConsoleService


class FakeRemote:
    def __init__(self) -> None:
        self.observations: dict[str, RemoteObservation] = {}
        self.cancelled: list[str] = []
        self.log_text = "line one\nline two\n"
        self.file_data = b"result-data"

    def resources(self, profile_name, profile):
        return {
            "observed_at": utc_now(),
            "gpus": [
                {
                    "index": 0,
                    "name": "GPU-0",
                    "memory_total_mb": 24000,
                    "memory_used_mb": 100,
                    "memory_free_mb": 23900,
                    "utilization": 0,
                    "available": True,
                },
                {
                    "index": 1,
                    "name": "GPU-1",
                    "memory_total_mb": 24000,
                    "memory_used_mb": 23000,
                    "memory_free_mb": 1000,
                    "utilization": 95,
                    "available": False,
                },
            ],
        }

    def launch(self, job, profile):
        observation = RemoteObservation(
            state="running",
            observed_at=utc_now(),
            pid=1000 + len(self.observations),
            pgid=1000 + len(self.observations),
            start_ticks="12345",
            total_runs=job.total_runs,
            completed_runs=0,
        )
        self.observations[job.job_id] = observation
        return observation

    def inspect(self, job, profile):
        return self.observations[job.job_id]

    def cancel(self, job, profile):
        self.cancelled.append(job.job_id)
        observation = RemoteObservation(
            state="cancelled",
            observed_at=utc_now(),
            pid=job.remote_pid,
            pgid=job.remote_pgid,
            start_ticks=job.remote_start_ticks,
            exit_code=-15,
        )
        self.observations[job.job_id] = observation
        return observation

    def logs(self, job, profile, *, stream, offset, limit, tail=False):
        data = self.log_text
        if tail:
            offset = max(0, len(data) - limit)
        text = data[offset : offset + limit]
        return LogChunk(
            stream=stream,
            offset=offset,
            next_offset=offset + len(text),
            eof=offset + len(text) >= len(data),
            text=text,
        )

    def fetch(self, job, profile, *, path, offset, limit):
        data = self.file_data[offset : offset + limit]
        return FileChunk(
            path=f"{job.cwd}/{path}",
            offset=offset,
            next_offset=offset + len(data),
            eof=offset + len(data) >= len(self.file_data),
            size=len(self.file_data),
            data_base64=base64.b64encode(data).decode(),
        )


@pytest.fixture
def profile() -> ServerProfile:
    return ServerProfile(
        ssh_target="test-host",
        allowed_roots=["/work"],
        state_root="/state/experiment-console-v3",
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        state_dir=tmp_path / "state",
        profiles_path=tmp_path / "unused.json",
        instance_id="test-console-v3",
        monitor_enabled=False,
    )


@pytest.fixture
def fake_remote() -> FakeRemote:
    return FakeRemote()


@pytest.fixture
def service(settings, profile, fake_remote) -> ConsoleService:
    return ConsoleService(settings, remote=fake_remote, profiles={"test": profile})
