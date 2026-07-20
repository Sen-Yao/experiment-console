from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

from .command import CommandFailed, CommandRunner
from .config import ServerProfile, Settings
from .models import FileChunk, JobRecord, LogChunk, RemoteObservation


INSTALL_HELPER = r"""
import hashlib, os, pathlib, sys, tempfile
target = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
source = sys.stdin.buffer.read()
if hashlib.sha256(source).hexdigest() != expected:
    raise SystemExit("helper digest mismatch")
target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
current = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
if current != expected:
    fd, temporary = tempfile.mkstemp(prefix=".remote-helper.", dir=target.parent)
    try:
        os.fchmod(fd, 0o500)
        with os.fdopen(fd, "wb") as handle:
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
""".strip()


class RemoteError(RuntimeError):
    pass


class RemoteExecutor:
    def __init__(self, settings: Settings, runner: CommandRunner | None = None) -> None:
        self.settings = settings
        self.runner = runner or CommandRunner()
        self._installed: set[tuple[str, str]] = set()
        helper_path = Path(__file__).with_name("remote_helper.py")
        self.helper_source = helper_path.read_text(encoding="utf-8")
        self.helper_digest = hashlib.sha256(self.helper_source.encode()).hexdigest()

    def _ssh_base(self, profile: ServerProfile) -> list[str]:
        command = [
            self.settings.ssh_path,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            f"ConnectTimeout={self.settings.ssh_timeout_seconds}",
            "-o",
            "StrictHostKeyChecking=yes",
        ]
        if self.settings.ssh_config_file:
            command.extend(["-F", str(self.settings.ssh_config_file)])
        if self.settings.ssh_key_file:
            command.extend(["-i", str(self.settings.ssh_key_file)])
        if self.settings.known_hosts_file:
            command.extend(
                ["-o", f"UserKnownHostsFile={self.settings.known_hosts_file}"]
            )
        command.append(profile.ssh_target)
        return command

    @staticmethod
    def _helper_path(profile: ServerProfile) -> str:
        return f"{profile.state_root.rstrip('/')}/remote_helper.py"

    def ensure_helper(self, profile_name: str, profile: ServerProfile) -> None:
        key = (profile_name, self.helper_digest)
        if key in self._installed:
            return
        remote = shlex.join(
            [
                profile.remote_python,
                "-c",
                INSTALL_HELPER,
                self._helper_path(profile),
                self.helper_digest,
            ]
        )
        try:
            self.runner.run(
                [*self._ssh_base(profile), remote],
                timeout=self.settings.command_timeout_seconds,
                input_text=self.helper_source,
            )
        except (CommandFailed, OSError) as exc:
            raise RemoteError(f"cannot install remote helper for {profile_name}: {exc}") from exc
        self._installed.add(key)

    def call(
        self,
        profile_name: str,
        profile: ServerProfile,
        action: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.ensure_helper(profile_name, profile)
        remote = shlex.join(
            [profile.remote_python, self._helper_path(profile), action]
        )
        try:
            result = self.runner.run(
                [*self._ssh_base(profile), remote],
                timeout=timeout or self.settings.command_timeout_seconds,
                input_text=json.dumps(payload, separators=(",", ":")),
            )
        except (CommandFailed, OSError) as exc:
            raise RemoteError(str(exc)) from exc
        try:
            decoded = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RemoteError(
                f"remote helper returned invalid JSON: {result.stdout[-1000:]}"
            ) from exc
        if not isinstance(decoded, dict):
            raise RemoteError("remote helper returned a non-object response")
        return decoded

    def resources(self, profile_name: str, profile: ServerProfile) -> dict[str, Any]:
        return self.call(
            profile_name,
            profile,
            "resources",
            {
                "gpu_query_argv": profile.gpu_query_argv,
                "gpu_min_free_mb": profile.gpu_min_free_mb,
                "gpu_max_utilization": profile.gpu_max_utilization,
            },
        )

    @staticmethod
    def _identity_payload(job: JobRecord, profile: ServerProfile) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "state_root": profile.state_root,
            "command_digest": job.request_hash,
        }

    def launch(self, job: JobRecord, profile: ServerProfile) -> RemoteObservation:
        payload = self._identity_payload(job, profile)
        payload["request"] = {
            "request_hash": job.request_hash,
            "command_digest": job.request_hash,
            "cwd": job.cwd,
            "allowed_roots": profile.allowed_roots,
            "argv": job.argv,
            "env": job.env,
            "gpu_indices": job.gpu_indices,
            "total_runs": job.total_runs,
            "bootstrap_argv": profile.bootstrap_argv,
        }
        return RemoteObservation.model_validate(
            self.call(job.profile, profile, "launch", payload)
        )

    def inspect(self, job: JobRecord, profile: ServerProfile) -> RemoteObservation:
        return RemoteObservation.model_validate(
            self.call(job.profile, profile, "inspect", self._identity_payload(job, profile))
        )

    def cancel(self, job: JobRecord, profile: ServerProfile) -> RemoteObservation:
        payload = self._identity_payload(job, profile)
        payload.update({"grace_seconds": self.settings.cancel_grace_seconds})
        return RemoteObservation.model_validate(
            self.call(
                job.profile,
                profile,
                "cancel",
                payload,
                timeout=(
                    self.settings.cancel_grace_seconds
                    + self.settings.ssh_timeout_seconds
                    + 5
                ),
            )
        )

    def logs(
        self,
        job: JobRecord,
        profile: ServerProfile,
        *,
        stream: str,
        offset: int,
        limit: int,
        tail: bool = False,
    ) -> LogChunk:
        payload = self._identity_payload(job, profile)
        payload.update(
            {"stream": stream, "offset": offset, "limit": limit, "tail": tail}
        )
        return LogChunk.model_validate(
            self.call(job.profile, profile, "logs", payload)
        )

    def fetch(
        self,
        job: JobRecord,
        profile: ServerProfile,
        *,
        path: str,
        offset: int,
        limit: int,
    ) -> FileChunk:
        payload = self._identity_payload(job, profile)
        payload.update({"path": path, "offset": offset, "limit": limit})
        return FileChunk.model_validate(
            self.call(job.profile, profile, "fetch", payload)
        )
