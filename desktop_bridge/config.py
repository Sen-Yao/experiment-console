from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    pass


def default_state_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "Experiment Wake Bridge"


@dataclass(frozen=True)
class BridgeConfig:
    ssh_target: str
    ssh_path: str = "/usr/bin/ssh"
    identity_file: str | None = None
    ssh_config_file: str | None = None
    known_hosts_file: str | None = None
    connect_timeout_seconds: int = 10
    server_alive_interval_seconds: int = 15
    server_alive_count_max: int = 3
    command_timeout_seconds: float = 15.0
    loop_interval_seconds: float = 5.0
    poll_interval_seconds: float = 30.0
    capture_lines: int = 80
    capture_max_chars: int = 12000
    max_events_per_poll: int = 20
    max_event_records: int = 256
    app_server_timeout_seconds: float = 10.0
    status_stale_seconds: float = 120.0
    status_file: Path = field(default_factory=lambda: default_state_dir() / "status.json")
    event_file: Path = field(default_factory=lambda: default_state_dir() / "events.json")
    lock_file: Path = field(default_factory=lambda: default_state_dir() / "bridge.lock")
    codex_command: tuple[str, ...] = ("codex", "app-server", "--stdio")

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "BridgeConfig":
        source = Path(path).expanduser()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read bridge config {source}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError("bridge config must be a JSON object")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BridgeConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ConfigError(f"unknown bridge config fields: {', '.join(unknown)}")
        values = dict(raw)
        for key in ("status_file", "event_file", "lock_file"):
            if key in values:
                values[key] = Path(str(values[key])).expanduser()
        if "codex_command" in values:
            command = values["codex_command"]
            if not isinstance(command, list) or not all(
                isinstance(item, str) and item for item in command
            ):
                raise ConfigError("codex_command must be a non-empty string array")
            values["codex_command"] = tuple(command)
        try:
            result = cls(**values)
        except TypeError as exc:
            raise ConfigError(str(exc)) from exc
        result.validate()
        return result

    def validate(self) -> None:
        if not self.ssh_target or self.ssh_target.startswith("-"):
            raise ConfigError("ssh_target is required")
        positive = (
            self.connect_timeout_seconds,
            self.server_alive_interval_seconds,
            self.server_alive_count_max,
            self.command_timeout_seconds,
            self.loop_interval_seconds,
            self.poll_interval_seconds,
            self.capture_lines,
            self.capture_max_chars,
            self.max_events_per_poll,
            self.max_event_records,
            self.app_server_timeout_seconds,
            self.status_stale_seconds,
        )
        if min(positive) <= 0:
            raise ConfigError("bridge numeric settings must be positive")
        if not self.codex_command:
            raise ConfigError("codex_command cannot be empty")
        for value in (
            self.identity_file,
            self.ssh_config_file,
            self.known_hosts_file,
        ):
            if value is not None and not Path(value).expanduser().is_absolute():
                raise ConfigError("credential and SSH paths must be absolute")

    def ssh_command(self, remote_command: str) -> list[str]:
        command = [
            self.ssh_path,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout_seconds}",
            "-o",
            f"ServerAliveInterval={self.server_alive_interval_seconds}",
            "-o",
            f"ServerAliveCountMax={self.server_alive_count_max}",
            "-o",
            "StrictHostKeyChecking=yes",
        ]
        if self.identity_file:
            command.extend(
                [
                    "-o",
                    "IdentitiesOnly=yes",
                    "-i",
                    str(Path(self.identity_file).expanduser()),
                ]
            )
        if self.ssh_config_file:
            command.extend(["-F", str(Path(self.ssh_config_file).expanduser())])
        if self.known_hosts_file:
            command.extend(
                [
                    "-o",
                    f"UserKnownHostsFile={Path(self.known_hosts_file).expanduser()}",
                ]
            )
        command.extend([self.ssh_target, remote_command])
        return command

    def app_server_command(self) -> list[str]:
        return list(self.codex_command)
