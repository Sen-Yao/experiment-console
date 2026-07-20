from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


class ConfigError(ValueError):
    pass


def default_state_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "Experiment Console Bridge v3"


@dataclass(frozen=True)
class BridgeConfig:
    ssh_target: str
    console_url: str = "http://127.0.0.1:5174"
    expected_instance_id: str = "yggdrasil-production-v3"
    consumer_id: str = field(
        default_factory=lambda: f"codex-{socket.gethostname().split('.', 1)[0]}"
    )
    local_host: str = "127.0.0.1"
    local_port: int = 5174
    remote_host: str = "127.0.0.1"
    remote_port: int = 5174
    ssh_path: str = "/usr/bin/ssh"
    identity_file: str | None = None
    ssh_config_file: str | None = None
    known_hosts_file: str | None = None
    console_token_file: str | None = None
    connect_timeout_seconds: int = 10
    server_alive_interval_seconds: int = 15
    server_alive_count_max: int = 3
    tunnel_probe_failures_before_restart: int = 3
    reconnect_initial_seconds: float = 1.0
    reconnect_max_seconds: float = 60.0
    loop_interval_seconds: float = 2.0
    poll_interval_seconds: float = 30.0
    poll_limit: int = 20
    lease_seconds: int = 120
    http_timeout_seconds: float = 5.0
    sleep_jump_seconds: float = 90.0
    app_server_timeout_seconds: float = 10.0
    status_stale_seconds: float = 120.0
    status_file: Path = field(default_factory=lambda: default_state_dir() / "status.json")
    lock_file: Path = field(default_factory=lambda: default_state_dir() / "bridge.lock")
    codex_command: tuple[str, ...] = ("codex", "app-server", "proxy")
    codex_socket: str | None = None

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
        for key in ("status_file", "lock_file"):
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
        parsed = urlsplit(self.console_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ConfigError("console_url must be loopback HTTP")
        if parsed.port != self.local_port:
            raise ConfigError("console_url port must match local_port")
        if not self.ssh_target or self.ssh_target.startswith("-"):
            raise ConfigError("ssh_target is required")
        if not self.expected_instance_id:
            raise ConfigError("expected_instance_id is required")
        if self.local_host not in {"127.0.0.1", "localhost", "::1"}:
            raise ConfigError("local_host must be loopback")
        positive = (
            self.connect_timeout_seconds,
            self.server_alive_interval_seconds,
            self.server_alive_count_max,
            self.tunnel_probe_failures_before_restart,
            self.reconnect_initial_seconds,
            self.reconnect_max_seconds,
            self.loop_interval_seconds,
            self.poll_interval_seconds,
            self.poll_limit,
            self.lease_seconds,
            self.http_timeout_seconds,
            self.sleep_jump_seconds,
            self.app_server_timeout_seconds,
            self.status_stale_seconds,
        )
        if min(positive) <= 0:
            raise ConfigError("bridge numeric settings must be positive")
        if self.reconnect_initial_seconds > self.reconnect_max_seconds:
            raise ConfigError("initial reconnect delay cannot exceed maximum delay")
        if not (
            1 <= self.local_port <= 65535 and 1 <= self.remote_port <= 65535
        ):
            raise ConfigError("bridge ports must be valid")
        if not self.codex_command:
            raise ConfigError("codex_command cannot be empty")
        for value in (
            self.identity_file,
            self.ssh_config_file,
            self.known_hosts_file,
            self.console_token_file,
        ):
            if value is not None and not Path(value).expanduser().is_absolute():
                raise ConfigError("credential and SSH paths must be absolute")

    def ssh_command(self) -> list[str]:
        forward = (
            f"{self.local_host}:{self.local_port}:"
            f"{self.remote_host}:{self.remote_port}"
        )
        command = [
            self.ssh_path,
            "-N",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ExitOnForwardFailure=yes",
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
            command.extend(["-i", str(Path(self.identity_file).expanduser())])
        if self.ssh_config_file:
            command.extend(["-F", str(Path(self.ssh_config_file).expanduser())])
        if self.known_hosts_file:
            command.extend(
                [
                    "-o",
                    f"UserKnownHostsFile={Path(self.known_hosts_file).expanduser()}",
                ]
            )
        command.extend(["-L", forward, self.ssh_target])
        return command

    def app_server_command(self) -> list[str]:
        result = list(self.codex_command)
        if self.codex_socket:
            result.extend(["--sock", str(Path(self.codex_socket).expanduser())])
        return result
