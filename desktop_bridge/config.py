from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


class ConfigError(ValueError):
    """Raised when bridge configuration is unsafe or incomplete."""


def _default_state_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "Experiment Console Bridge"


def _default_consumer_id() -> str:
    return f"codex-desktop-{socket.gethostname().split('.', 1)[0]}"


@dataclass(frozen=True)
class BridgeConfig:
    ssh_target: str
    console_url: str = "http://127.0.0.1:5174"
    consumer_id: str = field(default_factory=_default_consumer_id)
    expected_authority_role: str = "authoritative"
    expected_instance_id: str = "yggdrasil-production"
    local_host: str = "127.0.0.1"
    local_port: int = 5174
    remote_host: str = "127.0.0.1"
    remote_port: int = 5174
    ssh_path: str = "/usr/bin/ssh"
    identity_file: str | None = None
    ssh_config_file: str | None = None
    known_hosts_file: str | None = None
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
    state_file: Path = field(default_factory=lambda: _default_state_dir() / "state.json")
    status_file: Path = field(default_factory=lambda: _default_state_dir() / "status.json")
    lock_file: Path = field(default_factory=lambda: _default_state_dir() / "bridge.lock")
    acked_event_retention_seconds: int = 30 * 24 * 60 * 60
    max_acked_event_ids: int = 5000
    codex_command: tuple[str, ...] = ("codex", "app-server", "proxy")
    codex_socket: str | None = None
    app_server_timeout_seconds: float = 10.0
    inflight_retry_grace_seconds: float = 15.0
    max_handoff_bytes: int = 64 * 1024
    status_stale_seconds: float = 120.0
    console_token_file: str | None = None

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "BridgeConfig":
        config_path = Path(path).expanduser()
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"bridge config does not exist: {config_path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read bridge config {config_path}: {exc}") from exc
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
        if "state_file" in values:
            values["state_file"] = Path(str(values["state_file"])).expanduser()
        if "status_file" in values:
            values["status_file"] = Path(str(values["status_file"])).expanduser()
        if "lock_file" in values:
            values["lock_file"] = Path(str(values["lock_file"])).expanduser()
        if "codex_command" in values:
            command = values["codex_command"]
            if not isinstance(command, list) or not all(isinstance(arg, str) and arg for arg in command):
                raise ConfigError("codex_command must be a non-empty JSON string array")
            values["codex_command"] = tuple(command)
        try:
            config = cls(**values)
        except TypeError as exc:
            raise ConfigError(str(exc)) from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not self.ssh_target or self.ssh_target.startswith("-"):
            raise ConfigError("ssh_target is required and cannot begin with '-'")
        if not self.expected_authority_role or not self.expected_instance_id:
            raise ConfigError("expected_authority_role and expected_instance_id cannot be empty")
        parsed = urlsplit(self.console_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ConfigError("console_url must be a loopback HTTP URL reached through the SSH tunnel")
        if parsed.port != self.local_port:
            raise ConfigError("console_url port must match local_port")
        if self.local_host not in {"127.0.0.1", "localhost", "::1"}:
            raise ConfigError("local_host must be loopback")
        if not (1 <= self.local_port <= 65535 and 1 <= self.remote_port <= 65535):
            raise ConfigError("local_port and remote_port must be valid TCP ports")
        positive = {
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "server_alive_interval_seconds": self.server_alive_interval_seconds,
            "server_alive_count_max": self.server_alive_count_max,
            "tunnel_probe_failures_before_restart": self.tunnel_probe_failures_before_restart,
            "reconnect_initial_seconds": self.reconnect_initial_seconds,
            "reconnect_max_seconds": self.reconnect_max_seconds,
            "loop_interval_seconds": self.loop_interval_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "poll_limit": self.poll_limit,
            "lease_seconds": self.lease_seconds,
            "http_timeout_seconds": self.http_timeout_seconds,
            "sleep_jump_seconds": self.sleep_jump_seconds,
            "app_server_timeout_seconds": self.app_server_timeout_seconds,
            "inflight_retry_grace_seconds": self.inflight_retry_grace_seconds,
            "max_handoff_bytes": self.max_handoff_bytes,
            "status_stale_seconds": self.status_stale_seconds,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ConfigError(f"bridge config values must be positive: {', '.join(invalid)}")
        if self.reconnect_initial_seconds > self.reconnect_max_seconds:
            raise ConfigError("reconnect_initial_seconds cannot exceed reconnect_max_seconds")
        if not self.codex_command:
            raise ConfigError("codex_command cannot be empty")
        for path in (self.identity_file, self.ssh_config_file, self.known_hosts_file, self.console_token_file):
            if path is not None and not Path(path).expanduser().is_absolute():
                raise ConfigError("credential and SSH configuration paths must be absolute")

    def ssh_command(self) -> list[str]:
        forward = f"{self.local_host}:{self.local_port}:{self.remote_host}:{self.remote_port}"
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
            "TCPKeepAlive=yes",
            "-o",
            "StrictHostKeyChecking=yes",
        ]
        if self.identity_file:
            command.extend(["-i", str(Path(self.identity_file).expanduser())])
        if self.ssh_config_file:
            command.extend(["-F", str(Path(self.ssh_config_file).expanduser())])
        if self.known_hosts_file:
            command.extend(["-o", f"UserKnownHostsFile={Path(self.known_hosts_file).expanduser()}"])
        command.extend(["-L", forward, self.ssh_target])
        return command

    def app_server_command(self) -> list[str]:
        command = list(self.codex_command)
        if self.codex_socket:
            command.extend(["--sock", str(Path(self.codex_socket).expanduser())])
        return command
