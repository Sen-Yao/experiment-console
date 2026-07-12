"""Lightweight macOS bridge for Experiment Console and Codex Desktop."""

from .config import BridgeConfig, ConfigError
from .models import OutboxEvent
from .service import BridgeService

__all__ = ["BridgeConfig", "BridgeService", "ConfigError", "OutboxEvent"]
