from __future__ import annotations

import os
import re
from typing import Any


SECRET_PATTERNS = [
    re.compile(r"(WANDB_API_KEY=)[^\s'\"]+", re.IGNORECASE),
    re.compile(r"(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9_\-]{16,}", re.IGNORECASE),
    re.compile(r"(password[\"']?\s*[:=]\s*[\"']?)[^\s'\",}]+", re.IGNORECASE),
    re.compile(r"(token[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9_\-\.]{16,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]+", re.IGNORECASE),
]


def redact_text(text: str) -> str:
    out = text
    for key in ("WANDB_API_KEY", "WANDB_BASE_URL"):
        value = os.environ.get(key)
        if value:
            out = out.replace(value, "<redacted>")
    for pattern in SECRET_PATTERNS:
        out = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "<redacted>", out)
    return out


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if any(word in str(key).lower() for word in ("secret", "password", "token", "api_key", "apikey")):
                cleaned[key] = "<redacted>"
            else:
                cleaned[key] = redact_value(item)
        return cleaned
    return value

