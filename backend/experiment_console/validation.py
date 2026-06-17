from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigValidationError(ValueError):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return load_yaml_text(handle.read())


def load_yaml_text(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ConfigValidationError("config must be a YAML mapping")
    return data


def _values_for(spec: Any) -> list[Any]:
    if isinstance(spec, dict):
        if isinstance(spec.get("values"), list):
            return spec["values"]
        if "value" in spec:
            return [spec["value"]]
    return []


def expected_run_count(config: dict[str, Any]) -> int:
    params = config.get("parameters") or {}
    if not isinstance(params, dict):
        return 0
    total = 1
    seen = False
    for spec in params.values():
        values = _values_for(spec)
        if values:
            total *= max(1, len(values))
            seen = True
    return total if seen else 0


def validate_config_data(cfg: dict[str, Any], profile: str = "sweep", *, path_label: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    method = str(cfg.get("method") or "").lower()
    params = cfg.get("parameters") or {}
    if profile == "sweep":
        if method != "grid":
            errors.append("formal sweep profile requires method: grid")
        if not isinstance(params, dict):
            errors.append("parameters must be a mapping")
        seed_values = _values_for(params.get("seed")) if isinstance(params, dict) else []
        if seed_values and seed_values != [0, 1, 2, 3, 4]:
            warnings.append("formal sweep usually expects seed values [0, 1, 2, 3, 4]")
        dataset_values = _values_for(params.get("dataset")) if isinstance(params, dict) else []
        if len(dataset_values) > 1:
            errors.append("formal sweep profile requires a single dataset")
    if not cfg.get("program"):
        warnings.append("config has no program field; remote agent will rely on W&B defaults")
    if not cfg.get("name"):
        warnings.append("config has no name field")

    if errors:
        raise ConfigValidationError("; ".join(errors))

    return {
        "valid": True,
        "path": path_label,
        "profile": profile,
        "method": method or None,
        "program": cfg.get("program"),
        "name": cfg.get("name"),
        "expected_run_count": expected_run_count(cfg),
        "warnings": warnings,
    }



def validate_experiment_config(path: Path, profile: str = "sweep") -> dict[str, Any]:
    return validate_config_data(load_yaml(path), profile, path_label=str(path))


def validate_experiment_config_text(text: str, profile: str = "sweep", *, path_label: str | None = None) -> dict[str, Any]:
    return validate_config_data(load_yaml_text(text), profile, path_label=path_label)
