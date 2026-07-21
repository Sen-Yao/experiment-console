from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiment.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("experiment_runner_skill", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_surface_is_v3_only():
    module = load_runner()
    parser = module.parser()
    subparsers = next(
        action for action in parser._actions if isinstance(getattr(action, "choices", None), dict)
    )
    assert set(subparsers.choices) == {"resources", "run", "status", "logs", "fetch", "cancel"}


def test_structured_values_are_validated():
    module = load_runner()
    assert module.parse_gpus(["0,2"]) == [0, 2]
    assert module.parse_env(["MODE=test"]) == {"MODE": "test"}
