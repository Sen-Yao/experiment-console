from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skill" / "experiment-runner" / "scripts" / "experiment.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("experiment_runner_repo", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_surface_is_v3_only():
    module = load_runner()
    parser = module.parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(getattr(action, "choices", None), dict)
    )
    assert set(subparsers.choices) == {
        "resources",
        "run",
        "status",
        "logs",
        "fetch",
        "cancel",
    }


def test_repo_wrapper_delegates_directly_to_the_v3_skill():
    wrapper = (ROOT / "scripts" / "exp").read_text(encoding="utf-8")
    assert "skill/experiment-runner/scripts/experiment.py" in wrapper
    assert "EXPERIMENT_CONSOLE_URL" in wrapper
    assert "launch-sweep" not in wrapper


def test_logs_with_an_offset_do_not_force_tail_mode(monkeypatch):
    module = load_runner()
    captured = {}
    monkeypatch.setattr(module, "verify_console", lambda: {})

    def request_json(method, path, **kwargs):
        captured.update(kwargs["query"])
        return {"text": "chunk"}

    monkeypatch.setattr(module, "request_json", request_json)
    module.command_logs(
        SimpleNamespace(
            job_id="job_1234567890123456",
            stream="stdout",
            offset=10,
            limit=20,
            from_start=False,
            json=False,
        )
    )
    assert captured["offset"] == 10
    assert captured["tail"] == "false"
