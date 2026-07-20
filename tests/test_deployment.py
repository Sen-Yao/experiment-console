from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_launchd_module():
    path = ROOT / "scripts" / "manage_codex_bridge_launchd.py"
    spec = importlib.util.spec_from_file_location("bridge_launchd", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launchd_runs_only_the_v3_bridge(tmp_path):
    module = load_launchd_module()
    payload = module.make_plist(
        python=Path("/usr/bin/python3"), config=tmp_path / "bridge.json"
    )
    arguments = payload["ProgramArguments"]
    assert arguments[-1] == "run"
    assert "desktop_bridge" in arguments
    assert "repin-authority" not in arguments


def test_packaging_has_no_frontend_or_domain_controller():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yggdrasil.yaml").read_text(encoding="utf-8")
    local_start = (ROOT / "scripts" / "start_local_console.sh").read_text(
        encoding="utf-8"
    )
    assert "frontend" not in dockerfile.lower()
    assert "WANDB" not in compose
    assert "EXPERIMENT_CONSOLE_RESULTS" not in compose
    assert "runtime_console_server:create_app" in dockerfile
    assert "--factory" in dockerfile
    assert "experiment_console.api:create_app" in local_start
    assert "--factory" in local_start
    assert (ROOT / "config" / "server-profiles.json").is_file()
