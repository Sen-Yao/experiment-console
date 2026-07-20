from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4


HELPER = (
    Path(__file__).resolve().parents[1]
    / "backend"
    / "experiment_console"
    / "remote_helper.py"
)


def helper(action: str, payload: dict, *, timeout: float = 10) -> dict:
    result = subprocess.run(
        [sys.executable, str(HELPER), action],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def wait_terminal(payload: dict, timeout: float = 10) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = helper("inspect", payload)
        if state["state"] != "running":
            return state
        time.sleep(0.05)
    raise AssertionError("remote helper job did not reach terminal state")


def test_remote_helper_launch_progress_logs_and_fetch(tmp_path):
    cwd = tmp_path / "work"
    cwd.mkdir()
    state = tmp_path / "state"
    job_id = f"job_{uuid4().hex}"
    output = cwd / "result.json"
    code = (
        "import json,os,pathlib; "
        "pathlib.Path(os.environ['EXPERIMENT_CONSOLE_PROGRESS_FILE']).write_text("
        "json.dumps({'completed_runs': 1, 'total_runs': 1, 'message': 'done'})); "
        f"pathlib.Path({str(output)!r}).write_text('result'); print('hello')"
    )
    payload = {
        "job_id": job_id,
        "state_root": str(state),
        "request": {
            "request_hash": "hash",
            "command_digest": "digest",
            "cwd": str(cwd),
            "allowed_roots": [str(tmp_path)],
            "argv": [sys.executable, "-c", code],
            "env": {},
            "gpu_indices": [],
            "total_runs": 1,
            "bootstrap_argv": [],
        },
    }
    launched = helper("launch", payload)
    assert launched["state"] in {"running", "succeeded"}
    identity = {
        "job_id": job_id,
        "state_root": str(state),
        "command_digest": "digest",
    }
    terminal = wait_terminal(identity)
    assert terminal["state"] == "succeeded"
    assert terminal["completed_runs"] == 1

    logs = helper(
        "logs", {**identity, "stream": "stdout", "offset": 0, "limit": 1024}
    )
    assert logs["text"] == "hello\n"
    fetched = helper(
        "fetch", {**identity, "path": "result.json", "offset": 0, "limit": 1024}
    )
    assert fetched["size"] == 6

    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    denied = subprocess.run(
        [sys.executable, str(HELPER), "fetch"],
        input=json.dumps(
            {**identity, "path": str(outside), "offset": 0, "limit": 10}
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert denied.returncode != 0


def test_remote_helper_cancels_owned_process_group(tmp_path):
    cwd = tmp_path / "work"
    cwd.mkdir()
    state = tmp_path / "state"
    job_id = f"job_{uuid4().hex}"
    payload = {
        "job_id": job_id,
        "state_root": str(state),
        "request": {
            "request_hash": "hash",
            "command_digest": "digest",
            "cwd": str(cwd),
            "allowed_roots": [str(tmp_path)],
            "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
            "env": {},
            "gpu_indices": [],
            "bootstrap_argv": [],
        },
    }
    assert helper("launch", payload)["state"] == "running"
    cancelled = helper(
        "cancel",
        {
            "job_id": job_id,
            "state_root": str(state),
            "command_digest": "digest",
            "grace_seconds": 1,
        },
        timeout=5,
    )
    if cancelled["state"] == "running":
        cancelled = wait_terminal(
            {
                "job_id": job_id,
                "state_root": str(state),
                "command_digest": "digest",
            }
        )
    assert cancelled["state"] in {"cancelled", "lost"}


def test_remote_helper_rejects_log_receipt_path_outside_job_cwd(tmp_path):
    cwd = tmp_path / "work"
    cwd.mkdir()
    state = tmp_path / "state"
    job_id = f"job_{uuid4().hex}"
    directory = state / "jobs" / job_id
    directory.mkdir(parents=True)
    outside = tmp_path / "outside.log"
    outside.write_text("secret")
    (directory / "request.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "command_digest": "digest",
                "cwd": str(cwd),
                "total_runs": None,
            }
        )
    )
    (directory / "receipt.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "command_digest": "digest",
                "stdout_path": str(outside),
            }
        )
    )
    denied = subprocess.run(
        [sys.executable, str(HELPER), "logs"],
        input=json.dumps(
            {
                "job_id": job_id,
                "state_root": str(state),
                "command_digest": "digest",
                "stream": "stdout",
            }
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert denied.returncode != 0
    assert "log path is outside the job working directory" in denied.stderr


def test_remote_helper_honors_cancel_marker_before_child_spawn(tmp_path):
    cwd = tmp_path / "work"
    cwd.mkdir()
    state = tmp_path / "state"
    job_id = f"job_{uuid4().hex}"
    directory = state / "jobs" / job_id
    directory.mkdir(parents=True)
    request_payload = {
        "job_id": job_id,
        "state_root": str(state),
        "request_hash": "hash",
        "command_digest": "digest",
        "cwd": str(cwd),
        "allowed_roots": [str(tmp_path)],
        "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
        "env": {},
        "gpu_indices": [],
        "total_runs": None,
        "bootstrap_argv": [],
    }
    (directory / "request.json").write_text(json.dumps(request_payload))
    (directory / "cancel-requested.json").write_text(
        json.dumps({"requested_at": "now"})
    )
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(HELPER), "supervise", str(directory)],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    elapsed = time.monotonic() - started
    assert result.returncode != 0
    assert elapsed < 5
    assert json.loads((directory / "status.json").read_text())["state"] == "cancelled"
