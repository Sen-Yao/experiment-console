from __future__ import annotations

from experiment_console.command import CommandResult
from experiment_console.config import Settings
from experiment_console.ssh import SSHExecutor, build_failure_diagnostics, extract_error_signals


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def run(self, argv, *, timeout, input_text=None):
        self.calls.append({"argv": argv, "timeout": timeout, "input_text": input_text})
        return CommandResult(argv=argv, returncode=0, stdout='{"ok": true, "pid": 12345, "log": "/tmp/agent.log"}\n', stderr="")


def test_launch_agent_uses_conda_run_for_agent_process(tmp_path):
    runner = RecordingRunner()
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=runner)

    result = ssh.launch_agent(
        host="HCCS-25",
        remote_cwd="/home/linziyao/DualRefGAD",
        sweep_path="HCCS/DualRefGAD/abc123",
        gpu_index=3,
        conda_env="DualRefGAD",
        conda_sh="/opt/anaconda3/etc/profile.d/conda.sh",
        wandb_api_key="secret",
    )

    remote = runner.calls[0]["argv"][2]
    assert result["pid"] == "12345"
    assert runner.calls[0]["input_text"] == "secret\n"
    assert "CUDA_VISIBLE_DEVICES=3" in remote
    assert "source /opt/anaconda3/etc/profile.d/conda.sh" in remote
    assert "conda run -n DualRefGAD" in remote
    assert "wandb agent HCCS/DualRefGAD/abc123" in remote
    assert "python3 -c" in remote


def test_stop_agents_uses_single_layer_shell_matching(tmp_path):
    runner = RecordingRunner()
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=runner)

    ssh.stop_agents(host="HCCS-25", sweep_path="HCCS/DualRefGAD/abc123")

    remote = runner.calls[0]["argv"][2]
    assert "pgrep -af" in remote
    assert "kill -TERM" in remote
    assert "python3 -c" not in remote
    assert "wandb agent HCCS/DualRefGAD/abc123" in remote


def test_extract_error_signals_from_agent_log_tail():
    text = """wandb: syncing
Traceback (most recent call last):
  File "train.py", line 10, in <module>
    main()
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
"""

    signals = extract_error_signals(text, source="/tmp/agent.log")

    kinds = {signal["kind"] for signal in signals}
    assert "traceback" in kinds
    assert "cuda_oom" in kinds
    assert any("CUDA out of memory" in signal["excerpt"] for signal in signals)


def test_build_failure_diagnostics_reports_missing_process_and_log_tail():
    diagnostics = build_failure_diagnostics(
        host="HCCS-25",
        remote_cwd="/home/project",
        sweep_path="HCCS/DualRefGAD/abc123",
        launches=[{"gpu_index": 0, "pid": "123", "log": "/home/project/agent.log"}],
        pid_state={"tracked_pids": ["123"], "alive_pids": [], "pgrep": []},
        logs=[{
            "gpu_index": 0,
            "pid": "123",
            "path": "/home/project/agent.log",
            "exists": True,
            "tail": "ModuleNotFoundError: No module named 'torch_geometric'\n",
        }],
        command={"stdout": "ok"},
    )

    assert diagnostics["classification"] == "failure_signals_found"
    assert diagnostics["summary"]
    kinds = {signal["kind"] for signal in diagnostics["error_signals"]}
    assert "import_error" in kinds
    assert "agent_process_missing" in kinds
    assert diagnostics["log_tails"][0]["tail"].startswith("ModuleNotFoundError")
