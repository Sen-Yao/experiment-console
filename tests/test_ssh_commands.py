from __future__ import annotations

from experiment_console.command import CommandResult
from experiment_console.config import Settings
from experiment_console.ssh import SSHExecutor


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def run(self, argv, *, timeout, input_text=None):
        self.calls.append({"argv": argv, "timeout": timeout, "input_text": input_text})
        return CommandResult(argv=argv, returncode=0, stdout="12345\n", stderr="")


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
    assert "python -c" not in remote


def test_stop_agents_uses_single_layer_shell_matching(tmp_path):
    runner = RecordingRunner()
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=runner)

    ssh.stop_agents(host="HCCS-25", sweep_path="HCCS/DualRefGAD/abc123")

    remote = runner.calls[0]["argv"][2]
    assert "pgrep -af" in remote
    assert "kill -TERM" in remote
    assert "python3 -c" not in remote
    assert "wandb agent HCCS/DualRefGAD/abc123" in remote
