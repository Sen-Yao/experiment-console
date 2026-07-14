from __future__ import annotations

import json
import os
import signal
import subprocess
import time

import pytest

import experiment_console.ssh as ssh_module
from experiment_console.command import CommandFailed, CommandResult
from experiment_console.config import Settings
from experiment_console.ssh import SSHExecutor, build_failure_diagnostics, classify_argv_probe, extract_error_signals


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def run(self, argv, *, timeout, input_text=None):
        self.calls.append({"argv": argv, "timeout": timeout, "input_text": input_text})
        return CommandResult(argv=argv, returncode=0, stdout='{"ok": true, "pid": 12345, "log": "/tmp/agent.log"}\n', stderr="")


class LocalRemotePythonRunner:
    def run(self, argv, *, timeout, input_text=None):
        completed = subprocess.run(argv[-1], shell=True, text=True, capture_output=True, timeout=timeout, input=input_text)
        return CommandResult(argv=argv, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


class TransientThenSuccessRunner:
    def __init__(self, success_stdout: str):
        self.calls = []
        self.success_stdout = success_stdout

    def run(self, argv, *, timeout, input_text=None):
        self.calls.append({"argv": argv, "timeout": timeout, "input_text": input_text})
        if len(self.calls) == 1:
            raise CommandFailed(CommandResult(argv=argv, returncode=255, stdout="", stderr="Connection reset by peer"))
        return CommandResult(argv=argv, returncode=0, stdout=self.success_stdout, stderr="")


class LostAckOnceLocalRunner:
    def __init__(self):
        self.calls = []
        self.lost_ack = True

    def run(self, argv, *, timeout, input_text=None):
        self.calls.append({"argv": argv, "timeout": timeout, "input_text": input_text})
        completed = subprocess.run(argv[-1], shell=True, text=True, capture_output=True, timeout=timeout, input=input_text)
        result = CommandResult(argv=argv, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        if completed.returncode != 0:
            raise CommandFailed(result)
        if self.lost_ack:
            self.lost_ack = False
            raise CommandFailed(CommandResult(argv=argv, returncode=255, stdout=completed.stdout, stderr="Connection reset by peer"))
        return result


def write_fake_cmdline(proc_root, pid: int, argv: list[str]) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    (process_dir / "cmdline").write_bytes(b"\0".join(item.encode("utf-8") for item in argv) + b"\0")


def write_run_config(remote_root, run_id: str, output_path: str | None = None) -> None:
    run_files = remote_root / "wandb" / f"run-20260620_000000-{run_id}" / "files"
    run_files.mkdir(parents=True)
    text = "seed:\n  value: 1\n"
    if output_path is not None:
        text += f"out:\n  value: {output_path}\n"
    (run_files / "config.yaml").write_text(text, encoding="utf-8")


def test_ssh_host_rejects_leading_option_before_runner_execution(tmp_path):
    runner = RecordingRunner()
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=runner)

    with pytest.raises(ValueError, match="not an option"):
        ssh.probe_gpus("-oProxyCommand=touch_/tmp/pwned")

    assert runner.calls == []


def test_read_only_probe_retries_one_transient_disconnect(tmp_path):
    runner = TransientThenSuccessRunner("0, GPU, 10000, 100, 9900, 0\n")
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=runner)

    result = ssh.probe_gpus("HCCS-25")

    assert len(runner.calls) == 2
    assert result["eligible_count"] == 1


@pytest.mark.parametrize("operation", ["create_sweep", "reconcile_agent_capacity"])
def test_mutating_ssh_operations_do_not_retry_after_transient_disconnect(tmp_path, operation):
    runner = TransientThenSuccessRunner('{"ok":true,"pid":12345}\n')
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=runner)

    with pytest.raises(CommandFailed):
        if operation == "create_sweep":
            ssh.create_sweep(
                host="HCCS-25",
                remote_cwd="/home/linziyao/DualRefGAD",
                remote_config="/home/linziyao/DualRefGAD/config.yaml",
                entity="HCCS",
                project="DualRefGAD",
                wandb_api_key="secret",
            )
        else:
            ssh.reconcile_agent_capacity(
                host="HCCS-25",
                job_id="job-managed",
                remote_cwd="/home/linziyao/DualRefGAD",
                sweep_path="HCCS/DualRefGAD/abc123",
                desired_agents=1,
                eligible_gpu_indices=[0],
                conda_env="DualRefGAD",
                conda_sh="/opt/anaconda3/etc/profile.d/conda.sh",
                wandb_api_key="secret",
            )

    assert len(runner.calls) == 1


def test_agent_reconciler_uses_durable_receipts_and_conda_run(tmp_path):
    runner = RecordingRunner()
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=runner)

    result = ssh.reconcile_agent_capacity(
        host="HCCS-25",
        job_id="job-managed",
        remote_cwd="/home/linziyao/DualRefGAD",
        sweep_path="HCCS/DualRefGAD/abc123",
        desired_agents=1,
        eligible_gpu_indices=[3],
        conda_env="DualRefGAD",
        conda_sh="/opt/anaconda3/etc/profile.d/conda.sh",
        wandb_api_key="secret",
    )

    remote = runner.calls[0]["argv"][-1]
    assert runner.calls[0]["argv"][:3] == ["ssh", "--", "HCCS-25"]
    assert result["pid"] == 12345
    assert runner.calls[0]["input_text"] == "secret\n"
    assert "CUDA_VISIBLE_DEVICES" in remote
    assert "agent-receipts" in remote
    assert "generation" in remote
    assert "fcntl.flock" in remote
    assert "os.replace" in remote
    assert "/opt/anaconda3/etc/profile.d/conda.sh" in remote
    assert "DualRefGAD" in remote
    assert "HCCS/DualRefGAD/abc123" in remote
    assert "python3 -c" in remote


def test_agent_reconciler_recovers_lost_ssh_ack_from_remote_receipt_without_duplicate(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wandb = bin_dir / "wandb"
    wandb.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n", encoding="utf-8")
    wandb.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    receipt_root = tmp_path / "remote-state" / "agent-receipts"
    runner = LostAckOnceLocalRunner()
    ssh = SSHExecutor(Settings(state_dir=tmp_path, command_timeout_seconds=10), runner=runner)

    with pytest.raises(CommandFailed, match="command failed"):
        ssh.reconcile_agent_capacity(
            host="local",
            job_id="job-managed",
            remote_cwd=str(tmp_path),
            sweep_path="HCCS/DualRefGAD/receipt-test",
            desired_agents=1,
            eligible_gpu_indices=[3],
            conda_env=None,
            conda_sh="/unused/conda.sh",
            wandb_api_key="secret-not-for-receipt",
            receipt_root=str(receipt_root),
        )

    recovered = ssh.reconcile_agent_capacity(
        host="local",
        job_id="job-managed",
        remote_cwd=str(tmp_path),
        sweep_path="HCCS/DualRefGAD/receipt-test",
        desired_agents=1,
        eligible_gpu_indices=[3],
        conda_env=None,
        conda_sh="/unused/conda.sh",
        wandb_api_key="secret-not-for-receipt",
        receipt_root=str(receipt_root),
    )
    receipts = list(receipt_root.rglob("*.json"))
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    try:
        assert recovered["live_agents"] == 1
        assert recovered["launched"] == []
        assert len(receipts) == 1
        assert "secret-not-for-receipt" not in receipts[0].read_text(encoding="utf-8")
        assert receipt["gpu_index"] == 3
        assert receipt["generation"] == 1
    finally:
        try:
            os.killpg(int(receipt["session_id"]), signal.SIGTERM)
        except ProcessLookupError:
            pass


def test_stop_agents_uses_exact_proc_cmdline_matching(tmp_path):
    runner = RecordingRunner()
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=runner)

    ssh.stop_agents(host="HCCS-25", sweep_path="HCCS/DualRefGAD/abc123")

    remote = runner.calls[0]["argv"][-1]
    assert "python3 -c" in remote
    assert "proc_root" in remote
    assert "matches_sweep" in remote
    assert "unmatched_reused_pids" in remote
    assert "HCCS/DualRefGAD/abc123" in remote


def test_agent_probe_and_stop_never_treat_reused_pid_as_sweep_agent(tmp_path):
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())
    unrelated = subprocess.Popen(["/bin/sleep", "30"])
    proc_root = tmp_path / "proc"
    write_fake_cmdline(proc_root, unrelated.pid, ["/bin/sleep", "30"])
    try:
        probe = ssh.check_agent_processes(
            host="local",
            sweep_path="HCCS/DualRefGAD/not-this-process",
            pids=[str(unrelated.pid)],
            proc_root=str(proc_root),
        )
        assert probe["alive_pids"] == []
        assert probe["unmatched_reused_pids"] == [str(unrelated.pid)]

        stopped = ssh.stop_agents(
            host="local",
            sweep_path="HCCS/DualRefGAD/not-this-process",
            pids=[str(unrelated.pid)],
            proc_root=str(proc_root),
        )
        assert stopped["stopped_pids"] == []
        assert stopped["unmatched_reused_pids"] == [str(unrelated.pid)]
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_agent_probe_and_stop_match_exact_wandb_sweep_process(tmp_path):
    wandb = tmp_path / "wandb"
    wandb.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    wandb.chmod(0o755)
    sweep_path = "HCCS/DualRefGAD/exact-sweep"
    agent = subprocess.Popen([str(wandb), "agent", sweep_path])
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())
    proc_root = tmp_path / "proc"
    write_fake_cmdline(proc_root, agent.pid, ["python3", str(wandb), "agent", sweep_path])
    try:
        time.sleep(0.05)
        probe = ssh.check_agent_processes(
            host="local", sweep_path=sweep_path, pids=[str(agent.pid)], proc_root=str(proc_root),
        )
        assert probe["alive_pids"] == [str(agent.pid)]
        assert probe["unmatched_reused_pids"] == []

        stopped = ssh.stop_agents(
            host="local", sweep_path=sweep_path, pids=[str(agent.pid)], proc_root=str(proc_root),
        )
        assert stopped["stopped_pids"] == [str(agent.pid)]
        agent.wait(timeout=5)
    finally:
        if agent.poll() is None:
            agent.terminate()
            agent.wait(timeout=5)


def test_single_run_stop_requires_matching_status_receipt_and_cmdline(tmp_path):
    unrelated = subprocess.Popen(["/bin/sleep", "30"])
    status_path = tmp_path / "job.status.json"
    status_path.write_text(json.dumps({
        "job_id": "job-1",
        "child_pid": unrelated.pid,
        "command": ["python", "train.py"],
    }), encoding="utf-8")
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())
    proc_root = tmp_path / "proc"
    write_fake_cmdline(proc_root, unrelated.pid, ["/bin/sleep", "30"])
    try:
        result = ssh.stop_pids(
            host="local",
            pids=[str(unrelated.pid)],
            status_path=str(status_path),
            expected_job_id="job-1",
            proc_root=str(proc_root),
        )
        assert result["stopped_pids"] == []
        assert result["unmatched_reused_pids"] == [str(unrelated.pid)]
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_pull_results_reads_group_keys_from_wandb_config_yaml(tmp_path):
    run_files = tmp_path / "wandb" / "run-20260620_000000-run-a" / "files"
    run_files.mkdir(parents=True)
    (run_files / "wandb-summary.json").write_text('{"final_test_auc": 0.91}\n', encoding="utf-8")
    (run_files / "config.yaml").write_text(
        "_wandb:\n"
        "  value:\n"
        "    e:\n"
        "      run:\n"
        "        args:\n"
        "          - --batch_size=512\n"
        "          - --layers=2\n"
        "lr:\n"
        "  value: 0.001\n",
        encoding="utf-8",
    )
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())

    result = ssh.pull_results(
        host="local",
        remote_cwd=str(tmp_path),
        sweep_id="abc123",
        run_ids=["run-a"],
        budget_seconds=10,
        max_runs=1,
        metric_keys=["final_test_auc"],
        group_keys=["batch_size", "layers", "lr"],
    )

    assert result["rows"][0]["config"] == {"batch_size": 512, "layers": 2, "lr": 0.001}
    assert result["config_sources"]["run-a"]["batch_size"].endswith("config.yaml")
    assert result["missing_config_keys"] == {}


def test_pull_results_discovers_config_output_and_nested_metric_paths(tmp_path):
    run_files = tmp_path / "wandb" / "run-20260620_000000-run-a" / "files"
    run_files.mkdir(parents=True)
    output_dir = tmp_path / "outputs" / "foo"
    output_dir.mkdir(parents=True)
    final_path = output_dir / "run-a.json"
    progress_path = output_dir / "progress_run-a.json"
    (run_files / "wandb-summary.json").write_text('{"final_test_auc": 0.1}\n', encoding="utf-8")
    final_path.write_text(
        json.dumps({
            "final_test_auc": 0.91,
            "seed": 3,
            "comparisons": {"topo_gt_minus_mlp": 0.12},
            "representations": {
                "hybrid": {
                    "reader_metrics": {
                        "mlp_flat": {"auc": 0.88},
                        "linear": {"auc": 0.82},
                    }
                }
            },
        }),
        encoding="utf-8",
    )
    progress_path.write_text('{"final_test_auc": 0.01}\n', encoding="utf-8")
    (run_files / "config.yaml").write_text(
        "_wandb:\n"
        "  value:\n"
        "    e:\n"
        "      run:\n"
        "        args:\n"
        f"          - --out={final_path}\n"
        f"          - --progress_out={progress_path}\n",
        encoding="utf-8",
    )
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())

    result = ssh.pull_results(
        host="local",
        remote_cwd=str(tmp_path),
        sweep_id="abc123",
        run_ids=["run-a"],
        budget_seconds=10,
        max_runs=1,
        metric_keys=["final_test_auc"],
        group_keys=["seed"],
        metric_paths=["representations.*.reader_metrics.*.auc", "missing.path"],
        comparison_paths=["comparisons.*", "missing.comparison"],
        include_raw_artifacts=True,
    )

    row = result["rows"][0]
    assert row["metrics"]["final_test_auc"] == 0.91
    assert row["metrics"]["representations.hybrid.reader_metrics.linear.auc"] == 0.82
    assert row["metrics"]["representations.hybrid.reader_metrics.mlp_flat.auc"] == 0.88
    assert row["config"] == {"seed": 3}
    assert result["discovery_sources"]["run-a"]["selected_paths"] == [str(final_path)]
    assert str(progress_path) in result["discovery_sources"]["run-a"]["progress_paths"]
    assert result["missing_metric_paths"] == {"run-a": ["missing.path"]}
    assert row["comparisons"] == {"comparisons.topo_gt_minus_mlp": 0.12}
    assert result["missing_comparison_paths"] == {"run-a": ["missing.comparison"]}
    assert result["raw_artifacts"][0]["path"] == str(final_path)
    assert result["raw_artifacts"][0]["content"]["final_test_auc"] == 0.91
    assert all("progress" not in item["basename"] for item in result["raw_artifacts"])


def test_pull_results_output_globs_fill_when_config_has_no_result_path(tmp_path):
    run_files = tmp_path / "wandb" / "run-20260620_000000-run-a" / "files"
    run_files.mkdir(parents=True)
    custom_dir = tmp_path / "custom-results"
    custom_dir.mkdir()
    result_path = custom_dir / "science.json"
    result_path.write_text('{"final_test_auc": 0.73, "seed": 1}\n', encoding="utf-8")
    (run_files / "config.yaml").write_text("seed:\n  value: 1\n", encoding="utf-8")
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())

    result = ssh.pull_results(
        host="local",
        remote_cwd=str(tmp_path),
        sweep_id="abc123",
        run_ids=["run-a"],
        budget_seconds=10,
        max_runs=1,
        metric_keys=["final_test_auc"],
        group_keys=["seed"],
        output_globs=[str(custom_dir / "*.json")],
    )

    assert result["rows"][0]["metrics"] == {"final_test_auc": 0.73}
    assert result["rows"][0]["config"] == {"seed": 1}
    assert result["discovery_sources"]["run-a"]["selected_paths"] == [str(result_path)]
    assert result["raw_artifacts"] == []


def test_pull_results_rejects_absolute_wandb_config_artifact_outside_remote_cwd(tmp_path):
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"final_test_auc": 0.99}\n', encoding="utf-8")
    write_run_config(remote_root, "run-a", str(outside))
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())

    result = ssh.pull_results(
        host="local",
        remote_cwd=str(remote_root),
        sweep_id="abc123",
        run_ids=["run-a"],
        budget_seconds=10,
        max_runs=1,
        metric_keys=["final_test_auc"],
        group_keys=[],
        discovery_mode="wandb_config_result_path_v1",
        include_raw_artifacts=True,
    )

    assert result["valid_results"] == 0
    assert result["raw_artifacts"] == []
    assert result["discovery_sources"]["run-a"]["classification"] == "artifact_outside_remote_cwd"
    assert result["artifact_rejections"]["run-a"][0]["path"] == str(outside)


def test_pull_results_rejects_parent_traversal_from_wandb_config(tmp_path):
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    (tmp_path / "outside.json").write_text('{"final_test_auc": 0.99}\n', encoding="utf-8")
    write_run_config(remote_root, "run-a", "../outside.json")
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())

    result = ssh.pull_results(
        host="local",
        remote_cwd=str(remote_root),
        sweep_id="abc123",
        run_ids=["run-a"],
        budget_seconds=10,
        max_runs=1,
        metric_keys=["final_test_auc"],
        group_keys=[],
        discovery_mode="wandb_config_result_path_v1",
    )

    assert result["valid_results"] == 0
    assert result["discovery_sources"]["run-a"]["classification"] == "artifact_path_traversal"
    assert result["artifact_rejections"]["run-a"][0]["reason"] == "artifact_path_traversal"


def test_pull_results_rejects_glob_symlink_that_resolves_outside_remote_cwd(tmp_path):
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    write_run_config(remote_root, "run-a")
    outside = tmp_path / "outside.json"
    outside.write_text('{"final_test_auc": 0.99}\n', encoding="utf-8")
    outputs = remote_root / "outputs"
    outputs.mkdir()
    (outputs / "run-a.json").symlink_to(outside)
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())

    result = ssh.pull_results(
        host="local",
        remote_cwd=str(remote_root),
        sweep_id="abc123",
        run_ids=["run-a"],
        budget_seconds=10,
        max_runs=1,
        metric_keys=["final_test_auc"],
        group_keys=[],
        output_globs=[str(outputs / "{run_id}.json")],
        discovery_mode="run_id_output_globs_v1",
        include_raw_artifacts=True,
    )

    assert result["valid_results"] == 0
    assert result["raw_artifacts"] == []
    assert result["discovery_sources"]["run-a"]["classification"] == "artifact_outside_remote_cwd"


def test_pull_results_rejects_single_artifact_over_file_byte_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(ssh_module, "MAX_RESULT_ARTIFACT_FILE_BYTES", 64)
    monkeypatch.setattr(ssh_module, "MAX_RESULT_ARTIFACT_TOTAL_BYTES", 1024)
    write_run_config(tmp_path, "run-a")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    artifact = outputs / "run-a.json"
    artifact.write_text(json.dumps({"final_test_auc": 0.9, "padding": "x" * 100}), encoding="utf-8")
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())

    result = ssh.pull_results(
        host="local",
        remote_cwd=str(tmp_path),
        sweep_id="abc123",
        run_ids=["run-a"],
        budget_seconds=10,
        max_runs=1,
        metric_keys=["final_test_auc"],
        group_keys=[],
        output_globs=[str(outputs / "{run_id}.json")],
        discovery_mode="run_id_output_globs_v1",
        include_raw_artifacts=True,
    )

    assert result["valid_results"] == 0
    assert result["raw_artifacts"] == []
    assert result["artifact_bytes_read"] == 0
    assert result["discovery_sources"]["run-a"]["classification"] == "artifact_file_size_limit_exceeded"


def test_pull_results_enforces_total_artifact_byte_limit_across_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(ssh_module, "MAX_RESULT_ARTIFACT_FILE_BYTES", 1024)
    monkeypatch.setattr(ssh_module, "MAX_RESULT_ARTIFACT_TOTAL_BYTES", 100)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    artifact_sizes = []
    for run_id in ["run-a", "run-b"]:
        write_run_config(tmp_path, run_id)
        artifact = outputs / f"{run_id}.json"
        artifact.write_text(json.dumps({"final_test_auc": 0.9, "padding": "x" * 30}), encoding="utf-8")
        artifact_sizes.append(artifact.stat().st_size)
    assert artifact_sizes[0] < 100 < sum(artifact_sizes)
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())

    result = ssh.pull_results(
        host="local",
        remote_cwd=str(tmp_path),
        sweep_id="abc123",
        run_ids=["run-a", "run-b"],
        budget_seconds=10,
        max_runs=2,
        metric_keys=["final_test_auc"],
        group_keys=[],
        output_globs=[str(outputs / "{run_id}.json")],
        discovery_mode="run_id_output_globs_v1",
        include_raw_artifacts=True,
    )

    assert result["valid_results"] == 1
    assert result["missing_results"] == 1
    assert result["artifact_bytes_read"] == artifact_sizes[0]
    assert result["artifact_bytes_read"] <= result["artifact_limits"]["total_bytes"]
    assert result["discovery_sources"]["run-b"]["classification"] == "artifact_total_size_limit_exceeded"


def test_run_id_result_contract_rejects_canonical_path_claimed_by_two_runs(tmp_path):
    for run_id in ["run-a", "run-b"]:
        run_files = tmp_path / "wandb" / f"run-20260620_000000-{run_id}" / "files"
        run_files.mkdir(parents=True)
        (run_files / "config.yaml").write_text("seed:\n  value: 1\n", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    shared = output_dir / "shared.json"
    shared.write_text('{"final_test_auc": 0.9}\n', encoding="utf-8")
    (output_dir / "run-a.json").symlink_to(shared)
    (output_dir / "run-b.json").symlink_to(shared)
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())

    result = ssh.pull_results(
        host="local",
        remote_cwd=str(tmp_path),
        sweep_id="abc123",
        run_ids=["run-a", "run-b"],
        budget_seconds=10,
        max_runs=2,
        metric_keys=["final_test_auc"],
        group_keys=[],
        output_globs=[str(output_dir / "{run_id}.json")],
        discovery_mode="run_id_output_globs_v1",
        include_raw_artifacts=True,
    )

    assert result["valid_results"] == 1
    assert result["missing_results"] == 1
    assert result["discovery_sources"]["run-b"]["classification"] == "artifact_path_claimed_by_multiple_runs"
    assert len(result["raw_artifacts"]) == 1


def test_preflight_reports_remote_git_and_config_facts(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "--allow-empty", "-m", "init"],
        check=True,
        capture_output=True,
        text=True,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("method: grid\n", encoding="utf-8")
    ssh = SSHExecutor(Settings(state_dir=tmp_path), runner=LocalRemotePythonRunner())

    result = ssh.preflight(
        host="local",
        remote_cwd=str(tmp_path),
        conda_env=None,
        conda_sh=None,
        config_path=str(config_path),
    )

    assert result["checks"]["config_path_exists"] is True
    assert result["git"]["available"] is True
    assert result["git"]["head"]
    assert result["git"]["dirty"] is True
    assert any("config.yaml" in item for item in result["git"]["status_short"])
    assert result["runtime"]["requested_conda_env"] is None
    assert result["runtime"]["base_python"]


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


def test_extract_error_signals_detects_wandb_agent_killed_runs():
    text = "wandb: Agent received exit signal\nwandb: Killing runs and quitting\n"

    signals = extract_error_signals(text, source="/tmp/agent.log")

    assert any(signal["kind"] == "wandb_agent_killed_runs" for signal in signals)


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


def test_classify_argv_probe_detects_argparse_error():
    assert classify_argv_probe(0, "usage: main.py [-h]\n", "") == "argv_compatible"
    assert classify_argv_probe(2, "", "main.py: error: argument --wandb: expected one argument\n") == "argv_incompatible"
    assert classify_argv_probe(1, "", "RuntimeError: import failed\n") == "argv_probe_unavailable"
