from __future__ import annotations

import json
import subprocess

from experiment_console.command import CommandResult
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
        completed = subprocess.run(argv[2], shell=True, text=True, capture_output=True, timeout=timeout, input=input_text)
        return CommandResult(argv=argv, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


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


def test_classify_argv_probe_detects_argparse_error():
    assert classify_argv_probe(0, "usage: main.py [-h]\n", "") == "argv_compatible"
    assert classify_argv_probe(2, "", "main.py: error: argument --wandb: expected one argument\n") == "argv_incompatible"
    assert classify_argv_probe(1, "", "RuntimeError: import failed\n") == "argv_probe_unavailable"
