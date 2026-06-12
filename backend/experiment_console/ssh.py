from __future__ import annotations

import shlex
from typing import Any

from .command import CommandRunner
from .config import Settings


def quote_remote(command: str) -> str:
    return command


class SSHExecutor:
    def __init__(self, settings: Settings, runner: CommandRunner | None = None):
        self.settings = settings
        self.runner = runner or CommandRunner()

    def run(self, host: str, remote_command: str, *, timeout: int | None = None):
        argv = ["ssh", host, quote_remote(remote_command)]
        return self.runner.run(argv, timeout=timeout or self.settings.ssh_timeout_seconds)

    def probe_gpus(self, host: str, *, min_free_gb: float | None = None, max_util: int | None = None) -> dict[str, Any]:
        query = (
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu "
            "--format=csv,noheader,nounits"
        )
        result = self.run(host, query)
        gpus = []
        threshold_gb = min_free_gb if min_free_gb is not None else self.settings.gpu_min_free_gb
        util_limit = max_util if max_util is not None else self.settings.gpu_max_util
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue
            index, name, total, used, free, util = parts
            try:
                free_gb = float(free) / 1024
                util_pct = int(float(util))
                gpus.append({
                    "index": int(index),
                    "name": name,
                    "memory_total_mb": int(float(total)),
                    "memory_used_mb": int(float(used)),
                    "memory_free_mb": int(float(free)),
                    "utilization_gpu": util_pct,
                    "eligible": free_gb >= threshold_gb and util_pct <= util_limit,
                })
            except ValueError:
                continue
        return {"host": host, "gpus": gpus, "eligible_count": sum(1 for gpu in gpus if gpu["eligible"])}

    def launch_agent(self, *, host: str, remote_cwd: str, sweep_path: str, gpu_index: int, conda_env: str | None, conda_sh: str) -> dict[str, Any]:
        env_prefix = f"export CUDA_VISIBLE_DEVICES={gpu_index}; "
        conda_prefix = ""
        if conda_env:
            conda_prefix = f"source {shlex.quote(conda_sh)} && conda activate {shlex.quote(conda_env)} && "
        log_name = f"console_wandb_agent_{sweep_path.replace('/', '_')}_gpu{gpu_index}.log"
        remote = (
            f"cd {shlex.quote(remote_cwd)} && "
            f"{env_prefix}{conda_prefix}"
            f"nohup wandb agent {shlex.quote(sweep_path)} > {shlex.quote(log_name)} 2>&1 & echo $!"
        )
        result = self.run(host, remote, timeout=self.settings.command_timeout_seconds)
        pid = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        return {"host": host, "gpu_index": gpu_index, "pid": pid, "log": f"{remote_cwd.rstrip('/')}/{log_name}", "command": result.summary()}

    def stop_agents(self, *, host: str, sweep_path: str) -> dict[str, Any]:
        pattern = f"wandb agent {sweep_path}"
        remote = (
            "python3 -c "
            + shlex.quote(
                "import os, signal, subprocess\n"
                f"pattern={pattern!r}\n"
                "out=subprocess.run(['pgrep','-af','wandb agent'],text=True,capture_output=True)\n"
                "pids=[]\n"
                "for line in out.stdout.splitlines():\n"
                "    parts=line.split(None,1)\n"
                "    if len(parts)==2 and pattern in parts[1]:\n"
                "        pids.append(parts[0])\n"
                "for pid in pids:\n"
                "    os.kill(int(pid), signal.SIGTERM)\n"
                "print('\\n'.join(pids))\n"
            )
        )
        result = self.run(host, remote, timeout=self.settings.command_timeout_seconds)
        pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return {"host": host, "stopped_pids": pids, "command": result.summary()}

