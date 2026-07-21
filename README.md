# Experiment Wake Bridge

This repository now provides the small control-plane component Codex actually
needs for long HCCS experiments: a Mac bridge that wakes a blocked Codex Goal
when a tagged tmux session needs attention or reaches a terminal state.

## Active Runtime

```text
Codex agent
  -> SSH HCCS-25
  -> committed code in a detached worktree
  -> tmux panes on explicitly selected GPUs
  -> native W&B sweep/agents
  -> blocked Goal

Mac bridge
  -> fixed read-only tmux inspection over SSH
  -> session-level attention/terminal event
  -> Codex app-server Goal-aware delivery
```

The bridge does not understand W&B, schedule work, lock GPUs, terminate panes,
or certify scientific results. It retains only a bounded status file and a
mode-600 JSON event outbox for restart-safe delivery deduplication.

See [`docs/desktop-bridge.md`](docs/desktop-bridge.md) for the tmux registration
contract and deployment commands.

## Development

Requirements: Python 3.11+ and SSH access to HCCS-25.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
./scripts/run_tests.sh
python3 -m desktop_bridge \
  --config config/desktop-bridge.example.json dry-run
```

## Legacy Console v3 Rollback

The v3 durable executor remains under `backend/experiment_console`, with its
runner, Compose definition, and state format unchanged. It is not the active
agent path. Keep the old Yggdrasil deployment and Mac config disabled during
initial validation so the whole system can be rolled back without importing or
rewriting ledger state.

The retained v3 implementation includes its existing structured argv, allowed
root, request-id, receipt ownership, bounded read, cancellation, and GPU
process-classification safeguards. Do not extend it while it is in rollback
status.

Its launch resource check remains fail-closed: a GPU with a foreign or unknown
compute process is unavailable even when its free-memory threshold passes.
While a v3 job is running, a newly observed foreign process remains an
operational warning; Console does not kill or cancel an unrelated process.
