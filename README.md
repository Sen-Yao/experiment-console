# HCCS Experiment Runner Knowledge

There is no custom Experiment Console or wake bridge in the active research
path. Codex operates HCCS-25 directly using the installed `experiment-runner`
skill:

```text
Codex task
  -> committed detached HCCS worktree
  -> explicit tmux panes and GPUs
  -> native W&B sweep/agents
  -> one built-in one-shot thread heartbeat
  -> blocked Goal while external work runs
```

The heartbeat is a best-effort scheduled follow-up, not an experiment monitor.
When it fires, Codex reconstructs state from tmux, exact processes/GPU evidence,
W&B, the run manifest, and declared artifacts. A healthy incomplete sweep
updates the same heartbeat using bounded ETA backoff; a complete or invalid
sweep does not schedule another one.

The canonical operational workflow lives in
[`skill/experiment-runner/SKILL.md`](skill/experiment-runner/SKILL.md), with the
dynamic scheduling policy in
[`skill/experiment-runner/references/dynamic-heartbeat.md`](skill/experiment-runner/references/dynamic-heartbeat.md).

## Legacy Console v3

The v3 executor remains under `backend/experiment_console`, with its runner,
Compose definition, and tests retained only for an explicit whole-system
rollback. It is not deployed, monitored, or used by new experiments.

Its structured argv, allowed-root, request-id, receipt ownership, bounded read,
cancellation, and GPU process-classification safeguards remain intact. A GPU
with a foreign or unknown compute process is unavailable at v3 launch; a new
conflict observed after launch remains warning-only and never triggers automatic
process termination.

## Development

Requirements: Python 3.11+.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
./scripts/run_tests.sh
./scripts/exp --help
```
