# Experiment Console Agent Notes

## Local Runner Commands

When running the installed `experiment-runner` from this repository, use the
repo wrapper:

```bash
./scripts/exp <runner-command> [args...]
```

Do not invoke the runner as an inline environment assignment plus Python, such
as:

```bash
EXPERIMENT_CONSOLE_URL=http://127.0.0.1:5174 python3 /Users/oliver/.agents/skills/experiment-runner/scripts/experiment.py ...
```

The wrapper sets `EXPERIMENT_CONSOLE_URL`, `PYTHONPATH`, and
`PYTHONPYCACHEPREFIX` before delegating to the installed runner. Keeping commands
in the `./scripts/exp ...` form makes Codex approval prefix rules stable because
dynamic values such as job ids, sweep ids, and remote config paths stay ordinary
arguments instead of becoming part of the approval prefix.

## Production Authority

The default `http://127.0.0.1:5174` endpoint is the Desktop bridge's SSH tunnel
to the authoritative Yggdrasil Console. It is not a local production service.
Run `./scripts/exp authority` before mutating work. If the tunnel or authority
check fails, repair the bridge/tunnel; do not start a second local Console as a
fallback.

Long-running experiment waits belong to the Console monitor worker. Supply an
explicit `ResultContract` and `CODEX_THREAD_ID` when a job must wake Codex on a
terminal or attention event. Do not keep a Goal active and do not create Codex
Automations/heartbeat turns to poll a healthy experiment. A task may finish its
current useful analysis and yield while the Console continues monitoring.

For result downloads, `--artifact-dir` is a local runner destination. The
runner must fetch a controlled bundle from the Console artifact API and safely
extract it locally; never send a Mac `/Users/...` path in a Yggdrasil payload.

For read-only runner checks, prefer narrow subcommand prefixes such as:

```bash
./scripts/exp status --job-id job_...
./scripts/exp show-job --job-id job_...
./scripts/exp preflight --profile sweep --config /remote/config.yaml --remote-host HCCS-25 --remote-cwd /home/linziyao/DualRefGAD
```

For mutating commands such as `launch-sweep`, `launch-run`, `recover-agents`,
`stop`, or `cancel-sweep`, keep the same wrapper form and rely on Console
safety/idempotency semantics. Ask before acting only when the current request
does not already authorize the operation, or when the action is destructive,
credential-sensitive, concurrent/costly, or blocked by platform approval.

Managed sweeps use the Console agent-capacity reconciler. `recover-agents`
only triggers one immediate reconciliation pass for a managed job; it does not
accept GPU, max-agent, or conda launch overrides and cannot adopt historical
jobs. Do not use or recreate a separate `launch-agents` path.

## Multi-Dataset Experiment Ordering

When a batch needs experiments on multiple datasets, keep GPU time moving: after
the current dataset reaches a terminal state, immediately start or advance the
next dataset if one remains. Prepare the sweep YAML for every dataset in the
batch before launching the first dataset, validate them together, and include
them in the same GitHub sync/pull handoff. Do not wait for one dataset to finish
before creating and syncing the next dataset's sweep YAML. Confirm the next
dataset has entered a healthy running state, then come back to pull and organize
the previous dataset's results.
