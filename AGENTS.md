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

For read-only runner checks, prefer narrow subcommand prefixes such as:

```bash
./scripts/exp status --job-id job_...
./scripts/exp show-job --job-id job_...
./scripts/exp preflight --profile sweep --config /remote/config.yaml --remote-host HCCS-25 --remote-cwd /home/linziyao/DualRefGAD
```

For mutating commands such as `launch-sweep`, `launch-run`, `recover-agents`,
`stop`, or `cancel-sweep`, keep the same wrapper form but obtain explicit user
approval before starting, stopping, or recovering training.
