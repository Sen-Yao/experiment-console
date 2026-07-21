---
name: experiment-runner
description: "Use when Codex needs to inspect GPU resources, start or observe a durable remote experiment command, read its console output, fetch an explicit result file, or cancel an Experiment Console v3 job."
---

# Experiment Runner

Use the v3 runner as the only Experiment Console client. It submits structured
commands to the Yggdrasil executor; it does not execute SSH, W&B, scheduling, or
result-analysis side effects locally.

Within the Console repository, use the stable wrapper:

```bash
cd /Users/oliver/Developer/experiment-console
./scripts/exp <command> ...
```

## Boundary

Console owns durable process execution, explicit GPU locks, remote receipts,
bounded logs and file reads, fixed monitoring, cancellation, and terminal task
notification. Agent owns GPU choice, command construction, ordering, retries,
W&B access, and scientific result interpretation.

Do not add experiment-framework control, scheduling policy, result validation,
aggregation, local profile validation, or raw Console API commands to runner.

## Execution Unit

Default to one runner job per resolved trial attempt: one dataset, seed, and
method/config identity. Keep epochs and checkpoints from the same training
trajectory in that job, but split hidden loops over datasets, seeds, method
families, readers, or other independently retryable configurations.

Tightly coupled branches may remain in one job only when shared initialization,
state, or RNG is part of the scientific estimand. Such jobs must expose
per-branch progress, artifacts, timing, and bounded recovery. Before submission,
inspect the entrypoint and config for hidden scans; the top-level `argv` or a
seed-only W&B grid does not prove that a job is atomic.

## Commands

Inspect a server profile before choosing GPUs:

```bash
./scripts/exp resources --profile hccs-25
```

Run one durable command. Pass the command after `--`; runner automatically uses
`CODEX_THREAD_ID` when available.

```bash
./scripts/exp run \
  --profile hccs-25 \
  --cwd /home/linziyao/DualRefGAD \
  --gpu 0 \
  --total-runs 5 \
  -- python train.py --dataset elliptic
```

Read execution state, logs, or an explicit file:

```bash
./scripts/exp status job_<id>
./scripts/exp logs job_<id> --stream stderr
./scripts/exp fetch job_<id> results/summary.json --output ./summary.json
```

Cancel only when the user or current task has authorized termination:

```bash
./scripts/exp cancel job_<id> --reason "requested by user"
```

After `run`, retain the printed `request_id`. If submission times out, retry
with `--request-id <same-id>`; do not create a new id until status proves the
first request did not create a job.

## Progress

The command receives `EXPERIMENT_CONSOLE_JOB_ID` and
`EXPERIMENT_CONSOLE_PROGRESS_FILE`. A multi-run program may atomically replace
the progress file with:

```json
{"completed_runs": 3, "total_runs": 10, "message": "run 3/10"}
```

Console estimates ETA only after one completed run. Treat it as a transparent
arithmetic estimate, not a completion guarantee.

When a healthy job is the only remaining dependency, rely on Console's terminal
event and end the current turn. Do not keep a Codex Goal active solely for the
wait, and do not create model polling or heartbeat tasks.

## Gotchas

- **Wrong Console instance**: runner reports an instance mismatch when the
  loopback port targets local development or an old service. Repair the Desktop
  bridge/tunnel or set the expected local instance for isolated testing; do not
  mutate client identity state.
- **GPU rejected after resource inspection**: another job acquired the atomic
  lock between observations. Query `resources` again and let the agent choose;
  Console does not queue or auto-select another GPU.
- **Hidden experiment matrix**: do not package independently retryable trials
  into one job merely to reduce Console or W&B run count. Split the trials or
  record a scientifically necessary coupling exception before launch.
- **No ETA**: the command has not published a completed run. Check `logs` and
  the progress writer instead of asking Console to parse W&B or ordinary output.
- **Fetch denied**: the requested path resolves outside the job working
  directory. Fetch only an explicit result or `.experiment-console-v3` log path
  owned by that job.
