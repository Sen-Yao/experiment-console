# Experiment Console v3

Experiment Console is a small, durable remote command executor for Codex
agents. It runs a structured command on a named GPU server profile, records the
remote receipt, exposes bounded status/log/file reads, and notifies the task
that started the job when execution reaches a terminal state.

It is intentionally not a W&B controller, scheduler, result validator, artifact
store, or experiment policy engine. The agent chooses the server, GPU, command,
retry policy, and scientific interpretation.

## Runtime Shape

```text
Codex agent
  -> ./scripts/exp resources|run|status|logs|fetch|cancel
  -> SSH tunnel
  -> Console v3
  -> server_profile + remote helper
  -> GPU server
```

Console stores only a v3 SQLite ledger with `jobs`, `events`, `resource_locks`,
and `outbox`. Every job has a remote receipt. A fixed monitor worker checks
active jobs without per-job schedules or model polling.

## Runner

The repository wrapper is the canonical command shape:

```bash
./scripts/exp resources --profile hccs-25

./scripts/exp run \
  --profile hccs-25 \
  --cwd /home/linziyao/DualRefGAD \
  --gpu 0 \
  --total-runs 10 \
  --env MODE=experiment \
  -- python train.py --dataset elliptic

./scripts/exp status job_<id>
./scripts/exp logs job_<id> --stream stderr
./scripts/exp fetch job_<id> results/summary.json --output ./summary.json
./scripts/exp cancel job_<id> --reason "stop requested"
```

`run` uses `CODEX_THREAD_ID` automatically when available. Pass
`--task-id` outside Codex. A generated `request_id` is printed on every
submission; pass `--request-id` again after a network timeout to make the retry
idempotent.

The command receives `EXPERIMENT_CONSOLE_JOB_ID` and
`EXPERIMENT_CONSOLE_PROGRESS_FILE`. Write progress JSON there when the command
contains multiple runs. Console calculates ETA only after at least one
completed run.

## Local Development

Requirements: Python 3.11+, the project dependencies, and SSH access to the
profiled host for real execution.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
./scripts/run_tests.sh
```

Start an isolated local Console:

```bash
./scripts/start_local_console.sh
```

It serves `http://127.0.0.1:5174` with the local `config/server-profiles.json`
and a disposable `.state-v3` ledger. Set
`EXPERIMENT_CONSOLE_EXPECTED_INSTANCE_ID=local-experiment-console-v3` before
using the runner against it.

## Deployment

Yggdrasil runs one Compose service with one mounted v3 state directory, one
server-profile file, SSH configuration, and a bearer token. See
[`compose.yggdrasil.yaml`](compose.yggdrasil.yaml). The service may be stopped
for backup and replacement; v2 state is never imported.

The Desktop bridge keeps the SSH tunnel alive and delivers terminal events:

```bash
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json run
```

The bridge config example is [`config/desktop-bridge.example.json`](config/desktop-bridge.example.json).

## API Surface

- `GET /health`
- `GET /api/resources`
- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/logs`
- `GET /api/jobs/{job_id}/files`
- `POST /api/jobs/{job_id}/cancel`
- `POST /api/outbox/claim`
- `POST /api/outbox/{event_id}/ack`

There is no `/api/runner` compatibility namespace, W&B endpoint, intent API,
frontend, result aggregation endpoint, or artifact bundle endpoint.
