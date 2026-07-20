# Experiment Console Agent Notes

## v3 Boundary

This repository contains the v3 durable remote command executor. The only
agent-facing entry point is:

```bash
./scripts/exp <resources|run|status|logs|fetch|cancel> ...
```

The runner submits a structured `argv[]`, `env`, named `server_profile`, remote
working directory, explicit GPU indices, optional `total_runs`, and the current
Codex task id. Console owns only remote execution, receipt tracking, resource
locks, bounded log reads, progress/ETA calculation, file fetches, cancellation,
fixed-interval monitoring, and terminal outbox delivery.

Agent owns server/GPU choice, sequencing, retry decisions, result validity,
W&B interaction, and scientific interpretation. Do not add W&B, sweep, queue,
artifact, result-contract, scheduler, or experiment-policy concepts to Console.

## Safety

- `server_profile` is read-only deployment configuration. Runner may select a
  profile but cannot create or mutate one.
- `cwd` must be inside the profile's absolute `allowed_roots`.
- `argv` is structured. Do not add a free-form shell command API.
- API `env` rejects secret-like keys. Provision credentials in the remote
  profile/bootstrap instead of persisting them in the Console ledger.
- Console locks explicitly requested GPUs atomically and rejects conflicts; it
  never queues, swaps, or auto-selects GPUs.
- Remote receipts bind job id, command digest, PID, process group, and Linux
  `/proc` start time. Never terminate a process by a fuzzy command match.
- `cancel` sends `SIGTERM` to the owned process group, waits the configured
  grace period, then sends `SIGKILL`.
- `fetch` is limited to files under the job working directory and uses bounded
  chunks. Never accept a local Mac path as a remote path.
- v3 uses a new empty ledger. Do not import or reinterpret v2 state.

## Verification

Run the focused checks after changes:

```bash
./scripts/run_tests.sh
./scripts/exp --help
```

Use `python3 -m pytest` for a fast local run. Production uses the Yggdrasil
Compose service and the Desktop bridge tunnel; do not start a second mutating
Console on the production port.

## Progress Protocol

The remote command receives `EXPERIMENT_CONSOLE_JOB_ID` and
`EXPERIMENT_CONSOLE_PROGRESS_FILE`. A training program may atomically replace
that file with JSON such as:

```json
{"completed_runs": 3, "total_runs": 10, "message": "run 3/10"}
```

Console does not parse ordinary output or call W&B to infer progress. ETA is a
transparent estimate based on elapsed time and completed runs; with no completed
run it is `null`.

## Bridge

The bridge maintains only the SSH tunnel, a singleton lock, bounded status JSON,
and at-least-once outbox delivery. It uses the stable outbox `event_id` as
`clientUserMessageId`; repeated deliveries are safe to deduplicate. It does not
pin ledger ids or keep a delivery history.

## Cleanup Rule

Do not restore deleted v2 modules, compatibility aliases, old references, or
archived SOPs to active paths. Historical material belongs outside the active
runner skill and v3 runtime.
