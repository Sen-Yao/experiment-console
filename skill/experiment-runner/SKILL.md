---
name: experiment-runner
description: "Use when Codex needs to inspect HCCS-25 GPUs, prepare a committed execution worktree, launch or inspect tmux experiment panes, operate native W&B sweeps, stop a pane, or schedule dynamic follow-up for a long experiment."
---

# Experiment Runner

Treat this skill as operational knowledge, not a runtime service. Run commands
directly over the `HCCS-25` SSH alias. Use native Git, tmux, NVIDIA, conda, W&B,
and Codex's built-in thread heartbeat. Do not route new work through Experiment
Console v3 or build a custom watcher.

## Ownership Boundary

Codex owns server/GPU choice, Git state, worktree creation, commands, W&B,
retry/stop decisions, result validation, heartbeat scheduling, and scientific
interpretation. Native W&B owns assignments and aggregation; tmux owns durable
remote process state; the run manifest binds evidence across systems.

Do not add a queue, GPU allocator, experiment database, remote daemon, bridge,
watchdog, W&B adapter, or result aggregator. Keep repeated scientific operations
in project scripts and W&B sweep configs.

## Start With Live Facts

HCCS is shared and drift-prone. Re-probe before every launch:

```bash
ssh HCCS-25 hostname
ssh HCCS-25 tmux -V
ssh HCCS-25 nvidia-smi \
  --query-gpu=index,name,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
ssh HCCS-25 nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
  --format=csv,noheader,nounits
```

Choose GPUs explicitly from memory/utilization and live compute-process
evidence. Do not rely on a previous probe, kill another process, or assume that
an apparently idle GPU belongs to this task.

Read [references/hccs25-runtime.md](references/hccs25-runtime.md) when preparing
Git, conda, credentials, proxy, or storage on HCCS-25.

## Freeze Before Formal Work

Use `/home/linziyao/DualRefGAD` only as the mutable development checkout. Make
all code edits, tests, commits, pushes, and Git operations there. A formal run
must use a detached worktree created from the committed full SHA.

Before launch, commit and push both:

- the code/config commit used by the run;
- the research protocol/config and prepared run manifest.

The manifest binds the investigation, code SHA, research protocol commit,
config digest, seeds, W&B destination, execution start, expected trials, and
later the run/result digests. Use states `prepared`, `launched`, `completed`, or
`invalid`; do not require an atomic transaction across Git, HCCS, and W&B.

## Execution Unit

Default to one W&B run per independently retryable trial: one dataset, seed,
and resolved method/config identity. Keep epochs and checkpoints from the same
training trajectory together, but split hidden loops over datasets, seeds,
method families, readers, or other independently retryable configurations.

Keep tightly coupled branches together only when shared initialization, state,
or RNG is part of the scientific estimand. Expose per-branch progress,
artifacts, timing, and bounded recovery.

## Native W&B And Tmux

Read [references/wandb-native-sweeps.md](references/wandb-native-sweeps.md)
before creating a sweep or diagnosing network/auth failures. Direct HTTPS is
the default; OpenClash is only a diagnosed network fallback.

Create the sweep exactly once, record its ID, then launch one tmux pane per
explicitly selected GPU in the committed worktree. Activate the project conda
environment or put its `bin` first in `PATH` before `wandb agent`.

Use `remain-on-exit=on` when terminal evidence must survive. Record the exact
session, pane, GPU, task id, and first formal agent-pane start time in the
manifest. No `@codex_watch` options or Mac observer are required.

For SenyaoLab investigations, standing authorization permits HCCS execution
and transmission to `HCCS/DualRefGAD` of experiment config, seed, run status,
metadata, AUROC, and AP. It excludes raw data, source code, credentials, and
undeclared artifacts. Reauthorize only when destination, data classes, or the
investigation lifecycle changes.

## Dynamic Heartbeat Handoff

Read [references/dynamic-heartbeat.md](references/dynamic-heartbeat.md) before
handing off a long run. Use one active one-shot heartbeat for the Codex task.
After scheduling it, checkpoint material launch state, follow current Goal rules
to enter `blocked`, and end the turn. Do not keep a Goal active for polling.

For an ordinary healthy wake, use the reference's read-only fast path: load no
`research-investigation`, run one compact structured probe, schedule from a
fresh post-check clock reading, report one line, and stop. Do not capture healthy
logs, inspect Git/source, or repeat probes. Leave the fast path only when evidence
requires completion, diagnosis, or another material decision.

At every automatic or manual resume:

1. consume, cancel, or update the existing heartbeat before creating another;
2. re-probe tmux, exact processes/GPU state, W&B, manifest, and artifacts;
3. stop the sweep and diagnose if any result is failed, invalid, missing, or
   identity-inconsistent;
4. if incomplete and healthy, update the same one-shot heartbeat using the
   bounded ETA policy;
5. if complete, schedule nothing further and run full aggregate/replay.

Heartbeat delivery is best-effort while the Mac, Codex, or local network is
unavailable. If it does not resume a blocked Goal, fail closed and wait for
manual recovery; never restore a custom bridge or use an active Goal as a
poller.

## Inspect And Stop

Use bounded tmux output and exact pane IDs:

```bash
ssh HCCS-25 "tmux list-panes -t <session> -F '#{pane_id}|#{pane_dead}|#{pane_dead_status}|#{pane_current_command}'"
ssh HCCS-25 "tmux capture-pane -p -t %<pane-id> -S -80"
```

Codex may stop an abnormal pane after inspecting evidence:

```bash
ssh HCCS-25 "tmux kill-pane -t %<pane-id>"
```

Use only `tmux kill-pane`. Accept that child processes can survive, then
re-probe exact process identity and GPU state. Never fuzzy-match or broadly kill
processes. Healthy sibling panes continue.

## Gotchas

- Non-interactive SSH correctly has no proxy variables on the direct path.
  Source the mode-600 fallback environment only after a diagnosed network
  failure, never to mask `401` or `403`.
- Native `wandb agent` needs the backend for assignments. Offline runs do not
  preserve native sweep scheduling.
- An absolute conda `wandb` executable does not select the assigned trial's
  Python; put the conda environment first in pane `PATH`.
- A one-shot heartbeat may be delayed by Mac sleep or Codex downtime. HCCS and
  W&B remain authoritative; reconstruct state after recovery.
- Never leave more than one active heartbeat for a task. Manual resume must
  consume or replace the existing schedule so a stale follow-up cannot race.
- Keep first and final heartbeat prompts complete, but intermediate prompts
  minimal. Write Git only for material transitions, not every ETA recalculation.
- Never put W&B keys, proxy credentials, tokens, raw data, or source payloads in
  tmux options, automation prompts, argv, manifests, or logs.
