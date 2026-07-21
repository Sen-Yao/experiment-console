---
name: experiment-runner
description: "Use when Codex needs to inspect HCCS-25 GPUs, prepare a committed execution worktree, launch or inspect tmux experiment panes, operate native W&B sweeps, stop a pane, or hand a long experiment to the Goal wake bridge."
---

# Experiment Runner

Treat this skill as operational knowledge, not a runtime service. Run commands
directly over the `HCCS-25` SSH alias. Use native Git, tmux, NVIDIA, conda, and
W&B interfaces; do not route new work through Experiment Console v3.

## Ownership Boundary

Codex owns server/GPU choice, Git state, worktree creation, commands, W&B,
retry/stop decisions, result validation, and scientific interpretation. The Mac
bridge only observes tagged tmux panes and wakes the bound Codex Goal.

Do not add a queue, GPU allocator, experiment database, remote daemon, W&B
adapter, watchdog, or result aggregator to this skill. Keep repeated scientific
operations in project scripts and W&B sweep configs.

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

Choose GPUs explicitly from both memory/utilization and live compute-process
evidence. Do not rely on a previous probe, kill another process, or assume that
an apparently idle GPU belongs to this task.

Read [references/hccs25-runtime.md](references/hccs25-runtime.md) when preparing
Git, conda, credentials, proxy, or storage on HCCS-25.

## Freeze Before Formal Work

Use `/home/linziyao/DualRefGAD` only as the mutable development checkout. Make
all code edits, tests, commits, pushes, and Git operations there. A formal run
must use a detached worktree created from the committed full SHA; never run it
from the mutable checkout.

Before launch, commit and push both:

- the code/config commit used by the run;
- the research protocol/config and prepared run manifest.

The run manifest binds the investigation, code SHA, research protocol commit,
config digest, seeds, W&B destination, and later the sweep/run/result digests.
Do not require an atomic transaction across Git, HCCS, and W&B. Record the
manifest state as `prepared`, `launched`, `completed`, or `invalid`.

## Execution Unit

Default to one W&B run per independently retryable trial: one dataset, seed,
and resolved method/config identity. Keep epochs and checkpoints from the same
training trajectory together, but split hidden loops over datasets, seeds,
method families, readers, or other independently retryable configurations.

Keep tightly coupled branches together only when shared initialization, state,
or RNG is part of the scientific estimand. Expose per-branch progress,
artifacts, timing, and bounded recovery. Inspect the entrypoint and config for
hidden scans; a seed-only sweep grid does not prove the trial is atomic.

## W&B Native Sweeps

Prefer native W&B sweeps and agents for parameter assignment, run identity,
aggregation, and audit. Read
[references/wandb-native-sweeps.md](references/wandb-native-sweeps.md) before
creating a sweep or diagnosing network/auth failures.

Run the direct network/project preflight before `wandb.sweep`; OpenClash is a
diagnosed network fallback, not the default path and not an authentication
repair. Create a sweep exactly once, record its ID in the run manifest, then
start one tmux pane per explicitly selected GPU. W&B agents may run in
parallel. Do not implement local sweep scheduling or automatically downgrade
to offline mode.

For SenyaoLab investigations, the standing user authorization allows execution
on HCCS and transmission to `HCCS/DualRefGAD` of experiment config, seed, run
status, metadata, AUROC, and AP. It excludes raw data, source code, credentials,
and undeclared artifacts. Ask again only if destination, data classes, or the
investigation lifecycle changes; experiment count is not an authorization
limit unless the user or platform explicitly makes it one.

## Tmux And Goal Handoff

Read [references/tmux-goal-watch.md](references/tmux-goal-watch.md) for the
exact registration contract. In summary:

1. Create the session and all experiment panes in the detached worktree.
2. Set window `remain-on-exit=on`.
3. Set thread, generation, investigation, start, expected, and attention
   options on the session.
4. Set `@codex_watch=1` last.
5. Verify the bridge sees the session before handing off.

Use `expected_seconds` and `attention_after` as advisory timing. An overrun only
wakes Codex; it never stops work. When external execution is the only remaining
dependency, checkpoint the manifest/investigation, follow current Goal rules to
enter `blocked`, and end the turn. Do not keep a Goal active for model polling.

After a wake event, re-probe tmux, processes, GPUs, W&B, the run manifest, and
declared artifacts. A pane terminal state alone is not scientific readiness.

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

- Non-interactive SSH does not automatically load proxy variables. This is
  correct for the direct default. Source the mode-600 HCCS environment without
  printing it only after a diagnosed network failure selects proxy fallback.
- Native `wandb agent` needs the W&B backend for assignments. Offline runs do
  not preserve native sweep scheduling; any `offline-manual` fallback is a new
  explicit protocol.
- A native agent resolves assigned-trial `python` from `PATH`. Activate the
  project conda environment inside the tmux pane; invoking only the environment's
  `wandb` executable can still run trials with system Python.
- Set `@codex_watch=1` last. A fast command can finish before an incomplete
  registration becomes visible.
- `remain-on-exit` preserves normal terminal panes across bridge downtime.
  Killing the final pane may remove the session; cancellation is already an
  agent-observed action and does not rely on another wake.
- One generation emits at most one attention and one terminal event. Re-arm a
  continuing session with a new generation after Codex has handled attention.
- Never put W&B keys, proxy credentials, tokens, raw data, or source payloads in
  tmux options, argv, manifests, investigation prose, or logs.
