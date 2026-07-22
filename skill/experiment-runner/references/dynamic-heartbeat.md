# Dynamic Codex Heartbeat For Long Experiments

Use Codex's built-in thread heartbeat as a best-effort scheduled follow-up. It
does not observe HCCS continuously and does not prove result readiness.

## One-Shot Ownership

- Keep exactly one active heartbeat per Codex task.
- Use one-shot scheduling, never a recurring rule.
- Update the existing automation after each incomplete check.
- On manual resume, consume or cancel the pending heartbeat before rescheduling.
- Bind the prompt to the investigation, sweep, manifest, target task, and a
  schedule generation. A stale generation performs no experiment action.

Keep the first prompt complete enough to reconstruct the handoff. Keep ordinary
intermediate prompts to one short instruction that names the investigation,
manifest, and sweep and asks Codex to apply this policy. The final completion or
error turn may use a complete verification prompt again.

## Ordinary Heartbeat Fast Path

Use this read-only fast path when the previous checkpoint was healthy and no launch,
pane-topology change, error, completion, or other material transition is being
handled:

1. Load only `research`, `experiment-runner`, and this reference. Do not load
   `research-investigation` unless this wake escalates to a material record or
   scientific decision.
2. Read the existing heartbeat and run manifest.
3. Run one compact structured probe covering expected pane liveness and
   process/GPU ownership plus W&B status, declared artifacts, and manifest
   identities. Prefer an existing project validator and bounded machine-readable
   output containing counts, identities, and errors rather than per-run rows.
4. Classify the result as complete, error, zero-progress, or healthy incomplete.
5. For healthy incomplete, calculate the delay, obtain a fresh clock reading
   after all checks, update the same heartbeat for
   `ceil_to_minute(fresh_now + next_delay)`, give a one-line result, and stop.

On a healthy incomplete fast path, do not capture pane logs, inspect Git or
source, reread skills, repeat tmux/GPU/process probes, edit files, launch or stop
work, or interpret science. Escalate only when the evidence changes the decision;
then load the additional skills and diagnostics needed for that condition.

## Valid Progress

Let:

- `N` be the manifest's fixed expected trial count;
- `k` be the number of unique valid completed trials;
- `elapsed` be wall time since the first formal W&B agent pane launched.

A trial contributes to `k` only when all three agree:

1. W&B reports the expected terminal state;
2. every declared result artifact exists and parses;
3. trial, seed, run, config, registry, and code identities match the manifest.

Any failed/crashed run, missing artifact, parse failure, or identity mismatch
requires stopping the sweep and entering diagnosis. Whether to preserve, wait
for, or kill a pane remains Codex's evidence-based decision.

## Scheduling Policy

Before the first valid completion, select a comparable smoke duration `S_ref`:

- same dataset/trial family, main training budget, and GPU class when possible;
- median of two or three comparable successful smoke runs;
- the single observation when only one exists;
- scale a reduced smoke only when an explicit workload ratio is valid;
- use five minutes when no comparable smoke exists.

Set the first delay to:

```text
clamp(1.5 * S_ref, 5 minutes, 15 minutes)
```

If a wake still has `k = 0`, back off the previous delay by `1.5`, capped at 15
minutes, only when task-owned progress is visible. Otherwise diagnose now.

Once `k > 0`, use the deliberately naive cumulative estimator:

```text
speed = k / elapsed
ETA = (N - k) / speed
next_delay = clamp(ETA / 2, 5 minutes, 15 minutes)
```

Do not adjust the formula for GPU count, live agent count, or trial-count
changes. When ETA falls below five minutes, enter sticky final polling and keep
five-minute one-shot checks until complete or invalid. Always anchor the next
delay to a fresh time read immediately before updating the automation; time
spent loading skills or validating progress must not shorten the requested wait.

## Wake Actions

At a wake, first re-read live HCCS/W&B/manifest/artifact facts. Then:

- complete: cancel/consume the heartbeat, run full aggregate and independent
  replay, update the manifest, and continue the investigation;
- error: schedule nothing, stop the sweep, and diagnose;
- healthy incomplete: calculate the next delay and update the same heartbeat;
- zero progress without convincing task-owned activity: schedule nothing until
  diagnosis resolves the condition.

Only material transitions belong in Git: launch/first schedule, first valid
completion, entry into sticky five-minute mode, error/stop, completion, and
automation replacement/orphaning. Ordinary ETA checks stay out of Git.

## Availability Boundary

The heartbeat runs locally and may be delayed while the Mac, Codex, or network
is unavailable. Remote tmux and W&B work continues. On recovery, reconstruct
state from authoritative sources. If the heartbeat does not resume a blocked
Goal, leave the Goal blocked for manual recovery; do not add a bridge, external
scheduler, watchdog, or active model polling.
