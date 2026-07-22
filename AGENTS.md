# HCCS Experiment Runner Agent Notes

## Active Boundary

This repository has no active custom control-plane service. Codex launches and
inspects experiments directly on HCCS-25 according to the installed
`experiment-runner` skill. Native W&B owns sweep assignment and aggregation;
tmux owns durable remote execution; Codex's built-in thread heartbeat owns
scheduled follow-up.

Do not add a bridge, watcher, daemon, scheduler, queue, experiment ledger, W&B
adapter, GPU allocator, result validator, artifact store, or scientific policy
engine. Keep repeated scientific operations in project scripts and W&B sweep
configs.

## Scheduled Follow-Up

Use one active one-shot heartbeat per Codex task. After each wake, re-read tmux,
exact processes/GPU state, W&B, the run manifest, and declared artifacts before
deciding whether the sweep is complete, invalid, or still running.

For a healthy incomplete sweep:

- count only trials whose W&B terminal state, parseable artifacts, and manifest
  identity all agree;
- use elapsed wall time from the first formal agent pane launch;
- calculate `speed = valid_completed / elapsed` and
  `ETA = remaining / speed`;
- schedule the next one-shot heartbeat after `clamp(ETA / 2, 5m, 15m)`;
- once ETA is below five minutes, keep five-minute checks until terminal.

Before the first valid completion, use a comparable smoke duration when one
exists: `clamp(1.5 * smoke_time, 5m, 15m)`. If a wake still has zero valid
trials, back off by 1.5 only when task-owned progress is visible; otherwise
diagnose immediately. Any error result requires stopping the sweep and entering
diagnosis. Pane handling remains an agent decision based on exact evidence.

Keep first and final heartbeat prompts complete. Keep intermediate prompts
minimal. Update the existing heartbeat instead of creating another one, cancel
or consume it on manual resume, and write Git only for material transitions.
Heartbeats are best-effort while the Mac or Codex is unavailable; failure stays
blocked for manual recovery and never reactivates a custom bridge.

## Tmux And Secrets

Create explicit tmux panes in the committed detached worktree and enable
`remain-on-exit` when terminal evidence must survive. No `@codex_watch` options
are required. Codex owns process/GPU/W&B inspection and may choose exact
`tmux kill-pane` after reviewing evidence; never fuzzy-match or broadly kill
processes.

Do not place credentials, proxy URLs with embedded credentials, source code, or
raw data in tmux options, automation prompts, argv, manifests, or logs. W&B,
GitHub, and proxy credentials belong in mode-600 HCCS user credential stores.

## Legacy v3 Rollback

`backend/experiment_console`, `scripts/exp`, Compose files, and v3 tests are
retained as a disabled rollback implementation. Preserve their current user
changes, including foreign/unknown GPU process classification. Do not route new
experiments through v3 unless the user explicitly invokes whole-system rollback.
Never import or reinterpret old ledger state.

The retained resource-inspection rule is unchanged: classify GPU compute
processes against live v3 receipts; reject foreign or unknown ownership at v3
launch; keep a conflict observed after launch warning-only; never turn it into a
scientific failure or automatic process kill.

## Verification

Run:

```bash
./scripts/run_tests.sh
./scripts/exp --help
```

For scheduling changes, exercise a real one-shot Codex thread heartbeat and
verify that only one future automation exists for the target task.
