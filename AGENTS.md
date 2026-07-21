# Experiment Wake Bridge Agent Notes

## Active Boundary

The only active production component in this repository is the Mac
`desktop_bridge`. Codex launches experiments directly on HCCS-25 according to
the installed `experiment-runner` skill. The bridge performs fixed read-only
tmux inspection and Goal-aware Codex app-server delivery.

Do not add W&B, sweep, GPU scheduling, experiment policy, result validation,
artifact storage, remote process ownership, or scientific interpretation to
the bridge. It owns only:

- polling tmux sessions tagged with `@codex_watch=1`;
- detecting pane failure, attention deadlines, and all-pane terminal state;
- restart-safe local event deduplication;
- bounded pane capture for actionable events;
- blocked-Goal delivery through Codex app-server.

## Tmux Contract

A watched session supplies `@codex_thread_id`, `@codex_generation`,
`@codex_investigation_id`, `@codex_started_at`, `@codex_expected_seconds`, and
`@codex_attention_after`. Set window `remain-on-exit=on` and write
`@codex_watch=1` last. The bridge ignores incomplete registrations.

Attention only wakes Codex. It never kills a pane. Codex owns process/GPU/W&B
inspection and may choose `tmux kill-pane` after reviewing evidence. One
attention event and one terminal event are allowed per generation.

Do not place credentials, proxy URLs with embedded credentials, source code,
or raw data in tmux option values. W&B, GitHub, and proxy credentials belong in
mode-600 HCCS user credential stores.

## Goal Delivery

- blocked Goal: deliver with stable `clientUserMessageId`;
- active, paused, usage-limited, or budget-limited Goal: defer;
- complete Goal, missing Goal/task, or archived task: mark orphaned and never
  reopen it.

The tmux event proves only process state. W&B, run-manifest, ResultContract,
and artifact readiness remain agent-side checks after wake-up.

## Legacy v3 Rollback

`backend/experiment_console`, `scripts/exp`, Compose files, and v3 tests are
retained as a disabled rollback implementation. Preserve their current user
changes, including foreign/unknown GPU process classification. Do not route new
experiments through v3 unless the user explicitly invokes whole-system
rollback. Never import or reinterpret old ledger state.

The retained resource-inspection rule is unchanged: classify GPU compute
processes against live v3 receipts; reject foreign or unknown ownership at v3
launch; keep a conflict observed after launch warning-only; never turn it into
a scientific failure or automatic process kill.

## Verification

Run:

```bash
./scripts/run_tests.sh
python3 -m desktop_bridge \
  --config config/desktop-bridge.example.json dry-run
```

For production changes, exercise the real HCCS tmux and Codex Goal path,
including bridge restart recovery and duplicate-event delivery.
