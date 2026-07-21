# Tmux Goal Watch Contract

## Registration

Create all panes first. Then set the window/session options below, writing the
watch flag last:

```bash
tmux set-window-option -t <session> remain-on-exit on
tmux set-option -t <session> @codex_thread_id "$CODEX_THREAD_ID"
tmux set-option -t <session> @codex_generation <unique-generation>
tmux set-option -t <session> @codex_investigation_id <investigation-id>
tmux set-option -t <session> @codex_started_at <epoch-seconds>
tmux set-option -t <session> @codex_expected_seconds <seconds>
tmux set-option -t <session> @codex_attention_after <absolute-epoch-seconds>
tmux set-option -t <session> @codex_watch 1
```

For W&B agents, activate conda in the pane command or set
`PATH=<conda-env>/bin:$PATH` before `wandb agent`; an absolute `wandb` path does
not control the Python interpreter used for assigned trials.

The generation must change when Codex deliberately re-arms a session after an
attention event. Keep session and generation names to letters, digits, dots,
underscores, and hyphens.

## Event Semantics

- Any nonzero dead pane while a sibling remains alive: one attention event.
- Attention timestamp reached while work is alive: one attention event.
- All panes dead: one terminal event, independent of prior attention.
- A zero-exit pane while healthy siblings remain: no event.
- Multiple failures in one generation coalesce into the same attention event.

The bridge does not terminate panes, infer W&B state, or validate results.

## Goal Semantics

- blocked: deliver the event and resume the Goal;
- active, paused, usage-limited, or budget-limited: keep the event pending;
- complete, missing, or archived: mark the event orphaned and never reopen.

Delivery is at least once with the stable event ID as
`clientUserMessageId`. The local event JSON is mode `600` and survives bridge
restart.

## Inspection

Use exact pane IDs and bounded capture:

```bash
tmux list-panes -t <session> -F '#{pane_id}|#{pane_dead}|#{pane_dead_status}|#{pane_current_command}'
tmux capture-pane -p -t %<pane-id> -S -80
```

After wake-up, inspect tmux, exact processes, GPU state, W&B, run manifest, and
artifacts before deciding whether to wait, retry, or kill a pane.
