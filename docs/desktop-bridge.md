# Experiment Wake Bridge

The Mac bridge polls only tagged tmux sessions on HCCS-25. It does not launch
commands, allocate GPUs, inspect W&B, validate results, or maintain an
experiment ledger. A watched tmux session is the remote execution fact.

Register a session by setting these tmux user options after its panes exist:

```text
@codex_thread_id
@codex_generation
@codex_investigation_id
@codex_started_at
@codex_expected_seconds
@codex_attention_after
```

Set `remain-on-exit=on`, then set `@codex_watch=1` last. The bridge ignores
incomplete registrations. It emits one attention event and one terminal event
per generation. Attention never terminates a pane.

Events are kept in a mode-600 JSON outbox. Delivery uses the stable event id as
`clientUserMessageId`. A blocked Goal is resumed; active, paused, or limited
Goals are deferred; complete, missing, and archived Goals are marked orphaned.
Pending events remain until delivery is resolved. Delivered and orphaned
records remain only while their source tmux generation exists, and the outbox
has a hard record limit; reaching it makes bridge health fail explicitly.

For pending delivery, the bridge starts the bundled Codex
`app-server --stdio` process, checks the target task and Goal, and submits at
most one accepted turn per poll before closing the process. Remaining events
stay pending until a later poll rechecks the task and Goal. A delivered record
means app-server accepted `turn/start`; it does not mean Codex finished the
turn. The bridge does not require a standalone Codex daemon or app-server
control socket.

Create a config from `config/desktop-bridge.example.json`, then run:

```bash
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json dry-run
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json once
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json run
```

Install the in-place LaunchAgent replacement with:

```bash
python3 scripts/manage_codex_bridge_launchd.py install \
  --config ~/.config/experiment-console/bridge.json
```

The previous Console bridge config and LaunchAgent plist should be retained as
disabled rollback artifacts during the initial production validation.
