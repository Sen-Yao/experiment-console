# Desktop Bridge v3

The bridge runs on the Mac. It supervises one SSH local forward to Yggdrasil,
claims terminal events from Console, and starts a turn in the originating Codex
task. Healthy jobs produce no bridge messages.

The bridge has no ledger pin or delivery history. Console outbox events use a
stable `event_id`; the bridge sends it as `clientUserMessageId` and acknowledges
only after `turn/start` accepts the message. A failed ack causes the same event
to be claimed again, which is the intended at-least-once behavior.

Create a config from `config/desktop-bridge.example.json`, then run:

```bash
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json dry-run
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json once
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json run
```

`run` is the long-lived process. Use launchd or another local supervisor to
restart it; the bridge itself handles tunnel reconnect and sleep/wake gaps.

Render and install the provided LaunchAgent when desired:

```bash
python3 scripts/manage_codex_bridge_launchd.py render \
  --config ~/.config/experiment-console/bridge.json
python3 scripts/manage_codex_bridge_launchd.py install \
  --config ~/.config/experiment-console/bridge.json
```
