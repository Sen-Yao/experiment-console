# Experiment Console Desktop Bridge

The Desktop bridge is a small Python standard-library process. It keeps an SSH
local forward to the authoritative Yggdrasil Console and only contacts Codex
when the Console claims an actionable outbox event. Empty polls never connect
to app-server and never start a model turn.

The Codex handshake and method fields follow the official
[Codex app-server protocol](https://learn.chatgpt.com/docs/app-server):
`initialize`, `initialized`, `thread/read`, `thread/goal/get`, `thread/resume`,
and `turn/start`.

## Configuration

Copy the shape in `config/desktop-bridge.example.json` to
`~/.config/experiment-console/bridge.json`, replace the placeholder absolute
paths, and set mode `0600`. The SSH key must be dedicated to local forwarding;
the config contains paths only and must not contain private-key or bearer-token
contents.

`codex_command` should use an absolute Codex executable path under `launchd`.
The default command is `codex app-server proxy`, which connects to the running
Codex Desktop app-server control socket. It does not start the app-server daemon.
Set `codex_socket` only when Desktop uses a non-default control socket.
`console_token_file` must point to the local mode `0600` copy of the production
Console API bearer token. The token value is never placed in the bridge JSON,
plist, command line, event journal, or logs.

## Commands

```bash
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json dry-run
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json once
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json status
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json health
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json run
```

`dry-run`, `status`, and `health` do not open SSH, claim events, or contact
Codex. `once` owns and closes its temporary SSH process before exiting.

The first healthy production connection pins the Console `ledger_id` in the
local delivery journal. A different role, instance id, or ledger id fails
closed. After an intentional production ledger rebuild, replace the pin
explicitly while the bridge is stopped:

```bash
python3 -m desktop_bridge --config ~/.config/experiment-console/bridge.json \
  repin-authority --ledger-id NEW_LEDGER_ID
```

## LaunchAgent

Render the plist for inspection without loading it:

```bash
python3 scripts/manage_codex_bridge_launchd.py render \
  --config "$HOME/.config/experiment-console/bridge.json" \
  --plist "$HOME/Library/LaunchAgents/org.senyaolab.experiment-console-bridge.plist"
```

Install and load it after inspection:

```bash
python3 scripts/manage_codex_bridge_launchd.py install \
  --config "$HOME/.config/experiment-console/bridge.json"
```

Remove it with:

```bash
python3 scripts/manage_codex_bridge_launchd.py uninstall
```

`launchd` restarts a crashed bridge. The bridge itself supervises `ssh -N -L`
with `ExitOnForwardFailure`, server keepalives, health probes, and bounded
exponential reconnect. A sleep/wake time jump forces the old SSH process down
and immediately creates a fresh tunnel before polling resumes.
An advisory singleton lock prevents a manual `once`/`run` process from racing
the LaunchAgent and delivering the same event concurrently.
SSH always uses `IdentitiesOnly=yes`; with the configured `identity_file`, it
cannot silently authenticate with an administrator key from ssh-agent or a
default identity file.

## Delivery semantics

The Console remains authoritative. Each claim repeats the authority role,
instance id, and ledger id and grants a per-event opaque lease token. The
bridge validates and pins that identity on every claim. Ack includes both the
expected ledger id and exact lease token; a database replacement or stale
claim therefore fails closed.

The bridge merges new events by `thread_id`. Before `turn/start` it atomically
stores an `inflight` record with a deterministic `clientUserMessageId`. A
successful submission is not immediately acknowledged. On the next claim, or
after a timeout/restart, the bridge uses `thread/read(includeTurns=true)` and
requires exactly one persisted `UserMessage.clientId` match before marking the
events delivered and acking them. If no match exists, it waits through a short
grace period and retries the same id only while the thread is idle. Duplicate
matches, malformed history, and other ambiguity enter durable `uncertain`
state and are never automatically acked or retried.

Before a new submission, app-server calls are ordered as `thread/resume`,
`thread/goal/get`, and a final `thread/read` idle check immediately before
`turn/start`. Busy/offline/active-Goal/error paths do not ack, and the bridge
never clears a Goal itself.

Current app-server does not expose an atomic `rejectIfActive` or
`expectedStatus=idle` precondition on `turn/start`; it may steer input if a
regular turn becomes active after the final read. The ordering above narrows
that TOCTOU window but cannot eliminate it. Until app-server provides that
primitive, production should use a dedicated research task, keep Goals paused
during external waits, and treat concurrent manual turns as an operational
conflict rather than claiming strict exactly-once turn creation.

The local ledger retains bounded delivered/acked history. If ack fails after a
confirmed history match, a later claim is acked without starting another turn.
