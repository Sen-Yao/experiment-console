# Yggdrasil Production Runbook

Experiment Console runs as one authoritative control plane on Yggdrasil. The
Mac runs only the Desktop bridge and an SSH local-forward supervisor. It does
not hold a second ledger and does not monitor experiments with a model turn.

## Production Boundaries

- Yggdrasil API: published only on `127.0.0.1:5174` and reached through SSH.
- Authority: `EXPERIMENT_CONSOLE_AUTHORITY_ROLE=authoritative` and stable
  `EXPERIMENT_CONSOLE_INSTANCE_ID=yggdrasil-production`.
- Durable state: `console.sqlite3`, schedules, outbox, leases, and audit files
  live under `EXPERIMENT_CONSOLE_STATE_PATH`.
- SQLite uses
  `/mnt/user/appdata/experiment-console/state/console.sqlite3`, and temporary
  files use the sibling `state/sqlite-tmp` directory instead of the container's
  bounded `/tmp` tmpfs.
- Durable results: artifact bundles use the separate
  `EXPERIMENT_CONSOLE_RESULTS_PATH` bind mount. A Mac `/Users/...` artifact
  path must never be sent to the remote Console. The bridge downloads only
  controlled bundles exposed by the Console artifact API.
- One container owns the monitor worker. SQLite lease ownership prevents a
  second worker from acting if an accidental duplicate is started.
- W&B/process/artifact mismatches and external monitor failures are recorded on
  the first observation, but bridge events are emitted only after the configured
  consecutive-count and grace-time thresholds. Any mismatch blocks queue
  advancement immediately; the grace window delays escalation, not safety.
- W&B dependency health is aggregated by endpoint/credential scope and SSH
  health by remote host. Transient failures remain `*_unavailable_reconciling`
  until both 15 minutes and three attempts have elapsed; attention does not
  stop the bounded 10-minute retry loop. Explicit authentication and SSH
  configuration failures pause immediately for repair.

## Four-Credential Isolation

1. Mac to Yggdrasil tunnel key: used only by the Desktop bridge. Its
   Yggdrasil `authorized_keys` entry must restrict the key to the Console
   forward, for example:

   ```text
   restrict,port-forwarding,permitopen="127.0.0.1:5174",command="/bin/false" ssh-ed25519 PUBLIC_KEY_ONLY codex-console-bridge
   ```

   `restrict` disables PTY, agent forwarding, X11 forwarding, and user rc;
   `port-forwarding` re-enables only forwarding constrained by `permitopen`.
   OpenSSH `restrict` does not by itself forbid an exec request, so the forced
   `/bin/false` command is required. The bridge uses `ssh -N`, which opens no
   session command and remains able to forward. Do not reuse the deployment
   administrator identity.

2. Yggdrasil to HCCS service key: pre-provisioned at the path named by
   `EXPERIMENT_CONSOLE_HCCS_SSH_KEY_FILE`. Compose mounts it read-only. The
   non-root entrypoint copies it into a private tmpfs as mode `0600`. It is not
   the Mac tunnel key.

   Console reuses each OpenSSH transport for at most 60 seconds through a
   control socket in the container's private `/tmp` tmpfs. This avoids repeated
   HCCS key exchanges during one operation; the socket is not durable, does not
   contain credentials, and is not part of agent launch receipts.

3. W&B token: pre-provisioned as a one-line file named by
   `EXPERIMENT_CONSOLE_WANDB_SECRET_FILE`. Compose exposes it as a read-only
   Docker secret and sets only `WANDB_API_KEY_FILE`. The token is never placed
   in Compose environment values, `docker inspect`, the image, Git, SQLite, or
   deployment output.

4. Console API token: pre-provisioned as a separate high-entropy one-line file
   named by `EXPERIMENT_CONSOLE_API_TOKEN_SECRET_FILE`. Compose mounts it as
   `/run/secrets/console_api_token`; bridge claim/ack, artifact download,
   intent execution, and runner APIs require `Authorization: Bearer`. The Mac keeps a mode `0600`
   copy outside the repository. Do not reuse either SSH key or the W&B token.

No script in this repository creates, prints, or uploads a private key, W&B
token, or Console API token.

## One-Time Yggdrasil Provisioning

Create `/mnt/user/appdata/experiment-console/{config,secrets,state,results}`
outside the repository. Copy
`deploy/yggdrasil/production.env.example` to
`config/production.env`, populate the current non-secret HCCS routes from
`deploy/yggdrasil/hccs_ssh_config.example`, and install a pinned
`config/hccs_known_hosts` file. Provision the W&B, Console API, and HCCS secret
files through the approved credential workflow. Use UID/GID `10001:10001`;
secret files must be owner-readable only.

The production env file contains paths and identity settings, never secret
values. `activate-release.sh` refuses a non-authoritative role, a development
instance id, missing credential files, or any shared credential path.

## One-Time Fresh V2 Cutover

The v2 controller is a hard cutover. It does not import jobs, intents, monitor
schedules, wake events, dependency episodes, results, or W&B sweep references
from the previous ledger. There is no runtime compatibility switch.

Before applying the cutover, verify every job in the current authority is
terminal. Activation repeats this check inside the running source container and
refuses to stop it when any job is `planned`, `queued`, `validating`, `running`,
`finalizing`, `attention`, or `unknown`.

Activation then:

- stops the current Console and creates a paired, checksummed cold backup of
  state and results;
- empties both hot paths and lets the v2 application create a new SQLite
  ledger at `/mnt/user/appdata/experiment-console/state/console.sqlite3`;
- requires contract `runner_console_agent_v2`, schema version `2`, the two
  dependency-health tables, a changed ledger id, zero operational records, and
  no `cutover_committed_at` value;
- writes a read-only receipt under
  `/mnt/user/appdata/experiment-console/cutovers/` with the old/new ledger ids
  and backup path.

The operation is one-time. A second fresh cutover is refused after the first v2
Job write records `cutover_committed_at`.

## Deploy And Verify

All mutating wrappers are dry-run by default and use the direct Yggdrasil route
`ssh-direct.senyao.org:4622`. Review the output before adding `--apply`.

```bash
./scripts/deploy_yggdrasil_experiment_console.sh \
  --fresh-v2-ledger

./scripts/deploy_yggdrasil_experiment_console.sh --apply \
  --fresh-v2-ledger

./scripts/verify_yggdrasil_experiment_console.sh \
  --expected-instance yggdrasil-production \
  --require-fresh-v2-cutover \
  --require-empty-ledger
```

Fresh cutover and seed migration flags are mutually exclusive. Verification
before the smoke requires:

- healthy non-root `linux/amd64` container;
- loopback-only published API port;
- read-only credential/config mounts and writable state/results mounts;
- `authority_role=authoritative`, matching `instance_id`, contract
  `runner_console_agent_v2`, schema version `2`, and a ledger id different from
  the receipt's previous id;
- worker enabled, ready, running, and holding the singleton lease;
- non-empty Console API secret mounted read-only; protected APIs reject missing
  or incorrect bearer credentials;
- SQLite `quick_check=ok`, including `dependency_episodes` and
  `dependency_impacts`;
- zero jobs, intents, schedules, wake events, source observations, dependency
  episodes, and dependency impacts, with no `cutover_committed_at`;
- direct Console-to-W&B and Console-to-HCCS W&B authentication probes.

Only after this verification should the runner and stopped Desktop bridge be
repinned to the receipt's new ledger id. Then run the single-run/single-agent
smoke. After the smoke, run the verifier again without
`--require-empty-ledger`; the health response must now include a non-empty
`cutover_committed_at`. Do not run local and Yggdrasil mutating Consoles in
parallel.

## Backup And Rollback

Release activation builds before downtime, stops the current Console, then
creates a paired state/results archive in the same stopped window. Both
archives have SHA-256 checksums. A failed start or health check automatically
restores the previous release and both persistent paths.
On the first deployment, where no previous release exists, rollback restores
the pre-seed state/results snapshot, removes the `current` symlink, and leaves
the service stopped instead of attempting to restart the failed first release.

Manual backup is also dry-run first:

```bash
./scripts/backup_yggdrasil_experiment_console.sh
./scripts/backup_yggdrasil_experiment_console.sh --apply
```

Manual rollback requires an explicit backup path and `--apply`:

```bash
./scripts/rollback_yggdrasil_experiment_console.sh \
  --backup-dir /mnt/user/appdata/experiment-console/backups/BACKUP_ID

./scripts/rollback_yggdrasil_experiment_console.sh --apply \
  --backup-dir /mnt/user/appdata/experiment-console/backups/BACKUP_ID
```

Rollback retains the displaced state/results directories for a second recovery
path rather than deleting them.

For the fresh v2 cutover, full rollback is permitted only while
`cutover_committed_at` is absent. The first v2 Job write sets that metadata.
Afterward `rollback-release.sh` refuses the restore and operations must deploy a
forward fix against the new ledger.
