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

## Selective Runtime Migration

Freeze the local Console mutating path before migration. Run a dry-run first.
The allow-list contains exactly the current Elliptic training job and the
authorized Amazon queued job; names and broad `CJNV` matches are not accepted.

```bash
python3 scripts/migrate_runtime_to_yggdrasil.py \
  --source-state-dir /private/tmp/experiment-console-runtime \
  --target-state-dir /private/tmp/experiment-console-production-seed \
  --archive-dir "$HOME/Documents/Codex/experiment-console-archives" \
  --thread-id THREAD_ID_FOR_CURRENT_CJNV_TASK \
  --job-id job_20260711_233447_cjnv_a3_elliptic_fixedtail_training_permutation_audit_5x_85d0254a \
  --reconcile-job-id job_20260711_233447_cjnv_a3_elliptic_fixedtail_training_permutation_audit_5x_85d0254a \
  --result-contract job_20260711_233447_cjnv_a3_elliptic_fixedtail_training_permutation_audit_5x_85d0254a=deploy/yggdrasil/result-contracts/cjnv-a3-elliptic-training.json \
  --job-id job_20260711_225120_cjnv_a3_amazon_fixedtail_training_permutation_audit_5x5_4988b076 \
  --result-contract job_20260711_225120_cjnv_a3_amazon_fixedtail_training_permutation_audit_5x5_4988b076=deploy/yggdrasil/result-contracts/cjnv-a3-amazon-training.json
```

The dry-run must report `selected_count=2`, no ambiguities, Elliptic
`finished -> unknown`, and Amazon `queued -> queued`. Then repeat with:

```text
--apply --confirm-source-frozen
```

Migration behavior:

- archives the complete old runtime, including the large audit and historical
  results, plus a consistent SQLite backup;
- makes the archive and checksum read-only (`0400`);
- initializes the target schema through the current application, not copied
  legacy DDL;
- imports exactly the two allow-listed jobs and no old intents/events;
- clears operation payloads and cron/Hermes/OpenClaw/heartbeat metadata;
- preserves Amazon's queue group, payload, `created_at`, and sequential policy,
  while dropping the stale blocker id so the authoritative reconciler derives
  the real blocker;
- embeds a validated `ResultContract` for each job;
- creates one active monitor schedule per imported job with the same task id
  and `next_run_at` set to migration time;
- does not import the multi-gigabyte audit/results history into the hot state.

`migration_manifest.json` records the exact source-to-target status mapping,
contract digests, schedule count, reconcile list, original queue blocker, and
all excluded legacy categories. Any ambiguous terminal classification, missing
contract, missing explicit job id, source change during archive, or target
verification failure aborts the migration. Target installation is atomic.

## Deploy And Verify

All mutating wrappers are dry-run by default and use the direct Yggdrasil route
`ssh-direct.senyao.org:4622`. Review the output before adding `--apply`.

```bash
./scripts/deploy_yggdrasil_experiment_console.sh \
  --seed-state-dir /private/tmp/experiment-console-production-seed \
  --legacy-archive /path/to/legacy-runtime-MIGRATION_ID.tar.gz

./scripts/deploy_yggdrasil_experiment_console.sh --apply \
  --seed-state-dir /private/tmp/experiment-console-production-seed \
  --legacy-archive /path/to/legacy-runtime-MIGRATION_ID.tar.gz

./scripts/verify_yggdrasil_experiment_console.sh \
  --expected-instance yggdrasil-production
```

The seed is accepted only when the remote authoritative ledger does not yet
exist. Verification requires:

- healthy non-root `linux/amd64` container;
- loopback-only published API port;
- read-only credential/config mounts and writable state/results mounts;
- `authority_role=authoritative`, matching `instance_id`, and non-empty stable
  `ledger_id`;
- worker enabled, ready, running, and holding the singleton lease;
- non-empty Console API secret mounted read-only; protected APIs reject missing
  or incorrect bearer credentials;
- SQLite `quick_check=ok` with all control-plane tables;
- every imported migration job has a schedule, and every non-terminal/queued
  job with a result contract has an active schedule. Future sweep/register
  calls bind their schedule atomically when both `result_contract` and
  `thread_id` are supplied.

Only after this verification should `./scripts/exp` and the Desktop bridge use
the local SSH forward. Do not run local and Yggdrasil mutating Consoles in
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
