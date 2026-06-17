# Experiment Console Local Run

Last verified: 2026-06-16 on Oliver's MacBook Pro via Codex.

## Scope

This is the local control-plane handoff for Experiment Console. The stable local project path is:

```bash
/Users/oliver/Developer/experiment-console
```

Use this path instead of `/Users/oliver/Documents/Experiment Console`. The Documents checkout has shown macOS `dataless` placeholder behavior, which caused Python imports, git reads, and script reads to time out.

The local Console is intentionally a lightweight control-plane runtime:

- exposes the runner-facing API on `http://127.0.0.1:5174`
- reads W&B auth from a repo-external secret file
- can register/query/launch W&B sweeps through SSH on HCCS-25
- can launch managed single-run jobs from single-run configs
- keeps local runtime state in `/private/tmp/experiment-console-runtime`
- defaults remote W&B agent execution to the `DualRefGAD` conda environment

## Prerequisites

- SSH alias `HCCS-25` works.
- Bitwarden has already been used once to create the local secret file.
- Python dependencies are available either in `.local_deps` or `/private/tmp/experiment-console-deps`.

Install local dependencies when needed:

```bash
cd /Users/oliver/Developer/experiment-console
./scripts/install_local_deps.sh
```

If the Developer checkout is not writable from the current sandbox, install the same dependencies into the temporary fallback used by the startup script:

```bash
cd /Users/oliver/Developer/experiment-console
EXPERIMENT_CONSOLE_DEPS_DIR=/private/tmp/experiment-console-deps ./scripts/install_local_deps.sh
```

## Secrets

Do not store `WANDB_API_KEY` in the repo.

The local startup path reads:

```bash
~/.config/experiment-console/secrets.env
```

Expected shape:

```bash
WANDB_API_KEY=...
```

The file must be mode `600`. The current W&B value from Bitwarden is accepted by W&B when passed as the `WANDB_API_KEY` environment variable, even though old `wandb login` rejects it with a 40-character length check. Prefer environment authentication; do not run `wandb login --relogin` for this workflow.

The runtime also sets:

```bash
EXPERIMENT_CONSOLE_DEFAULT_CONDA_ENV=DualRefGAD
```

This is intentionally Console-owned. Runner may pass `--remote-conda-env`, but production launch/recover should still enter `DualRefGAD` when runner omits it.

## Start

```bash
cd /Users/oliver/Developer/experiment-console
./scripts/start_local_console.sh
```

Expected listener:

```bash
lsof -nP -iTCP:5174 -sTCP:LISTEN
```

Expected health:

```bash
curl -fsS http://127.0.0.1:5174/health
```

The browser URL is:

```bash
http://127.0.0.1:5174/
```

## Runner Smoke

Use the installed runner as a thin client:

```bash
EXPERIMENT_CONSOLE_URL=http://127.0.0.1:5174 \
PYTHONPATH="/Users/oliver/Developer/experiment-console/.local_deps:/private/tmp/experiment-console-deps:/Users/oliver/.agents/skills/experiment-runner/scripts" \
/opt/homebrew/bin/python3 /Users/oliver/.agents/skills/experiment-runner/scripts/experiment.py \
  status --job-id job_20260615_152258_prod_matguardgt_cleg3_v4_console_20260615 --json
```

Expected classification:

```text
ok
```

When local YAML dependencies are missing, use the zero-dependency Console client path:

```bash
EXPERIMENT_CONSOLE_URL=http://127.0.0.1:5174 \
/opt/homebrew/bin/python3 /Users/oliver/.agents/skills/experiment-runner/scripts/experiment.py \
  console get /health

EXPERIMENT_CONSOLE_URL=http://127.0.0.1:5174 \
/opt/homebrew/bin/python3 /Users/oliver/.agents/skills/experiment-runner/scripts/experiment.py \
  console post /api/runner/status --data '{"job_id":"job_20260615_152258_prod_matguardgt_cleg3_v4_console_20260615"}'
```

Current verified production sweep at the time of this handoff:

```text
HCCS/DualRefGAD/02f9u8wv
job_20260616_072248_prod_matguardgt_best_vega_block_spmm_repro_5seed_b9950df0
```

This job is a registered existing sweep for the completed best-Vega 5-seed reproduction. It should be used for status and pull-results smoke tests; do not launch a new production sweep for local runtime verification.

Console-owned agent health check:

```bash
EXPERIMENT_CONSOLE_URL=http://127.0.0.1:5174 \
PYTHONPATH="/Users/oliver/Developer/experiment-console/.local_deps:/private/tmp/experiment-console-deps:/Users/oliver/.agents/skills/experiment-runner/scripts" \
/opt/homebrew/bin/python3 /Users/oliver/.agents/skills/experiment-runner/scripts/experiment.py \
  watchdog-once --job-id job_20260615_152258_prod_matguardgt_cleg3_v4_console_20260615 --json
```

Current compact status smoke:

```bash
curl -fsS -X POST 'http://127.0.0.1:5174/api/runner/status?requested_by=local-smoke' \
  -H 'Content-Type: application/json' \
  -d '{"job_id":"job_20260616_072248_prod_matguardgt_best_vega_block_spmm_repro_5seed_b9950df0"}'
```

Expected classification is `ok`, with split state fields:

```text
job_status=finished
wandb_sweep_status=FINISHED
agent_health=terminal
```

Managed single-run smoke shape:

```bash
EXPERIMENT_CONSOLE_URL=http://127.0.0.1:5174 \
PYTHONPATH="/Users/oliver/Developer/experiment-console/.local_deps:/private/tmp/experiment-console-deps:/Users/oliver/.agents/skills/experiment-runner/scripts" \
/opt/homebrew/bin/python3 /Users/oliver/.agents/skills/experiment-runner/scripts/experiment.py \
  launch-run \
  --profile single-run \
  --name demo_single_run \
  --remote-config /absolute/remote/single-run.yaml \
  --remote-host HCCS-25 \
  --remote-cwd /absolute/remote/project \
  --json
```

Current bounded result smoke:

```bash
curl -fsS -X POST 'http://127.0.0.1:5174/api/runner/pull-results?requested_by=local-smoke' \
  -H 'Content-Type: application/json' \
  -d '{"job_id":"job_20260616_072248_prod_matguardgt_best_vega_block_spmm_repro_5seed_b9950df0","max_runs":1,"allow_partial":true}'
```

Expected result source is `remote_local_files`, with a scientific metric such as `final_test_auc`.

## Troubleshooting

- If `curl` from Codex sandbox says `Operation not permitted` or cannot connect while the browser works, retry with an approved command context. This has been a sandbox-local network permission issue.
- If Python import reads time out in the Documents checkout, do not debug Python first; check for dataless placeholders and use the Developer checkout.
- If the startup says `WANDB_API_KEY is not set`, check `~/.config/experiment-console/secrets.env` and file permissions.
- If agent launch times out over SSH, ensure background commands redirect stdin with `</dev/null`.
