# Experiment Console Local Run

Last verified: 2026-06-15 on Oliver's MacBook Pro via Codex.

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
- keeps local runtime state in `/private/tmp/experiment-console-runtime`

## Prerequisites

- SSH alias `HCCS-25` works.
- Bitwarden has already been used once to create the local secret file.
- Python dependencies are available either in `.local_deps` or `/private/tmp/experiment-console-deps`.

Install local dependencies when needed:

```bash
cd /Users/oliver/Developer/experiment-console
./scripts/install_local_deps.sh
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

Current production sweep at the time of this handoff:

```text
HCCS/DualRefGAD/ofzzsan4
```

Console-owned agent health check:

```bash
EXPERIMENT_CONSOLE_URL=http://127.0.0.1:5174 \
PYTHONPATH="/Users/oliver/Developer/experiment-console/.local_deps:/private/tmp/experiment-console-deps:/Users/oliver/.agents/skills/experiment-runner/scripts" \
/opt/homebrew/bin/python3 /Users/oliver/.agents/skills/experiment-runner/scripts/experiment.py \
  watchdog-once --job-id job_20260615_152258_prod_matguardgt_cleg3_v4_console_20260615 --json
```

## Troubleshooting

- If `curl` from Codex sandbox says `Operation not permitted` or cannot connect while the browser works, retry with an approved command context. This has been a sandbox-local network permission issue.
- If Python import reads time out in the Documents checkout, do not debug Python first; check for dataless placeholders and use the Developer checkout.
- If the startup says `WANDB_API_KEY is not set`, check `~/.config/experiment-console/secrets.env` and file permissions.
- If agent launch times out over SSH, ensure background commands redirect stdin with `</dev/null`.
