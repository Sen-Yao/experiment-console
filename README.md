# Experiment Console

A durable control plane for launching and supervising Weights & Biases sweeps
on SSH-accessible GPU machines.

The production authority runs continuously on Yggdrasil. Codex Desktop reaches
it through a loopback SSH forward; a lightweight local bridge wakes a specific
Codex task only for actionable Console events. Healthy experiment waiting does
not keep a Codex Goal active and does not create model-driven polling turns.

## Why It Exists

Running sweeps often means juggling a YAML config, W&B sweep creation, GPU selection, SSH sessions, detached `wandb agent` processes, and scattered notes about what happened. This project turns that into one auditable local workflow:

- preview an action before it runs;
- validate sweep configs locally;
- create a W&B sweep;
- probe remote GPUs with `nvidia-smi`;
- launch one W&B agent per eligible GPU;
- queue formal sweeps per remote GPU workspace so the active sweep uses all available GPUs before the next sweep starts;
- recover agents for an existing sweep;
- stop only the agents that match a tracked sweep;
- reconcile W&B, remote process, and per-run artifact evidence before declaring
  a result complete;
- keep one authoritative job ledger, redacted audit logs, durable monitor
  schedules, and downloadable artifact bundles.

## Runtime Model

- **Yggdrasil Console**: the only production ledger, monitor worker, W&B
  client, HCCS SSH client, and artifact bundler.
- **Codex Desktop bridge**: supervises the SSH tunnel and claims actionable
  wake events. Empty polls do not contact Codex app-server.
- **Runner**: `./scripts/exp <command>` is the stable command entry point. Its
  default `http://127.0.0.1:5174` endpoint is the SSH tunnel, not a second local
  Console.
- **Local Console**: development and test use only. Never run it as a mutating
  peer of production.

## Safety Model

- The browser UI never accepts API keys or arbitrary shell commands.
- Production secrets are mounted from separate owner-readable files. They are
  not placed in Compose environment values, images, SQLite, or bridge config.
- Audit logs redact common secret patterns before writing to disk.
- Actions with real side effects require the generated confirmation phrase.
- Stop actions only target remote commands matching the tracked `wandb agent <entity>/<project>/<sweep_id>` string.
- The API binds only to Yggdrasil loopback and requires a bearer token.
- Result readiness requires both terminal execution evidence and a complete,
  valid, run-id-associated artifact set under the job's explicit
  `ResultContract`.

## Project Layout

```text
.
├── backend/experiment_console/  # FastAPI app and control-plane services
├── examples/sweep.yaml          # Minimal W&B sweep example
├── frontend/                    # React + Vite operator UI
├── scripts/run_tests.sh         # Local verification helper
├── tests/                       # Backend unit tests
└── pyproject.toml               # Python package metadata
```

## Requirements

- Python 3.11+
- Node.js and npm
- W&B CLI installed and authenticated for real sweep launches
- SSH access to any remote GPU hosts you want to manage

## Production Use

Provision, migrate, deploy, verify, and roll back with
[`docs/yggdrasil-production.md`](docs/yggdrasil-production.md). Configure and
install the Mac bridge with [`docs/desktop-bridge.md`](docs/desktop-bridge.md).

After the bridge establishes the tunnel, verify authority before operating a
job:

```bash
./scripts/exp authority
./scripts/exp status --job-id job_...
./scripts/exp pull-results --job-id job_... --artifact-dir /absolute/local/path
```

The local artifact directory is a runner-side download destination. It is
never sent to Yggdrasil as a remote path.

## Local Development

Install backend dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

Configure isolated development defaults:

```bash
export EXPERIMENT_CONSOLE_STATE_DIR="$PWD/.state"
export EXPERIMENT_CONSOLE_DEFAULT_ENTITY="my-team"
export EXPERIMENT_CONSOLE_DEFAULT_PROJECT="my-project"
export WANDB_API_KEY="..."
export EXPERIMENT_CONSOLE_AUTHORITY_ROLE="local-development"
export EXPERIMENT_CONSOLE_INSTANCE_ID="local-development"
```

Start the backend:

```bash
uvicorn experiment_console.api:app --app-dir backend --reload --host 127.0.0.1 --port 8090
```

Start the frontend in another terminal:

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`.

Do not point the production runner or bridge at this process. See
[`LOCAL_RUN.md`](LOCAL_RUN.md) for local smoke checks.

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `EXPERIMENT_CONSOLE_STATE_DIR` | `.state` in the repo | SQLite database, audit log, W&B sweep cache |
| `EXPERIMENT_CONSOLE_DEFAULT_ENTITY` | `my-team` | Default W&B entity |
| `EXPERIMENT_CONSOLE_DEFAULT_PROJECT` | `my-project` | Default W&B project |
| `WANDB_API_KEY` | unset | W&B GraphQL/API authentication |
| `WANDB_API_KEY_FILE` | unset | Preferred production W&B secret file |
| `EXPERIMENT_CONSOLE_API_TOKEN_FILE` | unset | Production API bearer-token file |
| `EXPERIMENT_CONSOLE_AUTHORITY_ROLE` | `local-development` | Runtime authority role; production requires `authoritative` |
| `EXPERIMENT_CONSOLE_INSTANCE_ID` | `local-experiment-console` | Stable runtime identity |
| `EXPERIMENT_CONSOLE_SSH_TIMEOUT` | `20` | Timeout for SSH probes |
| `EXPERIMENT_CONSOLE_COMMAND_TIMEOUT` | `120` | Timeout for longer local/remote commands |
| `EXPERIMENT_CONSOLE_GPU_MIN_FREE_GB` | `2.0` | Minimum free GPU memory for eligibility |
| `EXPERIMENT_CONSOLE_GPU_MAX_UTIL` | `85` | Maximum GPU utilization for eligibility |

## API

- `GET /health`
- `GET /api/bridge/events`
- `POST /api/bridge/events/{event_id}/ack`
- `GET /api/artifacts/{snapshot_id}/download`
- `GET /api/overview`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `POST /api/intents/preview`
- `POST /api/intents/{intent_id}/confirm`
- `POST /api/intents/{intent_id}/execute`
- `GET /api/events`
- `GET /api/wandb/sweeps`
- `GET /api/hosts/gpus?host=gpu-host-1`
- `POST /api/runner/validate-config`
- `POST /api/runner/launch-sweep`
- `POST /api/runner/launch-run`
- `POST /api/runner/register-existing-sweep`
- `POST /api/runner/status`
- `POST /api/runner/recover-agents`
- `POST /api/runner/stop-job`
- `POST /api/runner/cancel-sweep`
- `POST /api/runner/auth-check`
- `POST /api/runner/preflight`
- `POST /api/runner/pull-results`
- `POST /api/runner/repair-watchdog`
- `POST /api/runner/schedule-monitor`
- `POST /api/runner/unschedule-monitor`
- `POST /api/runner/watchdog-once`
- `POST /api/runner/advance-queue`

The development runtime and production FastAPI runtime expose the same runner-facing contract. `experiment-runner` calls these endpoints instead of doing SSH, W&B, job-store, result aggregation, or monitor side effects locally.

For runner-facing launch and preflight calls, `config_path` means the YAML path on the remote host, not a local workstation path. The intended flow is to commit/push the config, pull it in the remote checkout, verify the remote file exists, then launch with a path such as `/home/linziyao/DualRefGAD/configs/demo.yaml`. The standalone `validate-config` endpoint remains a local validation helper and is separate from launch/preflight config handling.

Runner-facing `launch-sweep` defaults to `queue_policy=sequential`. If another sweep is already active in the same queue group, defaulting to `<remote_host>:<remote_cwd>`, Console records a queued job but does not create a W&B sweep or launch agents until `/api/runner/advance-queue` starts it. Use `queue_policy=immediate` only for explicitly approved concurrent launches.

For multi-dataset experiment batches, prepare the sweep YAML for every dataset before launching the first dataset, validate them together, and include them in the same GitHub sync/pull handoff. Do not wait for one dataset to finish before creating and syncing the next dataset's sweep YAML. When one dataset reaches a terminal state and another dataset still needs to run, prioritize starting or advancing the next dataset first. After the next dataset is confirmed running, return to the previous dataset's `pull-results` and result review.

## Development

Run all checks:

```bash
scripts/run_tests.sh
```

Skip dependency installation when your environment is already ready:

```bash
INSTALL_DEPS=0 scripts/run_tests.sh
```

Run the common checks directly:

```bash
python3 -m pytest
npm --prefix frontend run build
```

## Repository Hygiene

The repository tracks source, tests, docs, examples, and lockfiles. It intentionally ignores local state, virtual environments, dependency installs, build output, logs, SQLite databases, `.env` files, and private keys.

Before publishing your fork, review your own examples and config files for hostnames, usernames, W&B entities/projects, and local paths.
