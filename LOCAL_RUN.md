# Local v3 Run

Last verified: 2026-07-20 on Oliver's MacBook Pro with an isolated listener on
`127.0.0.1:5175`. The listener, v3 health contract, empty ledger, monitor worker,
and runner instance check all passed; the process was stopped after verification.

Use this path only for isolated rollback development. No Console or custom
bridge is active in production; formal experiments run directly on HCCS-25 and
use Codex thread heartbeats for scheduled follow-up.

```bash
cd /Users/oliver/Developer/experiment-console
python3 -m pip install -e ".[dev]"
./scripts/start_local_console.sh
```

The local service uses `config/server-profiles.json`, stores its empty ledger in
`.state-v3`, and listens on `http://127.0.0.1:5174`. For runner calls, set:

```bash
export EXPERIMENT_CONSOLE_EXPECTED_INSTANCE_ID=local-experiment-console-v3
```

Health and command smoke:

```bash
curl -fsS http://127.0.0.1:5174/health
./scripts/exp resources --profile hccs-25 --json
```

No local process should be used as a production peer. There is no W&B auth,
legacy state import, historical job migration, or frontend in v3.
