# Yggdrasil v3 Deployment

v3 is a clean replacement. Stop the old service, preserve its state as a
read-only archive if needed, and create a new empty state directory. Do not
import jobs, W&B metadata, schedules, artifacts, intents, or ledger fields.

## Required mounts

- v3 SQLite state directory
- read-only `server-profiles.json`
- SSH config, known-hosts file, and private key
- bearer-token file

Copy `.env.example` to an owner-readable deployment environment file, set the
absolute mount paths, then run:

```bash
docker compose -f compose.yggdrasil.yaml up -d --build
docker compose -f compose.yggdrasil.yaml ps
curl -fsS http://127.0.0.1:5174/health
```

The container binds loopback only. The Desktop bridge is the sole client path.
For a rollback, stop the service, restore the previous image and the backup of
the v3 state directory, then run the health check again. There is no schema
cutover or historical migration command.
