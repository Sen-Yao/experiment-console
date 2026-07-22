# Legacy Yggdrasil v3 Rollback

This deployment is disabled rollback material. Do not start it for new
experiments or use it as a scheduler. If the user explicitly requests a
whole-system rollback, preserve existing state and create a new empty state
directory; never import jobs, W&B metadata, schedules, artifacts, intents, or
ledger fields.

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

The container binds loopback only. A rollback requires an explicitly selected
client path and user authorization; no Desktop bridge exists. Stop the service,
restore the previous image and the backup of the v3 state directory, then run
the health check again. There is no schema cutover or historical migration
command.
