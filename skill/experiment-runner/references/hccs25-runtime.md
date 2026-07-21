# HCCS-25 Runtime

## Stable Access

- SSH alias: `HCCS-25`.
- Development checkout: `/home/linziyao/DualRefGAD`.
- Conda environment: `/home/linziyao/.conda/envs/DualRefGAD`.
- HCCS system Python is 3.8-era; do not assume newer `pathlib` or syntax.
- `/home` is near capacity. Use small detached worktrees and remove them only
  after their manifests and results are closed.

Probe current values instead of copying historical GPU or disk numbers:

```bash
ssh HCCS-25 "df -h /home"
ssh HCCS-25 "git -C /home/linziyao/DualRefGAD status --short --branch"
ssh HCCS-25 "git -C /home/linziyao/DualRefGAD rev-parse HEAD"
```

## Detached Execution Worktree

After tests and push from the clean development checkout:

```bash
ssh HCCS-25 "git -C /home/linziyao/DualRefGAD worktree add --detach /home/linziyao/worktrees/<name> <full-sha>"
ssh HCCS-25 "git -C /home/linziyao/worktrees/<name> status --short --branch"
```

Do not pull, edit, or switch a running worktree. Bind its full SHA and config
digest in the run manifest.

## Credentials And Network

- W&B credential: standard user netrc, mode `600`.
- GitHub: SSH key selected by the SSH/Git configuration; never embed a token in
  a remote URL.
- Default W&B path: direct HTTPS from HCCS-25.
- Optional OpenClash fallback: `~/.config/senyaolab/experiment.env`, mode
  `600`, only when a direct probe shows a network failure rather than an auth or
  project-permission failure.

Do not source the proxy file by default. When fallback is justified, load it
without echoing values:

```bash
set -a
. ~/.config/senyaolab/experiment.env
set +a
```

Verify only presence, permissions, fallback reachability, and authorized
project access. Do not print the environment file, netrc, tokens, or proxy
URLs. If the fallback file is absent or its endpoint is unreachable, record
that fact instead of inventing a proxy or silently switching W&B modes.

## W&B Project Probe

Run first from the project conda environment with proxy variables unset:

```bash
/home/linziyao/.conda/envs/DualRefGAD/bin/python -c \
  'import wandb; api=wandb.Api(timeout=10); print(bool(api.viewer)); print(bool(api.project("DualRefGAD", "HCCS")))'
```

A failed probe blocks native sweep creation. A `401`/`403` is an
authentication/project-permission problem and must not trigger proxy fallback.
For DNS/connect/timeout failures, retry a small bounded number of times; then,
if the optional mode-600 proxy environment is provisioned, source it and repeat
the same probe. Record which path passed in the run manifest.
