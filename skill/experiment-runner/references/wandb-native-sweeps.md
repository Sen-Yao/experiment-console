# W&B Native Sweeps On HCCS-25

## Preflight

Activate the project environment and test viewer/project access over direct
HTTPS first. Confirm the sweep config sends only authorized fields. Use the
private OpenClash environment only as a fallback for bounded, diagnosed
DNS/connect/timeout failures; never use it to mask `401`/`403`.

Create the sweep once from parsed YAML, not from a path string:

```python
import yaml
import wandb

with open("sweep.yaml", encoding="utf-8") as handle:
    sweep_id = wandb.sweep(
        yaml.safe_load(handle), entity="HCCS", project="DualRefGAD"
    )
print(sweep_id)
```

Immediately record the returned sweep ID in the run manifest. If the call times
out, query W&B before creating another sweep.

## Parallel Agents

Start one agent in each selected tmux pane with a distinct
`CUDA_VISIBLE_DEVICES` value:

```bash
CUDA_VISIBLE_DEVICES=<gpu> wandb agent HCCS/DualRefGAD/<sweep-id> --count <n>
```

Activate the project conda environment inside every pane, or explicitly put
its `bin` directory first in `PATH`. Starting the conda environment's `wandb`
binary by absolute path is insufficient: the native agent launches each trial
through `/usr/bin/env python`, which resolves from `PATH`.

Let W&B assign trials. Keep planned run counts advisory unless the user or
platform explicitly sets a hard limit. Do not hide independently retryable
trials inside one Python process to reduce run count.

## Network Failure

- Before sweep creation: fail closed and create an attention condition.
- Direct path: retry a small bounded number of times. If failures remain in the
  DNS/connect/timeout class and the optional proxy environment is provisioned,
  source it without printing values and repeat the identical project probe.
- Proxy fallback: record `network_path: openclash-fallback` in the manifest and
  revalidate during recovery; direct remains the default for later batches.
- During a run: preserve the local W&B directory and inspect client retry/log
  state before stopping compute.
- Between assignments: an agent waiting on W&B can remain alive; the next
  one-shot Codex heartbeat rechecks it for judgment.
- Persistent outage: Codex may wait, retry the same sweep/agent, or create an
  explicit `offline-manual` protocol. Never switch silently.

Offline W&B runs are useful local records but do not replace native sweep
assignment, aggregation, or audit semantics. Sync only after checking run IDs,
config identity, and manifest provenance.
