from __future__ import annotations

from .config import Settings
from .models import (
    CommandPreview,
    ExecutionPlan,
    IntentType,
    LaunchSweepPayload,
    RecoverAgentsPayload,
    StatusQueryPayload,
    StopJobPayload,
    ValidateConfigPayload,
    parse_payload,
)


def build_plan(intent: IntentType, payload: dict, settings: Settings) -> ExecutionPlan:
    parsed = parse_payload(intent, payload)
    if isinstance(parsed, ValidateConfigPayload):
        return ExecutionPlan(
            summary=f"Validate {parsed.profile} config at {parsed.config_path}.",
            risk_level="read_only",
            commands=[
                CommandPreview(
                    label="validate_config",
                    argv=["local-validator", "--profile", parsed.profile, "--config", parsed.config_path],
                    reason="Check YAML shape and formal sweep guardrails before any W&B or SSH side effects.",
                )
            ],
        )
    if isinstance(parsed, LaunchSweepPayload):
        entity = parsed.entity or settings.default_entity
        project = parsed.project or settings.default_project
        commands = [
            CommandPreview(
                label="validate_config",
                argv=["local-validator", "--profile", parsed.profile, "--config", parsed.config_path],
                reason="Validate the experiment configuration before creating a W&B sweep.",
            ),
            CommandPreview(
                label="create_wandb_sweep",
                argv=["wandb", "sweep", "--entity", entity, "--project", project, parsed.config_path],
                reason="Create exactly one W&B sweep for this launch intent.",
                side_effect=True,
            ),
            CommandPreview(
                label="probe_gpus",
                argv=["ssh", parsed.remote_host, "nvidia-smi --query-gpu=..."],
                host=parsed.remote_host,
                reason="Select eligible GPUs using auto mode unless max_agents constrains the count.",
            ),
            CommandPreview(
                label="start_agents",
                argv=["ssh", parsed.remote_host, "cd <remote_cwd> && nohup wandb agent <entity/project/sweep_id> ..."],
                host=parsed.remote_host,
                reason="Start one W&B agent per eligible GPU, capped only by max_agents.",
                side_effect=True,
            ),
        ]
        return ExecutionPlan(
            summary=f"Launch sweep {parsed.job_name} on {parsed.remote_host} for {entity}/{project}.",
            risk_level="remote_side_effect",
            commands=commands,
            warnings=["P0 does not schedule cron/watchdog or aggregate results."],
            expected_side_effects=["Create a W&B sweep", "Start remote wandb agent processes", "Write local job/audit state"],
        )
    if isinstance(parsed, StatusQueryPayload):
        target = parsed.job_id or parsed.sweep_id
        return ExecutionPlan(
            summary=f"Query status for {target}.",
            risk_level="read_only",
            commands=[
                CommandPreview(
                    label="query_status",
                    argv=["local-store", "and", "wandb-graphql"],
                    reason="Combine local job state with W&B sweep state when available.",
                )
            ],
        )
    if isinstance(parsed, StopJobPayload):
        return ExecutionPlan(
            summary=f"Stop job {parsed.job_id}.",
            risk_level="destructive",
            commands=[
                CommandPreview(
                    label="stop_matching_agents",
                    argv=["ssh", "<job.remote_host>", "terminate only pids matching 'wandb agent <entity/project/sweep_id>'"],
                    reason="Stop only agent processes for the target sweep.",
                    side_effect=True,
                )
            ],
            warnings=["cancel_wandb is recorded but not implemented in P0; P0 stops remote agents and marks the job cancelled."],
            expected_side_effects=["Terminate matching remote agent processes", "Mark local job cancelled"],
        )
    if isinstance(parsed, RecoverAgentsPayload):
        return ExecutionPlan(
            summary=f"Recover agents for existing job {parsed.job_id}.",
            risk_level="remote_side_effect",
            commands=[
                CommandPreview(
                    label="load_existing_job",
                    argv=["local-store", "get-job", parsed.job_id],
                    reason="Recover must reuse the existing sweep id and must not create a duplicate sweep.",
                ),
                CommandPreview(
                    label="probe_gpus",
                    argv=["ssh", "<job.remote_host>", "nvidia-smi --query-gpu=..."],
                    reason="Find eligible GPUs for replacement agents.",
                ),
                CommandPreview(
                    label="start_agents",
                    argv=["ssh", "<job.remote_host>", "cd <remote_cwd> && nohup wandb agent <existing_sweep> ..."],
                    reason="Start replacement agents for the existing sweep only.",
                    side_effect=True,
                ),
            ],
            expected_side_effects=["Start remote wandb agent processes for an existing sweep", "Update local job/audit state"],
        )
    raise ValueError(f"unsupported intent: {intent}")


def confirmation_phrase(intent_id: str, intent: IntentType) -> str:
    return f"EXECUTE {intent.value} {intent_id[-8:]}"

