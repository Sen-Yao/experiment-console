from __future__ import annotations

from .config import Settings
from .models import (
    CommandPreview,
    ExecutionPlan,
    AuthCheckPayload,
    CancelSweepPayload,
    IntentType,
    LaunchRunPayload,
    LaunchSweepPayload,
    PreflightPayload,
    PullResultsPayload,
    RepairWatchdogPayload,
    RecoverAgentsPayload,
    RegisterExistingSweepPayload,
    ScheduleMonitorPayload,
    StatusQueryPayload,
    StopJobPayload,
    UnscheduleMonitorPayload,
    ValidateConfigPayload,
    WatchdogOncePayload,
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
    if isinstance(parsed, LaunchRunPayload):
        commands = [
            CommandPreview(
                label="validate_single_run_config",
                argv=["remote-validator", "--profile", "single-run", "--config", parsed.config_path],
                reason="Validate that the remote config expands to exactly one training command.",
            ),
            CommandPreview(
                label="remote_preflight",
                argv=["ssh", parsed.remote_host or settings.default_remote_host, "check cwd/python/wandb/nvidia-smi/conda"],
                host=parsed.remote_host or settings.default_remote_host,
                reason="Verify the target host before starting a managed run.",
            ),
            CommandPreview(
                label="launch_single_run",
                argv=["ssh", parsed.remote_host or settings.default_remote_host, "cd <remote_cwd> && nohup <single-run command> ..."],
                host=parsed.remote_host or settings.default_remote_host,
                reason="Start one background training process and record pid/log/status metadata.",
                side_effect=True,
            ),
        ]
        return ExecutionPlan(
            summary=f"Launch single run {parsed.job_name} on {parsed.remote_host or settings.default_remote_host}.",
            risk_level="remote_side_effect",
            commands=commands,
            expected_side_effects=["Start one remote training process", "Write local job/audit state"],
        )
    if isinstance(parsed, RegisterExistingSweepPayload):
        return ExecutionPlan(
            summary=f"Register existing sweep {parsed.sweep_id} on {parsed.remote_host}.",
            risk_level="writes_local_state",
            commands=[
                CommandPreview(
                    label="register_existing_sweep",
                    argv=["local-store", "register", parsed.sweep_id],
                    reason="Bind an already-created W&B sweep to Console state without creating a duplicate sweep.",
                )
            ],
            expected_side_effects=["Write local job/audit state"],
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
    if isinstance(parsed, CancelSweepPayload):
        entity = parsed.entity or settings.default_entity
        project = parsed.project or settings.default_project
        return ExecutionPlan(
            summary=f"{parsed.mode.title()} W&B sweep {entity}/{project}/{parsed.sweep_id} from {parsed.remote_host}.",
            risk_level="destructive",
            commands=[
                CommandPreview(
                    label="cancel_wandb_sweep",
                    argv=["ssh", parsed.remote_host, f"wandb sweep --{parsed.mode} {entity}/{project}/{parsed.sweep_id}"],
                    host=parsed.remote_host,
                    reason="Stop a W&B sweep lifecycle without creating new runs.",
                    side_effect=True,
                )
            ],
            expected_side_effects=["Update W&B sweep lifecycle", "Write audit event"],
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
    if isinstance(parsed, RepairWatchdogPayload):
        return ExecutionPlan(
            summary=f"Repair watchdog metadata for existing job {parsed.job_id}.",
            risk_level="writes_local_state",
            commands=[
                CommandPreview(
                    label="repair_watchdog_metadata",
                    argv=["local-store", "update-job", parsed.job_id, "--remote-cwd", parsed.remote_cwd],
                    reason="Normalize remote path/conda metadata in Console state without creating or restarting any sweep.",
                )
            ],
            expected_side_effects=["Update local job/audit state"],
        )
    if isinstance(parsed, ScheduleMonitorPayload):
        return ExecutionPlan(
            summary=f"Schedule Console-owned watchdog monitor for {parsed.job_id}.",
            risk_level="writes_local_state",
            commands=[
                CommandPreview(
                    label="schedule_monitor",
                    argv=["console-store", "schedule-monitor", parsed.job_id, "--every", parsed.every],
                    reason="Record authoritative cron/watchdog metadata in Console state.",
                    side_effect=True,
                )
            ],
            expected_side_effects=["Update Console job monitor metadata", "Write audit event"],
        )
    if isinstance(parsed, UnscheduleMonitorPayload):
        return ExecutionPlan(
            summary=f"Unschedule Console-owned watchdog monitor for {parsed.job_id}.",
            risk_level="writes_local_state",
            commands=[
                CommandPreview(
                    label="unschedule_monitor",
                    argv=["console-store", "unschedule-monitor", parsed.job_id],
                    reason="Mark the Console-owned watchdog schedule inactive idempotently.",
                    side_effect=True,
                )
            ],
            expected_side_effects=["Update Console job monitor metadata", "Write audit event"],
        )
    if isinstance(parsed, WatchdogOncePayload):
        return ExecutionPlan(
            summary=f"Run one Console-owned watchdog status check for {parsed.job_id}.",
            risk_level="read_only",
            commands=[
                CommandPreview(
                    label="watchdog_once",
                    argv=["console-status", parsed.job_id],
                    reason="Map job/sweep status to a quiet or attention-worthy watchdog event.",
                )
            ],
        )
    if isinstance(parsed, AuthCheckPayload):
        return ExecutionPlan(
            summary="Check W&B auth locally and on the remote host.",
            risk_level="read_only",
            commands=[
                CommandPreview(
                    label="auth_check",
                    argv=["ssh", parsed.remote_host or "<job.remote_host>", "WANDB_API_KEY=<stdin> python -c <api probe>"],
                    host=parsed.remote_host,
                    reason="Verify remote W&B auth without printing or storing the key.",
                )
            ],
        )
    if isinstance(parsed, PreflightPayload):
        return ExecutionPlan(
            summary=f"Run remote preflight on {parsed.remote_host}.",
            risk_level="read_only",
            commands=[
                CommandPreview(
                    label="remote_preflight",
                    argv=["ssh", parsed.remote_host, "check cwd/python/wandb/nvidia-smi/conda"],
                    host=parsed.remote_host,
                    reason="Verify the target host is ready before creating or recovering agents.",
                )
            ],
        )
    if isinstance(parsed, PullResultsPayload):
        return ExecutionPlan(
            summary=f"Pull bounded experiment results for {parsed.job_id or parsed.sweep_id}.",
            risk_level="read_only",
            commands=[
                CommandPreview(
                    label="pull_results",
                    argv=["ssh", parsed.remote_host or "<job.remote_host>", "scan project outputs and wandb run summaries"],
                    host=parsed.remote_host,
                    reason="Return an agent-readable partial or complete summary within the requested time budget.",
                )
            ],
        )
    raise ValueError(f"unsupported intent: {intent}")


def confirmation_phrase(intent_id: str, intent: IntentType) -> str:
    return f"EXECUTE {intent.value} {intent_id[-8:]}"
