from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from .command import CommandRunner
from .config import Settings
from .validation import expected_run_count, load_yaml


class WandBUnavailable(RuntimeError):
    pass


def parse_sweep_id(text: str) -> str | None:
    patterns = [
        r"wandb agent ([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)",
        r"wandb sweep .*?([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)",
        r"Created sweep with ID:\s*([A-Za-z0-9_.\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            return value.rsplit("/", 1)[-1]
    return None


class WandBClient:
    endpoint = "https://api.wandb.ai/graphql"

    def __init__(self, settings: Settings, runner: CommandRunner | None = None):
        self.settings = settings
        self.runner = runner or CommandRunner()

    def _api_key(self) -> str:
        return os.environ.get(self.settings.wandb_api_key_env, "")

    def _headers(self) -> dict[str, str]:
        api_key = self._api_key()
        if not api_key:
            raise WandBUnavailable(f"{self.settings.wandb_api_key_env} is not set")
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = requests.post(self.endpoint, headers=self._headers(), json={"query": query, "variables": variables}, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if data.get("errors"):
                raise WandBUnavailable(str(data["errors"]))
            return data["data"]
        except Exception as exc:
            if isinstance(exc, WandBUnavailable):
                raise
            raise WandBUnavailable(str(exc)) from exc

    def create_sweep(self, config_path: Path, *, entity: str, project: str) -> dict[str, Any]:
        argv = ["wandb", "sweep", "--entity", entity, "--project", project, str(config_path)]
        result = self.runner.run(argv, timeout=self.settings.command_timeout_seconds)
        sweep_id = parse_sweep_id(result.stdout + "\n" + result.stderr)
        if not sweep_id:
            raise WandBUnavailable("failed to parse sweep id from wandb sweep output")
        return {"sweep_id": sweep_id, "entity": entity, "project": project, "command": result.summary()}

    def get_sweep_state(self, entity: str, project: str, sweep_id: str) -> dict[str, Any]:
        query = """
        query SweepState($entity: String!, $project: String!, $sweep: String!) {
          project(name: $project, entityName: $entity) {
            sweep(sweepName: $sweep) {
              name
              state
              createdAt
              runCount
              config
              runs(first: 200) {
                edges { node { name state createdAt heartbeatAt summaryMetrics config } }
              }
            }
          }
        }
        """
        data = self.graphql(query, {"entity": entity, "project": project, "sweep": sweep_id})
        sweep = ((data.get("project") or {}).get("sweep") or {})
        runs = []
        for edge in ((sweep.get("runs") or {}).get("edges") or []):
            node = edge.get("node") or {}
            runs.append({
                "name": node.get("name"),
                "state": node.get("state"),
                "created_at": node.get("createdAt"),
                "heartbeat_at": node.get("heartbeatAt"),
                "summary_metrics": node.get("summaryMetrics"),
                "config": node.get("config"),
            })
        return {
            "id": sweep_id,
            "entity": entity,
            "project": project,
            "name": sweep.get("name") or sweep_id,
            "state": sweep.get("state") or "UNKNOWN",
            "createdAt": sweep.get("createdAt"),
            "runCount": sweep.get("runCount") or 0,
            "expectedRunCount": expected_run_count_from_wandb_config(sweep.get("config")),
            "runs": runs,
        }

    def discover_sweeps(self, entity: str, project: str | None = None, days: int = 7) -> list[dict[str, Any]]:
        threshold = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if project:
            return self._project_sweeps(entity, project, threshold)
        query = """
        query EntityProjectsSweeps($entity: String!) {
          entity(name: $entity) {
            projects(first: 50) {
              edges { node { name sweeps(first: 100) { edges { node { name state createdAt runCount config } } } } }
            }
          }
        }
        """
        data = self.graphql(query, {"entity": entity})
        out = []
        for project_edge in (((data.get("entity") or {}).get("projects") or {}).get("edges") or []):
            project_name = (project_edge.get("node") or {}).get("name")
            for sweep_edge in ((((project_edge.get("node") or {}).get("sweeps") or {}).get("edges")) or []):
                node = sweep_edge.get("node") or {}
                if (node.get("createdAt") or "") >= threshold:
                    out.append(format_sweep(entity, project_name, node))
        return sorted(out, key=lambda item: item.get("createdAt") or "", reverse=True)

    def _project_sweeps(self, entity: str, project: str, threshold: str) -> list[dict[str, Any]]:
        query = """
        query ProjectSweeps($entity: String!, $project: String!) {
          project(name: $project, entityName: $entity) {
            sweeps(first: 100) { edges { node { name state createdAt runCount config } } }
          }
        }
        """
        data = self.graphql(query, {"entity": entity, "project": project})
        out = []
        for edge in ((((data.get("project") or {}).get("sweeps") or {}).get("edges")) or []):
            node = edge.get("node") or {}
            if (node.get("createdAt") or "") >= threshold:
                out.append(format_sweep(entity, project, node))
        return sorted(out, key=lambda item: item.get("createdAt") or "", reverse=True)


def expected_run_count_from_wandb_config(config: Any) -> int:
    if not config:
        return 0
    try:
        import yaml
        if isinstance(config, str):
            data = yaml.safe_load(config) or {}
        elif isinstance(config, dict):
            data = config
        else:
            return 0
        return expected_run_count(data)
    except Exception:
        return 0


def format_sweep(entity: str, project: str, node: dict[str, Any]) -> dict[str, Any]:
    run_count = int(node.get("runCount") or 0)
    expected = expected_run_count_from_wandb_config(node.get("config")) or run_count
    state = node.get("state") or "UNKNOWN"
    progress = 1.0 if state == "FINISHED" and expected else min(run_count / expected, 0.999) if expected else 0.0
    return {
        "id": node.get("name"),
        "entity": entity,
        "project": project,
        "name": node.get("name"),
        "state": state,
        "createdAt": node.get("createdAt"),
        "runCount": run_count,
        "expectedRunCount": expected,
        "progress": progress,
        "run_state_counts_source": "wandb_sweep_runCount",
    }

