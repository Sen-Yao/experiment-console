from __future__ import annotations

import importlib.util
import hashlib
import os
import sqlite3
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

from experiment_console.models import ResultContract


ROOT = Path(__file__).resolve().parents[1]


def test_compose_is_nonroot_loopback_only_and_uses_readonly_secrets():
    compose = yaml.safe_load((ROOT / "compose.yggdrasil.yaml").read_text(encoding="utf-8"))
    service = compose["services"]["console"]
    assert service["platform"] == "linux/amd64"
    assert service["read_only"] is True
    assert service["user"] != "root"
    assert service["restart"] == "unless-stopped"
    assert service["ports"] == ["127.0.0.1:${EXPERIMENT_CONSOLE_PORT:-5174}:5174"]
    assert service["environment"]["EXPERIMENT_CONSOLE_AUTHORITY_ROLE"].startswith("${EXPERIMENT_CONSOLE_AUTHORITY_ROLE:?")
    assert service["environment"]["SQLITE_TMPDIR"] == "/var/lib/experiment-console/state/sqlite-tmp"
    assert service["environment"]["WANDB_API_KEY_FILE"] == "/run/secrets/wandb_api_key"
    assert service["environment"]["EXPERIMENT_CONSOLE_API_TOKEN_FILE"] == "/run/secrets/console_api_token"
    assert "EXPERIMENT_CONSOLE_API_TOKEN" not in service["environment"]
    assert "WANDB_API_KEY" not in service["environment"]
    assert set(service["secrets"]) == {"wandb_api_key", "hccs_ssh_key", "console_api_token"}
    assert compose["secrets"]["console_api_token"]["file"].startswith(
        "${EXPERIMENT_CONSOLE_API_TOKEN_SECRET_FILE:?"
    )
    mounts = {item["target"]: item for item in service["volumes"]}
    assert mounts["/var/lib/experiment-console/state/results"]["source"].startswith("${EXPERIMENT_CONSOLE_RESULTS_PATH:?")
    assert mounts["/run/config/hccs_ssh_config"]["read_only"] is True
    assert service["healthcheck"]["test"] == ["CMD", "/usr/local/bin/experiment-console-healthcheck"]


def test_dockerfile_builds_amd64_target_and_drops_root():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG BASE_REGISTRY=docker.m.daocloud.io/library" in dockerfile
    assert "FROM --platform=$TARGETPLATFORM ${BASE_REGISTRY}/python:" in dockerfile
    assert "FROM --platform=$BUILDPLATFORM ${BASE_REGISTRY}/node:" in dockerfile
    assert "USER console:console" in dockerfile
    assert "openssh-client" in dockerfile
    assert "WANDB_API_KEY=" not in dockerfile
    assert "EXPERIMENT_CONSOLE_API_TOKEN=" not in dockerfile
    entrypoint = (ROOT / "deploy" / "yggdrasil" / "container-entrypoint.sh").read_text(encoding="utf-8")
    assert "export WANDB_API_KEY" not in entrypoint
    assert "WANDB_API_KEY_FILE" in entrypoint
    assert "EXPERIMENT_CONSOLE_API_TOKEN_FILE" in entrypoint
    assert "console_api_token" in entrypoint
    assert "API_TOKEN_LENGTH" in entrypoint
    assert 'mkdir -p "$SSH_DIR" "$SQLITE_TMPDIR"' in entrypoint
    assert 'chmod 0700 "$SQLITE_TMPDIR"' in entrypoint
    production_env = (ROOT / "deploy" / "yggdrasil" / "production.env.example").read_text(encoding="utf-8")
    assert "EXPERIMENT_CONSOLE_API_TOKEN_SECRET_FILE=" in production_env


def test_result_contracts_use_run_id_discovery_and_backend_schema():
    for path in sorted((ROOT / "deploy" / "yggdrasil" / "result-contracts").glob("*.json")):
        contract = ResultContract.model_validate_json(path.read_text(encoding="utf-8"))
        assert contract.discovery_mode == "run_id_output_globs_v1"
        assert contract.expected_runs == 25
        assert contract.max_runs == 25
        assert all(pattern.count("{run_id}") == 1 for pattern in contract.output_globs)


def test_container_health_requires_authority_ledger_and_worker_lease():
    path = ROOT / "deploy" / "yggdrasil" / "container-healthcheck.py"
    spec = importlib.util.spec_from_file_location("container_healthcheck_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    assert module.worker_ready({
        "monitor_worker": {"enabled": True, "ready": True, "running": True, "lease_held": True}
    }, require_lease=True)
    assert not module.worker_ready({
        "monitor_worker": {"enabled": True, "ready": True, "running": True, "lease_held": False}
    }, require_lease=True)
    source = path.read_text(encoding="utf-8")
    assert 'payload.get("contract") != "runner_console_agent_v2"' in source
    assert 'str(payload.get("ledger_schema_version")) != "2"' in source


def test_production_activation_supports_fresh_v2_cutover_and_verifier_probes_real_dependencies():
    activation = (ROOT / "deploy" / "yggdrasil" / "activate-release.sh").read_text(encoding="utf-8")
    assert "first authoritative deployment requires an explicit migration seed or --fresh-v2-ledger" in activation
    assert "migration seed refused because the authoritative ledger already exists" in activation
    assert "fresh v2 cutover refused while" in activation
    assert activation.count("docker exec -i") == 2
    assert "dependency_episodes" in activation
    assert "dependency_impacts" in activation
    assert "verified_empty=1" in activation
    assert 'NEW_RELEASE="$RELEASE"' in activation
    assert 'RELEASE="$NEW_RELEASE"' in activation
    verifier = (ROOT / "deploy" / "yggdrasil" / "verify-release.sh").read_text(encoding="utf-8")
    rollback = (ROOT / "deploy" / "yggdrasil" / "rollback-release.sh").read_text(encoding="utf-8")
    assert "WandBClient(settings).discover_sweeps" in verifier
    assert "SSHExecutor(settings).auth_check" in verifier
    assert 'health.get("contract") == "runner_console_agent_v2"' in verifier
    assert 'str(health.get("ledger_schema_version")) == "2"' in verifier
    assert "require_empty_ledger" in verifier
    assert "docker exec -i" in verifier
    assert "result_snapshot_verification_missing" in verifier
    assert "find \"$BASE/cutovers\"" in verifier
    assert "docker exec -i" in rollback
    assert '"hccs_wandb_auth_probe": "ok"' in verifier


def test_shell_scripts_parse_and_local_mutating_wrappers_are_dry_by_default(tmp_path: Path):
    shell_scripts = [
        ROOT / "scripts" / "deploy_yggdrasil_experiment_console.sh",
        ROOT / "scripts" / "verify_yggdrasil_experiment_console.sh",
        ROOT / "scripts" / "backup_yggdrasil_experiment_console.sh",
        ROOT / "scripts" / "rollback_yggdrasil_experiment_console.sh",
        *sorted((ROOT / "deploy" / "yggdrasil").glob("*.sh")),
    ]
    for script in shell_scripts:
        result = subprocess.run(["/bin/bash", "-n", str(script)], text=True, capture_output=True, check=False)
        assert result.returncode == 0, f"{script}: {result.stderr}"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "ssh-called"
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 99\n", encoding="utf-8")
    fake_ssh.chmod(0o755)
    environment = {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin"}
    cases = [
        [str(ROOT / "scripts" / "deploy_yggdrasil_experiment_console.sh")],
        [str(ROOT / "scripts" / "backup_yggdrasil_experiment_console.sh")],
        [
            str(ROOT / "scripts" / "rollback_yggdrasil_experiment_console.sh"),
            "--backup-dir", "/mnt/user/appdata/experiment-console/backups/example",
        ],
    ]
    for arguments in cases:
        result = subprocess.run(["/bin/bash", *arguments], cwd=ROOT, env=environment, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "DRY RUN" in result.stdout
    assert not marker.exists()


def test_each_release_uses_an_immutable_image_tag(tmp_path: Path):
    base = tmp_path / "experiment-console"
    config = base / "config"
    release_a = base / "releases" / "20260712T010000Z-aaaaaaa"
    release_b = base / "releases" / "20260712T020000Z-bbbbbbb"
    for directory in (config, release_a, release_b):
        directory.mkdir(parents=True, exist_ok=True)
    for release in (release_a, release_b):
        (release / "compose.yggdrasil.yaml").write_text("services: {}\n", encoding="utf-8")
    env_values = {
        "EXPERIMENT_CONSOLE_UID": "10001",
        "EXPERIMENT_CONSOLE_GID": "10001",
        "EXPERIMENT_CONSOLE_IMAGE_TAG": "production",
        "EXPERIMENT_CONSOLE_STATE_PATH": base / "state",
        "EXPERIMENT_CONSOLE_RESULTS_PATH": base / "results",
        "EXPERIMENT_CONSOLE_WANDB_SECRET_FILE": base / "secrets" / "wandb",
        "EXPERIMENT_CONSOLE_API_TOKEN_SECRET_FILE": base / "secrets" / "api",
        "EXPERIMENT_CONSOLE_HCCS_SSH_KEY_FILE": base / "secrets" / "hccs",
        "EXPERIMENT_CONSOLE_HCCS_SSH_CONFIG_FILE": config / "ssh_config",
        "EXPERIMENT_CONSOLE_HCCS_KNOWN_HOSTS_FILE": config / "known_hosts",
        "EXPERIMENT_CONSOLE_AUTHORITY_ROLE": "authoritative",
        "EXPERIMENT_CONSOLE_INSTANCE_ID": "test-production",
    }
    (config / "production.env").write_text(
        "".join(f"{key}={value}\n" for key, value in env_values.items()),
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "COMMON": str(ROOT / "deploy" / "yggdrasil" / "release-common.sh"),
        "BASE": str(base),
        "RELEASE_A": str(release_a),
        "RELEASE_B": str(release_b),
    }
    result = subprocess.run(
        [
            "/bin/bash", "-c",
            'source "$COMMON"; '
            'load_production_context "$BASE" "$RELEASE_A"; printf "%s|%s\\n" "$IMAGE_TAG" "${COMPOSE[*]}"; '
            'load_production_context "$BASE" "$RELEASE_B"; printf "%s|%s\\n" "$IMAGE_TAG" "${COMPOSE[*]}"',
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("release-20260712T010000Z-aaaaaaa|")
    assert lines[1].startswith("release-20260712T020000Z-bbbbbbb|")
    assert "EXPERIMENT_CONSOLE_IMAGE_TAG=release-20260712T010000Z-aaaaaaa" in lines[0]
    assert "EXPERIMENT_CONSOLE_IMAGE_TAG=release-20260712T020000Z-bbbbbbb" in lines[1]
    assert lines[0] != lines[1]


def test_runtime_server_reuses_backend_app_lifecycle():
    runtime_source = (ROOT / "scripts" / "runtime_console_server.py").read_text(encoding="utf-8")
    assert "from experiment_console.api import app, service" in runtime_source
    assert "FastAPI(" not in runtime_source
    assert "include_router" not in runtime_source


def test_first_deploy_rollback_restores_data_and_leaves_no_current_release(tmp_path: Path):
    base = tmp_path / "experiment-console"
    release = base / "releases" / "new"
    config = base / "config"
    state = base / "state"
    results = base / "results"
    backup = base / "backups" / "first"
    for directory in (release, config, state, results, backup):
        directory.mkdir(parents=True, exist_ok=True)
    (release / "compose.yggdrasil.yaml").write_text("services: {}\n", encoding="utf-8")
    (base / "current").symlink_to(release)
    (state / "new-state").write_text("discard", encoding="utf-8")
    (results / "new-result").write_text("discard", encoding="utf-8")
    env_values = {
        "EXPERIMENT_CONSOLE_UID": os.getuid(),
        "EXPERIMENT_CONSOLE_GID": os.getgid(),
        "EXPERIMENT_CONSOLE_STATE_PATH": state,
        "EXPERIMENT_CONSOLE_RESULTS_PATH": results,
        "EXPERIMENT_CONSOLE_WANDB_SECRET_FILE": base / "secrets" / "wandb",
        "EXPERIMENT_CONSOLE_API_TOKEN_SECRET_FILE": base / "secrets" / "console_api_token",
        "EXPERIMENT_CONSOLE_HCCS_SSH_KEY_FILE": base / "secrets" / "hccs",
        "EXPERIMENT_CONSOLE_HCCS_SSH_CONFIG_FILE": config / "ssh_config",
        "EXPERIMENT_CONSOLE_HCCS_KNOWN_HOSTS_FILE": config / "known_hosts",
        "EXPERIMENT_CONSOLE_AUTHORITY_ROLE": "authoritative",
        "EXPERIMENT_CONSOLE_INSTANCE_ID": "test-production",
    }
    (config / "production.env").write_text(
        "".join(f"{key}={value}\n" for key, value in env_values.items()),
        encoding="utf-8",
    )

    archived_state = tmp_path / "archived-state"
    archived_results = tmp_path / "archived-results"
    archived_state.mkdir()
    archived_results.mkdir()
    (archived_state / "old-state").write_text("restored", encoding="utf-8")
    (archived_results / "old-result").write_text("restored", encoding="utf-8")
    state_tar = backup / "state.tar.gz"
    results_tar = backup / "results.tar.gz"
    with tarfile.open(state_tar, "w:gz") as archive:
        archive.add(archived_state / "old-state", arcname="old-state")
    with tarfile.open(results_tar, "w:gz") as archive:
        archive.add(archived_results / "old-result", arcname="old-result")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (backup / "backup.meta").write_text(
        "\n".join([
            "backup_version=1",
            "release=__none__",
            f"state_path={state}",
            f"results_path={results}",
            f"state_sha256={digest(state_tar)}",
            f"results_sha256={digest(results_tar)}",
            "instance_id=test-production",
            "",
        ]),
        encoding="utf-8",
    )

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_docker.chmod(0o755)
    fake_sha = fake_bin / "sha256sum"
    fake_sha.write_text("#!/bin/sh\nexec /usr/bin/shasum -a 256 \"$@\"\n", encoding="utf-8")
    fake_sha.chmod(0o755)
    environment = {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin"}
    result = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "deploy" / "yggdrasil" / "rollback-release.sh"),
            "--base", str(base),
            "--backup-dir", str(backup),
            "--apply",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (base / "current").exists()
    assert (state / "old-state").read_text(encoding="utf-8") == "restored"
    assert (results / "old-result").read_text(encoding="utf-8") == "restored"
    assert "previous_release=__none__" in result.stdout


def _write_production_context(base: Path, release: Path, state: Path, results: Path) -> None:
    config = base / "config"
    secrets = base / "secrets"
    for directory in (config, secrets, release):
        directory.mkdir(parents=True, exist_ok=True)
    (release / "compose.yggdrasil.yaml").write_text("services: {}\n", encoding="utf-8")
    secret_paths = {
        "EXPERIMENT_CONSOLE_WANDB_SECRET_FILE": secrets / "wandb",
        "EXPERIMENT_CONSOLE_API_TOKEN_SECRET_FILE": secrets / "console_api_token",
        "EXPERIMENT_CONSOLE_HCCS_SSH_KEY_FILE": secrets / "hccs",
        "EXPERIMENT_CONSOLE_HCCS_SSH_CONFIG_FILE": config / "ssh_config",
        "EXPERIMENT_CONSOLE_HCCS_KNOWN_HOSTS_FILE": config / "known_hosts",
    }
    for path in secret_paths.values():
        path.write_text("test-value\n", encoding="utf-8")
    env_values = {
        "EXPERIMENT_CONSOLE_UID": os.getuid(),
        "EXPERIMENT_CONSOLE_GID": os.getgid(),
        "EXPERIMENT_CONSOLE_STATE_PATH": state,
        "EXPERIMENT_CONSOLE_RESULTS_PATH": results,
        **secret_paths,
        "EXPERIMENT_CONSOLE_AUTHORITY_ROLE": "authoritative",
        "EXPERIMENT_CONSOLE_INSTANCE_ID": "test-production",
    }
    (config / "production.env").write_text(
        "".join(f"{key}={value}\n" for key, value in env_values.items()),
        encoding="utf-8",
    )


def _write_fake_activation_commands(fake_bin: Path) -> None:
    fake_bin.mkdir(parents=True, exist_ok=True)
    commands = {
        "docker": """#!/bin/bash
if [[ "${FAIL_ACTIVATION_STEP:-}" == "start" && " $* " == *" up "* ]]; then
  exit 43
fi
if [[ "${FAKE_DOCKER_MODE:-}" == "fresh" ]]; then
  if [[ "$1" == "compose" && " $* " == *" ps "* ]]; then
    echo fake-console-cid
  elif [[ "$1" == "compose" && " $* " == *" up "* ]]; then
    touch "$FAKE_DOCKER_STATE"
  elif [[ "$1" == "inspect" && "$*" == *".State.Running"* ]]; then
    echo true
  elif [[ "$1" == "inspect" && "$*" == *".State.Health"* ]]; then
    echo healthy
  elif [[ "$1" == "exec" && -e "$FAKE_DOCKER_STATE" ]]; then
    echo ledger_new_v2
  elif [[ "$1" == "exec" ]]; then
    echo ledger_old_v1
    echo "${FAKE_NONTERMINAL_JOBS:-0}"
    echo __none__
  fi
fi
exit 0
""",
        "cp": """#!/bin/bash
if [[ "${FAIL_ACTIVATION_STEP:-}" == "seed" ]]; then
  exit 41
fi
exec /bin/cp "$@"
""",
        "ln": """#!/bin/bash
if [[ "${FAIL_ACTIVATION_STEP:-}" == "symlink" ]]; then
  exit 42
fi
exec /bin/ln "$@"
""",
        "sha256sum": """#!/bin/sh
exec /usr/bin/shasum -a 256 "$@"
""",
    }
    for name, source in commands.items():
        path = fake_bin / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)


def _write_paired_backup(backup: Path, state: Path, results: Path) -> None:
    backup.mkdir(parents=True, exist_ok=True)
    archived_state = backup.parent / f"{backup.name}-state-source"
    archived_results = backup.parent / f"{backup.name}-results-source"
    archived_state.mkdir()
    archived_results.mkdir()
    (archived_state / "old-state").write_text("backup-state", encoding="utf-8")
    (archived_results / "old-result").write_text("backup-result", encoding="utf-8")
    state_tar = backup / "state.tar.gz"
    results_tar = backup / "results.tar.gz"
    with tarfile.open(state_tar, "w:gz") as archive:
        archive.add(archived_state / "old-state", arcname="old-state")
    with tarfile.open(results_tar, "w:gz") as archive:
        archive.add(archived_results / "old-result", arcname="old-result")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (backup / "backup.meta").write_text(
        "\n".join([
            "backup_version=1",
            "release=__none__",
            f"state_path={state}",
            f"results_path={results}",
            f"state_sha256={digest(state_tar)}",
            f"results_sha256={digest(results_tar)}",
            "instance_id=test-production",
            "",
        ]),
        encoding="utf-8",
    )


def test_activation_and_rollback_traps_are_armed_before_mutation():
    activation = (ROOT / "deploy" / "yggdrasil" / "activate-release.sh").read_text(encoding="utf-8")
    rollback = (ROOT / "deploy" / "yggdrasil" / "rollback-release.sh").read_text(encoding="utf-8")
    assert activation.index("trap 'rollback_activation_once $?' ERR") < activation.index('if [[ -r "$SEED_LEDGER" ]]')
    assert activation.count('"$HERE/rollback-release.sh" --base "$BASE" --backup-dir "$BACKUP_DIR" --apply') == 1
    assert rollback.index("trap 'restore_displaced $?' ERR") < rollback.index('mv "$STATE_PATH" "$DISPLACED_STATE"')
    assert rollback.index("STATE_MOVED=1") < rollback.index('mv "$RESULTS_PATH" "$DISPLACED_RESULTS"')
    assert "fail " not in activation.split("trap 'rollback_activation_once $?' ERR", 1)[1]
    assert "fail " not in rollback.split("trap 'restore_displaced $?' ERR", 1)[1]


@pytest.mark.parametrize("failure_step", ["seed", "symlink", "start"])
def test_post_backup_activation_failure_rolls_back_exactly_once(tmp_path: Path, failure_step: str):
    base = tmp_path / "experiment-console"
    release = base / "releases" / "new"
    state = base / "state"
    results = base / "results"
    state.mkdir(parents=True)
    results.mkdir(parents=True)
    (state / "original-state").write_text("preserve", encoding="utf-8")
    (results / "original-result").write_text("preserve", encoding="utf-8")
    _write_production_context(base, release, state, results)
    seed = release / "migration-seed"
    seed.mkdir()
    with sqlite3.connect(seed / "console.sqlite3") as connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata VALUES ('ledger_id', 'ledger_seed', 'now')")
    (seed / "migration_manifest.json").write_text("{}\n", encoding="utf-8")

    fake_bin = tmp_path / "fake-bin"
    _write_fake_activation_commands(fake_bin)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
        "FAIL_ACTIVATION_STEP": failure_step,
    }
    result = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "deploy" / "yggdrasil" / "activate-release.sh"),
            "--base", str(base),
            "--release", str(release),
            "--apply",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert result.stdout.count("Rollback complete;") == 1, result.stdout + result.stderr
    assert (state / "original-state").read_text(encoding="utf-8") == "preserve"
    assert (results / "original-result").read_text(encoding="utf-8") == "preserve"
    assert not (state / "console.sqlite3").exists()
    assert not (base / "current").exists()
    assert len([path for path in (base / "backups").iterdir() if not path.name.startswith(".")]) == 1


def test_fresh_v2_activation_backs_up_then_starts_empty_ledger(tmp_path: Path):
    base = tmp_path / "experiment-console"
    old_release = base / "releases" / "old"
    new_release = base / "releases" / "new"
    state = base / "state"
    results = base / "results"
    state.mkdir(parents=True)
    results.mkdir(parents=True)
    (state / "console.sqlite3").write_text("old-ledger-placeholder", encoding="utf-8")
    (state / "old-audit.jsonl").write_text("old", encoding="utf-8")
    (results / "old-result.json").write_text("old", encoding="utf-8")
    _write_production_context(base, new_release, state, results)
    old_release.mkdir(parents=True)
    (old_release / "compose.yggdrasil.yaml").write_text("services: {}\n", encoding="utf-8")
    (base / "current").symlink_to(old_release)

    fake_bin = tmp_path / "fake-bin"
    _write_fake_activation_commands(fake_bin)
    fake_state = tmp_path / "console-started"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
        "FAKE_DOCKER_MODE": "fresh",
        "FAKE_DOCKER_STATE": str(fake_state),
    }
    result = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "deploy" / "yggdrasil" / "activate-release.sh"),
            "--base", str(base),
            "--release", str(new_release),
            "--fresh-v2-ledger",
            "--apply",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (base / "current").resolve() == new_release.resolve()
    assert not list(state.iterdir())
    assert not list(results.iterdir())
    backups = [path for path in (base / "backups").iterdir() if not path.name.startswith(".")]
    assert len(backups) == 1
    receipt = base / "cutovers" / "new.meta"
    receipt_text = receipt.read_text(encoding="utf-8")
    assert "previous_ledger_id=ledger_old_v1" in receipt_text
    assert "new_ledger_id=ledger_new_v2" in receipt_text
    assert "verified_empty=1" in receipt_text


def test_fresh_v2_activation_refuses_nonterminal_jobs(tmp_path: Path):
    base = tmp_path / "experiment-console"
    old_release = base / "releases" / "old"
    new_release = base / "releases" / "new"
    state = base / "state"
    results = base / "results"
    state.mkdir(parents=True)
    results.mkdir(parents=True)
    (state / "console.sqlite3").write_text("old-ledger-placeholder", encoding="utf-8")
    (state / "preserve").write_text("yes", encoding="utf-8")
    _write_production_context(base, new_release, state, results)
    old_release.mkdir(parents=True)
    (old_release / "compose.yggdrasil.yaml").write_text("services: {}\n", encoding="utf-8")
    (base / "current").symlink_to(old_release)
    fake_bin = tmp_path / "fake-bin"
    _write_fake_activation_commands(fake_bin)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
        "FAKE_DOCKER_MODE": "fresh",
        "FAKE_DOCKER_STATE": str(tmp_path / "console-started"),
        "FAKE_NONTERMINAL_JOBS": "2",
    }
    result = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "deploy" / "yggdrasil" / "activate-release.sh"),
            "--base", str(base),
            "--release", str(new_release),
            "--fresh-v2-ledger",
            "--apply",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refused while 2 jobs are nonterminal" in result.stderr
    assert (state / "preserve").read_text(encoding="utf-8") == "yes"
    assert not (base / "backups").exists()


@pytest.mark.parametrize("failed_directory", ["state", "results"])
def test_rollback_restores_only_data_directories_that_were_moved(tmp_path: Path, failed_directory: str):
    base = tmp_path / "experiment-console"
    state = base / "state"
    results = base / "results"
    backup = base / "backups" / "partial-move"
    state.mkdir(parents=True)
    results.mkdir(parents=True)
    (state / "live-state").write_text("state", encoding="utf-8")
    (results / "live-result").write_text("result", encoding="utf-8")
    _write_paired_backup(backup, state, results)

    fake_bin = tmp_path / "fake-bin"
    _write_fake_activation_commands(fake_bin)
    fake_mv = fake_bin / "mv"
    fake_mv.write_text(
        """#!/bin/bash
if [[ "$1" == "${FAIL_MV_SOURCE:-}" ]]; then
  exit 44
fi
exec /bin/mv "$@"
""",
        encoding="utf-8",
    )
    fake_mv.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
        "FAIL_MV_SOURCE": str(state if failed_directory == "state" else results),
    }
    result = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "deploy" / "yggdrasil" / "rollback-release.sh"),
            "--base", str(base),
            "--backup-dir", str(backup),
            "--apply",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 44, result.stdout + result.stderr
    assert (state / "live-state").read_text(encoding="utf-8") == "state"
    assert (results / "live-result").read_text(encoding="utf-8") == "result"
    assert not list(base.glob("state.pre-rollback-*"))
    assert not list(base.glob("results.pre-rollback-*"))


def test_rollback_refuses_after_cutover_commit(tmp_path: Path):
    base = tmp_path / "experiment-console"
    release = base / "releases" / "current"
    state = base / "state"
    results = base / "results"
    backup = base / "backups" / "pre-cutover"
    state.mkdir(parents=True)
    results.mkdir(parents=True)
    _write_production_context(base, release, state, results)
    (base / "current").symlink_to(release)
    with sqlite3.connect(state / "console.sqlite3") as connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO metadata VALUES ('cutover_committed_at', '2026-07-15T01:02:03+00:00', '2026-07-15T01:02:03+00:00')"
        )
    (results / "live-result").write_text("preserve", encoding="utf-8")
    _write_paired_backup(backup, state, results)
    fake_bin = tmp_path / "fake-bin"
    _write_fake_activation_commands(fake_bin)
    environment = {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin"}
    result = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "deploy" / "yggdrasil" / "rollback-release.sh"),
            "--base", str(base),
            "--backup-dir", str(backup),
            "--apply",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "rollback refused after cutover_committed_at=" in result.stderr
    assert (state / "console.sqlite3").exists()
    assert (results / "live-result").read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("layout", ["same", "nested", "reverse-nested", "outside"])
def test_activation_and_rollback_reject_unsafe_data_path_layouts(tmp_path: Path, layout: str):
    base = tmp_path / "experiment-console"
    release = base / "releases" / "new"
    if layout == "same":
        state = results = base / "runtime"
    elif layout == "nested":
        state = base / "runtime"
        results = state / "results"
    elif layout == "reverse-nested":
        results = base / "runtime"
        state = results / "state"
    else:
        state = base / "state"
        results = tmp_path / "outside-results"
    _write_production_context(base, release, state, results)

    activation = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "deploy" / "yggdrasil" / "activate-release.sh"),
            "--base", str(base),
            "--release", str(release),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert activation.returncode != 0
    assert "DRY RUN" not in activation.stdout

    backup = base / "backups" / f"invalid-{layout}"
    backup.mkdir(parents=True)
    (backup / "backup.meta").write_text(
        "\n".join([
            "backup_version=1",
            "release=__none__",
            f"state_path={state}",
            f"results_path={results}",
            "state_sha256=unused",
            "results_sha256=unused",
            "instance_id=test-production",
            "",
        ]),
        encoding="utf-8",
    )
    rollback = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "deploy" / "yggdrasil" / "rollback-release.sh"),
            "--base", str(base),
            "--backup-dir", str(backup),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rollback.returncode != 0
    assert "DRY RUN" not in rollback.stdout
