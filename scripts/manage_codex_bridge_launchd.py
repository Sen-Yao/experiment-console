#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path


LABEL = "org.senyaolab.experiment-console-bridge"
ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Install or remove the HCCS tmux Goal wake bridge LaunchAgent."
    )
    result.add_argument("command", choices=["render", "install", "uninstall"])
    result.add_argument(
        "--config",
        default=str(Path.home() / ".config" / "experiment-console" / "bridge.json"),
        help="Absolute bridge config path (contains paths, not secret values).",
    )
    result.add_argument(
        "--python", default=sys.executable, help="Absolute Python 3 executable."
    )
    result.add_argument(
        "--plist",
        default=str(Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"),
    )
    result.add_argument(
        "--no-load",
        action="store_true",
        help="Write/remove the plist without calling launchctl.",
    )
    return result


def make_plist(*, python: Path, config: Path) -> dict[str, object]:
    log_dir = Path.home() / "Library" / "Logs" / "Experiment Wake Bridge"
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(python),
            "-m",
            "desktop_bridge",
            "--config",
            str(config),
            "run",
        ],
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "ThrottleInterval": 5,
        "LowPriorityIO": True,
        "StandardOutPath": str(log_dir / "stdout.log"),
        "StandardErrorPath": str(log_dir / "stderr.log"),
    }


def launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["/bin/launchctl", *args], check=check)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = Path(args.config).expanduser()
    python = Path(args.python).expanduser()
    plist = Path(args.plist).expanduser()
    if not config.is_absolute() or not python.is_absolute() or not plist.is_absolute():
        parser().error("--config, --python, and --plist must be absolute paths")
    domain = f"gui/{os.getuid()}"

    if args.command in {"render", "install"}:
        if args.command == "install" and not config.is_file():
            parser().error(f"bridge config does not exist: {config}")
        plist.parent.mkdir(parents=True, exist_ok=True)
        if args.command == "install":
            log_dir = Path.home() / "Library" / "Logs" / "Experiment Wake Bridge"
            log_dir.mkdir(parents=True, exist_ok=True)
        with plist.open("wb") as handle:
            plistlib.dump(
                make_plist(python=python, config=config), handle, sort_keys=True
            )
        os.chmod(plist, 0o600)
        print(plist)
        if args.command == "install" and not args.no_load:
            launchctl("bootout", domain, str(plist), check=False)
            launchctl("bootstrap", domain, str(plist))
            launchctl("kickstart", "-k", f"{domain}/{LABEL}")
        return 0

    if not args.no_load:
        launchctl("bootout", domain, str(plist), check=False)
    try:
        plist.unlink()
    except FileNotFoundError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
