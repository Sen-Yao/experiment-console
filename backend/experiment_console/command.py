from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Mapping

from .redaction import redact_text


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    def summary(self, max_chars: int = 4000) -> dict:
        return {
            "argv": self.argv,
            "returncode": self.returncode,
            "stdout": redact_text(self.stdout[-max_chars:]),
            "stderr": redact_text(self.stderr[-max_chars:]),
        }


class CommandFailed(RuntimeError):
    def __init__(self, result: CommandResult):
        self.result = result
        super().__init__(f"command failed ({result.returncode}): {' '.join(result.argv)}")


class CommandRunner:
    def run(self, argv: list[str], *, timeout: int, env: Mapping[str, str] | None = None, cwd: str | None = None) -> CommandResult:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        result = CommandResult(argv=argv, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        if result.returncode != 0:
            raise CommandFailed(result)
        return result

