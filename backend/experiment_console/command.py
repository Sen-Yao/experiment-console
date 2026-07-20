from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    returncode: int


class CommandFailed(RuntimeError):
    def __init__(self, result: CommandResult):
        self.result = result
        detail = (result.stderr or result.stdout)[-2000:].strip()
        super().__init__(f"command failed with exit code {result.returncode}: {detail}")


class CommandRunner:
    def run(
        self,
        argv: list[str],
        *,
        timeout: float,
        input_text: str | None = None,
    ) -> CommandResult:
        completed = subprocess.run(
            argv,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        result = CommandResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
        if result.returncode != 0:
            raise CommandFailed(result)
        return result
