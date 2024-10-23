"""Windows-compatible command execution tool using PowerShell."""

import asyncio
import os
from typing import ClassVar, Literal

from anthropic.types.beta import BetaToolBash20241022Param

from .base import BaseAnthropicTool, CLIResult, ToolError, ToolResult


class _PowerShellSession:
    """A session of a PowerShell shell."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "powershell.exe"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        # Start PowerShell with UTF-8 encoding for better output handling
        self._process = await asyncio.create_subprocess_exec(
            self.command,
            "-NoLogo",  # Don't show the logo
            "-NoProfile",  # Don't load the PowerShell profile
            "-NonInteractive",  # Non-interactive mode
            "-OutputFormat", "Text",  # Output as text
            "-Command", "-",  # Accept commands from stdin
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Ensure proper encoding for Windows
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )

        self._started = True

    def stop(self):
        """Terminate the PowerShell session."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str) -> ToolResult:
        """Execute a command in the PowerShell session."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return ToolResult(
                system="tool must be restarted",
                error=f"PowerShell has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: PowerShell has not returned in {self._timeout} seconds and must be restarted",
            )

        # Ensure stdin/stdout are available
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # Wrap command to capture output and errors
        wrapped_command = (
            f'$OutputEncoding = [Console]::OutputEncoding = '
            f'[Text.Encoding]::UTF8; $ErrorActionPreference = "Continue"; '
            f'try {{ {command} }} catch {{ $_.Exception.Message }}; '
            f'Write-Output "{self._sentinel}"'
        )

        # Send command to the process
        self._process.stdin.write(wrapped_command.encode() + b"\n")
        await self._process.stdin.drain()

        # Read output until sentinel is found
        output = []
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    line = await self._process.stdout.readline()
                    if not line:
                        break
                    decoded_line = line.decode('utf-8').rstrip()
                    if decoded_line == self._sentinel:
                        break
                    output.append(decoded_line)

        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: PowerShell has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        # Get any error output
        error = []
        while True:
            try:
                line = await self._process.stderr.readline()
                if not line:
                    break
                error.append(line.decode('utf-8').rstrip())
            except Exception:
                break

        return CLIResult(
            output="\n".join(output) if output else None,
            error="\n".join(error) if error else None
        )


class CommandTool(BaseAnthropicTool):
    """
    A tool that allows the agent to run PowerShell commands on Windows.
    This tool replaces the Linux bash tool with Windows-compatible functionality.
    """

    _session: _PowerShellSession | None
    name: ClassVar[Literal["bash"]] = "bash"  # Keep name as 'bash' for compatibility
    api_type: ClassVar[Literal["bash_20241022"]] = "bash_20241022"

    def __init__(self):
        self._session = None
        super().__init__()

    async def __call__(
        self, command: str | None = None, restart: bool = False, **kwargs
    ) -> ToolResult:
        if restart:
            if self._session:
                self._session.stop()
            self._session = _PowerShellSession()
            await self._session.start()
            return ToolResult(system="tool has been restarted.")

        if self._session is None:
            self._session = _PowerShellSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ToolError("no command provided.")

    def to_params(self) -> BetaToolBash20241022Param:
        return {
            "type": self.api_type,
            "name": self.name,
        }
