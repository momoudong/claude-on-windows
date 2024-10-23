"""Base classes for computer use tools."""

from dataclasses import dataclass
from typing import Any, ClassVar, Optional


@dataclass
class ToolResult:
    """Result from running a tool."""
    output: Optional[str] = None
    error: Optional[str] = None
    system: Optional[str] = None
    base64_image: Optional[str] = None

    def replace(self, **kwargs: Any) -> "ToolResult":
        """Create a new ToolResult with some fields replaced."""
        return ToolResult(**{**self.__dict__, **kwargs})


@dataclass
class CLIResult(ToolResult):
    """Result from running a CLI command."""
    pass


class ToolError(Exception):
    """Error raised by tools."""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


class BaseAnthropicTool:
    """Base class for Anthropic tools."""

    name: ClassVar[str]
    api_type: ClassVar[str]

    def to_params(self) -> dict[str, Any]:
        """Convert the tool to API parameters."""
        raise NotImplementedError

    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given parameters."""
        raise NotImplementedError
