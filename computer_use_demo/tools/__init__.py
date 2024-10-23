"""Tool implementations for computer use."""

from .base import BaseAnthropicTool, CLIResult, ToolError, ToolResult
from .command import CommandTool
from .computer import ComputerTool


class ToolCollection:
    """Collection of tools available to the agent."""

    def __init__(self, *tools: BaseAnthropicTool):
        """Initialize with a list of tools."""
        self.tools = {tool.name: tool for tool in tools}

    def to_params(self):
        """Convert all tools to API parameters."""
        return [tool.to_params() for tool in self.tools.values()]

    async def run(self, name: str, tool_input: dict):
        """Run a tool by name with the given input."""
        if name not in self.tools:
            raise ToolError(f"unknown tool: {name}")
        return await self.tools[name](**tool_input)


__all__ = [
    "BaseAnthropicTool",
    "CLIResult",
    "CommandTool",
    "ComputerTool",
    "ToolCollection",
    "ToolError",
    "ToolResult",
]
