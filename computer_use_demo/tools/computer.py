"""Windows-compatible computer interaction tool using pyautogui."""

import asyncio
import base64
import io
import os
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

import pyautogui
from PIL import Image
from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult

# Ensure PyAutoGUI fails safely
pyautogui.FAILSAFE = True

# Configure typing speed
TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

# Configure screenshot directory
OUTPUT_DIR = os.path.join(os.getenv('TEMP', '.'), 'outputs')

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]

class Resolution(TypedDict):
    width: int
    height: int

# Add ultrawide resolution support
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
    "ULTRAWIDE": Resolution(width=5120, height=1440),  # 32:9
}

class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"

class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None

def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]

class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse using PyAutoGUI.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 0.5
    _scaling_enabled = False  # Disable scaling for ultrawide monitors

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self):
        super().__init__()
        # Use the environment variables for resolution if provided
        self.width = int(os.getenv("WIDTH") or 5120)  # Default to ultrawide
        self.height = int(os.getenv("HEIGHT") or 1440)
        self.display_num = None  # Windows handles multi-monitor differently

        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ) -> ToolResult:
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = self.scale_coordinates(ScalingSource.API, coordinate[0], coordinate[1])

            if action == "mouse_move":
                pyautogui.moveTo(x, y)
                return await self.take_screenshot()
            elif action == "left_click_drag":
                pyautogui.dragTo(x, y, button='left')
                return await self.take_screenshot()

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(f"{text} must be a string")

            if action == "key":
                # Convert key names to PyAutoGUI format
                key_parts = text.split('+')
                if len(key_parts) > 1:
                    # Handle key combinations (e.g., 'ctrl+c')
                    pyautogui.hotkey(*key_parts)
                else:
                    pyautogui.press(text)
                return await self.take_screenshot()
            elif action == "type":
                results = []
                for chunk in chunks(text, TYPING_GROUP_SIZE):
                    pyautogui.write(chunk, interval=TYPING_DELAY_MS/1000.0)
                    results.append(ToolResult(output=chunk))
                screenshot = await self.take_screenshot()
                return ToolResult(
                    output="".join(result.output or "" for result in results),
                    error="".join(result.error or "" for result in results),
                    base64_image=screenshot.base64_image,
                )

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.take_screenshot()
            elif action == "cursor_position":
                pos = pyautogui.position()
                x, y = self.scale_coordinates(ScalingSource.COMPUTER, pos.x, pos.y)
                return ToolResult(output=f"X={x},Y={y}")
            else:
                click_map = {
                    "left_click": lambda: pyautogui.click(button='left'),
                    "right_click": lambda: pyautogui.click(button='right'),
                    "middle_click": lambda: pyautogui.click(button='middle'),
                    "double_click": lambda: pyautogui.doubleClick(),
                }
                click_map[action]()
                return await self.take_screenshot()

        raise ToolError(f"Invalid action: {action}")

    async def take_screenshot(self) -> ToolResult:
        """Take a screenshot and return it as a base64 encoded string."""
        await asyncio.sleep(self._screenshot_delay)
        
        # Take screenshot using PyAutoGUI
        screenshot = pyautogui.screenshot()
        
        if self._scaling_enabled:
            x, y = self.scale_coordinates(ScalingSource.COMPUTER, self.width, self.height)
            screenshot = screenshot.resize((x, y), Image.Resampling.LANCZOS)

        # Save to bytes buffer
        img_buffer = io.BytesIO()
        screenshot.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Convert to base64
        base64_image = base64.b64encode(img_buffer.getvalue()).decode()
        
        return ToolResult(base64_image=base64_image)

    def scale_coordinates(self, source: ScalingSource, x: int, y: int) -> tuple[int, int]:
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
            
        ratio = self.width / self.height
        target_dimension = None
        
        for dimension in MAX_SCALING_TARGETS.values():
            # Allow some error in the aspect ratio
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                break
                
        if target_dimension is None:
            return x, y
            
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # Scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # Scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)
