"""Screenshot command implementation."""

from __future__ import annotations
from typing import List, Any, TYPE_CHECKING
from termcolor import colored

from agent.commands.base_command import BaseCommand, CommandResponse, CommandResult

if TYPE_CHECKING:
    import argparse

class ScreenshotCommand(BaseCommand):
    """Command to take screenshots for visual analysis."""

    @property
    def aliases(self) -> List[str]:
        return ['/img', '/screenshot']

    @property
    def description(self) -> str:
        return 'Take a screenshot for visual analysis'

    async def execute(
        self,
        args: argparse.Namespace,
        parts: List[str],
        **kwargs: Any
    ) -> CommandResponse:
        """Execute screenshot capture."""
        self.logger.info(colored("# cli-agent: Taking screenshot.", "green"))

        try:
            # Import here to avoid startup overhead

            base64_images = await self._handle_screenshot_capture()

            return CommandResponse(
                result=CommandResult.SUCCESS,
                message="Screenshot captured successfully",
                base64_images=base64_images,
                user_input=""  # Empty input to trigger normal processing
            )

        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            return CommandResponse(
                result=CommandResult.ERROR,
                message=f"Screenshot capture failed: {str(e)}"
            )

    async def _handle_screenshot_capture(self) -> List[str]:
        """Handle screenshot capture logic (extracted from main.py)."""
        # Import screenshot utility
        from agent.core.input_handlers import handle_screenshot_capture

        # Take screenshot using the utility (it returns base64 images directly)
        base64_images = await handle_screenshot_capture()

        if base64_images:
            self.logger.info("Screenshot captured successfully")
        else:
            self.logger.warning("Screenshot capture may have failed")

        return base64_images