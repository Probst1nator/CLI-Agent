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
        from pathlib import Path
        import base64
        
        # Import screenshot utility
        from utils.takescreenshot import takescreenshot
        
        screenshot_folder = Path("screenshots")
        screenshot_folder.mkdir(exist_ok=True)
        
        filename = f"screenshot_{len(list(screenshot_folder.glob('*.png'))) + 1}.png"
        screenshot_path = screenshot_folder / filename
        
        # Take screenshot using the utility
        result = takescreenshot(str(screenshot_path))
        
        base64_images = []
        if result and screenshot_path.exists():
            # Convert to base64
            with open(screenshot_path, 'rb') as img_file:
                base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                base64_images.append(base64_data)
                
            self.logger.info(f"Screenshot saved: {screenshot_path}")
        else:
            self.logger.warning("Screenshot capture may have failed")
            
        return base64_images