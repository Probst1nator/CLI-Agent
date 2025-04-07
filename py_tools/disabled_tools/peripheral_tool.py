from typing import Any, Dict, List, Union
import pyautogui
import time
from termcolor import colored

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse

class PeripheralTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="peripheral",
            description="Execute keyboard inputs for hands-free control",
            detailed_description="""Use this tool to execute keyboard inputs in sequence or parallel.

Input Format:
1. Single key: "space"
2. Key combination (parallel press): ["ctrl", "s"]
3. Sequence of inputs: [["ctrl", "a"], "x"] (select all, then cut)

Common keys:
- Media: space, left, right (seek), up, down (volume)
- Navigation: tab, enter, esc, pageup, pagedown
- Modifiers: ctrl, alt, shift
- Text: a-z, 0-9, backspace, delete
- Function: f1-f12

Examples of key sequences:
- [["alt", "tab"], ["alt", "tab"]] : Switch between two windows
- ["left", "left", "left"] : Skip back 3 times
- [["ctrl", "a"], ["ctrl", "c"]] : Select all and copy
- ["space", ["alt", "tab"]] : Pause media and switch window""",
            constructor="""
def run(action: str, device: str) -> None:
    \"\"\"Control peripheral devices.
    
    Args:
        action: The action to perform ('open', 'close', 'connect', 'disconnect')
        device: The peripheral device to control ('camera', 'microphone', 'speaker', 'printer')
    \"\"\"
"""
        )

    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        """Execute keyboard inputs in sequence."""
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: inputs"
            )

        parameters = params["parameters"]
        inputs = parameters["inputs"]

        try:
            summary_parts = []
            for input_item in inputs:
                if isinstance(input_item, str):
                    # Single key press
                    print(colored(f"Pressing key: {input_item}", "cyan"))
                    pyautogui.press(input_item)
                    summary_parts.append(f"Pressed '{input_item}'")
                elif isinstance(input_item, (list, tuple)):
                    # Parallel key press (hotkey)
                    print(colored(f"Pressing keys together: {'+'.join(input_item)}", "cyan"))
                    pyautogui.hotkey(*input_item)
                    summary_parts.append(f"Pressed '{'+'.join(input_item)}' together")
                time.sleep(0.1)  # Small delay between actions

            return self.format_response(
                status="success",
                summary=f"Successfully executed keyboard inputs:\n" + "\n".join(f"- {part}" for part in summary_parts)
            )

        except Exception as e:
            return self.format_response(
                status="error",
                summary=f"Error executing keyboard inputs: {str(e)}"
            ) 