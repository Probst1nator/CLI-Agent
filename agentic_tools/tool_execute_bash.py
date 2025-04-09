import subprocess
from typing import Any, Dict, List
import re
import sys
import time
from termcolor import colored
import asyncio
import os
import traceback
import builtins  # Import builtins to access the input function

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse, ToolStatus
from py_classes.cls_llm_router import LlmRouter, AIStrengths
from py_classes.cls_chat import Chat
from py_classes.globals import g

class ExecuteBashTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="execute_bash",
            description="Execute bash commands safely",
            detailed_description="""Use this tool when you need to:
- Execute system commands
- Manage files and directories
- Get system information
- Run command-line utilities
- Install packages
Perfect for:
- File operations
- System queries
- Process management
- Package installation
- Running system utilities""",
            constructor="""
def run(command: str, mirror_output_to_user: bool = False) -> None:
    \"\"\"Execute a bash command safely.
    
    Args:
        command: The bash command to execute
        mirror_output_to_user: When True, the raw output of the command will also be printed directly to the user.
    \"\"\"
"""
        )

    def _check_command_safety(self, command: str, force_local: bool = False) -> bool:
        """Check command safety using LLM"""
        command_guard_prompt = (
            "The following command must follow these guidelines:\n"
            "1. After execution it must exit fully automatically.\n"
            "2. It must not modify the operating system in major ways, although it is allowed to install trusted apt packages and updated software.\n"
            "Respond only with 'Safe' or 'Unsafe'\n\n"
            f"Command: {command}"
        )
        
        print("🛡")
        safety_response = LlmRouter.generate_completion(
            command_guard_prompt, 
            strength=AIStrengths.GUARD,
            force_local=force_local
        )
        
        # Consider command safe if response indicates safe or has acceptable security levels
        is_safe = (
            "safe" in safety_response.lower() or
            "S8" in safety_response or  # Ignore Intellectual Property
            "S7" in safety_response     # Ignore Privacy
        )
        
        return is_safe

    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status=ToolStatus.ERROR,
                summary="Invalid parameters for bash tool."
            )
        
        # Extract parameters
        parameters = params.get("parameters", {})
        command: str = parameters.get("command")
        mirror_output_to_user: bool = parameters.get("mirror_output_to_user", False)
        
        if not command:
            return self.format_response(
                status=ToolStatus.ERROR,
                summary="Missing required parameter: 'command'"
            )

        safe_commands: List[str] = ["cat", "ls", "pwd", "clear", "cls", "grep", "sed", "awk", "find", "head", "tail", "wc", "sort", "uniq", "diff", "echo", "date", "whoami", "hostname", "uname", "df", "du", "ps", "top", "which", "whereis", "file", "pip", "apt install"]
        is_safe: bool = any(command.startswith(cmd) for cmd in safe_commands) and "&&" not in command
            
        # Print command for visibility before safety check
        print(colored(f"Proposed command: {command}", 'magenta'))

        if not is_safe:
            # LLM safety check
            is_safe = self._check_command_safety(command, force_local=g.FORCE_LOCAL)
        
        # Handle unsafe commands with user confirmation
        if not is_safe:
            print(colored(f"Warning: Command '{command}' was flagged as potentially unsafe.", 'yellow'), file=sys.stderr)
            confirm: str
            try:
                # Use builtins.input to avoid potential conflicts if input is shadowed
                confirm = builtins.input(colored("Do you want to execute this command anyway? (y/N): ", 'yellow')).strip().lower()
            except EOFError: # Handle cases where input might not be available (e.g., non-interactive script)
                 confirm = 'n'

            if confirm != 'y':
                return self.format_response(
                    status=ToolStatus.ERROR, # Or potentially a SKIPPED status if available
                    summary=f"Command '{command}' execution cancelled by user due to safety concerns."
                )
            else:
                 print(colored("Executing potentially unsafe command due to user confirmation.", 'yellow'), file=sys.stderr)

        # Execute command
        try:
            # Re-print the command right before execution if it passed safety checks or confirmation
            print(colored(f"Executing: {command}", 'magenta'))
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=g.AGENTS_SANDBOX_DIR
            )
            
            # Format output first
            output_summary: str = f"Exit code: {result.returncode}\n"
            if result.stdout:
                output_summary += f"Stdout:\n{result.stdout}\n" # Label stdout
            if result.stderr:
                output_summary += f"Stderr:\n{result.stderr}\n" # Label stderr

            # Optionally mirror output to user's console
            if mirror_output_to_user:
                # Print labels for clarity when mirroring
                print(colored("--- Command Output Start ---", "cyan"), file=sys.stderr)
                if result.stdout:
                     print(colored("Stdout:", "cyan"), file=sys.stderr)
                     print(result.stdout, file=sys.stderr)
                if result.stderr:
                     print(colored("Stderr:", "cyan"), file=sys.stderr)
                     print(result.stderr, file=sys.stderr)
                print(colored(f"Exit code: {result.returncode}", "cyan"), file=sys.stderr)
                print(colored("--- Command Output End ---", "cyan"), file=sys.stderr)

            return self.format_response(
                status=ToolStatus.SUCCESS if result.returncode == 0 else ToolStatus.ERROR,
                summary=output_summary.strip() # Return the exact summary, stripped of trailing whitespace
            )

        except Exception as e:
            return self.format_response(
                status=ToolStatus.ERROR,
                summary=f"Error executing command: {str(e)}\n{traceback.format_exc()}"
            )