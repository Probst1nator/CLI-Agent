# tools/bash.py
"""
Tool to execute bash commands safely.
"""
import subprocess
import shlex


class bash:
    """Execute bash commands in the current working directory."""

    @staticmethod
    def get_delim() -> str:
        """Delimiter for this tool."""
        return 'bash'

    @staticmethod
    def get_tool_info() -> dict:
        """Tool documentation for the agent's system prompt."""
        return {
            "name": "bash",
            "description": (
                "Execute shell commands (npm, pip, pytest, build tools, etc.). "
                "Commands run in the current working directory. "
                "Use for package management, testing, building, and system operations. "
                "SAFETY: Avoid destructive commands (rm -rf, etc.) without explicit user request."
            ),
            "example": (
                "<bash>ls -la</bash>\n"
                "<bash>npm install</bash>\n"
                "<bash>pytest tests/</bash>\n"
                "<bash>python -m pip install -r requirements.txt</bash>"
            )
        }

    @staticmethod
    def run(content: str) -> str:
        """
        Execute a bash command and return its output.

        Args:
            content: The command to execute.

        Returns:
            Command output or error message.
        """
        command = content.strip()

        if not command:
            return "Error: No command provided."

        # Basic safety checks
        dangerous_patterns = [
            'rm -rf /',
            'rm -fr /',
            '> /dev/sda',
            'mkfs',
            'dd if=',
            'format ',
        ]

        command_lower = command.lower()
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return f"Error: Potentially destructive command blocked: '{pattern}'. Please verify with user before running."

        try:
            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )

            # Combine stdout and stderr
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]:\n{result.stderr}"

            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"

            return output if output.strip() else "[No output]"

        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 60 seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"
