from typing import Any, Dict, Tuple
import os
import sys
import traceback

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse, ToolStatus
from py_classes.cls_chat import Chat
from py_methods.cmd_execution import select_and_execute_commands

class PythonExecuteTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="python_execute",
            description="Execute an existing Python script from a file path.",
            detailed_description="""Use this tool when you need to:
- Run an existing Python script without making any changes
- Execute a previously created or edited script
Perfect for:
- Running a script after editing
- Re-executing a script to see updated outputs
- Running scripts from specific file paths""",
            constructor="""
def run(script_path: str) -> None:
    \"\"\"
    Execute an existing Python script.
    
    Args:
        script_path: The path to the Python script to execute.
    \"\"\"
""",
            default_followup_tools=["python_edit", "read_file", "python_execute"],
            is_followup_only=True
        )

    def _truncate_output(self, output: str, max_length: int = 1000, tail_length: int = 200) -> str:
        """
        Truncate long output to a reasonable length, keeping the last part.
        
        Args:
            output: The output string to truncate
            max_length: Maximum allowed length before truncation
            tail_length: Number of characters to keep from the end when truncating
            
        Returns:
            Truncated string with "..." prefix if it was truncated
        """
        if len(output) <= max_length:
            return output
            
        # Try to find the last newline within the tail section
        tail_start = len(output) - tail_length
        tail = output[tail_start:]
        last_newline = tail.rfind('\n')
        
        if last_newline != -1:
            # Found a newline in the tail, adjust tail_start to start at that newline
            tail_start = tail_start + last_newline + 1
            tail = output[tail_start:]
            
        return f"...{tail}"

    def execute_script(self, script_path: str) -> Tuple[str, bool]:
        """
        Execute a Python script and handle the results.
        
        Args:
            script_path: Path to the script to execute
        
        Returns:
            Tuple of (execution_details: str, success: bool)
        """
        execution_details, execution_summary = select_and_execute_commands(
            [f"python3 {script_path}"],
            auto_execute=True,
            detached=False
        )
        
        if "error" in execution_summary.lower() or "error" in execution_details.lower():
            return self._truncate_output(execution_details), False
        
        return self._truncate_output(execution_details), True

    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status=ToolStatus.ERROR,
                summary="Invalid parameters for python execute tool."
            )
        
        try:
            # Get script path from parameters
            parameters = params.get("parameters", {})
            script_path = parameters.get("script_path")
                
            if not script_path:
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary="Missing required parameter: script_path"
                )
            
            # Check if the script exists
            if not os.path.exists(script_path):
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary=f"The script at path '{script_path}' does not exist."
                )
            
            # Execute the script
            execution_details, success = self.execute_script(script_path)
            
            status = ToolStatus.SUCCESS if success else ToolStatus.ERROR
            execution_message = "Returned: " + execution_details if execution_details else ""
            status_message = "successfully" if success else "with errors"
            
            return self.format_response(
                status=status,
                summary=f"Executed {script_path} {status_message}!\n{execution_message}",
            )
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            return self.format_response(
                status=ToolStatus.ERROR,
                summary=f"Error calling python execute tool: {error_traceback}"
            )

# Test code when script is run directly
if __name__ == "__main__":
    import asyncio
    
    async def test_python_execute_tool():
        tool = PythonExecuteTool()
        chat = Chat()  # Create an empty chat for the context
        
        # Example parameters
        params = {
            "parameters": {
                "script_path": "/path/to/your/script.py"
            }
        }
        
        # Execute the tool
        try:
            response = await tool.run(params, chat)
            print("Tool Response Status:", response["status"])
            print("Tool Response Summary:", response["summary"])
        except Exception as e:
            print(f"Error testing PythonExecuteTool: {str(e)}")

    # Run the async test function
    asyncio.run(test_python_execute_tool()) 