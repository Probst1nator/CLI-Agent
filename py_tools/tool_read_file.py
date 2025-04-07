from typing import Any, Dict, Optional
import os
import sys
import traceback

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse, ToolStatus
from py_classes.cls_chat import Chat

class ReadFileTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="read_file",
            description="Read the contents of a file from a specified path.",
            detailed_description="""Use this tool when you need to:
- Read the contents of a file
- Check the results of a Python script execution
- Analyze log files or output files
- View text files for further processing
Perfect for:
- Examining output files
- Viewing logs or error reports
- Reading configuration files
- Opening any text-based file on the system""",
            constructor="""
def run(file_path: str, max_lines: Optional[int] = None) -> None:
    \"\"\"
    Read the contents of a file from a specified path.
    
    Args:
        file_path: The path to the file to read.
        max_lines: Optional limit on the number of lines to read. If None, reads the entire file.
    \"\"\"
""",
            default_followup_tools=["read_file"],
            is_followup_only=True
        )

    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status=ToolStatus.ERROR,
                summary="Invalid parameters for file read tool."
            )
        
        try:
            # Get file path from parameters or from context metadata
            parameters = params.get("parameters", {})
            file_path = parameters.get("file_path")
            
            if (".py" in file_path):
                additional_followup_tools = ["python_edit", "python_execute"]
            else:
                additional_followup_tools = []
                
            if not file_path:
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary="No file_path provided and no previous Python script found in context."
                )
                
            max_lines = parameters.get("max_lines")
            
            # Check if the file exists
            if not os.path.exists(file_path):
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary=f"The file at path '{file_path}' does not exist."
                )
            
            # Check if the file is readable
            if not os.access(file_path, os.R_OK):
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary=f"The file at path '{file_path}' is not readable."
                )
            
            # Read the file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if max_lines is not None:
                        lines = []
                        for i, line in enumerate(f):
                            if i >= max_lines:
                                break
                            lines.append(line)
                        content = ''.join(lines)
                        if i >= max_lines:
                            content += f"\n... (truncated, showing first {max_lines} lines)"
                    else:
                        content = f.read()
                
                # If content is very large, truncate it
                if len(content) > 10000:  # Limit to 10K characters
                    content = content[:10000] + "\n... (content truncated)"
                
                return self.format_response(
                    status=ToolStatus.SUCCESS,
                    summary=f"Contents of {file_path}:\n\n{content}",
                    followup_tools=additional_followup_tools
                )
            except UnicodeDecodeError:
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary=f"The file at path '{file_path}' is not a text file or has an unsupported encoding."
                )
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            return self.format_response(
                status=ToolStatus.ERROR,
                summary=f"Error calling file read tool: {error_traceback}"
            )

# Test code when script is run directly
if __name__ == "__main__":
    import asyncio
    
    async def test_read_file_tool():
        tool = ReadFileTool()
        chat = Chat()  # Create an empty chat for the context
        
        # Example parameters
        params = {
            "parameters": {
                "file_path": __file__,  # Read this very file
                "max_lines": 10
            }
        }
        
        # Execute the tool
        try:
            response = await tool.run(params, chat)
            print("Tool Response Status:", response["status"])
            print("Tool Response Summary:", response["summary"])
        except Exception as e:
            print(f"Error testing FileReadTool: {str(e)}")

    # Run the async test function
    asyncio.run(test_read_file_tool()) 