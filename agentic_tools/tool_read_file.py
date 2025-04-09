from typing import Any, Dict, Optional
import os
import sys
import traceback

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse, ToolStatus
from py_classes.cls_chat import Chat
from py_classes.globals import g


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
def run(file_path: str, read_start_line: int = 0, read_end_line: Optional[int] = None) -> None:
    \"\"\"
    Read the contents of a file from a specified path.
    
    Args:
        file_path: The path to the file to read.
        read_start_line: Optional start line to read. If None, reads the entire file.
        read_end_line: Optional end line to read. If None, reads the entire file.
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
                            
            if not file_path:
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary="No file_path provided and no previous Python script found in context."
                )
                
            # Remove max_lines, get start and end lines
            read_start_line = parameters.get("read_start_line", 0) # Default to 0 if not provided
            read_end_line = parameters.get("read_end_line")
            
            # Check if the file exists
            if not os.path.exists(file_path):
                file_path = os.path.join(g.AGENTS_SANDBOX_DIR, file_path)
                
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
                if (not os.path.exists(file_path)):
                    return self.format_response(
                        status=ToolStatus.ERROR,
                        summary=f"The file at path '{file_path}' does not exist."
                    )
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                if (len(lines) == 0):
                    return self.format_response(
                        status=ToolStatus.SUCCESS,
                        summary=f"The file at path '{file_path}' is empty."
                    )
                
                start_index = 0
                end_index = len(lines)
                

                # Process start line (defaults to 0 -> index 0)
                start_index = max(0, read_start_line - 1) # Convert 1-based (or 0) to 0-based index
                
                if read_end_line is not None:
                    end_index = min(len(lines), read_end_line) # End line is exclusive in slicing

                if start_index >= end_index:
                     return self.format_response(
                        status=ToolStatus.ERROR,
                        summary=f"Invalid line range: read_start_line ({read_start_line}) must be less than read_end_line ({read_end_line})."
                    )

                content_lines = lines[start_index:end_index]
                content = "".join(content_lines)
                
                # Add information about slicing if applicable
                summary_prefix = f"Contents of {file_path}"
                # Show range unless it's the full file (start_index=0, end_index=len(lines))
                if start_index != 0 or end_index != len(lines):
                    summary_prefix += f" (lines {start_index + 1} to {end_index})"
                summary_prefix += ":\n\n"

                # If content is very large, truncate it
                if len(content) > 10000:  # Limit to 10K characters
                    content = content[:10000] + "\n... (content truncated)"
                
                return self.format_response(
                    status=ToolStatus.SUCCESS,
                    summary=f"{summary_prefix}{content}"
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
                "read_start_line": 5,
                "read_end_line": 15
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