from typing import Any, Dict, Optional
import os
import sys
import traceback

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse, ToolStatus
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import AIStrengths, LlmRouter
from py_classes.globals import g


class WriteFileTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="write_file",
            description="Write content to a file at a specified path.",
            detailed_description="""Use this tool when you need to:
- Create a new text file with specific content
- Modify an existing text file with new content
- Save generated text to a file
Perfect for:
- Creating configuration files
- Saving output from tool executions
- Writing documentation
- Writing scripts
- Creating or modifying any text-based file on the system""",
            constructor="""
def run(file_path: str, content_prompt: str, create_if_missing: Optional[bool] = True) -> None:
    \"\"\"
    Write content to a file at a specified path.
    
    Args:
        file_path: The path to the file to write to.
        content_prompt: A description of the content to write or the content itself.
        create_if_missing: Whether to create the file if it doesn't exist. Defaults to True.
    \"\"\"
""",
            default_followup_tools=["read_file", "write_file"],
            is_followup_only=False
        )

    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status=ToolStatus.ERROR,
                summary="Invalid parameters for file write tool."
            )
        
        try:
            # Get parameters
            parameters = params.get("parameters", {})
            file_path = parameters.get("file_path")
            content_prompt = parameters.get("content_prompt", "")
            raw_content = parameters.get("raw_content", "")
            create_if_missing = parameters.get("create_if_missing", True)
                
            if not file_path:
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary="Missing required parameter: file_path"
                )
                
            if not content_prompt and not raw_content:
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary="Missing required parameter: content_prompt or raw_content"
                )
            
            # Check if the file exists
            file_exists = os.path.exists(file_path)
            
            # If the file doesn't exist, use the sandbox directory
            if not file_exists:
                file_path = os.path.join(g.AGENTS_SANDBOX_DIR, file_path)
            
            file_exists = os.path.exists(file_path)
            
            # If file doesn't exist and we shouldn't create it
            if not file_exists and not create_if_missing:
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary=f"The file at path '{file_path}' does not exist and create_if_missing is set to False."
                )
            
            # Determine if we're modifying an existing file or creating a new one
            existing_content = ""
            if file_exists:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        existing_content = f.read()
                except UnicodeDecodeError:
                    return self.format_response(
                        status=ToolStatus.ERROR,
                        summary=f"The file at path '{file_path}' is not a text file or has an unsupported encoding."
                    )
                except Exception as e:
                    return self.format_response(
                        status=ToolStatus.ERROR,
                        summary=f"Error reading file at '{file_path}': {str(e)}"
                    )
            
            if (not raw_content):
                # Create prompt for the AI to generate content
                implement_chat = Chat()
                if file_exists:
                    implement_chat.add_message(
                        Role.USER, 
                        f"Modify the following file according to these instructions: {content_prompt}\n\nExisting content:\n\n```\n{existing_content}\n```"
                    )
                else:
                    implement_chat.add_message(
                        Role.USER, 
                        f"Create a file with the following specifications: {content_prompt}"
                    )
                
                # Generate the modified content
                generated_content = LlmRouter.generate_completion(implement_chat, strength=AIStrengths.CODE)
                
                # Clean response to ensure we only get content
                if "```" in generated_content:
                    # Extract content between first ``` pair
                    start_idx = generated_content.find("```")
                    # Check if there is a language specifier
                    next_newline = generated_content.find("\n", start_idx)
                    if next_newline != -1:
                        content_start = next_newline + 1
                    else:
                        content_start = start_idx + 3
                    
                    end_idx = generated_content.find("```", content_start)
                    if end_idx != -1:
                        generated_content = generated_content[content_start:end_idx]
                    else:
                        generated_content = generated_content[content_start:]
                
                # If there were no code blocks, use the raw generated content
                generated_content = generated_content.strip()
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            else:
                generated_content = raw_content
                
            # Write the content to the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(generated_content)
            
            action = "Modified" if file_exists else "Created"
            return self.format_response(
                status=ToolStatus.SUCCESS,
                summary=f"{action} file at {file_path} successfully!"
            )
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            return self.format_response(
                status=ToolStatus.ERROR,
                summary=f"Error calling file write tool: {error_traceback}"
            )

# Test code when script is run directly
if __name__ == "__main__":
    import asyncio
    
    async def test_write_file_tool():
        tool = WriteFileTool()
        chat = Chat()  # Create an empty chat for the context
        
        # Example parameters
        params = {
            "parameters": {
                "file_path": "test_output.txt",
                "content_prompt": "Create a simple text file with a greeting and the current date."
            }
        }
        
        # Execute the tool
        try:
            response = await tool.run(params, chat)
            print("Tool Response Status:", response["status"])
            print("Tool Response Summary:", response["summary"])
        except Exception as e:
            print(f"Error testing WriteFileTool: {str(e)}")

    # Run the async test function
    asyncio.run(test_write_file_tool()) 