import argparse
import os

from termcolor import colored

from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_methods.cmd_execution import select_and_execute_commands
from py_classes.globals import g

def handle_python_tool(tool: dict, context_chat: Chat, args: argparse.Namespace) -> None:
    """
    Handle the Python tool execution within the agent's tool use system.
    Args:
    tool: The tool dictionary containing script details
    context_chat: The conversation context
    args: Command line arguments
    """
    try:
        script_title = tool.get('title', '')
        implement_script_chat = context_chat.deep_copy()
        implement_script_chat.add_message(
            Role.USER,
            "Can you implement the python script in a single block while using typing?"
        )
        implement_script_chat.add_message(
            Role.ASSISTANT,
            "Absolutely!\n```python\nimport"
        )
        implementation_response = "import" + LlmRouter.generate_completion(implement_script_chat, ["claude-3-5-sonnet-latest", "gpt-4o","qwen2.5-coder:7b-instruct"], force_local=args.local)
        # extract the python script from the response
        script_content = implementation_response[implementation_response.find('import'):implementation_response.find('```')]

        if not script_title or not script_content:
            context_chat.add_message(
                Role.USER,
                "Error: Python script requires both title and content."
            )
        
        # Write and execute the script
        tool_dir = os.path.join(g.PROJ_VSCODE_DIR_PATH, "python_tool")
        os.makedirs(tool_dir, exist_ok=True)

        # write script_title + script_content to script_path
        script_path = os.path.join(tool_dir, script_title)
        with open(script_path, "w") as f:
            f.write(script_content)

        # execute the script
        cmd_context_augmentation, execution_summarization = select_and_execute_commands([f"python3 {script_path}"], auto_execute=True)

        context_chat.add_message(Role.USER, cmd_context_augmentation)
        context_chat.add_message(Role.ASSISTANT, f"The python tool has been executed for the script '{script_title}'.")
    except Exception as e:
        error_msg = f"Error in Python tool execution: {str(e)}"
        print(colored(error_msg, "red"))