import argparse
import json
import os

from termcolor import colored

from py_classes.cls_chat import Chat, Role
from py_classes.cls_few_shot_provider import FewShotProvider
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
        
        
        # Write and execute the script
        tool_dir = os.path.join(g.PROJ_VSCODE_DIR_PATH, "python_tool")
        os.makedirs(tool_dir, exist_ok=True)

        # write script_title + script_content to script_path
        script_path = os.path.join(tool_dir, script_title)
        

        implement_script_chat = context_chat.deep_copy()

        if os.path.exists(script_path):
            print(colored(f"Existing script found at '{script_path}'.", "yellow"))
            with open(script_path, "r") as f:
                file_content = f.read()
                implement_script_chat.add_message(
                    Role.USER,
                    # f"I have found a existing script with the title '{script_title}'.\nPlease check if this code needs to be modified or our requirements as is. ßn```python\n{file_content}\n```"
                    f"```python\n{file_content}\n```\n\nI have found a existing script with the title '{script_title}'.\nPlease check if this implementation already fulfills our requirements as is. Please respond in a json object with a 'reasoning' key-value first and a 'answer' key with one of the following as values: ('rewrite', 'overwrite' or 'accept')"
                )
                implement_script_chat.add_message(
                    Role.ASSISTANT,
                    "```json\n{\n  \"reasoning\":"
                )
                adjust_script_response = "{\"reasoning\":" + LlmRouter.generate_completion(implement_script_chat, force_local=args.local)
                adjust_script_json = adjust_script_response[adjust_script_response.find('{'):adjust_script_response.find('}')+1]
                adjust_script_obj = json.loads(adjust_script_json)
                adjust_script_reasoning = adjust_script_obj.get("reasoning", "")
                adjust_script_answer = adjust_script_obj.get("answer", 'overwrite')
        else:
            adjust_script_answer = 'overwrite'
        
        if adjust_script_answer == 'overwrite' or adjust_script_answer == 'rewrite':
            # Illusion of choice
            # if adjust_script_answer == 'overwrite': 
            #     implement_script_chat = context_chat.deep_copy()
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
            final_script = implementation_response[implementation_response.find('import'):implementation_response.find('```')]
        else:
            final_script = file_content
        
        if not script_title or not final_script:
            context_chat.add_message(
                Role.USER,
                "Error: Python script requires both title and content."
            )
        
        with open(script_path, "w") as f:
            f.write(final_script)

        # execute the script
        execution_details, execution_summarization = select_and_execute_commands([f"python3 {script_path}"], auto_execute=True, detached=True)
        
        if FewShotProvider.few_shot_YesNo(f"Did the executed python encounter an error?\n{execution_details}", force_local=args.local)[0]:
            
            context_chat.add_message(Role.USER, "The python script has encountered an error.")
            context_chat.add_message(Role.ASSISTANT, f"The python tool has been executed for the script '{script_title}'.")
            handle_python_tool(tool, context_chat, args)
            return

        context_chat.add_message(Role.USER, execution_details)
        context_chat.add_message(Role.ASSISTANT, f"The python tool has been executed for the script '{script_title}'.")
    except Exception as e:
        error_msg = f"Error in Python tool execution: {str(e)}"
        print(colored(error_msg, "red"))