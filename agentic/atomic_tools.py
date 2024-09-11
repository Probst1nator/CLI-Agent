
import os

from termcolor import colored

from py_methods.assistants import extract_single_snippet
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import AIStrengths, LlmRouter
from py_classes.globals import g

def implement_new_method(method_title: str, method_requirements: str) -> str:
    """
    Creates a new method in the `tools.py` file.
    Args:
    - method_title (str): The title of the new method.
    - method_requirements (str): The detailed functional requirements for the new method.
    Returns:
    - Creation Success or failure message.
    """
    sandbox_tools_path = os.path.join(g.PROJ_AGENTIC_SANDBOX_PATH, "tools.py")
    
    context_chat = Chat("You are an expert Python programmer. Your task is to create a new Python method based on given requirements.")
    context_chat.add_message(Role.USER, f"Please create a Python method named '{method_title}' that meets the following requirements:\n{method_requirements}\n\nProvide the full method implementation, including a detailed docstring. Do not add an example usage, only provide the imports and the single method. Classes are not allowed. Usage of APIs requiring API keys is not allowed.")
    
    response = LlmRouter.generate_completion(context_chat, preferred_models=["llama-3.1-405b-reasoning", "claude-3-5-sonnet", "gpt-4o"], strength=AIStrengths.STRONG)
    
    new_method = extract_single_snippet(response, allow_no_end=True)
    
    if new_method:
        with open(sandbox_tools_path, 'a') as file:
            file.write(f"\n\n{new_method}")
        return f"New method '{method_title}' has been created and added to tools.py", "green"
    else:
        return "Failed to generate a valid method. Please try again with more specific requirements.", "red"