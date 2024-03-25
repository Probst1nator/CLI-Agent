import os
import re
import shutil
import subprocess
import time
from typing import List, Union

from interface.cls_chat_session import ChatSession
from interface.cls_llm_messages import Chat, Role
from interface.cls_ollama_client import OllamaClient
from interface.enum_available_models import AvailableModels

session = OllamaClient()


def setup_sandbox() -> str:
    """
    Sets up a sandbox directory for code execution.

    Returns:
    str: The path to the created sandbox directory.
    """
    sandbox_dir = "./sandbox/"
    if os.path.exists(sandbox_dir):
        shutil.rmtree(sandbox_dir)
    os.mkdir(sandbox_dir)
    return sandbox_dir


def run_code_in_sandbox(code: str, sandbox_dir: str) -> str:
    """
    Executes a given piece of code in a sandbox environment and captures its output.

    Parameters:
    code (str): The Python code to be executed.
    sandbox_dir (str): The directory path of the sandbox where the code will be executed.

    Returns:
    str: The output from executing the code.
    """
    code_path = os.path.join(sandbox_dir, "temp_code.py")
    with open(code_path, "w") as file:
        file.write(code)

    try:
        output = subprocess.check_output(
            ["python", code_path], stderr=subprocess.STDOUT, text=True
        )
    except subprocess.CalledProcessError as e:
        output = e.output  # Capture output even if the code execution fails

    return output


def llm_decide(choices: Union[List[str], str]) -> Union[int, None]:
    """
    Processes a list of choices or a single choice string, prompts the language model to decide on the best choice,
    and returns the index of that choice.

    Parameters:
    choices (Union[List[str], str]): A list of strings or a single string representing the choices.

    Returns:
    Union[int, None]: The index of the chosen option as an integer, or None if no valid choice is identified.
    """
    if isinstance(choices, list):
        prompt = "\n".join(choices)
    elif isinstance(choices, str):
        prompt = choices
    prompt += "\nGiven the above, which of the following options is best?\n"
    for i, choice in enumerate(choices):
        prompt += f"{i+1}. {choice}\n"
    prompt += "Please provide the number of the best option."

    chat = Chat()
    chat.add_message(Role.USER, prompt)
    completion = session.generate_completion(
        "phi", chat, "Let's think about this, step by step. "
    )
    chat.add_message(Role.ASSISTANT, completion)
    chat.add_message(
        Role.USER,
        "Can you please summarize your reasoning by explicitly responding with only the established index of your picked choice?",
    )
    completion = session.generate_completion(
        "phi", completion, "Sure! Based on the previous reasoning my chosen index is: "
    )
    match = re.match(r"(\d{1,3})", completion)
    if match:
        completion_number = int(match.group(1))
    else:
        print("ERROR: No completion number was found:\t" + completion)
        completion_number = None
    return completion_number


def extract_code_from_response(response: str) -> Union[str, None]:
    """
    Extracts a code snippet from a given response string.

    Parameters:
    response (str): The response string containing the code snippet.

    Returns:
    Union[str, None]: The extracted code snippet as a string, or None if no code snippet is found.
    """
    pattern = r"'''(.*?)'''"
    matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        return None
    elif len(matches) == 1:
        return matches[0].strip()
    else:
        return "Multiple code snippets detected. Please specify which one to use."


def handle_development_cycle(chat: Chat, session: OllamaClient, sandbox_dir: str):
    generated_code = design_and_implement(chat, session)
    test_generated_code(chat, generated_code, sandbox_dir)


def design_and_implement(chat: Chat, session: OllamaClient) -> tuple[str, str]:
    # Generate project design
    additional_content = "\n\nAs a senior expert developer, design this Python tool idea bit by bit using plain-text and full keypoint descriptions, detailing its main functionalities and how they interconnect."
    chat.update_last_message(chat.messages[-1][1] + additional_content)
    project_design = session.generate_completion(chat, start_response_with="Sure! As requested I will provide a project outline without showing any python snippets. Instead I will use keypoints and types for clarity.\n", include_start_response_with=False)

    # Generate Python code based on the project design
    code_prompt = f"Please implement a single full python script according to the following specifications: '''script_specificiation\n{project_design}'''"
    generated_code = session.generate_completion(code_prompt, AvailableModels.CODING)
    generated_code = extract_code_from_response(generated_code)

    return generated_code



def test_generated_code(chat: Chat, code: str, sandbox_dir: str):
    if code:
        test_result = run_code_in_sandbox(code, sandbox_dir)
        chat.add_message(Role.ASSISTANT, test_result)
    else:
        chat.add_message(Role.ASSISTANT, "No valid code snippet found.")


def log_conversation(chat: Chat):
    """
    Logs the conversation from the chat to a file.

    Parameters:
    chat (Chat): An instance of Chat whose messages will be logged.
    """
    with open("generated_chat.txt", "a") as file:
        for message in chat.messages:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            file.write(f"[{timestamp}] {message[0].name}: {message[0].value}\n")


sandbox_dir = setup_sandbox()

if __name__ == "__main__":
    chat = Chat()

    while True:
        # user_input = input("Prompt: ") #commented out for dev
        user_input = "Please implement a python project to convert hexadecimal to binary and print it out when simply run, as a test."
        chat.add_message(Role.USER, user_input)
        handle_development_cycle(chat, session, sandbox_dir)
        log_conversation(chat)
