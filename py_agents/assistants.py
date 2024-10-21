import argparse
from collections import defaultdict
import hashlib
import os
import random
import re
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import chromadb
import git
import pyperclip
from termcolor import colored

from py_classes.ai_providers.cls_ollama_interface import OllamaClient
from py_classes.cls_chat import Chat, Role
from py_classes.cls_few_shot_provider import FewShotProvider
from py_classes.cls_llm_router import AIStrengths, LlmRouter
from py_classes.cls_pptx_presentation import PptxPresentation
from py_classes.cls_web_scraper import WebTools
from py_methods.tooling import create_rag_prompt, extract_pdf_content, get_joined_pdf_contents, list_files_recursive, pdf_or_folder_to_database, run_python_script, split_string_into_chunks
from py_classes.globals import g


# # # helper methods
def extract_first_snippet(response: str, allow_no_end: bool = False) -> str:
    """
    Extracts the first code snippet from a given response string.
    This function searches for code blocks enclosed in triple backticks (```)
    and returns the content within the first found block.

    Args:
    response (str): The input string containing potential code snippets.
    allow_no_end (bool, optional): If True, allows extraction even if no end
                                   marker is found. Defaults to False.

    Returns:
    str: The extracted first code snippet, or an empty string if no valid snippet is found.

    Example:
    >>> response = "Here's some code:\n```python\nprint('Hello')\n```\nAnother snippet:\n```\nprint('World')\n```"
    >>> extract_first_snippet(response)
    "print('Hello')"
    """
    # Find the index of the first triple backtick
    start_index = response.find("```")
    if start_index != -1:
        # Find the end of the line containing the start marker
        start_line_end = response.find("\n", start_index)
        if start_line_end != -1:
            # Find the next triple backtick after the start marker
            end_index = response.find("```", start_line_end)
            if end_index != -1:
                # Extract and return the content between start and end markers
                return response[start_line_end + 1:end_index].strip()
            elif allow_no_end:
                # If no end marker is found and allow_no_end is True,
                # return everything after the start marker
                return response[start_line_end + 1:].strip()
    
    # Return an empty string if no valid snippet is found
    return ""
# # # helper methods


# # # assistants
def code_assistant(context_chat: Chat, file_path: str = "", pre_chosen_option: str = "", preferred_models: List[str] = [], force_local: bool = False) -> str:
    """
    A function that assists with code-related tasks such as adding docstrings, refactoring, and explaining code.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by argparse.
        context_chat (Chat): A Chat object representing the conversation context.

    This function performs the following main tasks:
    1. Reads code from a file if the --edit argument is provided.
    2. Optionally adds clipboard content to the code.
    3. Processes the code based on user input or automatic mode.
    4. Generates and returns modified code or explanations.
    
    Will return the modified code if pre_chosen_option is set to something other than "1" or "", else it will prompt the user for input and continue indefinetely.
    """
    snippets_to_process: List[str] = []
    result = ""
    preferred_models_before = preferred_models
    if file_path:
        # Read the content of the file specified by the --edit argument
        with open(file_path, 'r') as file:
            file_ending = os.path.splitext(file_path)[1]
            file_content = file.read()
            file_snippet = f"```{file_ending}\n{file_content}\n```"
            snippets_to_process = [file_snippet]
    
    while True:
        next_prompt = ""
        chunking_delimiter = ""
        if len(file_content)/2 > 8192 * 0.8: # 0.8 because works better
            if "py" in file_ending:
                chunking_delimiter = "def "
            elif "ts" in file_ending:
                chunking_delimiter = "function "
            else:
                chunking_delimiter = FewShotProvider.few_shot_objectFromTemplate([{"code_seperator": "def "}, {"code_seperator": "function "}], target_description=f"\n{file_snippet}\n\nA delimiter to split this code into smaller chunks", preferred_models=preferred_models, force_local=force_local)
                print(colored(f"Using delimiter: {chunking_delimiter}", "yellow"))
                try:
                    print(colored("Press Ctrl+C to abort.", 'yellow'))
                    for remaining in range(5, 0, -1):
                        sys.stdout.write("\r" + colored(f"Continuing in {remaining}s... ", 'yellow'))
                        sys.stdout.flush()
                        time.sleep(1)
                    sys.stdout.write("\n")  # Ensure we move to a new line after countdown
                except KeyboardInterrupt:
                    print(colored("Enter a custom delimiter to split the code into digestible chunks. Type '--f' on a new line when finished.", "blue"))
                    lines = []
                    while True:
                        line = input()
                        if line == "--f":
                            break
                        lines.append(line)
                    chunking_delimiter = "\n".join(lines)
                
        if pre_chosen_option == "1":
            # Automatic mode: Generate code overview and prepare prompt for docstring addition
            abstract_code_overview = LlmRouter.generate_completion("Please explain the below code step by step, provide a short abstract overview of its stages.\n\n" + snippets_to_process[0],  preferred_models=["llama-3.1-405b-reasoning", LlmRouter.last_used_model, "llama-3.1-70b-versatile"], strength=AIStrengths.STRONG, force_free=True, use_reasoning=False)
            if len(abstract_code_overview)/4 >= 2048:
                abstract_code_overview = LlmRouter.generate_completion(f"Summarize this code analysis, retaining the most important features and minimal details:\n{abstract_code_overview}",  preferred_models=[LlmRouter.last_used_model, "llama-3.1-70b-versatile"], strength=AIStrengths.STRONG, force_free=True, use_reasoning=False)
            next_prompt = "Augment the below code snippet with docstrings focusing on a concise overview of usage and parameters, alos add explanatory comments where the code seems highly complex. Do not include empty newlines in your docstrings. Retain all original comments, modifying them slightly only if essential. It is crucial that you do not modify the code's logic or structure; present it in full."
            next_prompt += f"\nTo help you get started, here's an handwritten overview of the code: \n{abstract_code_overview}"
            pre_chosen_option = ""
        else:
            # Second turn, do not process the same snippets again
            if result:
                snippets_to_process = []
            
            if pre_chosen_option:
                user_input = pre_chosen_option
                pre_chosen_option = ""
            else:
                # Manual mode: Present options to the user
                print(colored("Please choose an option:", 'cyan', attrs=["bold"]))
                print(colored("1. Add docstrings", 'yellow'))
                print(colored("2. Refactor", 'yellow'))
                print(colored("3. Explain", 'yellow'))
                print(colored("4. Perform web search", 'yellow'))
                print(colored("5. Use a custom delimiter for splitting the code into chunks", 'yellow'))
                print(colored("6. Add clipboard contents", 'yellow'))
                print(colored("7. Enable the most intelligent model", 'yellow'))
                print(colored("Write the prompt yourself", 'yellow') + " " + colored("(Use --m for multiline input)", 'grey'))
                user_input = input(colored("Enter your choice: ", 'blue'))
            
            # Process user input and set the appropriate prompt
            if user_input == "1":
                next_prompt = "Augment the following code snippet with thorough docstrings and step-by-step explanatory comments. Retain all original comments, modifying them slightly only if essential. It is crucial that you do not modify the code's logic or structure; present it in full."
            elif user_input == "2":
                next_prompt = "Please refactor the code, ensuring it remains functionally equivalent. You may change the code structure, variable names, and comments."
            elif user_input == "3":
                next_prompt = "Please explain the code, providing a concise explanation of the code's functionality and use."
            elif user_input == "4":
                web_query = input(colored("Enter your search query (defaults to the topic of the last 3 messages): ", 'blue'))
                if not web_query:
                    recent_context_str = context_chat.get_messages_as_string(-3)
                    query = FewShotProvider.few_shot_TextToQuery(recent_context_str)
                web_search_result = WebTools.search_brave(query, 2)
                context_chat.add_message(Role.USER, f"I found something on the web, please relate it to the code we're working on:\n```web_search\n{web_search_result}```")
                response = LlmRouter.generate_completion(context_chat, preferred_models=[LlmRouter.last_used_model, "llama-3.1-405b-reasoning", "claude-3-5-sonnet", "gpt-4o"], strength=AIStrengths.STRONG, use_reasoning=False)
                context_chat.add_message(Role.ASSISTANT, response)
                continue
            elif user_input == "5":
                # Process code in chunks
                print(colored("Enter your custom delimiter to split the code into digestible chunks. Type '--f' on a new line when finished.", "blue"))
                lines = []
                while True:
                    line = input()
                    if line == "--f":
                        break
                    lines.append(line)
                chunking_delimiter = "\n".join(lines)
            elif user_input == "6":
                clipboard_content = pyperclip.paste()
                snippets_to_process.insert(0, f"Here have a look at my clipboard:\n```\n{clipboard_content}\n```")
            elif user_input == "7":
                if preferred_models == [g.CURRENT_MOST_INTELLIGENT_MODEL_KEY]:
                    preferred_models = preferred_models_before
                    print(colored("Disabled the most intelligent model.", 'yellow'))
                else:
                    preferred_models_before = preferred_models
                    preferred_models = [g.CURRENT_MOST_INTELLIGENT_MODEL_KEY]
                    print(colored(f"Enabled the most intelligent model: {preferred_models[0]}", 'yellow'))
            else:
                next_prompt = user_input
            
            # Handle multiline input
            if next_prompt == "--m":
                print(colored("Enter your multiline input. Type '--f' on a new line when finished.", "blue"))
                lines = []
                while True:
                    line = input()
                    if line == "--f":
                        break
                    lines.append(line)
                next_prompt = "\n".join(lines)

            if not next_prompt:
                continue
        
        if chunking_delimiter:
            # Create regex pattern: match delimiter at start of string or after newline
            pattern = f'(^|\n)(?={re.escape(chunking_delimiter)})'
            # Split file_content using regex, keeping delimiter with following content
            snippets_to_process = re.split(pattern, file_content)
            # Clean snippets: remove empty strings, strip whitespace
            snippets_to_process = [snippet.strip() for snippet in snippets_to_process if snippet.strip()]
            # re-add delimiters to all snippets but first
            for i in range(1, len(snippets_to_process)):
                snippets_to_process[i] = chunking_delimiter + snippets_to_process[i]

        
        
        #! Actual code processing and generation starts here
        # check for additional snippets to add for processing
        generated_snippets: List[str] = []
        if snippets_to_process:
            print(colored(f"Processing {len(snippets_to_process)} snippets.", "yellow"))
            for snippet in snippets_to_process:
                # Prepare the prompt based on the number of snippets
                if len(snippets_to_process) == 1:
                    # Add the snippet only if it hasn't been added before
                    if not any(snippet in message for message in context_chat.messages):
                        next_prompt_i = next_prompt + f"\n\n{snippet}"
                else:
                    next_prompt_i = next_prompt + f"Please provide your modified snippet without any additional code such that it can be a drop in replacement for the below snippet:\n\n{snippet}"
                # Add the prompt to the chat context
                context_chat.add_message(Role.USER, next_prompt_i)
                response = LlmRouter.generate_completion(context_chat, preferred_models=preferred_models + [LlmRouter.last_used_model, "llama-3.1-405b-reasoning", "claude-3-5-sonnet", "gpt-4o"], strength=AIStrengths.STRONG)
                extracted_snippet = extract_first_snippet(response, allow_no_end=True)
                # Check if the response is empty because markers weren't included, this can be intended behavior if no code is asked for
                if (extracted_snippet):
                    generated_snippets.append(extracted_snippet)
                # Add the assistant's response to the chat context
                context_chat.add_message(Role.ASSISTANT, response)
        else:
            # Only use prompt + context without adding snippets
            context_chat.add_message(Role.USER, next_prompt)
            response = LlmRouter.generate_completion(context_chat, preferred_models=preferred_models+[LlmRouter.last_used_model, "llama-3.1-405b-reasoning", "claude-3-5-sonnet", "gpt-4o"], strength=AIStrengths.STRONG)
            extracted_snippet = extract_first_snippet(response, allow_no_end=True)
            # Check if the response is empty because markers weren't included, this can be intended behavior if no code is asked for
            if (extracted_snippet):
                generated_snippets.append(extracted_snippet)
            # Add the assistant's response to the chat context
            context_chat.add_message(Role.ASSISTANT, response)
        
        # Check if generated snippets can automatically replace the original snippets
        if len(snippets_to_process) == len(generated_snippets):
            # Replace the original snippets with the generated snippets
            result = file_content
            for source_snippet, generated_snippet in zip(snippets_to_process, generated_snippets):
                # remove first and last line from source_snippet
                clean_snippet = "\n".join(source_snippet.split("\n")[1:-1])
                result = result.replace(clean_snippet, generated_snippet)
            pyperclip.copy(result)
            print(colored("Result copied to clipboard.", 'green'))
        # Snippets were generated but need to be handled by hand
        elif len(snippets_to_process) <= len(generated_snippets):
            result = "\n".join(generated_snippets)
            pyperclip.copy(result)
            print(colored("Result copied to clipboard.", 'green'))
        # Something likely went wrong
        else:
            print(colored("INFO: Snippets were not reimplemented by the assistant.", 'yellow'))
            if (len(snippets_to_process) > 1):
                context_chat.add_message(Role.USER, "Please summarize your reasoning step by step and provide a short discussion.")
                response = LlmRouter.generate_completion(context_chat, preferred_models=preferred_models+[LlmRouter.last_used_model, "llama-3.1-405b-reasoning", "claude-3-5-sonnet", "gpt-4o"], strength=AIStrengths.STRONG)
        # Save chat
        if context_chat:
            context_chat.save_to_json()


def presentation_assistant(args: argparse.Namespace, context_chat: Chat, user_input: str = ""):
    """
    An assistant function that helps create and modify PowerPoint presentations based on user input.

    Args:
        args (argparse.Namespace): Command-line arguments containing LLM preferences and other settings.
        context_chat (Chat): The conversation context (not used in this function, but kept for consistency).
        user_input (str, optional): Initial user input for the presentation topic. Defaults to an empty string.

    Returns:
        None: The function modifies the presentation in-place and saves it to disk.

    This function performs the following main tasks:
    1. Collects user input for the presentation topic if not provided.
    2. Generates a structured presentation based on the input.
    3. Allows for iterative refinement of the presentation through user interaction.
    """

    # Collect user input if not provided
    if not user_input:
        print(colored("Please enter your presentation topic(s) and supplementary data if present. *Multiline mode* Type '--f' on a new line when finished.", "magenta"))
        lines = []
        while True:
            line = input()
            if line == "--f":
                break
            lines.append(line)
        user_input = "\n".join(lines)
    
    # Rephrase user input using few-shot learning
    rephrased_user_input = FewShotProvider.few_shot_rephrase(user_input, preferred_models=[args.llm], force_local=args.local)

    # Generate a prompt for decomposing the topic into subtopics
    decomposition_prompt = FewShotProvider.few_shot_rephrase("Please decompose the following into 3-6 subtopics and provide step by step explanations + a very short discussion:", preferred_models=[args.llm], force_local=args.local)

    # Generate detailed presentation content based on the decomposed topic
    presentation_details = LlmRouter.generate_completion(f"{decomposition_prompt}: '{rephrased_user_input}'", strength=AIStrengths.STRONG, use_cache=False, preferred_models=[args.llm], force_local=args.local)
    
    # Convert the generated content into a presentation format
    chat, response = FewShotProvider.few_shot_textToPresentation(presentation_details, preferred_models=[args.llm], force_local=args.local)

    while True:
        # Attempt to create and save the presentation from the generated JSON
        while True:
            try:
                presentation_json = response.split("```")[1].split("```")[0]
                presentation = PptxPresentation.from_json(presentation_json)
                presentation.save()
                break
            except Exception as e:
                # Handle errors in JSON format and regenerate the presentation
                chat.add_message(Role.USER, "Your json object did not follow the expected format, please try again.\nError: " + str(e))
                response = LlmRouter.generate_completion(chat, strength=AIStrengths.STRONG, use_cache=False, preferred_models=[args.llm], force_local=args.local)
                chat.add_message(Role.ASSISTANT, response)
                
        print(colored("Presentation saved.", 'green'))

        # Present options for further modifications
        print(colored("Please choose an option:", 'cyan', attrs=["bold"]))
        print(colored("1. Add details", 'yellow'))
        print(colored("2. Regenerate", 'yellow'))
        print(colored("3. Add Images, this may take a while...", 'yellow'))
        print(colored("Write the prompt yourself", 'yellow') + " " + colored("(Use --m for multiline input)", 'grey'))
        user_input = input(colored("Enter your choice: ", 'blue'))

        # Process user's choice
        if user_input == "1":
            # Generate and add more details to the presentation
            add_details_prompt = FewShotProvider.few_shot_rephrase(f"Please think step by step to add relevant/ missing details to the following topic: {presentation_details}", preferred_models=[args.llm])
            suggested_details = LlmRouter.generate_completion(f"{add_details_prompt} {presentation_details}", strength=AIStrengths.STRONG, preferred_models=[args.llm], force_local=args.local)
            next_prompt = f"Please add the following details to the presentation: \n{suggested_details}"
        elif user_input == "2":
            # Regenerate the entire presentation
            next_prompt = "I am unhappy with your suggested presentation, please try again."
        else:
            # Use custom user input as the next prompt
            next_prompt = user_input
            
        # Rephrase the next prompt and generate a new response
        next_prompt = FewShotProvider.few_shot_rephrase(next_prompt, preferred_models=[args.llm], force_local=args.local)
        chat.add_message(Role.USER, next_prompt)
        response = LlmRouter.generate_completion(chat, strength=AIStrengths.STRONG, preferred_models=[args.llm], force_local=args.local)

def search_folder_assistant(args: argparse.Namespace, context_chat: Chat, user_input: str = ""):
    """
    An assistant function that searches through files in the current working directory and its subdirectories,
    processes the content, and answers user queries based on the collected information.

    Args:
        args (argparse.Namespace): Command-line arguments containing LLM preferences and other settings.
        context_chat (Chat): The conversation context (not used in this function, but kept for consistency).
        user_input (str, optional): Initial user input for the search query. Defaults to an empty string.

    This function performs the following main tasks:
    1. Collects user input for the search query if not provided.
    2. Processes files in the current directory and subdirectories, extracting and embedding their content.
    3. Performs a similarity search based on the user's query.
    4. Generates responses using an AI model based on the relevant documents found.
    5. Engages in an interactive conversation with the user, allowing for follow-up questions or new searches.
    """
    client = chromadb.PersistentClient(g.PROJ_VSCODE_DIR_PATH)
    collection = client.get_or_create_collection(name="documents")

    # Collect user input if not provided
    if not user_input:
        print(colored("Please enter your search request. *Multiline mode* Type '--f' on a new line when finished.", "blue"))
        lines = []
        while True:
            line = input()
            if line == "--f":
                break
            lines.append(line)
        user_input = "\n".join(lines)
    
    # Generate embedding for the user's input query
    user_input_embedding = OllamaClient.generate_embedding(user_input)
    
    # Prepare the instruction for the AI assistant using few-shot learning
    instruction = FewShotProvider.few_shot_rephrase(f"This is a chat between a user and his private artificial intelligence assistant. The assistant uses the documents to answer the users questions factually, detailed and reliably. The assistant indicates if the answer cannot be found in the documents.", preferred_models=[args.llm], force_local=args.local, silent_reason="No given reason")
    chat = Chat(instruction)

    while True:
        collected_data = ""
        # List all files in the current directory and its subdirectories (up to 2 levels deep)
        files = list_files_recursive(os.getcwd(), 2)
        
        # Process each file
        for file_index, file_path in enumerate(files, start=1):
            print(colored(f"({file_index}/{len(files)}) Processing file: {file_path}", "yellow"))
            file_name = os.path.basename(file_path).replace(" ", "_")
            
            # Extract content from PDF files
            if file_path.endswith(".pdf"):
                text_content = extract_pdf_content(file_path)
                digestible_contents = split_string_into_chunks(text_content)
                
                # Process and embed each chunk of the PDF content
                for digestible_content in digestible_contents:
                    digestible_content_hash = hashlib.md5(digestible_content.encode()).hexdigest()
                    digestible_content_id = f"{digestible_content_hash}"
                    
                    # Add the content to the collection if it doesn't exist
                    if not collection.get(digestible_content_id)['documents']:
                        embedding = OllamaClient.generate_embedding(digestible_content)
                        collection.add(
                            ids=[digestible_content_id],
                            embeddings=embedding,
                            documents=[digestible_content]
                        )

        # Perform a similarity search based on the user's query
        results = collection.query(
            query_embeddings=user_input_embedding,
            n_results=10
        )

        if results['documents']:
            # Collect relevant data from the search results
            for ids, relevant_data in zip(results['ids'][0], results['documents'][0]):
                file_name = "".join(ids.split("_")[1:])
                collected_data += f"```{file_name}\n{relevant_data}\n```\n"

            collected_data = collected_data.strip().strip("\n").strip()
            print(colored(f"DEBUG: collected_data token count: {len(collected_data)/4}", "yellow"))

        # Prepare the chat message with the user's question and relevant documents
        chat.add_message(Role.USER, f"### Question:\n{user_input}\n\n### Documents:\n{collected_data}\n\n### Question:\n{user_input}")
        print(chat.messages[-1][1])

        # Generate responses and engage in conversation with the user
        while True:
            response = LlmRouter.generate_completion(chat, preferred_models=[args.llm], force_local=args.local)
            chat.add_message(Role.ASSISTANT, response)
            user_input = input(colored("Enter your response, (Type '--f' to start a new search): ", "blue")).lower()
            if ("--f" in user_input):
                user_input = input(colored("Enter your search request, previous context is still available: ", "blue")).lower()
                break
            chat.add_message(Role.USER, user_input)

def majority_response_assistant(context_chat: Chat, force_local: bool = False, preferred_models: List[str] = [], allow_costly_models: bool = False) -> Tuple[Chat, str]:
    """
    An assistant function that leverages multiple AI models to provide comprehensive and consensus-based answers to user queries.
    This assistant, called "majority_vote_assistant", operates by consulting various models and synthesizing their outputs.

    Args:
        args (argparse.Namespace): Command-line arguments containing settings for model selection, voting mechanisms, and other parameters.
        context_chat (Chat): The conversation context (maintained for consistency but not directly used).
        user_input (str, optional): Initial user input for the query. Defaults to an empty string.
    """
    if len(preferred_models) > 1:
        models = preferred_models
    else:
        if force_local:
            models = LlmRouter.get_models(force_local=force_local)
            strong_models = [model for model in models if model.strength == AIStrengths.STRONG]
            if len(strong_models) > 1:
                models = strong_models
        else:
            if allow_costly_models:
                models = [LlmRouter.get_model(force_free=True, strength=AIStrengths.STRONG)] + LlmRouter.get_models(["claude-3-5-sonnet", "gpt-4o"])
                final_response_models: List[str] = ["gpt-4o"]
            else:
                models = [LlmRouter.get_model(force_free=True, strength=AIStrengths.STRONG)] + LlmRouter.get_models(["gemma2-9b-it", "claude-3-haiku", "gpt-4o-mini"])
                final_response_models: List[str] = [models[0]]

    while True:
        # Distribute query to all available models and gather responses
        model_responses_str = ""
        model_responses = []
        for i, model in enumerate(models):
            try:
                response = LlmRouter.generate_completion(context_chat, preferred_models=[model])
                if not response:
                    continue
                model_responses_str += f"EXPERT {i}:\n{response}\n\n```"
                model_responses.append(response)
            except Exception as e:
                print(colored(f"Error getting response from {model.model_key}: {str(e)}", "red"))

        print(colored(f"Received responses from {len(model_responses)} models. Summarizing...", "yellow"))

        chat = Chat("You are a data scientist tasked with performing a comprehensive meta analysis of responses from various experts on a given topic. Please summarize the responses, highlighting the key points and areas of agreement or disagreement. Be thorough and work step by step to grasp and reveal each relevant nuance of the conversation.")
        chat.add_message(Role.USER, f"CONTEXT:\n{model_responses_str}\nUSER_QUERY:\n{context_chat.messages[-1][1]}")
        response = LlmRouter.generate_completion(chat=chat, preferred_models=final_response_models, force_local=force_local)
        chat.add_message(Role.ASSISTANT, response)
        chat.add_message(Role.USER, f"Please condense all, insights and nuances into a deeply helpful response to the following user query: {context_chat.messages[-1][1]}")
        response = LlmRouter.generate_completion(chat=chat, preferred_models=final_response_models, force_local=force_local)
        chat.add_message(Role.ASSISTANT, response)
        
        context_chat.add_message(Role.ASSISTANT, response)
        return context_chat, response

# # # assistants

# # # agents
def python_error_agent(context_chat: Chat, script_path: str):
    """
    An agent function that iteratively fixes and executes a Python script.

    This function attempts to fix errors in a given Python script by analyzing the error output,
    generating fixes, and re-executing the script until it runs successfully or reaches a maximum
    number of iterations.

    Args:
        args (argparse.Namespace): Command-line arguments containing the path to the Python script to fix.
        context_chat (Chat): A Chat object to maintain conversation context for error analysis and fixes.

    The function performs the following main tasks:
    1. Executes the Python script and captures the output or error.
    2. If an error occurs, it analyzes the error and generates a fix.
    3. Writes the fixed script to a new file and re-executes it.
    4. Continues this process until the script executes successfully or reaches the iteration limit.
    5. Allows the user to overwrite the original script with the fixed version and clean up temporary files.
    """
    
    fix_iteration = 0
    while True:
        # Execute the Python script
        print(colored(f"Executing Python file at: {script_path}\n" + "# " * 10, 'green'))
        py_script = ""
        with open(script_path, 'r') as file:
            py_script = file.read()
        output, error = run_python_script(script_path)
        analysis_amalgam = ""
        user_input_insights = ""

        if error:
            # Handle script execution error
            print(colored(f"Error: {error}", 'red'))
            
            # Prepare context for error analysis based on iteration
            if len(context_chat.messages) == 0:  # first iteration
                context_chat.add_message(Role.USER, FewShotProvider.few_shot_rephrase(f"Please analyze the following error step by step and inspect how it can be fixed in the appended script, please do not suggest a fixed implementation instead focus on understanding and explaining the issue step by step.") + f"\n```error\n{error}\n\n```python\n{py_script}\n```")
            elif user_input_insights:
                context_chat.add_message(Role.USER, FewShotProvider.few_shot_rephrase(f"{user_input_insights}\nAgain, do not suggest a fixed implementation instead for now, solely focus on understanding and explaining the issue step by step.") + f"\n```error\n{error}```")
            else:  # default case
                context_chat.add_message(Role.USER, FewShotProvider.few_shot_rephrase(f"Reflect on your past steps in the light of this new error, what did you miss? Only reflect, combine and infer for now. Do not provide the full reimplementation yet!") + f"\n```error\n{error}")
            
            # Generate error analysis
            error_analysis = LlmRouter.generate_completion(context_chat)
            context_chat.add_message(Role.ASSISTANT, error_analysis)
            
            # Accumulate analysis for later review
            analysis_amalgam += f"Analysis {fix_iteration}: {error_analysis}\n"

            # Request fixed script
            context_chat.add_message(Role.USER, "Seems reasonable. Now, please provide the fixed script in full.")
            script_fix = LlmRouter.generate_completion(context_chat, preferred_models=["llama-3.1-70b-versatile", "gpt-4o", "claude-3-5-sonnet"])
            context_chat.add_message(Role.ASSISTANT, script_fix)
            fixed_script = extract_first_snippet(script_fix)
            
            # Write fixed script to a new file
            script_path = script_path.replace(".py", f"_patchV{fix_iteration}.py")
            with open(script_path, 'w') as file:
                file.write(fixed_script)
                print(colored(f"Iteration {fix_iteration}: Patched script written to {script_path}", 'yellow'))
            fix_iteration += 1

            # Handle maximum iteration limit
            if fix_iteration > 3:
                print(colored(f"fixpy summary" + 10 * "# " + f"\n{analysis_amalgam}", 'light_magenta'))
                user_input = input(colored(f"3 Unsuccessful iterations, continue? (Y/n).", 'yellow')).lower()
                if user_input != "y" and user_input != "":
                    pyperclip.copy(f"```issue_report\n{analysis_amalgam}\n```\n\n```python\n{fixed_script}\n```")
                    print(colored(f"A summary of the analysis has been copied to the clipboard.", 'green'))
                    exit(1)
                user_input_insights = input(colored("Do you have any additional insight to enlighten the agent before we continue? (Press enter or type your insights): ", 'yellow'))
                fix_iteration = 0
            continue
        else:
            # Handle successful script execution
            print(colored(f"Execution success!\n```output\n{output}\n```", 'green'))
            
            # Prompt user to overwrite original script
            user_input = input(colored("Do you wish to overwrite the original script with the successfully executed version? (Y/n) ", 'yellow')).lower()
            if user_input == "y" or user_input == "":
                with open(script_path, 'w') as file:
                    file.write(fixed_script)
                    print(colored(f"Script overwritten with patched version.", 'green'))
            
            # Prompt user to remove deprecated patched versions
            user_input = input(colored("Do you wish to remove the other deprecated patched versions? (Y/n) ", 'yellow')).lower()
            if user_input == "y" or user_input == "":
                for i in range(fix_iteration):
                    os.remove(script_path.replace(".py", f"_patchV{i}.py"))
                    print(colored(f"Removed deprecated patched version {i}.", 'green'))
            exit(0)


def file_modification_agent(file_path: str, modification_request: str) -> bool:
    """
    An agent function that modifies a specified file based on a given modification request.

    Args:
        file_path (str): The path to the file to be modified.
        modification_request (str): A description of the modifications to be made.

    Returns:
        bool: True if the modification was successful, False otherwise.
    """
    try:
        # Create a Chat object for the code_assistant
        chat = Chat()
        # Use the code_assistant to generate the modified code
        script_content = code_assistant(chat, file_path, modification_request)
        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(script_content)
        print(colored(f"Successfully modified {file_path}", "green"))
        return True
    except Exception as e:
        print(colored(f"An unexpected error occurred: {str(e)}", "red"))
    return False

def project_agent(args: argparse.Namespace, modification_request: str, context_chat: Chat, project_directory_path: str):
    """
    An agent function that works on an entire project, integrating the capabilities of
    the code_agent and code_assistant for comprehensive project management and code improvement.

    Args:
        args (argparse.Namespace): Command-line arguments containing settings and preferences.
        context_chat (Chat): A Chat object to maintain conversation context.
        project_directory_path (str): Path to the project directory.
    """

    def create_project_directory():
        """Creates the project directory if it doesn't exist."""
        if not os.path.exists(project_directory_path):
            os.makedirs(project_directory_path)
            print(colored(f"Created new project directory: {project_directory_path}", "green"))
        else:
            print(colored(f"Using existing project directory: {project_directory_path}", "yellow"))

    def analyze_project_structure() -> Dict[str, List[str]]:
        """Analyzes the project structure and returns a dictionary of files grouped by type."""
        project_structure: Dict[str,List[str]] = {}
        for root, _, files in os.walk(project_directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_type = os.path.splitext(file)[1][1:]  # Get file extension without the dot
                if file_type not in project_structure:
                    project_structure[file_type] = []
                project_structure[file_type].append(os.path.relpath(file_path, project_directory_path))
        return project_structure


    def implement_plan(filepath_instruction_tuplelist: List[Tuple[str, str]]):
        """
        Implements the generated plan
        """
        for file_path, instruction in filepath_instruction_tuplelist:
            python_error_agent(Chat(), file_path)

    def execute_code_agent(file_path: str) -> bool:
        """
        Executes the code_agent on a specific file.
        Returns True if the code_agent successfully fixed the file, False otherwise.
        """
        try:
            # Assuming code_agent is a function that takes similar arguments
            python_error_agent(Chat(), file_path)
            return True
        except Exception as e:
            print(colored(f"Code agent failed for {file_path}: {str(e)}", "yellow"))
            return False

    def execute_code_assistant(file_path: str, content: str) -> str:
        """
        Executes the code_assistant on a specific file content.
        Returns the modified content.
        """
        assistant_chat = Chat()
        assistant_chat.add_message(Role.USER, f"Please improve the following code:\n\n{content}")
        response = LlmRouter.generate_completion(assistant_chat, preferred_models=[args.llm], force_local=args.local)
        return response

    def process_file(operation: str, file_path: str, content: str) -> None:
        """Processes a single file using the appropriate agent."""
        full_path = os.path.join(project_directory_path, file_path)

        if operation == "DELETE":
            if os.path.exists(full_path):
                os.remove(full_path)
                print(colored(f"Deleted file: {file_path}", "yellow"))
            return

        if operation == "CREATE" or operation == "MODIFY":
            if operation == "CREATE" or not os.path.exists(full_path):
                # For new files or non-existent files, use code_assistant directly
                improved_content = execute_code_assistant(file_path, content)
            else:
                # For existing files, try code_agent first, then fall back to code_assistant
                if not execute_code_agent(full_path):
                    with open(full_path, 'r') as file:
                        existing_content = file.read()
                    improved_content = execute_code_assistant(file_path, existing_content)
                else:
                    # If code_agent succeeded, read the improved content from the file
                    with open(full_path, 'r') as file:
                        improved_content = file.read()

            # Write the improved content to the file
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as file:
                file.write(improved_content)

            print(colored(f"{'Created' if operation == 'CREATE' else 'Modified'} file: {file_path}", "green"))

    create_project_directory()
    
    project_structure = analyze_project_structure()
    filepath_instruction_tuplelist, planner_chat = FewShotProvider.few_shot_projectModificationPlanning(project_structure, modification_request, preferred_models=[args.llm], force_local=args.local)

    while True:
        print(colored("Generated Project Plan:", "cyan"))
        print(filepath_instruction_tuplelist)
        user_input = input(colored("Do you want to implement this plan? (yes/no/modify): ", "yellow")).lower()
        if user_input == 'yes':
            implement_plan(filepath_instruction_tuplelist)
        elif user_input == 'modify':
            change_request = input(colored("Enter your change request: ", "yellow"))
            planner_chat.add_message(Role.USER, f"The user entered a change request, please reflect on it and provide the adjusted plan in the same format as before: {change_request}")
            adjusted_plan_str = LlmRouter.generate_completion(planner_chat, preferred_models=[args.llm], force_local=args.local)
            filepath_instruction_tuplelist = FewShotProvider._parse_projectModificationPlanningResponse(adjusted_plan_str)
            continue
        elif user_input == 'no':
            print(colored("Plan implementation skipped.", "yellow"))
        else:
            print(colored("Invalid input. Skipping plan implementation.", "red"))

        user_input = input(colored("Do you want to continue working on the project? (yes/no): ", "yellow")).lower()
        if user_input != 'yes':
            break

        print(colored("Project agent finished working on the project.", "green"))
# # # agents

# # # pipelines
from git import Optional, Repo, GitCommandError

def git_message_generator(project_path: str, user_input: str = "", preferred_models: List[str] = [], force_local: bool = False):
    repo = Repo(project_path)
    skipped_merges = []
    processing_errors = []
    conflicts = []

    def get_current_branch() -> str:
        return repo.active_branch.name

    def get_branch_base_commit(branch_name: str) -> str:
        try:
            return repo.merge_base(branch_name, 'main')[0].hexsha
        except GitCommandError:
            try:
                return repo.merge_base(branch_name, 'master')[0].hexsha
            except GitCommandError:
                return list(repo.iter_commits(branch_name, max_parents=0))[0].hexsha

    def get_all_commit_history(branch_name: str) -> List[str]:
        try:
            base_commit = get_branch_base_commit(branch_name)
            commits = list(repo.iter_commits(f'{base_commit}..HEAD'))
            if not commits:
                print(colored("No commits found on this branch.", "yellow"))
                return []
            return [commit.hexsha for commit in commits]
        except Exception as e:
            print(colored(f"Warning: {str(e)}. Falling back to all commits on the branch.", "yellow"))
            return [commit.hexsha for commit in repo.iter_commits(branch_name)]

    def get_commit_files(commit_hash: str) -> List[str]:
        commit = repo.commit(commit_hash)
        return [item.a_path for item in commit.diff(commit.parents[0])] if commit.parents else []

    def get_file_diff(commit_hash: str, file_path: str) -> str:
        try:
            return repo.git.show('--format=', '--no-color', f'{commit_hash}:{file_path}')
        except GitCommandError as e:
            if "fatal: path" in str(e) and "does not exist in" in str(e):
                print(colored(f"Warning: File {file_path} does not exist in commit {commit_hash[:7]}. It may have been deleted or renamed.", "yellow"))
                return ""  # Return empty string for deleted files
            else:
                raise  # Re-raise the exception if it's not a file-not-found error

    def generate_commit_message(diff: str, topic: str, file_path: str) -> str:
        prompt = f"Based on the following git diff for file '{file_path}' and the general topic '{topic}', generate a concise and informative commit message:\n\n{diff}"
        print(colored(prompt, "yellow"))
        message = LlmRouter.generate_completion(prompt, instruction="This is a commit message generator. The system is given a commit diff by the user to which it responds with an extremely short and concise commit message, describing the overall changes in 2-10 words. The system does not respond with anything else than the exact concise commit message.", preferred_models=preferred_models, force_local=force_local)

        detected_commit_topic = ""
        template = ""
        for word in ["glados", "ra", "amun", "radiosystem", "firmware"]:
            if f"/{word}/" in file_path.lower():
                detected_commit_topic = word
                break
        if detected_commit_topic:
            parts = file_path.split(f"/{detected_commit_topic}/", 1)
            if len(parts) > 1:
                relative_path = parts[1]
                path_parts = relative_path.split('/')
                template = f"{detected_commit_topic}: "
                for part in path_parts[:-1]:
                    template += part[0] + "/"
                template += os.path.splitext(path_parts[-1])[0] + ": "

        return f"{template}{message}"

    def apply_new_commit(file_path: str, file_content: str, new_message: str):
        if not file_path:
            print(colored("Warning: Encountered an empty file path. Skipping this commit.", "yellow"))
            return

        full_file_path = os.path.join(project_path, file_path)
        directory = os.path.dirname(full_file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        try:
            with open(full_file_path, 'w') as f:
                f.write(file_content)
            repo.index.add([file_path])
            repo.index.commit(new_message)
            print(colored(f"Successfully created commit for {file_path}", "green"))
        except Exception as e:
            print(colored(f"Error creating commit for {file_path}: {str(e)}", "red"))
            processing_errors.append(f"Failed to create commit for {file_path}: {str(e)}")

    def cherry_pick_user_commits(source_branch: str, target_branch: str, user_email: str):
        print(colored(f"Starting cherry-pick process from '{source_branch}' to '{target_branch}' for user {user_email}", "cyan"))

        try:
            current_branch = repo.active_branch.name
            repo.git.checkout(source_branch)

            base_commit = get_branch_base_commit(source_branch)
            all_commits = list(repo.iter_commits(f'{base_commit}..HEAD'))
            user_commits = [commit for commit in all_commits if commit.author.email == user_email]

            if not user_commits:
                print(colored(f"No commits found by user {user_email} on branch {source_branch}", "yellow"))
                return

            print(colored(f"Found {len(user_commits)} commits by {user_email}", "green"))

            target_exists = target_branch in repo.refs
            if target_exists:
                print(colored(f"Target branch '{target_branch}' already exists. It will be reset.", "yellow"))
                repo.git.checkout(target_branch)
                repo.git.reset('--hard', base_commit)
            else:
                print(colored(f"Creating new target branch '{target_branch}'", "green"))
                repo.git.checkout('-b', target_branch, base_commit)

            for commit in reversed(user_commits):
                try:
                    if len(commit.parents) > 1:
                        print(colored(f"Skipping merge commit {commit.hexsha[:7]}", "yellow"))
                        skipped_merges.append(commit.hexsha[:7])
                        continue

                    repo.git.cherry_pick(commit.hexsha)
                    print(colored(f"Successfully cherry-picked commit {commit.hexsha[:7]}", "green"))
                except GitCommandError as e:
                    if "CONFLICT" in str(e):
                        print(colored(f"Conflict in cherry-picking commit {commit.hexsha[:7]}", "yellow"))
                        conflicts.append(commit.hexsha[:7])
                        repo.git.cherry_pick('--abort')
                        print(colored("Cherry-pick aborted. Moving to next commit.", "yellow"))
                    else:
                        print(colored(f"Error cherry-picking commit {commit.hexsha[:7]}: {str(e)}", "red"))
                        processing_errors.append(f"Failed to cherry-pick commit {commit.hexsha[:7]}: {str(e)}")
                        repo.git.cherry_pick('--abort')

            print(colored(f"Cherry-pick process completed. Now on branch '{target_branch}'", "cyan"))

        except Exception as e:
            print(colored(f"An error occurred: {str(e)}", "red"))
            processing_errors.append(f"Error during cherry-pick process: {str(e)}")
        finally:
            try:
                repo.git.checkout(current_branch)
                print(colored(f"Returned to original branch '{current_branch}'", "green"))
            except GitCommandError:
                print(colored(f"Could not return to original branch '{current_branch}'", "yellow"))

    def split_and_rewrite_commits(commit_hashes: List[str], topic: str, target_branch: str) -> None:
        original_branch = get_current_branch()
        temp_branch = f"temp_branch_{random.randbytes(4).hex()}"
        
        try:
            # Create and checkout a new temporary branch
            repo.git.checkout('-b', temp_branch)

            for commit_hash in reversed(commit_hashes):
                commit = repo.commit(commit_hash)
                if len(commit.parents) > 1:
                    print(colored(f"Skipping merge commit {commit_hash[:7]}", "yellow"))
                    skipped_merges.append(commit_hash[:7])
                    continue

                files = get_commit_files(commit_hash)
                for file_path in files:
                    if not file_path.strip():
                        continue
                    file_content = get_file_diff(commit_hash, file_path)
                    if not file_content:  # Skip if file was deleted or not found
                        continue
                    diff = repo.git.diff('--no-color', f'{commit_hash}^..{commit_hash}', '--', file_path)
                    new_message = generate_commit_message(diff, topic, file_path)
                    apply_new_commit(file_path, file_content, new_message)
                    print(colored(f"Created commit for {file_path} from original commit {commit_hash[:7]}", "green"))

            # Check if target branch exists
            if target_branch in repo.refs:
                # If it exists, reset it to the temporary branch
                repo.git.checkout(target_branch)
                repo.git.reset('--hard', temp_branch)
            else:
                # If it doesn't exist, rename the temporary branch to the target branch
                repo.git.branch('-m', temp_branch, target_branch)
            
            print(colored(f"Commits have been applied to the '{target_branch}' branch", "green"))

        except Exception as e:
            print(colored(f"Error during commit splitting: {str(e)}", "red"))
            processing_errors.append(f"Error during commit splitting: {str(e)}")
            raise
        finally:
            # Always try to return to the original branch
            repo.git.checkout(original_branch)
            print(colored(f"Returned to original branch '{original_branch}'", "green"))
            
            # Try to delete the temporary branch if it still exists
            try:
                repo.git.branch('-D', temp_branch)
            except GitCommandError:
                print(colored(f"Warning: Could not delete temporary branch {temp_branch}. You may need to delete it manually.", "yellow"))

    # Main pipeline logic
    print(colored("Starting Integrated Git Message Generator Pipeline", "cyan"))

    try:
        if not repo.bare:
            print(colored("Repository detected.", "green"))
        else:
            raise ValueError("Not a git repository.")

        if not user_input.strip():
            raise ValueError("User input (commit topic) cannot be empty.")

        # Ask user for operation choice with default
        choice = input(colored("Choose operation:\n1. Create temporary branch with cherry-picked commits\n2. Rename commits on existing branch\nEnter 1 or 2 [Default: 1]: ", "cyan")) or "1"

        user_email = input(colored("Enter user email for filtering commits: ", "cyan"))

        if choice == "1":
            source_branch = input(colored("Enter source branch name:", "cyan"))
            target_branch = input(colored("Enter target branch name: ", "cyan"))
            cherry_pick_user_commits(source_branch, target_branch, user_email)
            current_branch = target_branch
        elif choice == "2":
            current_branch = get_current_branch()
            target_branch = input(colored(f"Enter target branch name [Default: {current_branch}_rewritten]: ", "cyan")) or f"{current_branch}_rewritten"
        else:
            raise ValueError("Invalid choice. Please enter 1 or 2.")

        print(colored(f"Current branch: {current_branch}", "yellow"))

        # Retrieve commit history
        print(colored("Retrieving commit history...", "yellow"))
        all_commit_hashes = get_all_commit_history(current_branch)
        if not all_commit_hashes:
            print(colored("No commits found on this branch. Nothing to process.", "yellow"))
            return

        print(colored(f"Found {len(all_commit_hashes)} total commits on this branch", "green"))

        # Filter commits by user email
        user_commits = [commit for commit in repo.iter_commits(current_branch) if commit.author.email == user_email]
        user_commit_hashes = [commit.hexsha for commit in user_commits]

        if not user_commit_hashes:
            print(colored(f"No commits found by user {user_email} on this branch. Nothing to process.", "yellow"))
            return

        print(colored(f"Found {len(user_commit_hashes)} commits by {user_email}", "green"))

        # Split commits and rewrite history
        print(colored("Splitting commits and rewriting history...", "yellow"))
        split_and_rewrite_commits(user_commit_hashes, user_input, target_branch)

        print(colored("Git Message Generator Pipeline completed successfully", "cyan"))
    except GitCommandError as e:
        print(colored(f"Git command failed: {e}", "red"))
        processing_errors.append(f"Git command failed: {str(e)}")
    except ValueError as e:
        print(colored(f"Error: {str(e)}", "red"))
        processing_errors.append(f"Value error: {str(e)}")
    except Exception as e:
        print(colored(f"Unexpected error: {str(e)}", "red"))
        processing_errors.append(f"Unexpected error: {str(e)}")
    finally:
        # Summary of skipped merges and errors
        if skipped_merges:
            print(colored("\nSkipped merge commits:", "yellow"))
            for commit in skipped_merges:
                print(f"- {commit}")

        if conflicts:
            print(colored("\nConflicts occurred in the following commits:", "yellow"))
            for commit in conflicts:
                print(f"- {commit}")
            print(colored("These commits were skipped. You may need to apply them manually.", "yellow"))

        if processing_errors:
            print(colored("\nErrors encountered during processing:", "red"))
            for error in processing_errors:
                print(f"- {error}")

        print(colored("\nGit Message Generator Pipeline finished", "cyan"))
        exit(0)
# # # pipelines