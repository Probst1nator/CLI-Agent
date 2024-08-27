import argparse
from collections import defaultdict
import hashlib
import os
import re
import sys
import time
from typing import List, Tuple

import chromadb
import pyperclip
from termcolor import colored

from classes.ai_providers.cls_ollama_interface import OllamaClient
from classes.cls_chat import Chat, Role
from classes.cls_few_shot_factory import FewShotProvider
from classes.cls_llm_router import AIStrengths, LlmRouter
from classes.cls_pptx_presentation import PptxPresentation
from classes.cls_web_scraper import search_brave
from tooling import create_rag_prompt, extract_pdf_content, list_files_recursive, pdf_or_folder_to_database, run_python_script, split_string_into_chunks
from globals import g


# # # helper methods

# Note: The following section is reserved for helper methods.
# Helper methods are typically small, reusable functions that perform specific tasks
# to support the main functionality of the script. They can include utility functions,
# data processing functions, or any other auxiliary operations needed by the main program.

# Examples of potential helper methods (not implemented here) could include:
# - Functions for data validation
# - Text processing utilities
# - File handling operations
# - Logging and error handling routines

# When implementing helper methods, consider:
# 1. Keeping them focused on a single task
# 2. Using descriptive names that indicate their purpose
# 3. Adding appropriate docstrings to explain their functionality and parameters
# 4. Ensuring they are easily testable and maintainable

# The actual implementation of helper methods would follow this comment block,
# each with its own documentation and explanatory comments.


def extract_single_snippet(response: str, allow_no_end: bool = False) -> str:
    """
    Extracts a single code snippet from a given response string.

    This function searches for code blocks enclosed in triple backticks (```) 
    and returns the content within the first found block. If no end marker is found
    and allow_no_end is True, it returns the content from the start marker to the end.

    Args:
        response (str): The input string containing potential code snippets.
        allow_no_end (bool, optional): If True, allows extraction even if no end 
                                        marker is found. Defaults to False.

    Returns:
        str: The extracted code snippet, or an empty string if no valid snippet is found.

    Example:
        >>> response = "Here's some code:\n```python\nprint('Hello')\n```\nEnd."
        >>> extract_single_snippet(response)
        "print('Hello')"
    """
    # Find the index of the first triple backtick
    start_index = response.find("```")
    if start_index != -1:
        # Find the end of the line containing the start marker
        start_line_end = response.find("\n", start_index)
        # Find the index of the last triple backtick
        end_index = response.rfind("```")
        
        if end_index != -1 and end_index > start_index:
            # Find the start of the line containing the end marker
            end_line_start = response.rfind("\n", start_index, end_index)
            if start_line_end != -1 and end_line_start != -1:
                # Extract and return the content between start and end markers
                return response[start_line_end + 1:end_line_start]
        elif allow_no_end:
            # If no end marker is found and allow_no_end is True,
            # return everything after the start marker
            return response[start_line_end + 1:]
    
    # Return an empty string if no valid snippet is found
    return ""
# # # helper methods


# # # assistants
def code_assistant(context_chat: Chat, file_path: str = "", pre_chosen_option: str = "", preferred_model_keys: List[str] = [], force_local: bool = False) -> str:
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
    if file_path:
        # Read the content of the file specified by the --edit argument
        with open(file_path, 'r') as file:
            file_ending = os.path.splitext(file_path)[1]
            file_content = file.read()
            file_snippet = f"```{file_ending}\n{file_content}\n```"
            snippets_to_process = [file_snippet]

    # Prompt user to add clipboard content
    try:
        print(colored("Add clipboard? Press (ctrl+c) to add.","yellow"))
        for remaining in range(3, 0, -1):
            sys.stdout.write("\r" + colored(f"Ignoring clipboard in {remaining}s... ", 'yellow'))
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\n")  # Ensure we move to a new line after countdown
    except KeyboardInterrupt:
        clipboard_content = pyperclip.paste()
        snippets_to_process[0] += f"```userClipboard\n{clipboard_content}\n```"
    
    while True:
        next_prompt = ""
        chunking_delimiter = ""
        if len(file_content)/2 > 8192 * 0.8: # 0.8 because works better
            if "py" in file_ending:
                chunking_delimiter = "def "
            elif "ts" in file_ending:
                chunking_delimiter = "function "
            else:
                chunking_delimiter = FewShotProvider.few_shot_objectFromTemplate([{"py": "def "}, {"typescript": "function "}], target_description="A delimiter to split the code into smaller chunks.", preferred_model_keys=preferred_model_keys, force_local=force_local)
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
            abstract_code_overview = LlmRouter.generate_completion("Please explain the below code step by step, provide a short abstract overview of its stages.\n\n" + snippets_to_process[0],  preferred_model_keys=["llama-3.1-405b-reasoning", LlmRouter.last_used_model, "llama-3.1-70b-versatile"], strength=AIStrengths.STRONG, force_free=True)
            if len(abstract_code_overview)/4 >= 2048:
                abstract_code_overview = LlmRouter.generate_completion(f"Summarize this code analysis, retaining the most important features and minimal details:\n{abstract_code_overview}",  preferred_model_keys=[LlmRouter.last_used_model, "llama-3.1-70b-versatile"], strength=AIStrengths.STRONG, force_free=True)
            next_prompt = "Augment the below code snippet with thorough docstrings and step-by-step explanatory comments. Retain all original comments, modifying them slightly only if essential. It is crucial that you do not modify the code's logic or structure; present it in full."
            next_prompt += f"\nTo help you get started, here's an handwritten overview of the code: \n{abstract_code_overview}"
        else:
            # Second turn, do not process the same snippets again
            if result:
                snippets_to_process = []
            
            if not pre_chosen_option:
                # Manual mode: Present options to the user
                print(colored("Please choose an option:", 'cyan', attrs=["bold"]))
                print(colored("1. Add docstrings", 'yellow'))
                print(colored("2. Refactor", 'yellow'))
                print(colored("3. Explain", 'yellow'))
                print(colored("4. Perform web search", 'yellow'))
                print(colored("5. Use a custom delimiter for splitting the code into chunks", 'yellow'))
                print(colored("Write the prompt yourself", 'yellow') + " " + colored("(Use --m for multiline input)", 'grey'))
                user_input = input(colored("Enter your choice: ", 'blue'))
            user_input = pre_chosen_option
            
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
                web_search_result = search_brave(query, 2)
                context_chat.add_message(Role.USER, f"I found something on the web, please relate it to the code we're working on:\n```web_search\n{web_search_result}```")
                response = LlmRouter.generate_completion(context_chat, preferred_model_keys=[LlmRouter.last_used_model, "llama-3.1-405b-reasoning", "claude-3-5-sonnet", "gpt-4o"], strength=AIStrengths.STRONG)
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
        
        # workaround this is used for vscode hotkey functionality and will also work for pipeline choices but if will break/(interrupt for user input) if the pipeline wants to do "1"
        if pre_chosen_option == "1":
            pre_chosen_option = ""
        
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

        # Print number of processed snippets
        print(colored(f"Processing {len(snippets_to_process)} snippets.", "yellow"))
        
        # # Rephrase the prompt using few-shot learning
        # next_prompt = FewShotProvider.few_shot_rephrase(next_prompt, [LlmRouter.last_used_model, "llama-3.1-70b-versatile", "llama-3.1-405b-reasoning", "gpt-4o", "claude-3-5-sonnet"])

        generated_snippets: List[str] = []
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
            
            # Generate a response using the LlmRouter
            response = LlmRouter.generate_completion(context_chat, preferred_model_keys=[LlmRouter.last_used_model, "llama-3.1-405b-reasoning", "claude-3-5-sonnet", "gpt-4o"], strength=AIStrengths.STRONG)
            extracted_snippet = extract_single_snippet(response, allow_no_end=True)
            # Check if the response is empty because markers weren't included, this can be intended behavior if no code is asked for
            if (extracted_snippet):
                generated_snippets.append(extracted_snippet)
            
            # Add the assistant's response to the chat context
            context_chat.add_message(Role.ASSISTANT, response)
        
        # Check if all snippets were regenerated
        if len(snippets_to_process) == len(generated_snippets):
            # Replace the original snippets with the generated snippets
            result = file_content
            for source_snippet, generated_snippet in zip(snippets_to_process, generated_snippets):
                # remove first and last line from source_snippet
                clean_snippet = "\n".join(source_snippet.split("\n")[1:-1])
                result = result.replace(clean_snippet, generated_snippet)
                
            pyperclip.copy(result)
            print(colored("Result copied to clipboard.", 'green'))
        else:
            print(colored("INFO: Snippets were not reimplemented by the assistant.", 'yellow'))
            if (len(snippets_to_process) > 1):
                context_chat.add_message(Role.USER, "Please summarize your reasoning step by step and provide a short discussion.")
                response = LlmRouter.generate_completion(context_chat, preferred_model_keys=[LlmRouter.last_used_model, "llama-3.1-405b-reasoning", "claude-3-5-sonnet", "gpt-4o"], strength=AIStrengths.STRONG)


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
    rephrased_user_input = FewShotProvider.few_shot_rephrase(user_input, preferred_model_keys=[args.llm], force_local=args.local)

    # Generate a prompt for decomposing the topic into subtopics
    decomposition_prompt = FewShotProvider.few_shot_rephrase("Please decompose the following into 3-6 subtopics and provide step by step explanations + a very short discussion:", preferred_model_keys=[args.llm], force_local=args.local)

    # Generate detailed presentation content based on the decomposed topic
    presentation_details = LlmRouter.generate_completion(f"{decomposition_prompt}: '{rephrased_user_input}'", strength=AIStrengths.STRONG, use_cache=False, preferred_model_keys=[args.llm], force_local=args.local)
    
    # Convert the generated content into a presentation format
    chat, response = FewShotProvider.few_shot_textToPresentation(presentation_details, preferred_model_keys=[args.llm], force_local=args.local)

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
                response = LlmRouter.generate_completion(chat, strength=AIStrengths.STRONG, use_cache=False, preferred_model_keys=[args.llm], force_local=args.local)
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
            add_details_prompt = FewShotProvider.few_shot_rephrase(f"Please think step by step to add relevant/ missing details to the following topic: {presentation_details}", preferred_model_keys=[args.llm])
            suggested_details = LlmRouter.generate_completion(f"{add_details_prompt} {presentation_details}", strength=AIStrengths.STRONG, preferred_model_keys=[args.llm], force_local=args.local)
            next_prompt = f"Please add the following details to the presentation: \n{suggested_details}"
        elif user_input == "2":
            # Regenerate the entire presentation
            next_prompt = "I am unhappy with your suggested presentation, please try again."
        else:
            # Use custom user input as the next prompt
            next_prompt = user_input
            
        # Rephrase the next prompt and generate a new response
        next_prompt = FewShotProvider.few_shot_rephrase(next_prompt, preferred_model_keys=[args.llm], force_local=args.local)
        chat.add_message(Role.USER, next_prompt)
        response = LlmRouter.generate_completion(chat, strength=AIStrengths.STRONG, preferred_model_keys=[args.llm], force_local=args.local)

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
    instruction = FewShotProvider.few_shot_rephrase(f"This is a chat between a user and his private artificial intelligence assistant. The assistant uses the documents to answer the users questions factually, detailed and reliably. The assistant indicates if the answer cannot be found in the documents.", preferred_model_keys=[args.llm], force_local=args.local, silent=True)
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
                text_content, image_content = extract_pdf_content(file_path)
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
            response = LlmRouter.generate_completion(chat, preferred_model_keys=[args.llm], force_local=args.local)
            chat.add_message(Role.ASSISTANT, response)
            user_input = input(colored("Enter your response, (Type '--f' to start a new search): ", "blue")).lower()
            if ("--f" in user_input):
                user_input = input(colored("Enter your search request, previous context is still available: ", "blue")).lower()
                break
            chat.add_message(Role.USER, user_input)

def majority_response_assistant(args: argparse.Namespace, context_chat: Chat, user_input: str = "", preferred_model_keys=["phi3.5:3.8b"]):
    """
    An assistant function that leverages multiple AI models to provide comprehensive and consensus-based answers to user queries.
    This assistant, called "majority_vote_assistant", operates by consulting various models and synthesizing their outputs.

    Args:
        args (argparse.Namespace): Command-line arguments containing settings for model selection, voting mechanisms, and other parameters.
        context_chat (Chat): The conversation context (maintained for consistency but not directly used).
        user_input (str, optional): Initial user input for the query. Defaults to an empty string.
    """
    force_local = True
    models = LlmRouter.get_models(force_local=force_local)

    while True:
        # Collect user input if not provided
        if not user_input:
            user_input = input(colored("Enter your request: ", "blue"))

        # Distribute query to all available models and gather responses
        model_responses_str = ""
        model_responses = []
        for i, model in enumerate(models):
            try:
                response = LlmRouter.generate_completion_raw(user_input, model=model)
                if not response:
                    continue
                model_responses_str += f"\n\n'''expert_opinion_{i}\n{response}\n'''"
                model_responses.append(response)
            except Exception as e:
                print(colored(f"Error getting response from {model.model_key}: {str(e)}", "red"))

        print(colored(f"Received responses from {len(model_responses)} models. Summarizing...", "yellow"))

        chat = Chat("You are a data scientist tasked with performing a comprehensive meta analysis of responses from various experts on a given topic. Please summarize the responses, highlighting the key points and areas of agreement or disagreement. Be thorough and work step by step to grasp and reveal each relevant nuance of the conversation.")
        chat.add_message(Role.USER, f"Question: {user_input}\n\n{model_responses_str}")
        response = LlmRouter.generate_completion(chat=chat, preferred_model_keys=preferred_model_keys, force_local=force_local)
        chat.add_message(Role.ASSISTANT, response)
        chat.add_message(Role.USER, f"Please provide a final, concise and accurate answer to the question: {user_input}")
        response = LlmRouter.generate_completion(chat=chat, preferred_model_keys=preferred_model_keys, force_local=force_local)
        chat.add_message(Role.ASSISTANT, response)
        while True:
            user_input = input(colored("Enter your response: ", "blue"))
            chat.add_message(Role.USER, user_input)
            response = LlmRouter.generate_completion(chat=chat, preferred_model_keys=preferred_model_keys, force_local=force_local)
            chat.add_message(Role.ASSISTANT, response)


def documents_assistant(question_context: Chat|str, pdf_or_folder_path: str = "") -> Tuple[str, Chat]:
    client = chromadb.PersistentClient(g.PROJ_VSCODE_DIR_PATH)
    collection = client.get_or_create_collection(name=hashlib.md5(pdf_or_folder_path.encode()).hexdigest())
    # This is going to take a while and should be done seperately, before runtime
    pdf_or_folder_to_database(pdf_or_folder_path, collection)
    
    if isinstance(question_context, str):
        chat = Chat("This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, reliable and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.")
        chat.add_message(Role.USER, question_context)
    else:
        chat = question_context

    user_query = chat.messages[-1][1]
    # Generate embedding for the user's input query
    user_input_embedding = OllamaClient.generate_embedding(user_query)
    
    # Perform a similarity search based on the user's query
    results = collection.query(
        query_embeddings=user_input_embedding,
        n_results=10
    )
    
    prompt = create_rag_prompt(results, user_query)
    
    chat.messages[-1] = (Role.USER, prompt)
    
    response = LlmRouter.generate_completion(chat, temperature=0.6, force_local=True)
    chat.add_message(Role.ASSISTANT, response)
    
    return response, chat
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
            script_fix = LlmRouter.generate_completion(context_chat, preferred_model_keys=["llama-3.1-70b-versatile", "gpt-4o", "claude-3-5-sonnet"])
            context_chat.add_message(Role.ASSISTANT, script_fix)
            fixed_script = extract_single_snippet(script_fix)
            
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
        response = LlmRouter.generate_completion(assistant_chat, preferred_model_keys=[args.llm], force_local=args.local)
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
    filepath_instruction_tuplelist, planner_chat = FewShotProvider.few_shot_projectModificationPlanning(project_structure, modification_request, preferred_model_keys=[args.llm], force_local=args.local)

    while True:
        print(colored("Generated Project Plan:", "cyan"))
        print(filepath_instruction_tuplelist)
        user_input = input(colored("Do you want to implement this plan? (yes/no/modify): ", "yellow")).lower()
        if user_input == 'yes':
            implement_plan(filepath_instruction_tuplelist)
        elif user_input == 'modify':
            change_request = input(colored("Enter your change request: ", "yellow"))
            planner_chat.add_message(Role.USER, f"The user entered a change request, please reflect on it and provide the adjusted plan in the same format as before: {change_request}")
            adjusted_plan_str = LlmRouter.generate_completion(planner_chat, preferred_model_keys=[args.llm], force_local=args.local)
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
import subprocess
import os
from typing import List, Tuple, Dict
from termcolor import colored

def git_message_generator(args: argparse.Namespace, context_chat: Chat, user_input: str = ""):
    def run_git_command(command: List[str], error_message: str) -> str:
        """Runs a git command and handles potential errors."""
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"{error_message}\nCommand: {' '.join(command)}\nError: {result.stderr}")
        return result.stdout.strip()

    def get_current_branch() -> str:
        """Retrieves the name of the current branch."""
        return run_git_command(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            "Failed to get current branch name."
        )

    def get_all_commit_history() -> List[str]:
        """Retrieves all commit hashes of the current branch."""
        result = run_git_command(
            ['git', 'log', '--format=%H'],
            "Failed to retrieve branch commit history."
        )
        commits = result.split('\n') if result else []
        if not commits or commits[0] == '':
            print(colored("No commits found on this branch.", "yellow"))
            return []
        return commits

    def get_commit_files(commit_hash: str) -> List[str]:
        """Retrieves the list of files changed in a specific commit."""
        return run_git_command(
            ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash],
            f"Failed to retrieve file list for commit {commit_hash}."
        ).split('\n')

    def get_file_diff(commit_hash: str, file_path: str) -> str:
        """Retrieves the diff for a specific file in a commit."""
        return run_git_command(
            ['git', 'show', '--format=', '--no-color', f'{commit_hash}:{file_path}'],
            f"Failed to retrieve diff for file {file_path} in commit {commit_hash}."
        )

    def generate_commit_message(diff: str, topic: str, file_path: str) -> str:
        """
        Generates a new commit message based on the diff, topic, and file path.
        The commit message includes a template based on the file path structure.
        """
        prompt = f"Based on the following git diff for file '{file_path}' and the general topic '{topic}', generate a concise and informative commit message:\n\n{diff}"
        print(colored(prompt, "yellow"))
        message = LlmRouter.generate_completion(prompt, preferred_model_keys=[args.llm], force_local=args.local)
        
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
                
                # Generate the template
                template = f"{detected_commit_topic}: "
                for part in path_parts[:-1]:  # Exclude the last part (file name)
                    template += part[0] + "/"
                
                # Add the full file name without extension
                template += os.path.splitext(path_parts[-1])[0] + ": "
        
        return f"{template}{message}"

    def apply_new_commit(file_path: str, file_content: str, new_message: str):
        """Creates a new commit for a single file with the given message."""
        if not file_path:
            print(colored("Warning: Encountered an empty file path. Skipping this commit.", "yellow"))
            return
        
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w') as f:
            f.write(file_content)
        run_git_command(['git', 'add', file_path], f"Failed to stage file {file_path}")
        escaped_message = new_message.replace('"', '\\"').replace('$', '\\$')
        run_git_command(['git', 'commit', '-m', escaped_message], f"Failed to create commit for {file_path}")

    def split_and_rewrite_commits(commit_hashes: List[str], topic: str) -> None:
        """Splits commits with multiple files and rewrites the commit history."""
        temp_branch = f"temp_branch_{os.urandom(4).hex()}"
        run_git_command(['git', 'checkout', '-b', temp_branch], f"Failed to create temporary branch {temp_branch}")
        
        try:
            for commit_hash in reversed(commit_hashes):
                files = get_commit_files(commit_hash)
                for file_path in files:
                    if not file_path.strip():
                        continue
                    file_content = get_file_diff(commit_hash, file_path)
                    diff = run_git_command(['git', 'diff', '--no-color', f'{commit_hash}^..{commit_hash}', '--', file_path],
                                            f"Failed to get diff for {file_path}")
                    new_message = generate_commit_message(diff, topic, file_path)
                    apply_new_commit(file_path, file_content, new_message)
                    print(colored(f"Created commit for {file_path} from original commit {commit_hash[:7]}", "green"))
            
            current_branch = get_current_branch()
            run_git_command(['git', 'checkout', current_branch], f"Failed to switch back to {current_branch}")
            run_git_command(['git', 'reset', '--hard', temp_branch], f"Failed to reset {current_branch} to {temp_branch}")
            run_git_command(['git', 'branch', '-D', temp_branch], f"Failed to delete temporary branch {temp_branch}")
        except Exception as e:
            print(colored(f"Error during commit splitting: {str(e)}", "red"))
            current_branch = get_current_branch()
            run_git_command(['git', 'checkout', current_branch], f"Failed to switch back to {current_branch}")
            run_git_command(['git', 'branch', '-D', temp_branch], f"Failed to delete temporary branch {temp_branch}")
            raise

    # Main pipeline logic
    print(colored("Starting Git Message Generator Pipeline with Commit Splitting", "cyan"))

    try:
        # Check if we're in a git repository
        run_git_command(['git', 'rev-parse', '--is-inside-work-tree'], "Not a git repository.")

        if not user_input.strip():
            raise ValueError("User input (commit topic) cannot be empty.")

        # Step 1: Identify the current branch
        current_branch = get_current_branch()
        if current_branch == 'HEAD':
            raise ValueError("You are in 'detached HEAD' state. Please checkout a branch.")

        print(colored(f"Current branch: {current_branch}", "yellow"))

        # Step 2: Retrieve all commit history
        print(colored("Retrieving all commit history...", "yellow"))
        commit_hashes = get_all_commit_history()
        if not commit_hashes:
            print(colored("No commits found on this branch. Nothing to process.", "yellow"))
            return
        print(colored(f"Found {len(commit_hashes)} commits on this branch", "green"))

        # Step 3: Split commits and rewrite history
        print(colored("Splitting commits and rewriting history...", "yellow"))
        split_and_rewrite_commits(commit_hashes, user_input)

        print(colored("Git Message Generator Pipeline with Commit Splitting completed successfully", "cyan"))
    except subprocess.CalledProcessError as e:
        print(colored(f"Git command failed: {e}", "red"))
        print(colored(f"Error output: {e.stderr}", "red"))
    except ValueError as e:
        print(colored(f"Error: {str(e)}", "red"))
    except Exception as e:
        print(colored(f"Unexpected error: {str(e)}", "red"))
    finally:
        print(colored("Git Message Generator Pipeline finished", "cyan"))
    exit(0)
# # # pipelines