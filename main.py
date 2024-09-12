#!/usr/bin/env python3

from pyfiglet import figlet_format
import speech_recognition as sr
from dotenv import load_dotenv
from termcolor import colored
import argparse
import pyaudio
import time
import sys
import re
import warnings

from py_methods.cmd_execution import select_and_execute_commands
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")

from py_methods.assistants import python_error_agent, code_assistant, git_message_generator, majority_response_assistant, presentation_assistant, documents_assistant
from py_methods.tooling import extract_blocks, pdf_or_folder_to_database,recolor, listen_microphone, remove_blocks, text_to_speech, update_cmd_collection, visualize_context
from py_classes.cls_web_scraper import WebTools
from py_classes.cls_llm_router import LlmRouter
from py_classes.cls_few_shot_factory import FewShotProvider
from py_classes.cls_chat import Chat, Role
from agentic.cls_AgenticPythonProcess import AgenticPythonProcess
from py_classes.globals import g



def parse_cli_args() -> argparse.Namespace:
    """Setup and parse CLI arguments, ensuring the script's functionality remains intact."""
    
    parser = argparse.ArgumentParser(
        description="AI CLI-Agent with backend options and more.",
        add_help=False  # Disable automatic help to manually handle unrecognized args
    )
    
    parser.add_argument("-m", "--message", type=str,
                        help="Enter your first message instantly.")
    parser.add_argument("-l", "--local", action="store_true",
                        help="Use the local Ollama backend for language processing.")
    parser.add_argument("-s", "--speak", action="store_true",
                        help="Enable text-to-speech for agent responses.")
    parser.add_argument("-c", action="store_true",
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-w", "--web_search", action="store_true",
                        help="Use web search to enhance responses.")
    parser.add_argument("-a", "--auto", nargs='?', const=10, type=int,
                        help="""Skip user confirmation for command execution.""", metavar="DELAY")
    parser.add_argument("-e", "--edit", nargs='?', const="", type=str, metavar="FILEPATH",
                        help="Edit either the file at the specified path or the contents of the clipboard.")
    parser.add_argument("-p", "--presentation", nargs='?', const="", type=str, metavar="TOPIC",
                        help="Interactively create a presentation.")    
    parser.add_argument("-h", "--help", action="store_true",
                        help="Display this help")
    parser.add_argument("-ma", "--majority", action="store_true",
                        help="Generate a response based on the majority of all local models.")
    parser.add_argument("-fp", "--fixpy", type=str,
                        help="Execute the Python file at the specified path and iterate if an error occurs.")
    parser.add_argument("-doc", "--documents", nargs='?', const="", type=str, metavar="PATH",
                        help="Uses a pdf or folder of pdfs to generate a response. Uses retrieval-based approach.")
    parser.add_argument("-doc2", "--documents2", nargs='?', const="", type=str, metavar="PATH",
                        help="Uses a pdf or folder of pdfs to generate a response. Uses needle in a haystack approach.")
    parser.add_argument("-stt", "--speech_to_text", action="store_true",
                        help="Enable microphone input and text-to-speech. (Wip: please split this up)")
    parser.add_argument("-vis", "--visualize", action="store_true",
                        help="Visualize the chat on a html page.")
    
    parser.add_argument("--exp", action="store_true",
                        help='Experimental agentic hierarchical optimization state machine.')
    parser.add_argument("--llm", nargs='?', const='phi3.5:3.8b', type=str, default="",
                        help='Specify model to use. Supported backends: Groq, Ollama, OpenAI. Examples: ["phi3.5:3.8b", "llama3.1:8b", "claude3.5", "gpt-4o"]')
    parser.add_argument("--preload", action="store_true",
                        help="Preload systems like embeddings and other resources.")
    parser.add_argument("--git_message_generator", nargs='?', const="", type=str, metavar="TOPIC",
                        help="Will rework all messages done by the user on the current branch. Enter the projects theme for better results.")
    
    # Parse known arguments and capture any unrecognized ones
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args or args.help:
        if not args.help:
            print(colored(f"Warning: Unrecognized arguments {' '.join(unknown_args)}.", "red"))
        parser.print_help()
        exit(1)
    
    return args


def main() -> None:
    print("Environment path: ", g.PROJ_ENV_FILE_PATH)
    load_dotenv(g.PROJ_ENV_FILE_PATH)
    
    args = parse_cli_args()
    print(args)
    
    if args.preload:
        print(colored("Preloading resources...", "green"))
        print(colored("Generating atuin-command-history embeddings...", "green"))
        update_cmd_collection()
        print(colored("Generating pdf embeddings for cli-agent directory...", "green"))
        pdf_or_folder_to_database(g.PROJ_DIR_PATH, force_local=False, preferred_models=["phi3.5:3.8b"])
        print(colored("Preloading complete.", "green"))
        exit(0)
    
    if args.exp:
        while True:
            print(colored("Experimental agentic hierarchical optimization state machine.", "green"))
            user_input = input(colored("Enter new user request or press enter to run an iteration of AgenticSelf, type 'exit' to exit: ", 'blue'))
            if user_input == "exit":
                exit(0)
            # agent = AgenticPythonProcess("human")
            agent = AgenticPythonProcess()
            agent.run(user_input)
        
    
    context_chat: Chat|None = None
    next_prompt = ""
    
    if args.c:
        context_chat = Chat.load_from_json()
    
    if args.speech_to_text:
        # setup microphone
        pyaudio_instance = pyaudio.PyAudio()
        default_microphone_info = pyaudio_instance.get_default_input_device_info()
        microphone_device_index = default_microphone_info['index']
        r = sr.Recognizer()
        source = sr.Microphone(device_index=microphone_device_index)
        if context_chat:
            if len(context_chat.messages) > 0:
                # tts last response
                last_response = context_chat.messages[-1][1]
                text_to_speech(last_response)
                print(colored(last_response, 'magenta'))
    
    if args.edit != None: # code edit mode
        if not context_chat:
            context_chat = Chat()
        pre_chosen_option = ""
        if (args.auto):
            pre_chosen_option = "1"
        code_assistant(context_chat, args.edit, pre_chosen_option)    
    
    if args.fixpy != None:
        if not context_chat:
            context_chat = Chat()
        python_error_agent(context_chat, args.fixpy)

    if args.presentation != None:
        if not context_chat:
            context_chat = Chat()
        presentation_assistant(args, context_chat, args.presentation)
    
    if args.git_message_generator:
        if not context_chat:
            context_chat = Chat()
        user_input = ""
        if args.message:
            user_input = args.message
        git_message_generator(args.git_message_generator, user_input)
    
    
    prompt_context_augmentation: str = ""
    temporary_prompt_context_augmentation: str = ""
    
    while True:
        if args.visualize and context_chat:
            visualize_context(context_chat, force_local=args.local, preferred_models=[args.llm])
        
        if args.message:
            next_prompt = args.message
            args.message = None
        elif args.speech_to_text:
            next_prompt = ""
            while not next_prompt:
                with source:
                    r.adjust_for_ambient_noise(source, 2)
                next_prompt = listen_microphone(source, r)[0]
        else:
            next_prompt = input(colored("Enter your request: ", 'blue', attrs=["bold"]))
        
        
        if args.documents != None:
            if args.documents:
                file_or_folder_path = args.documents
            else:
                file_or_folder_path = g.CURRENT_WORKING_DIR_PATH
            
            if (context_chat):
                context_chat.add_message(Role.USER, next_prompt)
                response, context_chat = documents_assistant(context_chat, file_or_folder_path)
            else:
                # Init
                response, context_chat = documents_assistant(next_prompt, file_or_folder_path)
            continue
        
        if args.documents2 != None:
            if args.documents2:
                file_or_folder_path = args.documents2
            else:
                file_or_folder_path = g.CURRENT_WORKING_DIR_PATH
            
            print(colored(f"Using needle in a haystack approach", "green"))
            print(colored(f"Searching for the file or folder at: {file_or_folder_path}", "green"))
            
            if (context_chat):
                context_chat.add_message(Role.USER, next_prompt)
                response, context_chat = documents_assistant(context_chat, file_or_folder_path, True)
            else:
                # Init
                response, context_chat = documents_assistant(next_prompt, file_or_folder_path, True)
            continue
        
        if next_prompt.lower().endswith('--q'):
            print(colored("Exiting...", "red"))
            break
        
        if next_prompt.endswith("--r") and context_chat:
            if len(context_chat.messages) < 2:
                print(colored(f"# cli-agent: No chat history found, cannot regenerate last response.", "red"))
                continue
            context_chat.messages.pop()
            next_prompt = context_chat.messages.pop()[1]
            print(colored(f"# cli-agent: KeyBinding detected: Regenerating last response, type (--h) for info", "green"))
            
        if next_prompt.startswith("--p"):
            next_prompt = next_prompt[:-3]
            print(colored(f"# cli-agent: KeyBinding detected: Starting ScreenCapture, type (--h) for info [NOT IMPLEMENTED]", "green"))
            # screenCapture = ScreenCapture()
            # region_image_base64 = screenCapture.return_captured_region_image()
            # fullscreen_image_base64 = screenCapture.return_fullscreen_image()
            # session.generate_completion("Put words to the contents of the image for a blind user.", "gpt-4o", )
        
        if next_prompt.endswith("--l"):
            next_prompt = next_prompt[:-3]
            args.local = not args.local
            print(colored(f"# cli-agent: KeyBinding detected: Local toggled {args.local}, type (--h) for info", "green"))
            continue
        
        if next_prompt.startswith("--llm"):
            user_input = input(colored("Enter the llm to use (phi3.5:3.8b, claude3.5, gpt-4o), or leave empty for automatic: ", 'blue'))
            if user_input:
                args.llm = user_input
            else:
                args.llm = None
            next_prompt = ""
            print(colored(f"# cli-agent: KeyBinding detected: LLM set to {args.llm}, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--a"):
            next_prompt = next_prompt[:-3]
            args.auto = not args.auto
            print(colored(f"# cli-agent: KeyBinding detected: Autonomous command execution toggled {args.auto}, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--w"):
            args.web_search = True
            print(colored(f"# cli-agent: KeyBinding detected: Websearch enabled, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--maj"):
            args.majority = True    
            print(colored(f"# cli-agent: KeyBinding detected: Running majority response assistant, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--vis"):
            print(colored(f"# cli-agent: KeyBinding detected: Visualize, this will generate a html site and display it, type (--h) for info", "green"))
            visualize_context(context_chat, force_local=args.local, preferred_models=[args.llm])
            continue
        
        if next_prompt.endswith("--m"):
            print(colored("Enter your multiline input. Type '--f' on a new line when finished.", "blue"))
            lines = []
            while True:
                line = input()
                if line == "--f":
                    break
                lines.append(line)
            next_prompt = "\n".join(lines)
        
        
        if next_prompt.endswith("--h"):
            next_prompt = next_prompt[:-3]
            
            print(figlet_format("cli-agent", font="slant"))
            print(colored(f"""# cli-agent: KeyBinding detected: Display help message:
# cli-agent: KeyBindings:
# cli-agent: --r: Regenerates the last response.
# cli-agent: --llm: Set the language model to use. (Examples: "phi3.5:3.8b", "claude3.5", "gpt-4o")
# cli-agent: --p: Add a screenshot to the next prompt.
# cli-agent: --l: Toggles local llm host mode.
# cli-agent: --a: Toggles autonomous command execution.
# cli-agent: --w: Perform a websearch before answering.
# cli-agent: --maj: Run the majority response assistant.
# cli-agent: --m: Multiline input mode.
# cli-agent: --h: Shows this help message.
# cli-agent: Type 'quit' to exit the program.
""", "yellow"))
            continue
        
        
        if args.web_search:
            # recent_context_str = context_chat.get_messages_as_string(-3)
            recent_context_str = next_prompt
            query = FewShotProvider.few_shot_TextToQuery(recent_context_str)
            results = WebTools.search_brave(query, 2)
            temporary_prompt_context_augmentation += f"\n\n```web_search_results\n{''.join(results)}\n```"
            args.web_search = False
        
        
        if "https://github.com/" in next_prompt and next_prompt.count("/") >= 4:
            match = re.search(r"(?P<url>https?://[^\s]+)", next_prompt)
            if match:
                github_repo_url = match.group("url")
                github_readme = WebTools.get_github_readme(github_repo_url)
                prompt_context_augmentation += f"\n\nHere's the readme from the github repo:\n```md\n{github_readme}\n```"

        
        next_prompt +=  temporary_prompt_context_augmentation  # appending relevant content to generate better responses
        next_prompt +=  prompt_context_augmentation  # appending relevant content to generate better responses
        
        if args.majority:
            # Majority voting behavior
            majority_chat, majority_response = majority_response_assistant(next_prompt, args.message)
            if not context_chat:
                context_chat = majority_chat
            else:
                context_chat.add_message(Role.USER, next_prompt)
                context_chat.add_message(Role.ASSISTANT, majority_response)
            continue
        elif context_chat and len(context_chat.messages) > 1:
            # Default continuation
            context_chat.add_message(Role.USER, next_prompt)
            llm_response = LlmRouter.generate_completion(context_chat, [args.llm], force_local=args.local)
            context_chat.add_message(Role.ASSISTANT, llm_response)
        else:
            # Default behavior initalization
            llm_response, context_chat = FewShotProvider.few_shot_CmdAgent(next_prompt, [args.llm], force_local=args.local)
        
        if (args.speak or args.speech_to_text):
            spoken_response = remove_blocks(llm_response, ["md"])
            text_to_speech(spoken_response)
        
        # remove temporary context augmentation from the last user message
        context_chat.messages[-1] = (Role.USER, context_chat.messages[-1][1].replace(temporary_prompt_context_augmentation, ""))

        # save the context_chat to a json file
        context_chat.save_to_json()

        reponse_blocks = extract_blocks(llm_response)
        
        bash_blocks = [block[1] for block in reponse_blocks if block[0] == "bash"]
        if not bash_blocks:
            continue  # or other logic to handle non-command responses
        
        if args.auto is None and not args.speech_to_text:
            # text_to_speech("Do you want me to execute these steps?")
            # if args.speech_to_text:
            #     print(colored("Do you want me to execute these steps? (Yes/no) ", 'yellow'))
            #     user_input = listen_microphone(source, r)[0]
            # else:
            user_input = input(colored("Do you want me to execute these steps? (Y/n) ", 'yellow')).lower()
            if not (user_input == "" or user_input == "y" or "yes" in user_input or "sure" in user_input or "ja" in user_input):
                continue
        else:
            if args.auto is None:
                args.auto = 10
            print(colored(f"Command will be executed in {args.auto} seconds, press Ctrl+C to abort.", 'yellow'))
            try:
                for remaining in range(args.auto, 0, -1):
                    sys.stdout.write("\r" + colored(f"Executing in {remaining} seconds... ", 'yellow'))
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\n")  # Ensure we move to a new line after countdown
            except KeyboardInterrupt:
                print(colored("\nExecution aborted by the user.", 'red'))
                continue  # Skip the execution of commands and start over
        
        temporary_prompt_context_augmentation, execution_summarization = select_and_execute_commands(bash_blocks, args.auto is not None) 
        print(recolor(execution_summarization, "\t#", "successfully", "green"))
        
if __name__ == "__main__":
    main()