#!/usr/bin/env python3

import json
import os
import select
import shutil
from typing import List
import chromadb
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
from pocketsphinx import LiveSpeech, get_model_path


from py_agents.make_agent import MakeErrorCollectorAgent
from py_classes.cls_html_server import HtmlServer
from py_classes.cls_rag_tooling import RagTooling
from py_classes.cls_youtube import YouTube
from py_methods.cmd_execution import select_and_execute_commands
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")

from py_agents.assistants import python_error_agent, code_assistant, git_message_generator, majority_response_assistant, presentation_assistant, documents_assistant
from py_methods.tooling import extract_blocks, pdf_or_folder_to_database,recolor, listen_microphone, remove_blocks, take_screenshot, text_to_speech, update_cmd_collection
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
    
    parser.add_argument("-d", "--daily", action="store_true",
                        help="Run the daily routine.")
    parser.add_argument("-m", "--message", type=str,
                        help="Enter your first message instantly.")
    parser.add_argument("-l", "--local", action="store_true",
                        help="Use the local Ollama backend for language processing.")
    parser.add_argument("-v", "--voice", action="store_true",
                        help="Enable microphone input and speech-to-text output.")
    parser.add_argument("-c", action="store_true",
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-r", "--regenerate", action="store_true",
                        help="Regenerate the last response.")
    parser.add_argument("-w", "--web_search", action="store_true",
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-i", "--intelligent", action="store_true",
                        help="Use the current most intelligent model for the agent.")
    parser.add_argument("-a", "--auto", nargs='?', const=5, type=int,
                        help="""Skip user confirmation for command execution.""", metavar="DELAY")
    parser.add_argument("-e", "--edit", nargs='?', const="", type=str, metavar="FILEPATH",
                        help="Edit either the file at the specified path or the contents of the clipboard.")
    parser.add_argument("-p", "--presentation", nargs='?', const="", type=str, metavar="TOPIC",
                        help="Interactively create a presentation.")    
    parser.add_argument("-h", "--help", action="store_true",
                        help="Display this help")
    parser.add_argument("-q", "--quick", action="store_true",
                        help="Disable reasoning for the agent.")
    parser.add_argument("-img", "--image", action="store_true",
                        help="Take a screenshot and generate a response based on the contents of the image.")
    parser.add_argument("-maj", "--majority", action="store_true",
                        help="Generate a response based on the majority of all local models.")
    parser.add_argument("-rag", action="store_true",
                        help="Will use rag as implemented in the given behavior.")
    parser.add_argument("-fpy", "--fixpy", type=str,
                        help="Execute the Python file at the specified path and iterate if an error occurs.")
    parser.add_argument("-doc", "--documents", nargs='?', const="", type=str, metavar="PATH",
                        help="Uses a pdf or folder of pdfs to generate a response. Uses retrieval-based approach.")
    parser.add_argument("-doc2", "--documents2", nargs='?', const="", type=str, metavar="PATH",
                        help="Uses a pdf or folder of pdfs to generate a response. Uses needle in a haystack approach.")
    parser.add_argument("-vis", "--visualize", action="store_true",
                        help="Visualize the chat on a html page.")
    parser.add_argument("--fmake", nargs=2, type=str, metavar=('MAKE_PATH', 'CHANGED_FILE_PATH'),
                        help="Runs the make command in the specified path and agentically fixes any errors. "
                            "MAKE_PATH: The path where to execute the make command. "
                            "CHANGED_FILE_PATH: The path of the file that was changed before the error occurred.")
    parser.add_argument("--exp", action="store_true",
                        help='Experimental agentic hierarchical optimization state machine.')
    parser.add_argument("--llm", nargs='?', const='phi3.5:3.8b', type=str, default="",
                        help='Specify model to use. Supported backends: Groq, Ollama, OpenAI. Examples: ["phi3.5:3.8b", "llama3.1:8b", "claude3.5", "gpt-4o"]')
    parser.add_argument("--preload", action="store_true",
                        help="Preload systems like embeddings and other resources.")
    parser.add_argument("--git_message_generator", nargs='?', const="", type=str, metavar="TOPIC",
                        help="Will rework all messages done by the user on the current branch. Enter the projects theme for better results.")
    parser.add_argument("--yt_slides", type=str, metavar="URL", help="Convert a youtube url to a slideshow")
    
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
        pdf_or_folder_to_database(g.PROJ_DIR_PATH, force_local=args.local)
        print(colored("Preloading complete.", "green"))
        exit(0)
    
    if args.fmake:
        retry_count: int = 0
        init_sandbox = False
        context_file_paths: List[str] = []
        if "[" in args.fmake[1] and "]" in args.fmake[1]:
            context_file_paths = json.loads(args.fmake[1])
        else:
            context_file_paths = [args.fmake[1]]
            
        makeAgent = MakeErrorCollectorAgent()
        while True: # TODO: automate checking if number of errors are reduced instead of increased 
            retry_count += 1
            success = makeAgent.execute(build_dir_path=args.fmake[0], context_file_paths=context_file_paths, init_sandbox=init_sandbox, force_local=args.local)
            init_sandbox = False
            if retry_count == 1 and not success:
                init_sandbox = True
                print(colored("First attempt failed, trying again with sandbox initialization...", "yellow"))
                continue
            # if success:
            #     exit(0)    
                
            do_continue = input(colored("Do you want to iterate once more? (Y/n): ", "yellow"))
            if "n" in do_continue.lower():
                exit(0)
    
    if args.daily:
        client = chromadb.PersistentClient(g.PROJ_VSCODE_DIR_PATH)
        collection = client.get_or_create_collection(name="long-term-memories")
        args.voice = True
    else:
        client = chromadb.PersistentClient(g.PROJ_VSCODE_DIR_PATH)
        collection = client.get_or_create_collection(name="commands")
    
    if args.quick:
        use_reasoning = False
        print(colored("Quick mode enabled: reasoning disabled.", "green"))
    else:
        use_reasoning = True
    
    if args.intelligent:
        args.llm = g.CURRENT_MOST_INTELLIGENT_MODEL_KEY
        print(colored(f"Enabling the current most intelligent model: {args.llm}", "green"))
    
    if args.exp:
        while True:
            print(colored("Experimental agentic hierarchical optimization state machine.", "green"))
            user_input = input(colored("Enter new user request or press enter to run an iteration of AgenticSelf, type 'exit' to exit: ", 'blue'))
            if user_input == "exit":
                exit(0)
            # agent = AgenticPythonProcess("human")
            agent = AgenticPythonProcess()
            agent.run(user_input)
            do_continue = input(colored("Do you want to continue? (Y/n): ", "yellow"))
            if "n" in do_continue.lower():
                exit(0)
            
    if args.yt_slides:
        from py_classes.ai_providers.cls_stable_diffusion import StableDiffusion
        from py_classes.ai_providers.cls_pyaihost_interface import PyAiHost
        print(colored("Converting youtube video to slideshow...", "green"))
        video_path = YouTube.download_video(args.yt_slides, g.PROJ_VSCODE_DIR_PATH)
        mp3_path = YouTube.convert_video_to_mp3(video_path)
        pyaihost = PyAiHost()
        pyaihost.initialize("small")
        songtext, _ = pyaihost.transcribe_audio(mp3_path)
        split_songtext = songtext.split("\n")
        stable_diffusion = StableDiffusion()
        print(colored(f"Downloaded models: {stable_diffusion.list_downloaded_models()}", "yellow"))
        stable_diffusion.load_model("CompVis/stable-diffusion-v1-4")
        for text in split_songtext:
            image_gen_prompt = FewShotProvider.few_shot_ToImageGenPrompt(text, "llama3.1:8b")
            stable_diffusion.generate_image(image_gen_prompt, g.PROJ_VSCODE_DIR_PATH, num_inference_steps=20, guidance_scale=2.0)
        exit(0)
    
    
    next_prompt: str = ""
    context_chat: Chat
    
    if args.c:
        context_chat = Chat.load_from_json()
        print(colored("# # # Recent executed actions # # #", "green"))
        print(colored("\n".join(g.get_recent_actions()), "yellow"))
    else:
        context_chat = Chat()
    
    if args.voice and context_chat and len(context_chat.messages) > 0:
        # tts last response (when continuing)
        last_response = context_chat.messages[-1][1]
        text_to_speech(last_response)
        print(colored(last_response, 'magenta'))

    if args.edit != None: # code edit mode
        pre_chosen_option = ""
        if (args.auto):
            pre_chosen_option = "1"
        code_assistant(context_chat, args.edit, pre_chosen_option)    
    
    if args.fixpy != None:
        python_error_agent(context_chat, args.fixpy)

    if args.presentation != None:
        presentation_assistant(args, context_chat, args.presentation)
    
    if args.git_message_generator:
        user_input = ""
        if args.message:
            user_input = args.message
        git_message_generator(args.git_message_generator, user_input)
    
    prompt_context_augmentation: str = ""
    temporary_prompt_context_augmentation: str = ""
    previous_model_key: str | None = None
    server: HtmlServer | None = None
    
    # remove empty context chat for few_shot_inits
    if len(context_chat.messages) == 0:
        context_chat = None
    
    # Main loop
    while True:
        # remove temporary context augmentation from the last user message
        if context_chat:
            if len(context_chat.messages) > 0:
                if context_chat.messages[-1][0] == Role.USER:
                    context_chat.messages[-1] = (Role.USER, context_chat.messages[-1][1].replace(temporary_prompt_context_augmentation, "").strip())
        
        # save the context_chat to a json file
        if context_chat:
            context_chat.save_to_json()
    
        if args.visualize and context_chat:
            if not server:
                server = HtmlServer(g.PROJ_VSCODE_DIR_PATH)
            server.visualize_context(context_chat, force_local=args.local, preferred_models=[args.llm])
        
        # cli args regenerate last message
        if args.regenerate:
            args.regenerate = False
            if context_chat and len(context_chat.messages) > 1:
                if context_chat.messages[-1][0] == Role.USER:
                    context_chat.messages.pop()
                    next_prompt = context_chat.messages.pop()[1]
                    print(colored(f"# cli-agent: Regenerating last response.", "green"))
                    print(colored(next_prompt, "blue"))
        
        # screen capture
        if args.image:
            args.image = False
            window_title = "Firefox" # Default window title
            print(colored(f"Capturing screenshots of '", "green") + colored(f"{window_title}", "yellow") + colored("' press any key to enter another title.", 'green'))
            try:
                for remaining in range(3, 0, -1):
                    sys.stdout.write("\r" + colored(f"Proceeding in {remaining} seconds... ", 'yellow'))
                    sys.stdout.flush()
                    time.sleep(1)
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        raise KeyboardInterrupt
                sys.stdout.write("\n")
            except KeyboardInterrupt:
                window_title = input(colored("\nEnter the title of the window you want to capture: ", 'blue'))
                
            base64_images = take_screenshot(window_title)
            if not base64_images:
                print(colored(f"# cli-agent: No images were returned.", "red"))
                continue
            for i, base64_image in enumerate(base64_images):
                print(colored(f"# cli-agent: Converting Image ({i}/{len(base64_images)}) into words...", "green"))
                image_response_str = LlmRouter.generate_completion("Put words to the contents of the image for a blind user.", base64_images=[base64_image], force_local=args.local, use_reasoning=use_reasoning, silent_reasoning=False)
                prompt_context_augmentation += f'\n\n```vision_{i}\n{image_response_str}\n```'
        
        # get controlled from the html server
        if server:
            waiting_counter = 900
            while not server.remote_user_input:
                time.sleep(0.01)
                waiting_counter += 1
                if waiting_counter % 1000 == 0:
                    print(colored(f"Waiting for user input from the html server at: {server.host_route}", "green"))
                    waiting_counter = 0
            next_prompt = server.remote_user_input
            server.remote_user_input = ""
        # cli args message
        elif args.message:
            next_prompt = args.message
            args.message = None
        # use microphone
        elif args.voice:
            import pyaudio
            import numpy as np
            from openwakeword import Model
            if args.daily:
                def listen_for_keyword(keywords=['hey_lucy', 'ok_lucy']):
                    # Initialize OpenWakeWord model
                    model = Model()

                    # Initialize PyAudio
                    pa = pyaudio.PyAudio()
                    audio_stream = pa.open(
                        rate=16000,
                        channels=1,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=1024
                    )

                    print(f"Listening for keywords: {', '.join(keywords)}...")
                    try:
                        while True:
                            audio_data = audio_stream.read(1024, exception_on_overflow=False)
                            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                            # Get prediction from openWakeWord model
                            prediction = model.predict(audio_array)

                            # Check if any wake word was detected
                            for wake_word in keywords:
                                print(prediction)
                                if wake_word in prediction and prediction[wake_word] > 0.5:  # You can adjust this threshold
                                    print(f"Detected wake word: {wake_word}")
                                    return True

                    except KeyboardInterrupt:
                        print('Stopping ...')
                    finally:
                        audio_stream.stop_stream()
                        audio_stream.close()
                        pa.terminate()

                    return False

                while True:
                    if not listen_for_keyword():
                        break
            next_prompt = listen_microphone()[0]
            
        # default user input
        else:
            next_prompt = input(colored("Enter your request: ", 'blue', attrs=["bold"]))
        
        if next_prompt.endswith('--q'):
            print(colored("Exiting...", "red"))
            break
        
        if next_prompt.endswith("--r") and context_chat:
            if len(context_chat.messages) < 2:
                print(colored(f"# cli-agent: No chat history found, cannot regenerate last response.", "red"))
                continue
            context_chat.messages.pop()
            next_prompt = context_chat.messages.pop()[1]
            print(colored(f"# cli-agent: KeyBinding detected: Regenerating last response, type (--h) for info", "green"))
            
        
        if next_prompt.endswith("--l"):
            next_prompt = next_prompt[:-3]
            args.local = not args.local
            print(colored(f"# cli-agent: KeyBinding detected: Local toggled {args.local}, type (--h) for info", "green"))
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
        
        if next_prompt.endswith("--img"):
            next_prompt = next_prompt[:-3]
            print(colored(f"# cli-agent: KeyBinding detected: Starting ScreenCapture, type (--h) for info", "green"))
            args.image = True
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
        
        if next_prompt.endswith("--maj"):
            args.majority = True    
            print(colored(f"# cli-agent: KeyBinding detected: Running majority response assistant, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--rea"):
            use_reasoning = not use_reasoning
            print(colored(f"# cli-agent: KeyBinding detected: Reasoning toggled to {use_reasoning}, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--rag"):
            args.rag = not args.rag
            print(colored(f"# cli-agent: KeyBinding detected: RAG toggled to {args.rag}, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--i"):
            previous_model_key = args.llm
            args.llm = g.CURRENT_MOST_INTELLIGENT_MODEL_KEY
            if previous_model_key:
                args.llm = previous_model_key
                previous_model_key = None
                print(colored(f"# cli-agent: KeyBinding detected: Disabled the current most intelligent model, now using: {args.llm}, type (--h) for info", "green"))
            else:    
                print(colored(f"# cli-agent: KeyBinding detected: Enabled the current most intelligent model: {args.llm}, type (--h) for info", "green"))
        
        if next_prompt.endswith("--vis"):
            print(colored(f"# cli-agent: KeyBinding detected: Visualize, this will generate a html site and display it, type (--h) for info", "green"))
            if not server:
                server = HtmlServer(g.PROJ_VSCODE_DIR_PATH)
            server.visualize_context(context_chat, force_local=args.local, preferred_models=[args.llm])
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
# cli-agent: --h: Shows this help message.
# cli-agent: --r: Regenerates the last response.
# cli-agent: --l: Toggles local llm host mode.
# cli-agent: --a: Toggles autonomous command execution.
# cli-agent: --w: Perform a websearch before answering.
# cli-agent: --m: Multiline input mode.
# cli-agent: --i: Toggles to the current most intelligent model.
# cli-agent: --img: Take a screenshot.
# cli-agent: --rag: Toggles retrieval augmented generation (RAG).
# cli-agent: --rea: Toggles reasoning.
# cli-agent: --vis: Visualize the chat on a html page.
# cli-agent: --maj: Run the majority response assistant.
# cli-agent: --llm: Set the language model to use. (Examples: "phi3.5:3.8b", "claude3.5", "gpt-4o")
# cli-agent: Type 'quit' to exit the program.
""", "yellow"))
            continue
        
        
        if args.web_search:
            if context_chat and len(context_chat.messages) > 0:
                recent_context_str = context_chat.get_messages_as_string(-3)
            else:
                recent_context_str = next_prompt
            
            # recent_context_str = next_prompt
            query = FewShotProvider.few_shot_TextToQuery(recent_context_str)
            results = WebTools().search_brave(query, 2)
            temporary_prompt_context_augmentation += f"\n\n```web_search_results\n{''.join(results)}\n```"
            args.web_search = False
        
        
        if "https://github.com/" in next_prompt and next_prompt.count("/") >= 4:
            match = re.search(r"(?P<url>https?://[^\s]+)", next_prompt)
            if match:
                github_repo_url = match.group("url")
                github_readme = WebTools.get_github_readme(github_repo_url)
                prompt_context_augmentation += f"\n\nHere's the readme from the github repo:\n```md\n{github_readme}\n```"

        
        next_prompt = temporary_prompt_context_augmentation + "\n" + prompt_context_augmentation + "\n\n" + next_prompt # add any context augmentation to the prompt
        next_prompt = next_prompt.strip()
        
        # Document assistant behavior
        if args.documents2 != None or args.documents != None:
            if args.documents != None:
                needle_in_haystack: bool = False
                if args.documents:
                    file_or_folder_path = args.documents
                else:
                    file_or_folder_path = g.CURRENT_WORKING_DIR_PATH
            if args.documents2 != None:
                needle_in_haystack: bool = True
                if args.documents2:
                    file_or_folder_path = args.documents2
                else:
                    file_or_folder_path = g.CURRENT_WORKING_DIR_PATH
                print(colored(f"Using needle in a haystack approach", "green"))
            print(colored(f"Document assistant running for: {file_or_folder_path}", "green"))
            
            if (context_chat):
                # Continuation
                context_chat.add_message(Role.USER, next_prompt)
                response, context_chat = documents_assistant(context_chat, file_or_folder_path, needle_in_haystack)
            else:
                # Initialization
                response, context_chat = documents_assistant(next_prompt, file_or_folder_path, needle_in_haystack)
            continue
        
        # Majority voting behavior
        if args.majority:
            if not context_chat:
                context_chat = Chat()
            context_chat.add_message(Role.USER, next_prompt)
            context_chat, majority_response = majority_response_assistant(context_chat, args.local, allow_costly_models=args.intelligent)
            continue
        
        # Retrieval Augmented Generation (RAG) behavior
        if args.rag:
            next_prompt = RagTooling.retrieve_augment(next_prompt, collection, 3)
        
        # Daily behavior
        if args.daily:
            if context_chat and len(context_chat.messages) > 1:
                # Continuation
                context_chat.add_message(Role.USER, next_prompt)
                llm_response = LlmRouter.generate_completion(context_chat, [args.llm], force_local=args.local, use_reasoning=use_reasoning, silent_reasoning=False)
                context_chat.add_message(Role.ASSISTANT, llm_response)
            else:
                # Initialization
                llm_response, context_chat = FewShotProvider.few_shot_DailyAssistant(next_prompt, [args.llm], force_local=args.local, use_reasoning=use_reasoning, silent_reasoning=False)
        else:
            # Default behavior (terminal assistant)
            if context_chat and len(context_chat.messages) > 1:
                # Continuation
                context_chat.add_message(Role.USER, next_prompt)
                llm_response = LlmRouter.generate_completion(context_chat, [args.llm], force_local=args.local, use_reasoning=use_reasoning, silent_reasoning=False)
                context_chat.add_message(Role.ASSISTANT, llm_response)
            else:
                # Initalization
                llm_response, context_chat = FewShotProvider.few_shot_TerminalAssistant(next_prompt, [args.llm], force_local=args.local, use_reasoning=use_reasoning, silent_reasoning=False)
        
        if (args.voice):
            spoken_response = remove_blocks(llm_response, ["md"])
            text_to_speech(spoken_response)
        
        reponse_blocks = extract_blocks(llm_response)
        
        bash_blocks = [block[1] for block in reponse_blocks if block[0] == "bash"]
        if not bash_blocks:
            continue  # or other logic to handle non-command responses
        
        if args.auto is None:
            # text_to_speech("Do you want me to execute these steps?")
            if args.voice:
                confirmation_response = "Do you want me to execute these steps? (Yes/no)"
                print(colored(confirmation_response, 'yellow'))
                text_to_speech(confirmation_response)
                user_input = listen_microphone(10)[0]
            else:
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
        
        # save context once before executing
        if context_chat:
            context_chat.save_to_json()
        
        temporary_prompt_context_augmentation, execution_summarization = select_and_execute_commands(bash_blocks, args.auto is not None) 
        print(recolor(execution_summarization, "\t#", "successfully", "green"))
        
if __name__ == "__main__":
    main()