#!/usr/bin/env python3

import json
import logging
import os
import random
import select
import string
import time
import traceback
from typing import List, Tuple
import chromadb
from pyfiglet import figlet_format
import speech_recognition as sr
from dotenv import load_dotenv
from termcolor import colored
import argparse
import sys
import socket
import re
import warnings


from py_agents.make_agent import MakeErrorCollectorAgent
from py_classes.cls_html_server import HtmlServer
from py_classes.cls_python_tooling import handle_python_tool
from py_classes.cls_rag_tooling import RagTooling
from py_classes.cls_youtube import YouTube
from py_methods.cmd_execution import select_and_execute_commands
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")

from py_agents.assistants import python_error_agent, code_assistant, git_message_generator, majority_response_assistant, presentation_assistant
from py_methods.tooling import extract_blocks, pdf_or_folder_to_database,recolor, listen_microphone, remove_blocks, take_screenshot, text_to_speech, update_cmd_collection
from py_classes.cls_web_scraper import WebTools
from py_classes.cls_llm_router import LlmRouter
from py_classes.cls_few_shot_provider import FewShotProvider
from py_classes.cls_chat import Chat, Role
from agentic.cls_AgenticPythonProcess import AgenticPythonProcess
from py_classes.globals import g

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
# Disable CUDA warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # This will force CPU usage


def get_local_ip():
    try:
        # Create a socket object and connect to an external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        print(f"Error: {e}")
        return None


def parse_cli_args() -> argparse.Namespace:
    """Setup and parse CLI arguments, ensuring the script's functionality remains intact."""
    
    parser = argparse.ArgumentParser(
        description="AI CLI-Agent with backend options and more.",
        add_help=False  # Disable automatic help to manually handle unrecognized args
    )
    
    parser.add_argument("-u", "--unsafe", nargs='?', const=5, type=int,
                        help="""Allow automatic command execution even when classified unsafe.""", metavar="DELAY")
    parser.add_argument("-s", "--safe", action="store_true",
                        help="""Safe command execution mode, will always prompt for confirmation before executing.""")
    parser.add_argument("-c", action="store_true",
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-d", "--daily", action="store_true",
                        help="Run the daily routine.")
    parser.add_argument("-e", "--edit", nargs='?', const="", type=str, metavar="FILEPATH",
                        help="Edit either the file at the specified path or the contents of the clipboard.")
    parser.add_argument("-h", "--help", action="store_true",
                        help="Display this help")
    parser.add_argument("-i", "--intelligent", action="store_true",
                        help="Use the current most intelligent model for the agent.")
    parser.add_argument("-l", "--local", action="store_true",
                        help="Use the local Ollama backend for processing.")
    parser.add_argument("-o", "--online", action="store_true",
                        help="Use online backends for processing.")
    parser.add_argument("-m", "--message", type=str,
                        help="Enter your first message instantly.")
    parser.add_argument("-p", "--presentation", nargs='?', const="", type=str, metavar="TOPIC",
                        help="Interactively create a presentation.")    
    parser.add_argument("-q", "--quick", action="store_true",
                        help="Disable reasoning for the agent.")
    parser.add_argument("-r", "--regenerate", action="store_true",
                        help="Regenerate the last response.")
    parser.add_argument("-v", "--voice", action="store_true",
                        help="Enable microphone input and text-to-speech output.")
    parser.add_argument("-spe", "--speak", action="store_true",
                        help="Text-to-speech output.")
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
    parser.add_argument("-vis", "--visualize", action="store_true",
                        help="Visualize the chat on a html page.")
    parser.add_argument("--fmake", nargs=2, type=str, metavar=('MAKE_PATH', 'CHANGED_FILE_PATH'),
                        help="Runs the make command in the specified path and agentically fixes any errors. "
                            "MAKE_PATH: The path where to execute the make command. "
                            "CHANGED_FILE_PATH: The path of the file that was changed before the error occurred.")
    parser.add_argument("--exp", action="store_true",
                        help='Experimental agentic hierarchical optimization state machine.')
    parser.add_argument("--llm", nargs='?', type=str, default="",
                        help='Specify model to use. Supported backends: Groq, Ollama, OpenAI. \nDefault: "llama3.2:3b", Examples: ["llama3.2:3b", "llama3.1:8b", "claude3.5", "gpt-4o"]')
    parser.add_argument("--preload", action="store_true",
                        help="Preload systems like embeddings and other resources.")
    parser.add_argument("--git_message_generator", nargs='?', const="", type=str, metavar="TOPIC",
                        help="Will rework all messages done by the user on the current branch. Enter the projects theme for better results.")
    parser.add_argument("--yt_slides", type=str, metavar="URL", help="Convert a youtube url to a slideshow")
    
    # Parse known arguments and capture any unrecognized ones
    args, unknown_args = parser.parse_known_args()
    
    if args.local and not args.llm:
        args.llm = "llama3.2:3b"

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
    
    if os.getenv("DEFAULT_FORCE_LOCAL") == get_local_ip() and not args.online:
        args.local = True
        if not args.llm:
            args.llm = "mistral-nemo:12b"
    
    
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
    
    
    user_input: str = ""
    context_chat: Chat
    
    if args.c:
        context_chat = Chat.load_from_json()
        print(colored("# # # Recent executed actions # # #", "green"))
        print(colored("\n".join(g.get_recent_actions()), "yellow"))
    else:
        context_chat = Chat()
    
    if (args.voice or args.speak) and context_chat and len(context_chat.messages) > 0:
        # tts last response (when continuing)
        last_response = context_chat.messages[-1][1]
        text_to_speech(last_response)
        print(colored(last_response, 'magenta'))

    if args.edit != None: # code edit mode
        pre_chosen_option = ""
        if (args.unsafe):
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
    prompt_context_augmentation: str = ""
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
                    context_chat.messages[-1] = (Role.USER, context_chat.messages[-1][1].replace(prompt_context_augmentation, "").strip())
        
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
                    user_input = context_chat.messages.pop()[1]
                    print(colored(f"# cli-agent: Regenerating last response.", "green"))
                    print(colored(user_input, "blue"))
        
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
        
        # # get controlled from the html server
        # if server:
        #     waiting_counter = 900
        #     while not server.remote_user_input:
        #         time.sleep(0.01)
        #         waiting_counter += 1
        #         if waiting_counter % 1000 == 0:
        #             print(colored(f"Waiting for user input from the html server at: {server.host_route}", "green"))
        #             waiting_counter = 0
        #     user_input = server.remote_user_input
        #     server.remote_user_input = ""
        # cli args message
        elif args.message:
            user_input = args.message
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
            user_input = listen_microphone()[0]
            continue_workflow_count = 0
        # default user input
        else:
            user_input = input(colored("Enter your request: ", 'blue', attrs=["bold"]))
            continue_workflow_count = 0
        
        # USER INPUT HANDLING - BEGIN
        if user_input.endswith('--q'):
            print(colored("Exiting...", "red"))
            break
        
        if user_input.endswith("--r") and context_chat:
            if len(context_chat.messages) < 2:
                print(colored(f"# cli-agent: No chat history found, cannot regenerate last response.", "red"))
                continue
            context_chat.messages.pop()
            user_input = context_chat.messages.pop()[1]
            print(colored(f"# cli-agent: KeyBinding detected: Regenerating last response, type (--h) for info", "green"))
            
        
        if user_input.endswith("--l"):
            user_input = user_input[:-3]
            args.local = not args.local
            print(colored(f"# cli-agent: KeyBinding detected: Local toggled {args.local}, type (--h) for info", "green"))
            continue
        
        if user_input.endswith("--u"):
            user_input = user_input[:-3]
            args.unsafe = not args.unsafe
            print(colored(f"# cli-agent: KeyBinding detected: Allow unsafe command execution toggled {args.unsafe}, type (--h) for info", "green"))
            continue
        
        if user_input.endswith("--s"):
            user_input = user_input[:-3]
            args.unsafe = not args.unsafe
            print(colored(f"# cli-agent: KeyBinding detected: Safe command execution toggled {args.safe}, type (--h) for info", "green"))
            continue
        
        if user_input.endswith("--img"):
            user_input = user_input[:-3]
            print(colored(f"# cli-agent: KeyBinding detected: Starting ScreenCapture, type (--h) for info", "green"))
            args.image = True
            continue
        
        if user_input.startswith("--llm"):
            user_input = input(colored("Enter the llm to use (phi3.5:3.8b, claude3.5, gpt-4o), or leave empty for automatic: ", 'blue'))
            if user_input:
                args.llm = user_input
            else:
                args.llm = None
            user_input = ""
            print(colored(f"# cli-agent: KeyBinding detected: LLM set to {args.llm}, type (--h) for info", "green"))
            continue
        
        if user_input.endswith("--maj"):
            args.majority = True    
            print(colored(f"# cli-agent: KeyBinding detected: Running majority response assistant, type (--h) for info", "green"))
            continue
        
        if user_input.endswith("--rea"):
            use_reasoning = not use_reasoning
            print(colored(f"# cli-agent: KeyBinding detected: Reasoning toggled to {use_reasoning}, type (--h) for info", "green"))
            continue
        
        if user_input.endswith("--rag"):
            args.rag = not args.rag
            print(colored(f"# cli-agent: KeyBinding detected: RAG toggled to {args.rag}, type (--h) for info", "green"))
            continue
        
        if user_input.endswith("--i"):
            previous_model_key = args.llm
            args.llm = g.CURRENT_MOST_INTELLIGENT_MODEL_KEY
            if previous_model_key:
                args.llm = previous_model_key
                previous_model_key = None
                print(colored(f"# cli-agent: KeyBinding detected: Disabled the current most intelligent model, now using: {args.llm}, type (--h) for info", "green"))
            else:    
                print(colored(f"# cli-agent: KeyBinding detected: Enabled the current most intelligent model: {args.llm}, type (--h) for info", "green"))
        
        if user_input.endswith("--vis"):
            print(colored(f"# cli-agent: KeyBinding detected: Visualize, this will generate a html site and display it, type (--h) for info", "green"))
            if not server:
                server = HtmlServer(g.PROJ_VSCODE_DIR_PATH)
            server.visualize_context(context_chat, force_local=args.local, preferred_models=[args.llm])
            continue
        
        if "--debug" in user_input:
            print(colored(f"# cli-agent: KeyBinding detected: Debug information:", "green"))
            context_chat.print_chat()
        
        if user_input.endswith("--m"):
            print(colored("Enter your multiline input. Type '--f' on a new line when finished.", "blue"))
            lines = []
            while True:
                line = input()
                if line == "--f":
                    break
                lines.append(line)
            user_input = "\n".join(lines)
        
        if "--h" in user_input:
            user_input = user_input[:-3]
            print(figlet_format("cli-agent", font="slant"))
            print(colored(f"""# cli-agent: KeyBinding detected: Display help message:
# cli-agent: KeyBindings:
# cli-agent: --h: Shows this help message.
# cli-agent: --r: Regenerates the last response.
# cli-agent: --l: Toggles local llm host mode.
# cli-agent: --u: Toggles unsafe command execution.
# cli-agent: --s: Toggles safe command execution.
# cli-agent: --m: Multiline input mode.
# cli-agent: --i: Toggles to the current most intelligent model.
# cli-agent: --debug: Display debug information.
# cli-agent: --img: Take a screenshot.
# cli-agent: --rag: Toggles retrieval augmented generation (RAG).
# cli-agent: --rea: Toggles reasoning.
# cli-agent: --vis: Visualize the chat on a html page.
# cli-agent: --maj: Run the majority response assistant.
# cli-agent: --llm: Set the language model to use. (Examples: "phi3.5:3.8b", "claude3.5", "gpt-4o")
# cli-agent: Type 'quit' to exit the program.
""", "yellow"))
            continue
        # USER INPUT HANDLING - END
        
        # AGENT INITIALIZEATION - BEGIN
        if not context_chat:
            # llm_response, context_chat = FewShotProvider.few_shot_TerminalAssistant(user_input, [args.llm], force_local=args.local, use_reasoning=use_reasoning, silent_reasoning=False)
            context_chat = Chat("You are a delightfully helpful ai assistant.")
        # AGENT INITIALIZEATION - END
        
        # AGENT TOOL USE - BEGIN
        context_chat.add_message(Role.USER, user_input)
        # 3. "majority": For solving very complex problems that require consulting high-value and high-cost expert language models.
        # {{"tool": "majority", "reasoning": "This problem is highly complex and would benefit from consulting multiple expert models for a more comprehensive solution."}}
        # AGENT TOOL USE - END


        def make_tools_chat(context_chat: Chat) -> Chat:
            tool_use_context_chat = context_chat.deep_copy()
            if tool_use_context_chat.messages[-1][0] == Role.USER:
                tool_use_context_chat.messages[-1] = (Role.USER, f"# # # USER INPUT # # #\n{tool_use_context_chat.messages[-1][1]}\n")
            tool_use_context_chat.add_message(Role.USER, f"""# # # TOOL USE INSTRUCTIONS # # #\nAnalyze the context to determine the most appropriate tool to use. Provide your reasoning with your tool choice in the below format. Respond with a single JSON object containing your reasoning, your selected tool, and any additional required parameters as shown below. Available tools are:
1. "web_search": When requiring up-to-date information or precise data a web search should be used. When used, include a "web_query" string for the search.
2. "bash": Bash commands can be used to execute commands on the users operating system. Use this for tasks like updating, file handling and information gathering.
3. "reply": Pick this tool to reply to the user. 'Reply' should be chosen when the user can be provided with a helpful response that does not include any code or if the given task cannot be achieved safely without further guidance.
4. "python": Use Python scripts for calculations, coding, and other tasks that benefit from a symbolic solver approach. Include a "title" string for the script name.

Example responses:
{{"reasoning": "The query requires up-to-date information on AI, which is best obtained through a web search.", "tool": "web_search", "web_query": "latest news on artificial intelligence"}}
{{"reasoning": "The websearch tool has already been executed successfully, and the results contain the required information. I can now inform the user.", "tool": "reply"}}
{{"reasoning": "The websearch tool has been executed successfully, but the information required is not included. I should try another websearch with a more specific query.", "tool": "web_search", "web_query": ""classical electrodynamics" fundamental "Maxwell equations" text:pdf before:2024 -quantum inurl:edu"}}

{{"reasoning": "The user requested me to update the Ubuntu system I am running on.", "tool": "bash", "command": "sudo apt-get update && sudo apt-get upgrade -y"}}
{{"reasoning": "The task requires real-time hardware information that can only be obtained through direct system queries.", "tool": "bash", "command": "lscpu"}}
{{"reasoning": "To fix the issue in the python script, I need to read from the main.py file.", "tool": "bash", "command": "cat ~/main.py"}}

{{"reasoning": "The bash tool has been executed succesfully and has returned the current time. I can now reply to the user.", "tool": "reply"}}
{{"reasoning": "The user is confused about my recent actions, I should reason through my recent actions and explain them to the user.", "tool": "reply"}}
{{"reasoning": "The python tool has been executed succesfully, it seems the user has closed the pygame window. I should ask the user if the game was fun.", "tool": "reply"}}

{{"reasoning": "The user wants to see an interactable mandelbrot set, this is too complex for a simple bash command, instead I will implement a python script to generate the mandelbrot set. The user will be able to interact with the set and zoom in and out.", "tool": "python", "title": "mandelbrot.py"}}
{{"reasoning": "The user is asking for the result of the calculation (3*(4+3). I will use python to ensure the correct result is returned.", "tool": "python", "title": "calculation.py"}}
{{"reasoning": "The user wants to reverse the mandelbrot generation. I will extend our previously implemented script to include buttons to control the flow of time.", "tool": "python", "title": "mandelbrot.py"}}

# Reminder: Your task is to work towards a helpful response, respond when you're ready. This is the original user input:\n{user_input}""")
            return tool_use_context_chat

        # AGENTIC IN-TURN LOOP - BEGIN
        action_counter = 0  # Initialize counter for consecutive actions
        MAX_ACTIONS = 10    # Maximum number of consecutive actions before forcing a reply

        while True:
            try:
                # Check if we've hit the maximum number of consecutive actions
                if action_counter >= MAX_ACTIONS:
                    # Ask use if to continue or not
                    user_input = input(colored(f"Warning: Agent has performed {MAX_ACTIONS} consecutive actions without replying. Do you want to continue? (Y/n) ", "yellow")).lower()
                    if (user_input == "" or user_input == "y" or "yes" in user_input or "sure" in user_input or "ja" in user_input):
                        MAX_ACTIONS += 3  # Increase the maximum number of consecutive actions
                    else:
                        context_chat.add_message(Role.USER, f"You have performed {MAX_ACTIONS} actions without replying and are being interrupted by the user. Please summarize your progress a respond intelligently to the user.")
                        break
                
                tool_use_context_chat = make_tools_chat(context_chat)
                tool_use_response = LlmRouter.generate_completion(tool_use_context_chat, [args.llm], force_local=args.local, use_reasoning=use_reasoning, silent_reasoning=True)
                # tool_use_context_chat.print_chat()
                
                # Use the framework's extract_blocks function
                tool_use_reponse_blocks = extract_blocks(tool_use_response)
                tool_call_json_list = [block[1] for block in tool_use_reponse_blocks if block[0] == r"first{}"]
                
                # If no JSON blocks found in extract_blocks result, try direct JSON parsing
                if not tool_call_json_list:
                    try:
                        # Clean the response of any leading/trailing whitespace and quotes
                        cleaned_response = tool_use_response.strip().strip('"\'')
                        # Try to parse it as JSON
                        json.loads(cleaned_response)  # Test if it's valid JSON
                        tool_call_json_list = [cleaned_response]
                    except json.JSONDecodeError:
                        print(colored("TOOL USE ERROR: No valid JSON found in response - DEBUG BEGIN", "red"))
                        tool_use_context_chat.print_chat()
                        print(colored("TOOL USE ERROR: No valid JSON found in response - DEBUG END", "red"))
                        break
                
                if not tool_call_json_list:
                    break
                
                selected_tools = [json.loads(tool_call_json) for tool_call_json in tool_call_json_list]
                print(colored(f"Selected tools: {[tool.get('tool', '') for tool in selected_tools]}", "green"))
                
                should_continue = False
                
                for tool in selected_tools:
                    
                    selected_tool = tool.get('tool', '').strip()
                    bash_command = tool.get('command', '')
                    
                    reasoning = tool.get('reasoning', 'No specific reasoning provided.')
                    if selected_tool and selected_tool != "reply":
                        context_chat.add_message(Role.ASSISTANT, reasoning)
                    
                    print(colored(f"Using tool: {selected_tool}", "green"))
                    print(colored(f"Reasoning: {reasoning}", "cyan"))

                    if selected_tool == 'bash':
                        print(colored(f"Executing bash command: {bash_command}", "yellow"))
                        cmd_context_augmentation, execution_summarization = run_bash_cmds([bash_command], args)
                        print(execution_summarization)
                        context_chat.add_message(Role.USER, cmd_context_augmentation)
                        context_chat.add_message(Role.ASSISTANT, f"The bash tool has been executed for the command: '{bash_command}'.")
                        should_continue = True
                        action_counter += 1  # Increment action counter
                    elif selected_tool == 'web_search':
                        web_query = tool.get('web_query', '')
                        results = WebTools().search_brave(web_query, 3, args.llm if args.llm else "")
                        web_search_context_chat = context_chat.deep_copy()
                        results_joined = '\n'.join(results)
                        web_search_context_chat.add_message(Role.USER, f"Please summarize the relevant information from these results:\n```web_search_results\n{results_joined}\n```")
                        web_search_summary = LlmRouter.generate_completion(web_search_context_chat, [args.llm], force_local=args.local, use_reasoning=use_reasoning, silent_reasoning=False)
                        context_chat.add_message(Role.USER, f"You just performed a websearch for '{web_query}', this is the returned result:\n```txt\n{web_search_summary}\n```")
                        context_chat.add_message(Role.ASSISTANT, "The websearch tool has been executed " + ("successfully" if any(results) else "unsuccessfully") + f" for the query: '{web_query}'.")
                        should_continue = True
                        action_counter += 1  # Increment action counter
                    elif selected_tool == 'python': # Implement and execute python script
                        title = tool.get('title', None)
                        if not title:
                            title = FewShotProvider.few_shot_TextToQuery(reasoning, force_local=args.local)
                        print(colored(f"Implementing and executing python script: {title}", "yellow"))
                        handle_python_tool(tool, context_chat, args)
                        should_continue = "ModuleNotFoundError" in context_chat.messages[-2][1]
                    elif selected_tool == 'reply':
                        should_continue = False
                        break
                    else:
                        should_continue = False
                
                if not should_continue:
                    break
                    
            except json.JSONDecodeError as e:
                print(colored(f"Error parsing tool selection response: {str(e)}", "red"))
                print(colored("Raw response causing error:", "red"))
                print(colored(tool_use_response, "red"))
                break
            except Exception as e:
                print(colored(f"An error occurred during tool selection: {str(e)}", "red"))
                traceback.print_exc()
                break
        # AGENT TOOL USE - END
        
        # REPLY - BEGIN
        # Final summarization
        if context_chat.messages[-1][0] == Role.ASSISTANT:
            if "python" in selected_tools:
                context_chat.add_message(Role.USER, "Please summarize a response to the user. Do not include any python code.")
            else:
                context_chat.add_message(Role.USER, "Please summarize a response to the user.")

        print(colored("# # # RESPONSE # # #", "green"))
        llm_response = LlmRouter.generate_completion(context_chat, [args.llm], force_local=args.local, use_reasoning=use_reasoning)

        context_chat.add_message(Role.ASSISTANT, llm_response)
        
        if (args.voice or args.speak):
            spoken_response = remove_blocks(llm_response, ["md"])
            text_to_speech(spoken_response)

        # save context once before executing
        if context_chat:
            context_chat.save_to_json()


def run_bash_cmds(bash_blocks: List[str], args) -> Tuple[str, str]:
    command_guard_prompt = f"The following command must follow these guidelines:\n1. After execution it must exit fully automatically.\n2. It must not modify the operating system in major ways, although it is allowed to install trusted apt packages and updated software.\nRespond only with 'Safe' or 'Unsafe'\n\nCommand: ",
    safe_bash_blocks: List[str] = []
    for bash_block in bash_blocks:
        print(colored(bash_block, 'magenta'))
        execute_actions_guard_response = LlmRouter.generate_completion(f"{command_guard_prompt}{bash_blocks}", ["llama-guard"], force_local=args.local, silent_reason="command guard")
        
        execute_actions_automatically: bool = not "unsafe" in execute_actions_guard_response.lower()
        if "S8" in execute_actions_guard_response or "S7" in execute_actions_guard_response : # Ignore: S7 - Privacy, S8 - Intellectual Property
            execute_actions_automatically = True
            
        if execute_actions_automatically:
            safe_bash_blocks.append(bash_block)
        else:
            pass
    
    if len(safe_bash_blocks) > 0 and len(safe_bash_blocks) != len(bash_blocks):
        bash_blocks = safe_bash_blocks
        execute_actions_automatically = True
    
    if (args.unsafe is None and not execute_actions_automatically) or args.safe:
        if args.voice or args.speak:
            confirmation_response = "Do you want me to execute these steps? (Yes/no)"
            print(colored(confirmation_response, 'yellow'))
            text_to_speech(confirmation_response)
            user_input = listen_microphone(10)[0]
        else:
            user_input = input(colored("Do you want me to execute these steps? (Y/n) ", 'yellow')).lower()
        if not (user_input == "" or user_input == "y" or "yes" in user_input or "sure" in user_input or "ja" in user_input):
            bash_blocks = safe_bash_blocks
    else:
        if not execute_actions_automatically:
            print(colored(f"Command will be executed in {args.unsafe} seconds, press Ctrl+C to abort.", 'yellow'))
            try:
                for remaining in range(args.unsafe, 0, -1):
                    sys.stdout.write("\r" + colored(f"Executing in {remaining} seconds... ", 'yellow'))
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\n")  # Ensure we move to a new line after countdown
            except KeyboardInterrupt:
                print(colored("\nExecution aborted by the user.", 'red'))
    return select_and_execute_commands(bash_blocks, args.unsafe is not None or execute_actions_automatically) 

if __name__ == "__main__":
    main()
