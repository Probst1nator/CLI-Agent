"""
Input handling utilities for CLI-Agent.

This module contains input handling functions extracted from main.py to provide
reusable input functionality including multiline input, screenshot capture,
and permission management.
"""

import asyncio
import argparse
import base64
import datetime
import logging
import math
import os
import re
import subprocess
import tempfile
import threading
import time
from typing import Dict, Any, List, Optional, Tuple

import pyperclip
from termcolor import colored
from prompt_toolkit import HTML

# Import dependencies from main.py context
try:
    from main import (
        get_listen_microphone, text_to_speech, llm_selection, 
        construct_engineered_prompt, save_current_llm_config_as_default,
        print_startup_summary, handle_screenshot_capture as original_handle_screenshot_capture
    )
except ImportError:
    # Fallback imports for standalone usage
    pass

# Import command system from refactored module
from agent.core.command_system import COMMAND_MAP, get_prompt_session, _COMMAND_DEFINITIONS

from core.globals import g
from core.chat import Chat


def handle_multiline_input() -> str:
    """Handles multiline input from the user."""
    print(colored("Enter your multiline input. Type '--f' on a new line when finished.", "blue"))
    lines = []
    while True:
        try:
            line = input()
            if line == "--f":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)


async def get_user_input_with_bindings(
    args: argparse.Namespace,
    context_chat: Optional[Chat],
    prompt: str = colored(" Enter your request: ", 'blue', attrs=["bold"]),
    input_override: str = None,
    force_input: bool = False,
    input_event: Optional[asyncio.Event] = None,
    input_lock: Optional[threading.Lock] = None,
    shared_input: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[str]]:
    """
    Interactively captures user input, including voice for confirmation prompts.

    This function serves as the primary Read-Eval-Print Loop (REPL). It
    returns the user's text prompt. If voice mode is active and this is a
    confirmation prompt, it will use TTS and STT.
    """
    
    # --- First Call Logic ---
    if get_user_input_with_bindings.is_first_call:
        get_user_input_with_bindings.is_first_call = False

    # --- VOICE-ENABLED CONFIRMATION PROMPTS ---
    # Check if this is an internal confirmation prompt, not the main REPL prompt.
    is_internal_prompt = "Enter your request" not in prompt

    if args.voice and is_internal_prompt:
        listen_microphone = get_listen_microphone()
        
        # --- FIX: Improved clean_prompt logic ---
        clean_prompt = re.sub(r'\x1b\[[0-9;]*m', '', prompt).strip()
        
        if clean_prompt == ">":
            clean_prompt = "Please state your choice."
        elif clean_prompt.endswith(":"):
            clean_prompt = clean_prompt[:-1]
        
        if '(y/n)' in clean_prompt:
            clean_prompt = clean_prompt.replace('(y/n)', 'Please say yes or no.')
        elif '(y=execute, n=abort, c=refine)' in clean_prompt:
            clean_prompt = clean_prompt.replace('(y=execute, n=abort, c=refine)', 'Please say execute, abort, or refine.')
        elif 'Permissions>' in clean_prompt:
            clean_prompt = "Permissions manager. What would you like to do?"

        logging.info(f"🎤 Asking: \"{clean_prompt}\"")
        text_to_speech(text=clean_prompt, speed=0.8, volume=0.7, use_piper=True)
        
        transcription, _, _ = listen_microphone(
            private_remote_wake_detection=args.private_remote_wake_detection, 
            remote_voice_enabled=args.online_voice
        )
        
        logging.info(f"🎤 User replied: \"{transcription}\"")

        # --- NEW: Keyword mapping for voice commands ---
        keyword_map = {
            'y': ['yes', 'yeah', 'yep', 'confirm', 'konfirm', 'allow', 'okay', 'ok', 'go', 'execute', 'proceed'],
            'n': ['no', 'nope', 'stop', 'abort', 'cancel', 'deny'],
            'a': ['always', 'always allow', 'always permit'],
            'd': ['always deny'],
            'm': ['manage', 'manager', 'manage permissions'],
            '?': ['help', 'options', 'what are the options'],
            'c': ['refine', 'change', 'correct', 'custom', 'custom pattern'],
            # Numeric options with voice-friendly keywords
            '1': ['exact', 'exact command', 'one'],
            '2': ['code', 'code execution', 'two'],
            '3': ['files', 'file management', 'three'],
            '4': ['shell', 'bash', 'shell operations', 'four'],
            '5': ['pattern', 'five'],
            # General option keywords
            'list': ['list', 'show'],
            'remove': ['remove', 'delete'],
            'clear': ['clear', 'reset'],
            'exit': ['exit', 'quit', 'back'],
            'x': ['cancel', 'exit', 'back']
        }

        user_reply = transcription.lower().strip()
        for command, keywords in keyword_map.items():
            for keyword in keywords:
                if keyword in user_reply:
                    logging.info(colored(f"🎤 Interpreted \"{transcription}\" as command: '{command}'", "cyan"))
                    return command, [] # Return the mapped single-letter command

        # If no keyword is found, return the raw transcription for invalid option handling
        return transcription, []

    # --- STANDARD TEXT-BASED INPUT LOOP ---
    while True:
        if prompt == "":
            user_input = ""
        elif input_override:
            user_input = input_override
        elif force_input:
            if input_event and input_lock and shared_input is not None:
                await input_event.wait()
                input_event.clear()
                with input_lock:
                    user_input = shared_input["value"]
            else:
                user_input = input(prompt)
        else:
            try:
                # --- FIX: Use the thread-safe event/lock mechanism ---
                if input_event and input_lock and shared_input is not None:
                    # In TUI mode, wait for the event that signals input is ready
                    agent_logger = logging.getLogger("agent_main")
                    agent_logger.info("Agent waiting for input event...")
                    await input_event.wait()
                    agent_logger.info("Input event received! Processing...")
                    input_event.clear() # Reset the event for the next input
                    with input_lock:
                        user_input = shared_input["value"]
                        agent_logger.info(f"Retrieved user input: '{user_input}'")
                else:
                    # Enhanced CLI behavior with command completion
                    try:
                        session = get_prompt_session()
                        # Convert to prompt_toolkit's HTML format for proper styling
                        # Strip ANSI codes from prompt and use the actual prompt parameter
                        import re
                        clean_prompt = re.sub(r'\x1b\[[0-9;]*m', '', prompt)
                        
                        if context_chat:
                            token_info = f"\n[Tokens: {math.ceil(len(context_chat.__str__())*3/4)} | Messages: {len(context_chat.messages)}] "
                            # Use very subtle coloring - just a gentle emphasis
                            html_prompt = HTML(token_info + f'<dim>{clean_prompt}</dim>')
                            user_input = await session.prompt_async(html_prompt)
                        else:
                            # Use very subtle coloring - just a gentle emphasis
                            html_prompt = HTML(f'<dim>{clean_prompt}</dim>')
                            user_input = await session.prompt_async(html_prompt)
                    except ImportError:
                        # Fallback to regular input if prompt_toolkit is not available
                        if context_chat:
                            user_input = input(f"\n[Tokens: {math.ceil(len(context_chat.__str__())*3/4)} | Messages: {len(context_chat.messages)}] " + prompt)
                        else:
                            user_input = input(prompt)
                    except Exception:
                        # Reset the prompt session if it becomes corrupted (e.g., after Ctrl+C)
                        global _prompt_session
                        _prompt_session = None
                        # Try once more with a fresh session
                        try:
                            session = get_prompt_session()
                            import re
                            clean_prompt = re.sub(r'\x1b\[[0-9;]*m', '', prompt)
                            
                            if context_chat:
                                token_info = f"\n[Tokens: {math.ceil(len(context_chat.__str__())*3/4)} | Messages: {len(context_chat.messages)}] "
                                html_prompt = HTML(token_info + f'<dim>{clean_prompt}</dim>')
                                user_input = await session.prompt_async(html_prompt)
                            else:
                                html_prompt = HTML(f'<dim>{clean_prompt}</dim>')
                                user_input = await session.prompt_async(html_prompt)
                        except Exception:
                            # Final fallback to regular input
                            if context_chat:
                                user_input = input(f"\n[Tokens: {math.ceil(len(context_chat.__str__())*3/4)} | Messages: {len(context_chat.messages)}] " + prompt)
                            else:
                                user_input = input(prompt)
            except KeyboardInterrupt:
                logging.warning(colored("\n# cli-agent: Exiting due to Ctrl+C...", "yellow"))
                exit()
        
        # --- Command Handling ---
        if user_input.startswith('/'):
            parts = user_input.split()
            command_alias = parts[0]
            canonical_command = COMMAND_MAP.get(command_alias)

            if not canonical_command:
                logging.error(colored(f"# cli-agent: Unknown command '{command_alias}'. Type '/help' for a list of commands.", "red"))
                continue

            # --- Command Dispatcher ---
            if canonical_command == 'screenshot':
                logging.info(colored("# cli-agent: Taking screenshot.", "green"))
                base64_images = await handle_screenshot_capture()
                return "", base64_images

            elif canonical_command == 'multiline':
                return handle_multiline_input(), []

            elif canonical_command == 'regenerate':
                if not context_chat or len(context_chat.messages) < 2:
                    logging.error(colored("# cli-agent: No chat history found, cannot regenerate.", "red"))
                    continue
                logging.info(colored("# cli-agent: Regenerating last response.", "green"))
                context_chat.messages.pop()
                return "", [] # Return empty to trigger regeneration

            elif canonical_command == 'local':
                args.local = not args.local
                g.FORCE_LOCAL = args.local
                logging.info(colored(f"# cli-agent: Local mode toggled {'on' if args.local else 'off'}.", "green"))
                continue

            elif canonical_command == 'llm':
                logging.info(colored("# cli-agent: Opening LLM selection.", "green"))
                await llm_selection(args, preselected_llms=g.SELECTED_LLMS)
                continue
            
            elif canonical_command == 'unsay':
                if context_chat and len(context_chat.messages) >= 2:
                    context_chat.messages.pop() # Assistant
                    context_chat.messages.pop() # User
                    logging.info(colored("# cli-agent: Removed last user message and AI response.", "green"))
                elif context_chat and len(context_chat.messages) == 1:
                    context_chat.messages.pop() # User
                    logging.info(colored("# cli-agent: Removed last user message.", "green"))
                else:
                    logging.warning(colored("# cli-agent: No messages to remove.", "yellow"))
                continue

            elif canonical_command == 'auto_execute':
                args.auto = not args.auto
                # (Your existing logic for displaying guard status can remain here)
                logging.info(colored(f"# cli-agent: Automatic execution toggled {'on' if args.auto else 'off'}.", "green"))
                continue

            elif canonical_command == 'fast_mode':
                args.fast = not args.fast
                logging.info(colored(f"# cli-agent: Fast mode toggled {'on' if args.fast else 'off'} (deprecated).", "green"))
                continue

            elif canonical_command == 'voice_mode':
                args.voice = not args.voice
                logging.info(colored(f"# cli-agent: Voice mode toggled {'on' if args.voice else 'off'}.", "green"))
                continue

            elif canonical_command == 'full_output':
                args.full_output_mode = not args.full_output_mode
                g.SUMMARY_MODE = not args.full_output_mode
                logging.info(colored(f"# cli-agent: Full output mode toggled {'on' if args.full_output_mode else 'off'}.", "green"))
                continue
            
            elif canonical_command == 'print':
                logging.info(colored("# cli-agent: Printing chat history.", "green"))
                os.system('clear')
                print(colored("Chat history:", "green"))
                if context_chat:
                    context_chat.print_chat()
                else:
                    print(colored("No chat history available.", "yellow"))
                continue

            elif canonical_command == 'online':
                args.online = not args.online
                logging.info(colored(f"# cli-agent: Online mode toggled {'on' if args.online else 'off'}.", "green"))
                continue

            elif canonical_command == 'exit':
                logging.info(colored("# cli-agent: Exiting...", "green"))
                exit(0)

            elif canonical_command == 'help':
                print_startup_summary(args)
                print(colored("\n--- Available Commands ---", "yellow"))
                CMD_WIDTH = 28
                for cmd_info in _COMMAND_DEFINITIONS:
                    aliases = ', '.join(cmd_info['aliases'])
                    line = f"  {colored(aliases.ljust(CMD_WIDTH), 'white')}{cmd_info['description']}"
                    print(line)
                continue
            
            elif canonical_command == 'save':
                await save_current_llm_config_as_default()
                continue

            elif canonical_command == 'clear':
                if context_chat:
                    context_chat.messages.clear()
                context_chat = await construct_engineered_prompt(args.voice)
                logging.info(colored("# cli-agent: Chat cleared.", "green"))
                continue
            
            elif canonical_command == 'compact':
                if context_chat and context_chat.messages:
                    logging.info(colored("# cli-agent: Compacting conversation...", "green"))
                    g.AGENT_IS_COMPACTING = True
                    automated_message = "Let's pause. Please summarize our conversation and progress into a markdown file at: {}/MEMORY/summary_{}.md".format(g.CLIAGENT_PERSISTENT_STORAGE_PATH, time.strftime("%Y%m%d_%H%M%S"))
                    return automated_message, []
                else:
                    logging.warning(colored("# cli-agent: No conversation to compact.", "yellow"))
                continue
            
            elif canonical_command == 'permissions':
                # The dispatcher calls the handler, passing the full user input
                await handle_permission_flags(user_input, get_user_input_with_bindings, args, input_event, input_lock, shared_input)
                continue

        # If it's not a command, it's a normal text prompt.
        return user_input, []

# Initialize the stateful flag for the first call.
get_user_input_with_bindings.is_first_call = True


async def handle_screenshot_capture() -> List[str]:
    """
    Handles the screenshot capture process.

    Returns:
        List[str]: The base64 encoded image data or an empty list if failed.
    """
    base64_images: List[str] = []
    screenshots_paths: List[str] = []
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        base64_images = []
        try:
            logging.info(colored("Attempting screenshot with Spectacle (region selection)...", "green"))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_filename = temp_file.name
            subprocess.run(['spectacle', '-rbno', temp_filename], check=True)
            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                with open(temp_filename, 'rb') as image_file:
                    png_data = image_file.read()
                base64_img = base64.b64encode(png_data).decode('utf-8')
                base64_images = [base64_img]
                pyperclip.copy(base64_img)
                logging.info(colored("Screenshot captured successfully. (+Copied to clipboard)", "green"))
            else:
                logging.warning(colored(f"No screenshot was captured (attempt {attempt}).", "yellow"))
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
        except subprocess.CalledProcessError:
            logging.error(colored(f"Spectacle command failed on attempt {attempt}. Is it installed?", "red"))
        except Exception as e:
            logging.exception(f"Unexpected error during screenshot capture attempt {attempt}: {e}")
        if base64_images:
            break
        if attempt < max_attempts and not base64_images:
            logging.warning(colored("Retrying screenshot capture...", "yellow"))
            await asyncio.sleep(2)
    if base64_images:
        images_dir = os.path.join(g.AGENTS_SANDBOX_DIR, "screenshots")
        os.makedirs(images_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, base64_img in enumerate(base64_images):
            try:
                img_data = base64.b64decode(base64_img)
                img_path = os.path.join(images_dir, f"screenshot_{timestamp}_{i+1}.png")
                screenshots_paths.append(img_path)
                with open(img_path, "wb") as img_file:
                    img_file.write(img_data)
                logging.info(colored(f"Screenshot saved to {img_path}", "green"))
            except Exception as e:
                logging.error(colored(f"Error saving screenshot {i+1}: {e}", "red"))
    else:
        logging.error(colored("No images were captured.", "red"))
    if not base64_images:
        logging.warning(colored("# cli-agent: No screenshot was captured after multiple attempts.", "yellow"))
        return []
    
    return base64_images


async def handle_permission_flags(user_input: str, get_user_input_func, args, input_event, input_lock, shared_input):
    """Handle permission management flags like /permissions, -p, --p."""
    from tools.permissions import permissions
    from termcolor import colored
    
    # Parse the command
    parts = user_input.split()
    command = parts[0]  # /permissions, -p, or --p
    
    try:
        if len(parts) == 1:
            # Just show the list
            print(colored("\n🔧 CLI-AGENT PERMISSION MANAGER", "yellow", attrs=['bold']))
            print(colored("Using flag-based interface for easy permission management", "white"))
            print(colored("─" * 60, "yellow"))
            result = permissions._list_permissions()
            print(result)
            return
        
        action = parts[1].lower()
        
        if action == "list":
            result = permissions._list_permissions()
            print(result)
            
        elif action == "remove" and len(parts) >= 3:
            perm_id = parts[2]
            result = permissions._remove_permission(perm_id)
            print(result)
            
        elif action == "clear":
            # Ask for confirmation
            print(colored("\n⚠️  WARNING: This will clear ALL saved permissions!", "red", attrs=['bold']))
            print(colored("Type 'yes' to confirm or anything else to cancel:", "yellow"))
            
            try:
                if get_user_input_func:
                    confirmation, _ = await get_user_input_func(
                        args, None, colored("> ", "cyan"),
                        input_event=input_event,
                        input_lock=input_lock,
                        shared_input=shared_input
                    )
                else:
                    confirmation = input(colored("> ", "cyan"))
            except KeyboardInterrupt:
                # User pressed Ctrl+C during clear confirmation - treat as cancel
                print(colored("\n❌ Clear operation cancelled (Ctrl+C).", "yellow"))
                return
            
            if confirmation.strip().lower() == 'yes':
                result = permissions._clear_all_permissions()
                print(result)
            else:
                print(colored("❌ Clear operation cancelled.", "green"))
                
        elif action == "interactive" or action == "manage":
            # Open interactive manager
            await interactive_permission_manager(get_user_input_func, args, input_event, input_lock, shared_input)
            
        else:
            print(colored(f"❓ Unknown permission action: {action}", "red"))
            print(colored("Usage:", "cyan"))
            print(colored("  /permissions list                 # List all permissions", "white"))
            print(colored("  /permissions remove <id>          # Remove permission by ID", "white"))
            print(colored("  /permissions clear                # Clear all permissions", "white"))
            print(colored("  /permissions interactive          # Open interactive manager", "white"))
            print()
            print(colored("Shortcuts:", "cyan"))
            print(colored("  /p list, etc.                    # Same as /permissions", "white"))
            
    except Exception as e:
        print(colored(f"❌ Error handling permission command: {str(e)}", "red"))


async def interactive_permission_manager(get_user_input_func, args, input_event, input_lock, shared_input):
    """Interactive permission management interface."""
    from tools.permissions import permissions
    from termcolor import colored
    
    print()
    print(colored("🔧 INTERACTIVE PERMISSION MANAGER", "yellow", attrs=['bold']))
    print(colored("═" * 50, "yellow"))
    
    while True:
        # Show current permissions
        result = permissions._list_permissions()
        print(result)
        
        print()
        print(colored("Commands:", "cyan"))
        print(colored("  list                    # Refresh permission list", "white"))
        print(colored("  remove <id>             # Remove permission by ID", "white"))
        print(colored("  clear                   # Clear all permissions", "white"))
        print(colored("  exit, quit, back        # Return to main CLI", "white"))
        print()
        
        try:
            if get_user_input_func:
                user_cmd, _ = await get_user_input_func(
                    args, None, colored("Permissions> ", "cyan"),
                    input_event=input_event,
                    input_lock=input_lock,
                    shared_input=shared_input
                )
            else:
                user_cmd = input(colored("Permissions> ", "cyan"))
            
            user_cmd = user_cmd.strip().lower()
        except KeyboardInterrupt:
            # User pressed Ctrl+C in interactive permission manager - exit gracefully
            print(colored("\n↩️  Returning to main CLI (Ctrl+C)...", "yellow"))
            break
        
        if user_cmd in ['exit', 'quit', 'back', '']:
            print(colored("↩️  Returning to main CLI...", "green"))
            break
            
        elif user_cmd == 'list':
            continue  # Will refresh at top of loop
            
        elif user_cmd.startswith('remove ') and len(user_cmd.split()) >= 2:
            perm_id = user_cmd.split()[1]
            result = permissions._remove_permission(perm_id)
            print(result)
            
        elif user_cmd == 'clear':
            print(colored("\n⚠️  WARNING: This will clear ALL saved permissions!", "red", attrs=['bold']))
            print(colored("Type 'yes' to confirm or anything else to cancel:", "yellow"))
            
            try:
                if get_user_input_func:
                    confirmation, _ = await get_user_input_func(
                        args, None, colored("> ", "cyan"),
                        input_event=input_event,
                        input_lock=input_lock,
                        shared_input=shared_input
                    )
                else:
                    confirmation = input(colored("> ", "cyan"))
            except KeyboardInterrupt:
                # User pressed Ctrl+C during clear confirmation - treat as cancel
                print(colored("\n❌ Clear operation cancelled (Ctrl+C).", "yellow"))
                continue
            
            if confirmation.strip().lower() == 'yes':
                result = permissions._clear_all_permissions()
                print(result)
                break  # Exit after clearing
            else:
                print(colored("❌ Clear operation cancelled.", "green"))
                
        else:
            print(colored("❓ Unknown command. Use: list, remove <id>, clear, or exit", "yellow"))