#!/usr/bin/env python3

import datetime
import logging
import os
import select
import time
import traceback
from typing import List, Optional, Tuple
from pyfiglet import figlet_format
from dotenv import load_dotenv
from termcolor import colored
import argparse
import sys
import socket
import warnings
import asyncio
import re
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.widgets import CheckboxList, Frame, Label, RadioList
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
import base64
import tempfile
import subprocess

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")
# Suppress phonemizer warnings
warnings.filterwarnings("ignore", message="words count mismatch on*", module="phonemizer", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")  # Catch all phonemizer warnings

# Import utils_audio which uses torch
from py_methods import utils_audio
from py_classes.cls_util_manager import UtilsManager
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_llm_router import Llm, LlmRouter
from py_classes.cls_chat import Chat, Role
from py_classes.utils.cls_utils_web_server import WebServer
from py_classes.globals import g
from py_classes.cls_python_sandbox import PythonSandbox
from py_classes.cls_ssh_sandbox import SSHSandbox
from py_classes.cls_text_stream_painter import TextStreamPainter

# Fix the import by using a relative or absolute import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#!/usr/bin/env python3

# Usage example for the unified PythonSandbox

from py_classes.cls_python_sandbox import PythonSandbox
from termcolor import colored

def filter_cmd_output(text: str) -> str:
    text = text.replace("ksshaskpass: Unable to parse phrase \"[sudo] password for prob: \"", "")
    return text

def update_python_environment(chunk: str, print_char: bool = True) -> str:
    response_buffer = ""
    for char in chunk:
        response_buffer += char
        if print_char:
            print(char, end="", flush=True)
        if response_buffer.count("```") == 2:
            final_response = response_buffer
            response_buffer = ""
            return final_response
    return None

try:
    # Create unified sandbox - handles both Python and shell commands
    sandbox = PythonSandbox()
    
    def stdout_callback(text: str) -> None:
        text = filter_cmd_output(text)
        print(text, end="")
    
    def stderr_callback(text: str) -> None:
        text = filter_cmd_output(text)
        print(colored(text, "red"), end="")
    
    def input_callback(prompt: str) -> str:
        print("🤖 LLM handling input prompt:")
        print(f"📝 Prompt: {prompt}")
        
        # Your existing LLM interaction logic
        konsole_interaction_chat = Chat()
        konsole_interaction_chat.add_message(Role.USER, 
            f"Your notebook execution was halted, please determine what keys to enter to continue execution. "
            f"Provide the key or the string to enter as the last line of your response:\n```bash\n{prompt}```")
        
        konsole_interaction_response = LlmRouter.generate_completion(
            konsole_interaction_chat,
            [g.SELECTED_LLMS[0]],
            temperature=0,
            generation_stream_callback=update_python_environment,
            strengths=g.LLM_STRENGTHS
        )
        
        response = konsole_interaction_response.split("\n")[-1].strip()
        print(f"🔤 LLM Response: {response}")
        return response
    
    print("🚀 Unified PythonSandbox Examples")
    print("=" * 40)
    
    # Example 1: Python code execution
    print("\n📊 Example 1: Python code execution")
    python_code = """
import subprocess
try:
    subprocess.run(["docker", "--version"], check=True, capture_output=True, text=True)
    print("Docker is installed! 🎉")
except FileNotFoundError:
    print("Docker is not installed. 🤔")
except subprocess.CalledProcessError as e:
    print(f"Docker command failed: {e}")
    print("Docker might be installed but not working correctly. 🤔")
"""
    
    stdout, stderr, result = sandbox.execute(
        python_code,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback
    )
    
    print(f"\n✅ Python execution result: {result}")
    
    # Example 2: Shell command execution (your original problem!)
    print("\n🐳 Example 2: Shell command with LLM interaction")
    
    # The exact command that was causing problems - now works with unified execute!
    stdout, stderr, exit_code = sandbox.execute(
        "!curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo -A gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg",
        input_callback=input_callback,  # Your LLM function works!
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback
    )
    
    print(f"\n✅ Shell command completed with exit code: {exit_code}")
    
    # Example 3: Simple shell commands
    print("\n🔍 Example 3: Simple shell commands")
    
    stdout, stderr, exit_code = sandbox.execute(
        "!which docker",
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback
    )
    
    print(f"\n✅ Command completed with exit code: {exit_code}")
    
    # Example 4: Python code with input
    print("\n👤 Example 4: Python code with input handling")
    
    interactive_python = """
name = input("What's your name? ")
age = input("What's your age? ")
print(f"Hello {name}! You are {age} years old.")
"""
    
    # Use a simple input callback for this example
    def simple_input_callback(prompt: str) -> str:
        responses = {
            "What's your name? ": "Alice",
            "What's your age? ": "25"
        }
        return responses.get(prompt.strip(), "Unknown")
    
    stdout, stderr, result = sandbox.execute(
        interactive_python,
        input_callback=simple_input_callback,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback
    )
    
    print(f"\n✅ Interactive Python result: {result}")
    
    # Example 5: Long-running shell command with timeout
    print("\n⏰ Example 5: Long command with timeout hints")
    
    stdout, stderr, exit_code = sandbox.execute(
        "!sleep 35 && echo 'Done sleeping!'",
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
        timeout=60,  # 1 minute timeout
        show_timeout_hints=True
    )
    
    print(f"\n✅ Long command completed with exit code: {exit_code}")
    
    print("\n🎉 All unified execution examples completed!")
    print("\n📋 Summary of the Unified Approach:")
    print("✅ One execute() method handles everything")
    print("✅ Automatic detection: Python code vs shell commands (!)")
    print("✅ Interactive input handling with LLM integration")
    print("✅ Environment consistency between Python and shell")
    print("✅ Timeout management with helpful hints") 
    print("✅ Real-time output streaming")
    print("✅ Proper error handling and interruption support")
    print("✅ No need for separate methods!")
    
    print("\n💡 Usage Tips:")
    print("• Python code: Execute directly")
    print("• Shell commands: Prefix with !")
    print("• Interactive prompts: Handled automatically via input_callback")
    print("• Long operations: Use timeout parameter for safety")
    print("• Real-time output: Use stdout_callback and stderr_callback")
    
except KeyboardInterrupt:
    print(colored("\n\n🛑 Main script interrupted by user (Ctrl+C)", "yellow"))
except Exception as e:
    print(colored(f"\n❌ Error: {e}", "red"))
    import traceback
    traceback.print_exc()
finally:
    if 'sandbox' in locals():
        sandbox.shutdown()
        print("\n🔒 Sandbox shut down successfully")