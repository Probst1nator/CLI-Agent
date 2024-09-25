import sys
import time
from importlib import import_module
from types import ModuleType

class ImportProfiler:
    def __init__(self, threshold=0.2):
        self.threshold = threshold
        self.import_times = {}

    def __call__(self, name, globals=None, locals=None, fromlist=(), level=0):
        start_time = time.time()
        
        # Use the original __import__ function to maintain correct behavior
        module = original_import(name, globals, locals, fromlist, level)
        
        end_time = time.time()
        
        import_time = end_time - start_time
        if import_time > self.threshold:
            self.import_times[name] = import_time
        
        return module

    def print_results(self):
        print(f"Imports taking longer than {self.threshold} seconds:")
        for name, duration in sorted(self.import_times.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {duration:.2f} seconds")

# Store the original import function
original_import = __import__

# Create an instance of the ImportProfiler
profiler = ImportProfiler(threshold=0.2)

# Replace the built-in __import__ function with our profiler
sys.modules['builtins'].__import__ = profiler

# Your existing imports
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

from py_classes.cls_html_server import HtmlServer
from py_classes.cls_rag_tooling import RagTooling
from py_classes.cls_youtube import YouTube
from py_methods.cmd_execution import select_and_execute_commands
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")

from py_agents.assistants import python_error_agent, code_assistant, git_message_generator, majority_response_assistant, presentation_assistant, documents_assistant
from py_methods.tooling import extract_blocks, pdf_or_folder_to_database,recolor, listen_microphone, remove_blocks, text_to_speech, update_cmd_collection
from py_classes.cls_web_scraper import WebTools
from py_classes.cls_llm_router import LlmRouter
from py_classes.cls_few_shot_factory import FewShotProvider
from py_classes.cls_chat import Chat, Role
from agentic.cls_AgenticPythonProcess import AgenticPythonProcess
from py_classes.globals import g


# Print the profiling results
profiler.print_results()

# Add an early exit after printing the results
sys.exit(0)

# The rest of your main.py code would go here, but it won't be executed due to the early exit