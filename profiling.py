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
#!/usr/bin/env python3

import datetime
import logging
import os
import select
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


from py_methods.utils import extract_blocks
from py_methods.utils import pdf_or_folder_to_database
from py_methods.utils import listen_microphone
from py_methods.utils import take_screenshot
from py_methods.utils import update_cmd_collection
from py_methods.utils import ScreenCapture

from py_methods import utils_audio
from py_classes.cls_util_manager import UtilsManager
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_llm_router import Llm, LlmRouter
from py_classes.cls_chat import Chat, Role
from py_classes.utils.cls_utils_web_server import WebServer
from py_classes.globals import g
from py_classes.cls_python_sandbox import PythonSandbox
from py_classes.cls_text_stream_painter import TextStreamPainter



# Print the profiling results
profiler.print_results()

# Add an early exit after printing the results
sys.exit(0)

# The rest of your main.py code would go here, but it won't be executed due to the early exit