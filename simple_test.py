#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from py_classes.cls_computational_notebook import ComputationalNotebook

def test_simple_streaming():
    """Test simple real-time streaming."""
    
    def my_stdout(text):
        print(text, end='', flush=True)
    
    print("Testing simple streaming...")
    
    notebook = ComputationalNotebook(stdout_callback=my_stdout)
    
    # Simple test with time delays
    python_code = """
import time
print("Starting...")
time.sleep(1)
print("Step 1")
time.sleep(1)
print("Step 2")
time.sleep(1)
print("Done!")
"""
    
    notebook.execute(python_code, is_python_code=True, persist_python_state=True)
    notebook.close()
    print("\nTest completed!")

if __name__ == "__main__":
    test_simple_streaming() 