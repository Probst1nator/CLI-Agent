#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from py_classes.cls_computational_notebook import ComputationalNotebook

def test_streaming():
    """Test real-time streaming of Python code output."""
    
    def my_stdout(text):
        print(text, end='', flush=True)  # Ensure immediate output
    
    print("Testing real-time streaming of Python code execution...")
    print("=" * 60)
    
    notebook = ComputationalNotebook(stdout_callback=my_stdout)
    
    # Test 1: Simple print statements with delays
    print("\n--- Test 1: Print statements with delays ---")
    python_code_1 = """
import time
print("Starting test...")
time.sleep(1)
print("After 1 second")
time.sleep(1)
print("After 2 seconds")
time.sleep(1)
print("After 3 seconds")
print("Test completed!")
"""
    
    notebook.execute(python_code_1, is_python_code=True, persist_python_state=True)
    
    # Test 2: Loop with incremental output
    print("\n--- Test 2: Loop with incremental output ---")
    python_code_2 = """
import time
for i in range(5):
    print(f"Iteration {i+1}")
    time.sleep(0.5)
print("Loop completed!")
"""
    
    notebook.execute(python_code_2, is_python_code=True, persist_python_state=True)
    
    # Test 3: Progress bar simulation
    print("\n--- Test 3: Progress bar simulation ---")
    python_code_3 = """
import time
import sys

print("Progress: ", end="")
for i in range(20):
    print("█", end="", flush=True)
    time.sleep(0.1)
print(" Done!")
"""
    
    notebook.execute(python_code_3, is_python_code=True, persist_python_state=True)
    
    # Test 4: Non-persistent execution
    print("\n--- Test 4: Non-persistent execution ---")
    python_code_4 = """
import time
print("Non-persistent execution starting...")
for i in range(3):
    print(f"Step {i+1}")
    time.sleep(0.5)
print("Non-persistent execution completed!")
"""
    
    notebook.execute(python_code_4, is_python_code=True, persist_python_state=False)
    
    notebook.close()
    print("\n" + "=" * 60)
    print("Streaming test completed!")

if __name__ == "__main__":
    test_streaming() 