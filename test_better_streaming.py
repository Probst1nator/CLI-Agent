#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from py_classes.cls_computational_notebook import ComputationalNotebook

def test_better_streaming():
    """Test the improved streaming with more complex output."""
    
    def my_stdout(text):
        print(text, end='', flush=True)
    
    print("Testing improved streaming...")
    
    notebook = ComputationalNotebook(stdout_callback=my_stdout)
    
    # Test with more complex output that might take time
    python_code = """
import time
import json

print("Starting SearchWeb simulation...")
time.sleep(1)

# Simulate some processing
for i in range(3):
    print(f"Processing step {i+1}...")
    time.sleep(0.5)

# Simulate the actual result
result = {
    "query": "current chancellor of Germany", 
    "result": "Olaf Scholz is the current Chancellor of Germany",
    "timestamp": "2025-05-30"
}

print("Search completed!")
print(json.dumps(result, indent=2))
print("Done!")
"""
    
    notebook.execute(python_code, is_python_code=True, persist_python_state=True)
    notebook.close()
    print("\nTest completed!")

if __name__ == "__main__":
    test_better_streaming() 