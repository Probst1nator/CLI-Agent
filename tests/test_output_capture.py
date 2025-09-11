#!/usr/bin/env python3
"""
Quick test for improved background process output capture.
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(__file__))

from tools.bash import bash
from tools.taskmanager import taskmanager

async def test_output_capture():
    """Test background process output capture"""
    
    print("=== TESTING IMPROVED OUTPUT CAPTURE ===")
    
    # Test with a command that produces output
    print("🧪 Testing command with output: 'echo Output && sleep 8'")
    
    result = await bash.run("echo 'Background process output test' && sleep 8")
    print(f"Bash result: {result}")
    
    # Extract process ID
    import re
    pid_match = re.search(r'Process ID: (bg_[a-f0-9]+)', result)
    if pid_match:
        process_id = pid_match.group(1)
        print(f"Process ID: {process_id}")
        
        # Wait a moment then check output
        await asyncio.sleep(2)
        
        stdout_result = taskmanager.run(f"<stdout>{process_id}</stdout>")
        print("Process output check:")
        print(stdout_result)
        
        # Kill the process
        kill_result = taskmanager.run(f"<kill>{process_id}</kill>")
        print(f"Kill result: {kill_result}")
    
    print("✅ Output capture test completed!")

if __name__ == "__main__":
    asyncio.run(test_output_capture())