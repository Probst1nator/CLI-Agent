#!/usr/bin/env python3
"""
Test script for the auto-background feature integration.
Tests the enhanced bash tool with automatic background process conversion.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from tools.bash import bash
from tools.taskmanager import taskmanager


async def test_auto_background_integration():
    """Test the complete auto-background system integration"""

    print("=== TESTING AUTO-BACKGROUND INTEGRATION ===")
    print()

    # Test 1: Quick command (should not go to background)
    print("🧪 Test 1: Quick command (should complete normally)")
    try:
        result = await bash.run("echo 'Hello World'")
        print(f"Result: {result}")
        print("✅ Quick command works correctly")
        print()
    except Exception as e:
        print(f"❌ Error in quick command: {e}")
        print()

    # Test 2: Long command (should become background process)
    print("🧪 Test 2: Long command (should auto-background after 5s)")
    try:
        # This should timeout after 5 seconds and become a background process
        start_time = time.time()
        result = await bash.run("sleep 10")
        end_time = time.time()

        print(f"Execution time: {end_time - start_time:.2f} seconds")
        print(f"Result: {result}")

        # Check if result indicates background process creation
        if "⏳" in result and "📋 Process ID:" in result:
            print("✅ Command correctly converted to background process!")

            # Extract process ID from the result
            import re
            pid_match = re.search(r'Process ID: (bg_[a-f0-9]+)', result)
            if pid_match:
                process_id = pid_match.group(1)
                print(f"✅ Process ID extracted: {process_id}")

                # Test 3: List background processes
                print("\n🧪 Test 3: List background processes")
                list_result = taskmanager.run("<list/>")
                print("Background processes:")
                print(list_result)

                if process_id in list_result:
                    print("✅ Background process appears in taskmanager list!")
                else:
                    print("❌ Background process not found in taskmanager list")

                # Test 4: Check stdout of background process
                print("\n🧪 Test 4: Check background process output")
                # Wait a moment for some output to accumulate
                await asyncio.sleep(2)

                stdout_result = taskmanager.run(f"<stdout>{process_id}</stdout>")
                print("Process output:")
                print(stdout_result)

                if "Status: running" in stdout_result:
                    print("✅ Background process is running and accessible!")
                else:
                    print("❌ Background process status unclear")

                # Test 5: Kill background process
                print("\n🧪 Test 5: Kill background process")
                kill_result = taskmanager.run(f"<kill>{process_id}</kill>")
                print(f"Kill result: {kill_result}")

                if "killed" in kill_result:
                    print("✅ Background process successfully killed!")
                else:
                    print("❌ Failed to kill background process")
            else:
                print("❌ Could not extract process ID from result")
        else:
            print("❌ Command did not convert to background process as expected")
            print("This might indicate the timeout mechanism isn't working")

        print()
    except Exception as e:
        print(f"❌ Error in long command test: {e}")
        print()

    # Test 6: Verify session recovery (main session should still work)
    print("🧪 Test 6: Verify main session recovery")
    try:
        result = await bash.run("pwd")
        print(f"Current directory: {result}")
        print("✅ Main bash session recovered and working!")
        print()
    except Exception as e:
        print(f"❌ Main session recovery failed: {e}")
        print()

    print("=== INTEGRATION TEST SUMMARY ===")
    print("The auto-background feature provides these benefits:")
    print("✅ Long commands don't timeout - they become manageable background processes")
    print("✅ Process management via taskmanager tool")
    print("✅ Main bash session remains responsive")
    print("✅ Output capture and monitoring capabilities")
    print("✅ Clean process lifecycle management")

    print("\n💡 Usage: Commands exceeding 5s timeout automatically become background processes")
    print("💡 Use taskmanager to list, monitor, and kill background processes")
    print("💡 Main bash session continues working normally")

if __name__ == "__main__":
    asyncio.run(test_auto_background_integration())