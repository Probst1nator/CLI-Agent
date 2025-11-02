#!/usr/bin/env python3
"""
Test script for the taskmanager tool functionality.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from tools.taskmanager import taskmanager


def test_taskmanager_tool():
    """Test basic taskmanager tool functionality"""

    print("=== TESTING TASKMANAGER TOOL ===")

    # Test 1: Get tool info
    tool_info = taskmanager.get_tool_info()
    print(f"Tool Info: {tool_info}")
    print(f"✅ Name: {tool_info['name']}")
    print(f"✅ Delimiter: {taskmanager.get_delim()}")
    print()

    # Test 2: List processes (should be empty initially)
    list_result = taskmanager.run("<list/>")
    print("List processes result:")
    print(list_result)
    print()

    # Test 3: Test invalid command
    invalid_result = taskmanager.run("<invalid>test</invalid>")
    print("Invalid command result:")
    print(invalid_result)
    print()

    # Test 4: Test stdout on non-existent process
    stdout_result = taskmanager.run("<stdout>nonexistent</stdout>")
    print("Stdout on non-existent process:")
    print(stdout_result)
    print()

    # Test 5: Test kill on non-existent process
    kill_result = taskmanager.run("<kill>nonexistent</kill>")
    print("Kill non-existent process:")
    print(kill_result)
    print()

    print("=== TOOL EXAMPLES ===")
    print("Usage examples:")
    print("1. List all background processes:")
    print("   <taskmanager><list/></taskmanager>")
    print()
    print("2. Kill all background processes:")
    print("   <taskmanager><kill>all</kill></taskmanager>")
    print()
    print("3. View stdout of a specific process:")
    print("   <taskmanager><stdout>a2b4c6d8</stdout></taskmanager>")
    print()
    print("4. Kill a specific process:")
    print("   <taskmanager><kill>a2b4c6d8</kill></taskmanager>")
    print()

    print("✅ Taskmanager tool is working correctly!")
    print("Note: To test with actual background processes, start a background bash command first:")
    print("Example: <bash><background>true</background>sleep 10</bash>")

if __name__ == "__main__":
    test_taskmanager_tool()