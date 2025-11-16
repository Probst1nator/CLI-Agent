#!/usr/bin/env python3
"""Quick test to verify all tools load correctly."""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.think import think
from tools.todos import todos
from tools.readfile import readfile
from tools.grepsearch import grepsearch

def test_tool(tool_class, tool_name):
    """Test that a tool has the required methods."""
    print(f"\nTesting {tool_name}...")

    # Check required methods
    assert hasattr(tool_class, 'get_delim'), f"{tool_name} missing get_delim()"
    assert hasattr(tool_class, 'get_tool_info'), f"{tool_name} missing get_tool_info()"
    assert hasattr(tool_class, 'run'), f"{tool_name} missing run()"

    # Test get_delim
    delim = tool_class.get_delim()
    assert delim == tool_name, f"Delimiter mismatch: {delim} != {tool_name}"
    print(f"  ✓ Delimiter: {delim}")

    # Test get_tool_info
    info = tool_class.get_tool_info()
    assert 'name' in info, f"{tool_name} info missing 'name'"
    assert 'description' in info, f"{tool_name} info missing 'description'"
    assert 'example' in info, f"{tool_name} info missing 'example'"
    print(f"  ✓ Tool info: {info['name']}")
    print(f"  ✓ Description: {info['description'][:50]}...")

    print(f"  ✓ {tool_name} passed all checks!")

def main():
    print("=" * 60)
    print("Minimal Agent - Tool Verification")
    print("=" * 60)

    tools = [
        (think, 'think'),
        (todos, 'todos'),
        (readfile, 'readfile'),
        (grepsearch, 'grepsearch')
    ]

    for tool_class, tool_name in tools:
        try:
            test_tool(tool_class, tool_name)
        except AssertionError as e:
            print(f"  ❌ {e}")
            return 1
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return 1

    print("\n" + "=" * 60)
    print("✓ All tools verified successfully!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
