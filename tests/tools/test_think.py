"""
Think Tool Tests

Tests for the think tool including basic functionality,
content handling, and tool contract verification.
The think tool is a cognitive primitive that performs no action.
"""

import os
import sys
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.primitives.think import think


def test_think_returns_none():
    """Test that think tool returns None (no-op)"""
    thought = "I need to analyze this problem carefully"
    result = think.run(thought)

    assert result is None, f"Expected None, got: {repr(result)}"
    return True


def test_think_empty_content():
    """Handle empty thoughts"""
    result = think.run("")

    assert result is None, f"Expected None for empty thought, got: {repr(result)}"
    return True


def test_think_multiline_thought():
    """Handle multiline thought processes"""
    thought = """First, I should check the file.
Then, I will process the data.
Finally, I will return the result."""

    result = think.run(thought)

    assert result is None, f"Expected None for multiline thought, got: {repr(result)}"
    return True


def test_think_special_characters():
    """Handle special characters in thoughts"""
    thought = "Special chars: <>&\"'ñáéíóú 🚀 @#$%^&*()"
    result = think.run(thought)

    assert result is None, f"Expected None, got: {repr(result)}"
    return True


def test_think_very_long_thought():
    """Handle very long thought content"""
    thought = "x" * 10000  # 10KB of text
    result = think.run(thought)

    assert result is None, f"Expected None for long thought, got: {repr(result)}"
    return True


def test_think_complex_reasoning():
    """Test complex reasoning thought"""
    thought = """
    Let me think through this step by step:
    1. The user asked about file permissions
    2. I need to check if the file exists first
    3. Then verify read permissions
    4. Finally, attempt to read the file
    This approach should prevent errors.
    """
    result = think.run(thought)

    assert result is None, f"Expected None for complex thought, got: {repr(result)}"
    return True


def test_think_with_code_snippets():
    """Handle thoughts containing code snippets"""
    thought = """
    I should use this pattern:
    ```python
    if condition:
        do_something()
    ```
    This will solve the problem.
    """
    result = think.run(thought)

    assert result is None, f"Expected None, got: {repr(result)}"
    return True


def test_think_with_xml_like_content():
    """Handle thoughts containing XML-like tags"""
    thought = "I notice <tag>content</tag> in the output, this might be important"
    result = think.run(thought)

    assert result is None, f"Expected None, got: {repr(result)}"
    return True


def test_think_unicode_content():
    """Handle unicode characters in thoughts"""
    thought = "日本語 Ελληνικά العربية עברית 中文 Русский"
    result = think.run(thought)

    assert result is None, f"Expected None for unicode thought, got: {repr(result)}"
    return True


def test_think_tool_contract():
    """Verify tool implements the required contract"""
    # Test get_delim
    delim = think.get_delim()
    assert delim == "think", f"Expected delimiter 'think', got: {delim}"

    # Test get_tool_info
    info = think.get_tool_info()
    assert isinstance(info, dict), "get_tool_info should return dict"
    assert "name" in info, "Tool info should have 'name'"
    assert "description" in info, "Tool info should have 'description'"
    assert "example" in info, "Tool info should have 'example'"
    assert info["name"] == "think", f"Expected name 'think', got: {info['name']}"

    return True


def test_think_idempotency():
    """Test that multiple calls with same content produce same result"""
    thought = "This is a consistent thought"

    result1 = think.run(thought)
    result2 = think.run(thought)
    result3 = think.run(thought)

    assert result1 is None, "First call should return None"
    assert result2 is None, "Second call should return None"
    assert result3 is None, "Third call should return None"
    assert result1 == result2 == result3, "All calls should return the same value"

    return True


# Test level assignments
test_think_returns_none.level = 2
test_think_empty_content.level = 2
test_think_multiline_thought.level = 2
test_think_special_characters.level = 2
test_think_very_long_thought.level = 3
test_think_complex_reasoning.level = 2
test_think_with_code_snippets.level = 2
test_think_with_xml_like_content.level = 2
test_think_unicode_content.level = 2
test_think_tool_contract.level = 2
test_think_idempotency.level = 2
