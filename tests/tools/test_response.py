"""
Response Tool Tests

Tests for the response tool including basic output,
empty content handling, and special character preservation.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.primitives.response import response


def test_response_basic_output():
    """Test basic response output without server"""
    content = "This is a response message to the user"

    # Mock the server to not exist (CLI mode)
    with patch('api.server.get_server', return_value=None):
        result = response.run(content)

    # In CLI mode, should return empty string
    assert result == "", f"Expected empty string in CLI mode, got: {repr(result)}"
    return True


def test_response_empty_content():
    """Handle empty content gracefully"""
    with patch('api.server.get_server', return_value=None):
        result = response.run("")

    assert result == "", f"Expected empty string for empty content, got: {repr(result)}"
    return True


def test_response_special_characters():
    """Preserve special characters in content"""
    content = "Special chars: <>&\"'ñáéíóú 🚀 @#$%^&*()"

    with patch('api.server.get_server', return_value=None):
        result = response.run(content)

    # Should not crash with special characters
    assert isinstance(result, str), "Should return a string"
    return True


def test_response_end_turn_flag_default():
    """Test that end_turn defaults to True"""
    content = "Regular response"

    with patch('api.server.get_server', return_value=None):
        with patch('core.tool_context.tool_context') as mock_context:
            result = response.run(content)

            # Verify set_end_turn was called with True (default)
            mock_context.set_end_turn.assert_called_once_with(True)

    return True


def test_response_end_turn_flag_false():
    """Test explicit end_turn=false parameter"""
    content = "<end_turn>false</end_turn>This is a partial response"

    with patch('api.server.get_server', return_value=None):
        with patch('core.tool_context.tool_context') as mock_context:
            result = response.run(content)

            # Verify set_end_turn was called with False
            mock_context.set_end_turn.assert_called_once_with(False)

    return True


def test_response_end_turn_flag_true():
    """Test explicit end_turn=true parameter"""
    content = "<end_turn>true</end_turn>This is a final response"

    with patch('api.server.get_server', return_value=None):
        with patch('core.tool_context.tool_context') as mock_context:
            result = response.run(content)

            # Verify set_end_turn was called with True
            mock_context.set_end_turn.assert_called_once_with(True)

    return True


def test_response_multiline_content():
    """Handle multiline content properly"""
    content = """This is a multiline response.
It has several lines.
And should be handled correctly."""

    with patch('api.server.get_server', return_value=None):
        result = response.run(content)

    # Should not crash with multiline content
    assert isinstance(result, str), "Should return a string"
    return True


def test_response_very_long_content():
    """Handle very long content without errors"""
    content = "x" * 10000  # 10KB of text

    with patch('api.server.get_server', return_value=None):
        result = response.run(content)

    # Should not crash with long content
    assert isinstance(result, str), "Should return a string"
    return True


def test_response_with_markdown():
    """Handle markdown formatting in content"""
    content = """# Heading

**Bold text** and *italic text*

- List item 1
- List item 2

```python
code_block = "example"
```
"""

    with patch('api.server.get_server', return_value=None):
        result = response.run(content)

    # Should handle markdown without issues
    assert isinstance(result, str), "Should return a string"
    return True


def test_response_tool_contract():
    """Verify tool implements the required contract"""
    # Test get_delim
    delim = response.get_delim()
    assert delim == "response", f"Expected delimiter 'response', got: {delim}"

    # Test get_tool_info
    info = response.get_tool_info()
    assert isinstance(info, dict), "get_tool_info should return dict"
    assert "name" in info, "Tool info should have 'name'"
    assert "description" in info, "Tool info should have 'description'"
    assert "example" in info, "Tool info should have 'example'"
    assert info["name"] == "response", f"Expected name 'response', got: {info['name']}"

    return True


# Test level assignments
test_response_basic_output.level = 2
test_response_empty_content.level = 2
test_response_special_characters.level = 2
test_response_end_turn_flag_default.level = 2
test_response_end_turn_flag_false.level = 2
test_response_end_turn_flag_true.level = 2
test_response_multiline_content.level = 2
test_response_very_long_content.level = 3
test_response_with_markdown.level = 2
test_response_tool_contract.level = 2
