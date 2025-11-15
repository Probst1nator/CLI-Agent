"""
Help Tool Tests

Tests for the help tool including tool listing,
specific tool help, and error handling.
"""

import sys
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.help import help


def test_help_tool_listing():
    """List all available tools"""
    result = help.run("")

    # Should return a listing of tools
    assert isinstance(result, str), "Should return a string"
    assert len(result) > 0, "Should not return empty string"

    # Should contain some known tools
    assert "readfile" in result.lower() or "readfile" in result, "Should list readfile"
    assert "bash" in result.lower() or "bash" in result, "Should list bash"

    # Should have formatting
    assert "AVAILABLE TOOLS" in result or "available" in result.lower(), "Should have header"

    return True


def test_help_tool_descriptions():
    """Show tool descriptions in listing"""
    result = help.run("")

    # Should contain descriptive information
    assert "description" in result.lower() or ":" in result, "Should show descriptions"

    # Should show count of tools
    assert "tools" in result.lower() or "Tools" in result, "Should mention tools count"

    return True


def test_help_specific_tool_readfile():
    """Get help for a specific tool - readfile"""
    result = help.run("readfile")

    # Should return help for readfile
    assert isinstance(result, str), "Should return a string"
    assert len(result) > 0, "Should not return empty string"
    assert "readfile" in result.lower(), "Should mention readfile"

    # Should show usage information
    assert "filepath" in result.lower() or "path" in result.lower(), "Should show filepath parameter"

    return True


def test_help_specific_tool_bash():
    """Get help for a specific tool - bash"""
    result = help.run("bash")

    # Should return help for bash
    assert isinstance(result, str), "Should return a string"
    assert "bash" in result.lower(), "Should mention bash"

    return True


def test_help_nonexistent_tool():
    """Handle non-existent tool gracefully"""
    result = help.run("fakeTool123_nonexistent")

    # Should return error message
    assert isinstance(result, str), "Should return a string"
    assert "not found" in result.lower() or "error" in result.lower(), \
        "Should indicate tool not found"

    # Should suggest available tools
    assert "available" in result.lower() or "use" in result.lower(), \
        "Should suggest how to see available tools"

    return True


def test_help_tool_with_detailed_info():
    """Get detailed help for tools that provide it"""
    # searchfile and readdirectory have get_detailed_tool_info()
    result = help.run("searchfile")

    assert isinstance(result, str), "Should return a string"
    assert len(result) > 100, "Detailed help should be substantial"
    assert "searchfile" in result.lower(), "Should mention the tool name"

    return True


def test_help_whitespace_handling():
    """Handle extra whitespace in tool names"""
    # Test with spaces
    result = help.run("  readfile  ")

    assert isinstance(result, str), "Should return a string"
    assert "readfile" in result.lower(), "Should find readfile despite whitespace"

    return True


def test_help_case_sensitivity():
    """Test case sensitivity of tool names"""
    # Tool names should be case-sensitive (lowercase)
    result_lower = help.run("readfile")
    result_upper = help.run("READFILE")

    # Upper case should fail
    assert "not found" in result_upper.lower() or "error" in result_upper.lower(), \
        "Should not find READFILE (case-sensitive)"

    # Lower case should succeed
    assert "not found" not in result_lower.lower() or "readfile" in result_lower.lower(), \
        "Should find readfile"

    return True


def test_help_tool_contract():
    """Verify tool implements the required contract"""
    # Test get_delim
    delim = help.get_delim()
    assert delim == "help", f"Expected delimiter 'help', got: {delim}"

    # Test get_tool_info
    info = help.get_tool_info()
    assert isinstance(info, dict), "get_tool_info should return dict"
    assert "name" in info, "Tool info should have 'name'"
    assert "description" in info, "Tool info should have 'description'"
    assert "example" in info, "Tool info should have 'example'"
    assert info["name"] == "help", f"Expected name 'help', got: {info['name']}"

    return True


def test_help_discover_tools_method():
    """Test the _discover_tools internal method"""
    tools = help._discover_tools()

    # Should return a dictionary
    assert isinstance(tools, dict), "Should return a dictionary"

    # Should find multiple tools
    assert len(tools) > 5, f"Should discover multiple tools, found: {len(tools)}"

    # Each tool should have required methods
    for tool_name, tool_class in tools.items():
        assert hasattr(tool_class, 'get_delim'), f"{tool_name} should have get_delim"
        assert hasattr(tool_class, 'get_tool_info'), f"{tool_name} should have get_tool_info"
        assert callable(tool_class.get_delim), f"{tool_name}.get_delim should be callable"
        assert callable(tool_class.get_tool_info), f"{tool_name}.get_tool_info should be callable"

    return True


def test_help_format_tool_list_method():
    """Test the _format_tool_list internal method"""
    # Create a minimal mock tools dict
    mock_tools = {
        'test_tool': type('MockTool', (), {
            'get_delim': lambda: 'test',
            'get_tool_info': lambda: {
                'name': 'test_tool',
                'description': 'A test tool',
                'example': '<test>example</test>'
            }
        })
    }

    result = help._format_tool_list(mock_tools)

    # Should format the list
    assert isinstance(result, str), "Should return a string"
    assert "test_tool" in result.upper() or "test_tool" in result, "Should include tool name"
    assert "A test tool" in result, "Should include description"

    return True


def test_help_empty_tools_dict():
    """Test handling of empty tools dictionary"""
    result = help._format_tool_list({})

    assert isinstance(result, str), "Should return a string"
    assert "no tools" in result.lower(), "Should indicate no tools available"

    return True


def test_help_output_formatting():
    """Test that help output is well-formatted"""
    result = help.run("")

    # Should have consistent line breaks
    lines = result.split('\n')
    assert len(lines) > 5, "Should have multiple lines"

    # Should not have excessive blank lines
    consecutive_blank = 0
    for line in lines:
        if line.strip() == "":
            consecutive_blank += 1
            assert consecutive_blank < 5, "Should not have too many consecutive blank lines"
        else:
            consecutive_blank = 0

    return True


# Test level assignments
test_help_tool_listing.level = 2
test_help_tool_descriptions.level = 2
test_help_specific_tool_readfile.level = 2
test_help_specific_tool_bash.level = 2
test_help_nonexistent_tool.level = 2
test_help_tool_with_detailed_info.level = 2
test_help_whitespace_handling.level = 2
test_help_case_sensitivity.level = 2
test_help_tool_contract.level = 2
test_help_discover_tools_method.level = 2
test_help_format_tool_list_method.level = 3
test_help_empty_tools_dict.level = 3
test_help_output_formatting.level = 3
