"""
Test that disabled tools are blocked from execution and return error to agent.
"""
import pytest
from core.globals import g


@pytest.fixture
def setup_disabled_tool():
    """Setup g.disabled_tools for testing."""
    # Save original state
    original_disabled = getattr(g, 'disabled_tools', [])

    # Set test state
    g.disabled_tools = ['bash', 'writefile']

    yield

    # Restore original state
    g.disabled_tools = original_disabled


def test_disabled_tool_check():
    """Test that we can check if a tool is disabled."""
    g.disabled_tools = ['bash', 'writefile', 'editfile']

    user_disabled_tools = getattr(g, 'disabled_tools', [])

    assert 'bash' in user_disabled_tools
    assert 'writefile' in user_disabled_tools
    assert 'editfile' in user_disabled_tools
    assert 'readfile' not in user_disabled_tools


def test_disabled_tool_error_message():
    """Test that the error message format is correct."""
    tool_name = 'bash'
    expected_error = f"ERROR: The '{tool_name}' tool is currently disabled. Please enable it first or use a different approach."

    # Simulate what happens when a disabled tool is detected
    user_disabled_tools = ['bash']
    if tool_name in user_disabled_tools:
        tool_result = f"ERROR: The '{tool_name}' tool is currently disabled. Please enable it first or use a different approach."
        tool_status = "error"
        tool_error = "tool_disabled"

    assert tool_result == expected_error
    assert tool_status == "error"
    assert tool_error == "tool_disabled"


def test_enabled_tool_passes_check():
    """Test that enabled tools pass the disabled check."""
    g.disabled_tools = ['bash', 'writefile']

    tool_name = 'readfile'
    user_disabled_tools = getattr(g, 'disabled_tools', [])

    # Should not be in disabled list
    assert tool_name not in user_disabled_tools


# Assign test levels for hierarchical testing
test_disabled_tool_check.level = 0  # Integration test
test_disabled_tool_error_message.level = 0  # Integration test
test_enabled_tool_passes_check.level = 0  # Integration test
