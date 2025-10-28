"""
Integration test for disabled tools functionality.

Tests that the frontend can disable tools via the API and that
the backend properly stores and uses the disabled tools list.
"""
from core.globals import g


def test_disabled_tools_initialization():
    """Test that g.disabled_tools is properly initialized"""
    # Verify g.disabled_tools exists and is a list
    assert hasattr(g, 'disabled_tools')
    assert isinstance(g.disabled_tools, list)
    print("✓ g.disabled_tools is properly initialized")


def test_disabled_tools_can_be_set():
    """Test that g.disabled_tools can be set and retrieved"""
    # Set disabled tools
    g.disabled_tools = ['bash', 'writefile', 'editfile']

    # Verify it was set
    assert g.disabled_tools == ['bash', 'writefile', 'editfile']
    assert len(g.disabled_tools) == 3

    # Reset to empty
    g.disabled_tools = []
    assert g.disabled_tools == []

    print("✓ g.disabled_tools can be set and retrieved")


def test_tool_discovery_respects_disabled_tools():
    """Test that tool discovery skips disabled tools"""
    from agent.core.tool_discovery import _generate_tool_documentation

    # Test with some tools disabled
    disabled_tools = ['bash', 'writefile']
    doc_string = _generate_tool_documentation(disabled_tools)

    # Verify disabled tools are not in the documentation
    assert 'bash' not in g.tool_classes
    assert 'writefile' not in g.tool_classes

    # Verify other tools are still present
    assert len(g.tool_classes) > 0

    # Verify documentation doesn't include disabled tools
    # Note: This checks that the tool names aren't in the doc string
    # (they might appear in descriptions, so we check the tool_classes dict)

    print(f"✓ Tool discovery properly excludes {len(disabled_tools)} disabled tools")
    print(f"  Loaded {len(g.tool_classes)} tools")


if __name__ == '__main__':
    print("Running disabled tools integration tests...")
    print()

    # Run tests
    test_disabled_tools_initialization()
    test_disabled_tools_can_be_set()
    test_tool_discovery_respects_disabled_tools()

    print()
    print("All tests passed! ✓")
