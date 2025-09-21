#!/usr/bin/env python3
"""
Test suite for XML tool routing and validation.

This module tests the XML validation and tool routing functionality
that prevents malformed XML from causing incorrect tool execution.
"""
import os
import sys
from unittest.mock import Mock
import asyncio

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from core.globals import g
from core.llm_router import StreamInterruptedException
from main import create_interruption_callback


class TestXMLValidation:
    """Test XML validation functionality for tool routing."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Mock tool classes for testing
        self.mock_tools = {
            'grepsearch': type('MockSearchTool', (), {
                'get_delim': lambda: 'search',
                'run': lambda content: f"Search executed with: {content}"
            })(),
            'editfile': type('MockEditTool', (), {
                'get_delim': lambda: 'editfile',
                'run': lambda content: f"Edit executed with: {content}"
            })(),
            'bash': type('MockBashTool', (), {
                'get_delim': lambda: 'bash',
                'run': lambda content: f"Bash executed with: {content}"
            })()
        }

        # Set up global state
        g.tool_classes = self.mock_tools
        g.DEBUG_MODE = False

    def test_valid_xml_blocks(self):
        """Test that valid, complete XML blocks are properly detected."""
        test_cases = [
            "<grepsearch><pattern>test</pattern></grepsearch>",
            "<editfile><filepath>test.py</filepath><grepsearch>old</grepsearch><replace>new</replace></editfile>",
            "<bash>echo hello</bash>",
        ]

        for xml_block in test_cases:
            response_buffer = [""]
            painter = Mock()
            callback = create_interruption_callback(response_buffer, painter)

            # Simulate streaming the complete XML block
            try:
                asyncio.run(self._simulate_stream(callback, xml_block))
                assert False, f"Expected StreamInterruptedException for valid XML: {xml_block}"
            except StreamInterruptedException as e:
                assert e.response == xml_block
            assert response_buffer[0] == xml_block

    def test_malformed_xml_rejection(self):
        """Test that malformed XML blocks are rejected and don't interrupt prematurely."""
        malformed_cases = [
            "<grepsearch>for (int i = 0;i<5;i++) {</grepsearch><context>",  # The actual bug case
            "<editfile><filepath>test.py</filepath><grepsearch>incomplete",
            "<bash>echo hello<nested>",
            "<grepsearch><pattern>test</grepsearch>",  # Wrong closing tag
        ]

        for malformed_xml in malformed_cases:
            response_buffer = [""]
            painter = Mock()
            callback = create_interruption_callback(response_buffer, painter)

            # Should NOT raise StreamInterruptedException for malformed XML
            try:
                asyncio.run(self._simulate_stream(callback, malformed_xml))
                # If we get here, no interruption occurred (which is good for malformed XML)
                assert response_buffer[0] == malformed_xml
            except StreamInterruptedException:
                # If interruption occurred, it should only be for valid, complete blocks
                assert False, f"Malformed XML should not trigger interruption: {malformed_xml}"

    def test_nested_xml_structures(self):
        """Test handling of nested XML structures."""
        nested_cases = [
            "<editfile><filepath>test.py</filepath><grepsearch><inner>nested</inner></grepsearch><replace>new</replace></editfile>",
            "<bash>echo '<tag>content</tag>'</bash>",
        ]

        for xml_block in nested_cases:
            response_buffer = [""]
            painter = Mock()
            callback = create_interruption_callback(response_buffer, painter)

            try:
                asyncio.run(self._simulate_stream(callback, xml_block))
                assert False, f"Expected StreamInterruptedException for nested XML: {xml_block}"
            except StreamInterruptedException as e:
                assert e.response == xml_block

    def test_partial_xml_no_interruption(self):
        """Test that partial XML blocks don't trigger premature interruption."""
        partial_cases = [
            "<grepsearch><pattern>test",  # Incomplete opening
            "<editfile><filepath>test.py</filepath>",  # Partial content
            "<bash>echo hello",  # No closing tag
        ]

        for partial_xml in partial_cases:
            response_buffer = [""]
            painter = Mock()
            callback = create_interruption_callback(response_buffer, painter)

            # Should not raise exception for partial XML
            asyncio.run(self._simulate_stream(callback, partial_xml))
            assert response_buffer[0] == partial_xml

    def test_xml_with_special_characters(self):
        """Test XML blocks containing special characters and code."""
        special_cases = [
            "<grepsearch><pattern>for (int i = 0; i < 5; i++)</pattern></grepsearch>",
            "<editfile><filepath>test.py</filepath><grepsearch>if x > 0 && y < 10:</grepsearch><replace>if 0 < x and y < 10:</replace></editfile>",
            "<bash>grep -E '^[0-9]+$' file.txt</bash>",
        ]

        for xml_block in special_cases:
            response_buffer = [""]
            painter = Mock()
            callback = create_interruption_callback(response_buffer, painter)

            try:
                asyncio.run(self._simulate_stream(callback, xml_block))
                assert False, f"Expected StreamInterruptedException for nested XML: {xml_block}"
            except StreamInterruptedException as e:
                assert e.response == xml_block

    def test_multiple_tool_blocks(self):
        """Test handling of multiple consecutive tool blocks."""
        multi_block_text = """<grepsearch><pattern>test</pattern></grepsearch>

Some text in between.

<bash>echo "done"</bash>"""

        response_buffer = [""]
        painter = Mock()
        callback = create_interruption_callback(response_buffer, painter)

        # Should interrupt on the first complete block
        try:
            asyncio.run(self._simulate_stream(callback, multi_block_text))
            assert False, "Expected StreamInterruptedException for multi-block text"
        except StreamInterruptedException as e:
            # Should interrupt on the search block
            assert "<grepsearch><pattern>test</pattern></grepsearch>" in e.response

    def test_tool_regex_pattern_building(self):
        """Test that the tool delimiter regex pattern is built correctly."""
        # Test with different tool configurations
        test_tool_configs = [
            {'grepsearch': Mock(), 'editfile': Mock(), 'bash': Mock()},
            {'python': Mock(), 'readfile': Mock()},
            {},  # Empty tools
        ]

        for tools in test_tool_configs:
            g.tool_classes = tools

            response_buffer = [""]
            painter = Mock()
            callback = create_interruption_callback(response_buffer, painter)

            # The callback should be created without errors
            assert callable(callback)

    def test_concurrent_callback_execution(self):
        """Test callback execution with asyncio locks for concurrent safety."""
        response_buffer = [""]
        painter = Mock()
        lock = asyncio.Lock()
        callback = create_interruption_callback(response_buffer, painter, lock)

        xml_block = "<grepsearch><pattern>test</pattern></grepsearch>"

        try:
            asyncio.run(self._simulate_stream(callback, xml_block))
            assert False, f"Expected StreamInterruptedException for concurrent test: {xml_block}"
        except StreamInterruptedException as e:
            assert e.response == xml_block

    def test_performance_impact(self):
        """Test that XML validation doesn't significantly impact performance."""
        import time

        response_buffer = [""]
        painter = Mock()
        callback = create_interruption_callback(response_buffer, painter)

        # Test with a large text block containing valid XML at the end
        large_text = "x" * 10000 + "<grepsearch><pattern>test</pattern></grepsearch>"

        start_time = time.time()
        try:
            asyncio.run(self._simulate_stream(callback, large_text))
            assert False, "Expected StreamInterruptedException for performance test"
        except StreamInterruptedException:
            pass  # Expected
        end_time = time.time()

        # Should complete within reasonable time (less than 1 second for this test)
        assert end_time - start_time < 1.0

    async def _simulate_stream(self, callback, text):
        """Helper method to simulate character-by-character streaming."""
        for char in text:
            await callback(char)


class TestToolRouting:
    """Test tool routing based on XML content."""

    def setup_method(self):
        """Set up test environment for tool routing tests."""
        self.setup_mock_tools()

    def setup_mock_tools(self):
        """Set up mock tools for testing routing."""
        # Mock search tool that expects <pattern> tags
        search_tool = Mock()
        search_tool.get_delim.return_value = 'search'
        search_tool.run.return_value = "Search result"

        # Mock editfile tool that uses <grepsearch> tags
        editfile_tool = Mock()
        editfile_tool.get_delim.return_value = 'editfile'
        editfile_tool.run.return_value = "Edit result"

        g.tool_classes = {
            'grepsearch': search_tool,
            'editfile': editfile_tool
        }

    def test_correct_tool_selection(self):
        """Test that tools are selected based on their XML tag structure."""
        # This would test the actual tool execution after interruption
        # For now, we verify that the interruption mechanism works correctly

        # Case 1: Valid search tool XML
        search_xml = "<grepsearch><pattern>test pattern</pattern></grepsearch>"
        response_buffer = [""]
        painter = Mock()
        callback = create_interruption_callback(response_buffer, painter)

        with pytest.raises(StreamInterruptedException) as exc_info:
            asyncio.run(self._simulate_stream(callback, search_xml))

        assert exc_info.value.response == search_xml

        # Case 2: Valid editfile tool XML
        edit_xml = "<editfile><filepath>test.py</filepath><grepsearch>old</grepsearch><replace>new</replace></editfile>"
        response_buffer = [""]
        callback = create_interruption_callback(response_buffer, painter)

        with pytest.raises(StreamInterruptedException) as exc_info:
            asyncio.run(self._simulate_stream(callback, edit_xml))

        assert exc_info.value.response == edit_xml

    def test_malformed_xml_no_routing(self):
        """Test that malformed XML doesn't trigger incorrect tool routing."""
        # The problematic case from the original issue
        malformed_xml = "<grepsearch>for (int i = 0;i<5;i++) {</grepsearch><context>"

        response_buffer = [""]
        painter = Mock()
        callback = create_interruption_callback(response_buffer, painter)

        # This should NOT trigger an interruption because the XML is malformed
        try:
            asyncio.run(self._simulate_stream(callback, malformed_xml))
            # If we get here, no interruption occurred (which is correct)
            assert response_buffer[0] == malformed_xml
        except StreamInterruptedException:
            pytest.fail("Malformed XML should not trigger tool routing")

    async def _simulate_stream(self, callback, text):
        """Helper method to simulate character-by-character streaming."""
        for char in text:
            await callback(char)


class TestRegressionCases:
    """Test cases to prevent regressions in existing functionality."""

    def setup_method(self):
        """Set up test environment for regression tests."""
        g.tool_classes = {
            'python': Mock(get_delim=lambda: 'python'),
            'bash': Mock(get_delim=lambda: 'bash'),
            'grepsearch': Mock(get_delim=lambda: 'search'),
            'editfile': Mock(get_delim=lambda: 'editfile'),
        }
        g.DEBUG_MODE = False

    def test_existing_valid_xml_still_works(self):
        """Ensure that existing valid XML blocks continue to work correctly."""
        existing_cases = [
            "<python>print('hello world')</python>",
            "<bash>ls -la</bash>",
            "<grepsearch><pattern>function.*main</pattern></grepsearch>",
            "<editfile><filepath>main.py</filepath><grepsearch>old_code</grepsearch><replace>new_code</replace></editfile>",
        ]

        for xml_block in existing_cases:
            response_buffer = [""]
            painter = Mock()
            callback = create_interruption_callback(response_buffer, painter)

            try:
                asyncio.run(self._simulate_stream(callback, xml_block))
                assert False, f"Expected StreamInterruptedException for nested XML: {xml_block}"
            except StreamInterruptedException as e:
                assert e.response == xml_block

    def test_original_bug_case_fixed(self):
        """Test that the original bug case is now handled correctly."""
        # The exact case from the original issue
        problematic_case = "<grepsearch>for (int i = 0;i<5;i++) {</grepsearch><context>"

        response_buffer = [""]
        painter = Mock()
        callback = create_interruption_callback(response_buffer, painter)

        # This should NOT raise an exception because the XML is malformed
        try:
            asyncio.run(self._simulate_stream(callback, problematic_case))
            # Success - no premature interruption
            assert response_buffer[0] == problematic_case
        except StreamInterruptedException:
            assert False, "The original bug case should be fixed - malformed XML should not interrupt"

    async def _simulate_stream(self, callback, text):
        """Helper method to simulate character-by-character streaming."""
        for char in text:
            await callback(char)


# Simple test runner for manual testing
def run_all_tests():
    """Run all test methods manually."""
    print("🧪 Running XML Tool Routing Tests...")

    # Instantiate test classes
    xml_test = TestXMLValidation()
    routing_test = TestToolRouting()
    regression_test = TestRegressionCases()

    tests = [
        ("XML Validation Tests", [
            xml_test.test_valid_xml_blocks,
            xml_test.test_malformed_xml_rejection,
            xml_test.test_nested_xml_structures,
            xml_test.test_partial_xml_no_interruption,
            xml_test.test_xml_with_special_characters,
            xml_test.test_multiple_tool_blocks,
            xml_test.test_tool_regex_pattern_building,
            xml_test.test_concurrent_callback_execution,
            xml_test.test_performance_impact,
        ]),
        ("Tool Routing Tests", [
            routing_test.test_correct_tool_selection,
            routing_test.test_malformed_xml_no_routing,
        ]),
        ("Regression Tests", [
            regression_test.test_existing_valid_xml_still_works,
            regression_test.test_original_bug_case_fixed,
        ])
    ]

    total_tests = 0
    passed_tests = 0

    for category, test_methods in tests:
        print(f"\n📋 {category}")
        for test_method in test_methods:
            total_tests += 1
            try:
                # Get the instance that test_method is bound to
                test_instance = test_method.__self__
                test_instance.setup_method()
                test_method()
                print(f"  ✅ {test_method.__name__}")
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {test_method.__name__}: {str(e)}")

    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)