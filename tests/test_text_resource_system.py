"""
Test suite for the Text Resource System

This module tests the TextResourceManager, searchresource tool, and integration
with taskmanager and readfile tools for handling large text content.
"""
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.utils.text_resource_manager import TextResourceManager
from tools.searchresource import searchresource

# Backward compatibility alias for tests
searchtext = searchresource


class TestTextResourceManager(unittest.TestCase):
    """Test the TextResourceManager functionality."""

    def setUp(self):
        """Set up test environment."""
        # Clear any existing resources using lazy access
        resource_manager = TextResourceManager._get_manager()
        resource_manager.clear_all_resources()

    def tearDown(self):
        """Clean up after tests."""
        resource_manager = TextResourceManager._get_manager()
        resource_manager.clear_all_resources()

    def test_singleton_pattern(self):
        """Test that TextResourceManager is a singleton."""
        manager1 = TextResourceManager._get_manager()
        manager2 = TextResourceManager._get_manager()
        self.assertIs(manager1, manager2)

    def test_create_resource_basic(self):
        """Test basic resource creation."""
        content = "Line 1\nLine 2\nLine 3"
        resource_manager = TextResourceManager._get_manager()
        resource_id = resource_manager.create_resource(
            content=content,
            source="test_file.txt",
            source_type="file"
        )

        self.assertIsInstance(resource_id, str)
        self.assertTrue(len(resource_id) > 0)

        # Verify resource exists
        metadata = resource_manager.get_metadata(resource_id)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['source'], "test_file.txt")
        self.assertEqual(metadata['source_type'], "file")
        self.assertEqual(metadata['total_lines'], 3)
        self.assertEqual(metadata['total_chars'], len(content))

    def test_create_resource_custom_id(self):
        """Test resource creation with custom ID."""
        content = "Custom content"
        custom_id = "my_custom_id"
        resource_manager = TextResourceManager._get_manager()

        resource_id = resource_manager.create_resource(
            content=content,
            source="test.txt",
            source_type="file",
            resource_id=custom_id
        )

        self.assertEqual(resource_id, custom_id)

        # Verify resource exists with custom ID
        metadata = resource_manager.get_metadata(custom_id)
        self.assertIsNotNone(metadata)

    def test_create_resource_empty_content(self):
        """Test that creating resource with empty content raises error."""
        resource_manager = TextResourceManager._get_manager()
        with self.assertRaises(ValueError):
            resource_manager.create_resource(
                content="",
                source="empty.txt",
                source_type="file"
            )

    def test_search_resource(self):
        """Test searching content in a resource."""
        content = """Line 1: Normal line
Line 2: ERROR occurred here
Line 3: Another normal line
Line 4: FATAL error happened
Line 5: Final line"""

        resource_manager = TextResourceManager._get_manager()
        resource_id = resource_manager.create_resource(
            content=content,
            source="test.log",
            source_type="file"
        )

        # Search for error patterns
        results = resource_manager.search_resource(
            resource_id=resource_id,
            pattern="error|ERROR",
            context_lines=1,
            max_results=10
        )

        self.assertNotIn("error", results)  # Should not have error key
        self.assertEqual(results["total_matches"], 2)
        self.assertEqual(len(results["matches"]), 2)

        # Check first match
        first_match = results["matches"][0]
        self.assertEqual(first_match["line_number"], 2)
        self.assertIn("ERROR occurred here", first_match["matched_line"])

        # Check context
        self.assertEqual(len(first_match["context"]), 3)  # 1 before + match + 1 after

    def test_search_resource_invalid_regex(self):
        """Test search with invalid regex pattern."""
        content = "Some content"
        resource_manager = TextResourceManager._get_manager()
        resource_id = resource_manager.create_resource(
            content=content,
            source="test.txt",
            source_type="file"
        )

        # Invalid regex pattern
        results = resource_manager.search_resource(
            resource_id=resource_id,
            pattern="[invalid",  # Unclosed bracket
            context_lines=0,
            max_results=10
        )

        # Should return error in matches list
        self.assertEqual(len(results["matches"]), 1)
        self.assertIn("error", results["matches"][0])

    def test_get_line_range(self):
        """Test retrieving specific line ranges."""
        content = "\n".join([f"Line {i}" for i in range(1, 11)])  # 10 lines

        resource_manager = TextResourceManager._get_manager()
        resource_id = resource_manager.create_resource(
            content=content,
            source="test.txt",
            source_type="file"
        )

        # Get lines 3-5
        result = resource_manager.get_line_range(
            resource_id=resource_id,
            start_line=3,
            end_line=5
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["start_line"], 3)
        self.assertEqual(result["end_line"], 5)
        self.assertEqual(len(result["lines"]), 3)
        self.assertEqual(result["lines"][0]["content"], "Line 3")
        self.assertEqual(result["lines"][2]["content"], "Line 5")

    def test_get_line_range_single_line(self):
        """Test retrieving a single line."""
        content = "Line 1\nLine 2\nLine 3"
        resource_manager = TextResourceManager._get_manager()
        resource_id = resource_manager.create_resource(
            content=content,
            source="test.txt",
            source_type="file"
        )

        result = resource_manager.get_line_range(
            resource_id=resource_id,
            start_line=2
        )

        self.assertEqual(result["start_line"], 2)
        self.assertEqual(result["end_line"], 2)
        self.assertEqual(len(result["lines"]), 1)
        self.assertEqual(result["lines"][0]["content"], "Line 2")

    def test_get_line_range_out_of_bounds(self):
        """Test line range with out-of-bounds values."""
        content = "Line 1\nLine 2\nLine 3"
        resource_manager = TextResourceManager._get_manager()
        resource_id = resource_manager.create_resource(
            content=content,
            source="test.txt",
            source_type="file"
        )

        # Start line too high
        result = resource_manager.get_line_range(
            resource_id=resource_id,
            start_line=10
        )

        self.assertIn("error", result)
        self.assertIn("out of range", result["error"])

    def test_list_resources(self):
        """Test listing all resources."""
        resource_manager = TextResourceManager._get_manager()
        # Create multiple resources
        resource1 = resource_manager.create_resource("Content 1", "file1.txt", "file")
        resource2 = resource_manager.create_resource("Content 2", "proc_123", "process")

        resources = resource_manager.list_resources()

        self.assertEqual(len(resources), 2)

        # Should be sorted by last_accessed (most recent first)
        resource_ids = [r["resource_id"] for r in resources]
        self.assertIn(resource1, resource_ids)
        self.assertIn(resource2, resource_ids)

    def test_remove_resource(self):
        """Test removing a resource."""
        resource_manager = TextResourceManager._get_manager()
        resource_id = resource_manager.create_resource("Content", "test.txt", "file")

        # Verify it exists
        self.assertIsNotNone(resource_manager.get_metadata(resource_id))

        # Remove it
        success = resource_manager.remove_resource(resource_id)
        self.assertTrue(success)

        # Verify it's gone
        self.assertIsNone(resource_manager.get_metadata(resource_id))

    def test_get_stats(self):
        """Test getting manager statistics."""
        resource_manager = TextResourceManager._get_manager()
        # Create some resources
        resource_manager.create_resource("Short content", "test1.txt", "file")
        resource_manager.create_resource("This is longer content\nwith multiple lines", "test2.txt", "file")

        stats = resource_manager.get_stats()

        self.assertEqual(stats["total_resources"], 2)
        self.assertGreater(stats["total_lines"], 0)
        self.assertGreater(stats["total_chars"], 0)
        self.assertIsInstance(stats["max_resources"], int)
        self.assertIsInstance(stats["max_age_seconds"], int)


class TestSearchResourceTool(unittest.TestCase):
    """Test the searchresource tool functionality."""

    def setUp(self):
        """Set up test environment."""
        resource_manager = TextResourceManager._get_manager()
        resource_manager.clear_all_resources()

        # Create test resources
        self.test_content = """Line 1: Application started
Line 2: Loading configuration
Line 3: ERROR: Database connection failed
Line 4: Retrying database connection
Line 5: SUCCESS: Database connected
Line 6: Processing requests
Line 7: FATAL: Out of memory
Line 8: Application crashed"""

        self.resource_id = resource_manager.create_resource(
            content=self.test_content,
            source="test.log",
            source_type="file"
        )

    def tearDown(self):
        """Clean up after tests."""
        resource_manager = TextResourceManager._get_manager()
        resource_manager.clear_all_resources()

    def test_get_delim(self):
        """Test tool delimiter."""
        self.assertEqual(searchresource.get_delim(), "searchresource")

    def test_get_tool_info(self):
        """Test tool info structure."""
        info = searchresource.get_tool_info()
        self.assertIn("name", info)
        self.assertIn("description", info)
        self.assertIn("example", info)
        self.assertEqual(info["name"], "searchresource")

    def test_list_resources(self):
        """Test listing resources."""
        result = searchtext.run("<list_resources/>")

        self.assertIn("Available Text Resources", result)
        self.assertIn(self.resource_id, result)
        self.assertIn("test.log", result)

    def test_search_pattern(self):
        """Test pattern searching."""
        xml_content = f"""<resource_id>{self.resource_id}</resource_id>
                          <pattern>ERROR|FATAL</pattern>
                          <context_lines>1</context_lines>"""

        result = searchtext.run(xml_content)

        self.assertIn("Search Results", result)
        self.assertIn("Found 2 matches", result)
        self.assertIn("Line 3", result)  # ERROR line
        self.assertIn("Line 7", result)  # FATAL line
        self.assertIn("Database connection failed", result)
        self.assertIn("Out of memory", result)

    def test_search_no_matches(self):
        """Test search with no matches."""
        xml_content = f"""<resource_id>{self.resource_id}</resource_id>
                          <pattern>NONEXISTENT</pattern>"""

        result = searchtext.run(xml_content)

        self.assertIn("No matches found", result)
        self.assertIn("NONEXISTENT", result)

    def test_line_range_query(self):
        """Test line range retrieval."""
        xml_content = f"""<resource_id>{self.resource_id}</resource_id>
                          <lines>3-5</lines>"""

        result = searchtext.run(xml_content)

        self.assertIn("Lines 3-5", result)
        self.assertIn("ERROR: Database connection failed", result)
        self.assertIn("SUCCESS: Database connected", result)
        self.assertIn("   3→", result)  # Line number formatting
        self.assertIn("   5→", result)

    def test_single_line_query(self):
        """Test single line retrieval."""
        xml_content = f"""<resource_id>{self.resource_id}</resource_id>
                          <lines>7</lines>"""

        result = searchtext.run(xml_content)

        self.assertIn("Lines 7-7", result)
        self.assertIn("FATAL: Out of memory", result)
        self.assertIn("   7→", result)

    def test_resource_not_found(self):
        """Test searching non-existent resource."""
        xml_content = """<resource_id>nonexistent_id</resource_id>
                         <pattern>test</pattern>"""

        result = searchtext.run(xml_content)

        self.assertIn("Error", result)
        self.assertIn("not found", result)
        self.assertIn("nonexistent_id", result)

    def test_no_resource_id(self):
        """Test command without resource_id."""
        xml_content = "<pattern>test</pattern>"

        result = searchtext.run(xml_content)

        self.assertIn("Error", result)
        self.assertIn("No resource_id specified", result)

    def test_resource_info(self):
        """Test getting resource info without specific operation."""
        xml_content = f"<resource_id>{self.resource_id}</resource_id>"

        result = searchtext.run(xml_content)

        self.assertIn("Resource Info", result)
        self.assertIn(self.resource_id, result)
        self.assertIn("test.log", result)
        self.assertIn("file", result)
        self.assertIn("8", result)  # Line count

    def test_invalid_line_range(self):
        """Test invalid line range specification."""
        xml_content = f"""<resource_id>{self.resource_id}</resource_id>
                          <lines>invalid-range</lines>"""

        result = searchtext.run(xml_content)

        self.assertIn("Error", result)
        self.assertIn("Invalid line specification", result)


class TestIntegrationWithReadfile(unittest.TestCase):
    """Test integration between readfile and text resource system."""

    def setUp(self):
        """Set up test environment."""
        resource_manager = TextResourceManager._get_manager()
        resource_manager.clear_all_resources()

    def tearDown(self):
        """Clean up after tests."""
        resource_manager = TextResourceManager._get_manager()
        resource_manager.clear_all_resources()

    def test_large_file_resource_creation(self):
        """Test that readfile creates resources for large files."""
        # Create a temporary large file
        large_content = "\n".join([f"Line {i}: This is line number {i} with some content" for i in range(1, 1001)])  # 1000 lines

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write(large_content)
            temp_file_path = temp_file.name

        try:
            # Import readfile here to avoid circular imports during module loading
            from tools.readfile import readfile

            # Read the large file
            result = readfile.run(temp_file_path)

            # Should create a resource instead of showing truncated content
            if len(large_content) > readfile.CHUNK_LIMIT:
                self.assertIn("loaded into searchable resource", result)
                self.assertIn("Resource ID:", result)
                self.assertIn("searchresource", result)

        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


class TestIntegrationWithTaskManager(unittest.TestCase):
    """Test integration between taskmanager and text resource system."""

    def setUp(self):
        """Set up test environment."""
        resource_manager = TextResourceManager._get_manager()
        resource_manager.clear_all_resources()

    def tearDown(self):
        """Clean up after tests."""
        resource_manager = TextResourceManager._get_manager()
        resource_manager.clear_all_resources()

    @patch('tools.taskmanager.bash._background_processes')
    def test_large_output_resource_creation(self, mock_processes):
        """Test that taskmanager creates resources for large outputs."""
        from tools.taskmanager import taskmanager
        import io

        # Mock a process with large output
        large_output = "\n".join([f"Output line {i}" for i in range(1, 201)])  # 200 lines
        mock_buffer = io.StringIO(large_output)

        mock_proc_info = {
            'source': 'bash',
            'child': MagicMock(),
            'command': 'test command',
            'status': 'finished',
            'output_buffer': mock_buffer
        }

        mock_processes.get.return_value = mock_proc_info

        # Test _read_stdout with large output
        result = taskmanager._read_stdout('test_pid')

        # Should create a resource for large output
        if len(large_output) > 4096:  # Default max_chars
            self.assertIn("loaded into searchable resource", result)
            self.assertIn("Resource ID:", result)
            self.assertIn("searchresource", result)

    def test_stats_and_cleanup(self):
        """Test resource manager stats and cleanup functionality."""
        resource_manager = TextResourceManager._get_manager()
        # Create several resources to test stats
        for i in range(5):
            resource_manager.create_resource(
                content=f"Content {i}\nLine 2\nLine 3",
                source=f"source_{i}.txt",
                source_type="file"
            )

        # Check stats
        stats = resource_manager.get_stats()
        self.assertEqual(stats["total_resources"], 5)
        self.assertEqual(stats["total_lines"], 15)  # 3 lines × 5 resources

        # Test cleanup
        resource_manager.clear_all_resources()
        stats = resource_manager.get_stats()
        self.assertEqual(stats["total_resources"], 0)


def test_level_0():
    """Level 0 tests - Critical functionality."""
    suite = unittest.TestSuite()

    # Core TextResourceManager functionality
    suite.addTest(TestTextResourceManager('test_singleton_pattern'))
    suite.addTest(TestTextResourceManager('test_create_resource_basic'))
    suite.addTest(TestTextResourceManager('test_search_resource'))
    suite.addTest(TestTextResourceManager('test_get_line_range'))

    # Core searchresource tool functionality
    suite.addTest(TestSearchTextTool('test_search_pattern'))
    suite.addTest(TestSearchTextTool('test_line_range_query'))

    return suite


def test_level_1():
    """Level 1 tests - Important functionality."""
    suite = unittest.TestSuite()

    # Extended TextResourceManager functionality
    suite.addTest(TestTextResourceManager('test_create_resource_custom_id'))
    suite.addTest(TestTextResourceManager('test_list_resources'))
    suite.addTest(TestTextResourceManager('test_remove_resource'))

    # Extended searchresource functionality
    suite.addTest(TestSearchTextTool('test_list_resources'))
    suite.addTest(TestSearchTextTool('test_resource_info'))

    return suite


def test_level_2():
    """Level 2 tests - Error handling and edge cases."""
    suite = unittest.TestSuite()

    # Error handling
    suite.addTest(TestTextResourceManager('test_create_resource_empty_content'))
    suite.addTest(TestTextResourceManager('test_search_resource_invalid_regex'))
    suite.addTest(TestTextResourceManager('test_get_line_range_out_of_bounds'))

    # Tool error handling
    suite.addTest(TestSearchTextTool('test_resource_not_found'))
    suite.addTest(TestSearchTextTool('test_no_resource_id'))
    suite.addTest(TestSearchTextTool('test_invalid_line_range'))

    return suite


def test_level_3():
    """Level 3 tests - Integration and advanced features."""
    suite = unittest.TestSuite()

    # Integration tests
    suite.addTest(TestIntegrationWithReadfile('test_large_file_resource_creation'))
    suite.addTest(TestIntegrationWithTaskManager('test_large_output_resource_creation'))
    suite.addTest(TestIntegrationWithTaskManager('test_stats_and_cleanup'))

    return suite


def test_level_4():
    """Level 4 tests - Performance and edge cases."""
    suite = unittest.TestSuite()

    # Additional functionality
    suite.addTest(TestTextResourceManager('test_get_stats'))
    suite.addTest(TestSearchTextTool('test_search_no_matches'))
    suite.addTest(TestSearchTextTool('test_single_line_query'))

    return suite


if __name__ == '__main__':
    # Run all tests when executed directly
    unittest.main(verbosity=2)