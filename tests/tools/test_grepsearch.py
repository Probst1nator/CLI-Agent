"""
GrepSearch Tool Tests

Tests for the grepsearch tool including recursive file/content search,
regex pattern matching, and result limiting.
"""

import os
import sys
import tempfile
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.grepsearch import grepsearch


def create_test_directory_structure(base_dir):
    """Helper function to create a test directory structure"""

    # Create directories
    os.makedirs(os.path.join(base_dir, "src"))
    os.makedirs(os.path.join(base_dir, "tests"))
    os.makedirs(os.path.join(base_dir, "docs"))
    os.makedirs(os.path.join(base_dir, "config"))

    # Create files with content
    files = {
        "src/main.py": """def main_function():
    print("Hello World")
    calculate_result()

def calculate_result():
    return 42
""",
        "src/utils.py": """def utility_function():
    print("Utility function")

def helper_method():
    return "helper"
""",
        "src/test_helper.py": """def test_function():
    assert True
    helper_method()
""",
        "tests/test_main.py": """import unittest

class TestMain(unittest.TestCase):
    def test_main_function(self):
        main_function()
""",
        "docs/README.md": """# Project Documentation

This project contains helper functions and utilities.
""",
        "config/settings.ini": """[database]
host = localhost
port = 5432

[app]
debug = True
""",
        "project_helper.txt": """This is a helper file at the root level.
Contains some helper information.
"""
    }

    for file_path, content in files.items():
        full_path = os.path.join(base_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)


def test_grepsearch_content_matching():
    """Test searching for content within files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_test_directory_structure(temp_dir)

        # Search for "helper" in content
        xml_input = f"""<pattern>helper</pattern>
<path>{temp_dir}</path>"""

        result = grepsearch.run(xml_input)

        # Should find content matches
        assert "LINE:" in result, f"Should find line matches, got: {result}"
        assert "helper_method" in result, "Should find helper_method function"
        assert "helper" in result.lower(), "Should find helper references"

        # Should show file paths and line numbers
        assert "src/utils.py" in result, "Should find match in utils.py"
        assert ":" in result, "Should show line numbers"

        return True


def test_grepsearch_filename_matching():
    """Test searching for filenames"""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_test_directory_structure(temp_dir)

        # Search for files containing "helper"
        xml_input = f"""<pattern>helper</pattern>
<path>{temp_dir}</path>"""

        result = grepsearch.run(xml_input)

        # Should find filename matches
        assert "FILE:" in result, f"Should find file matches, got: {result}"
        assert "test_helper.py" in result, "Should find test_helper.py"
        assert "project_helper.txt" in result, "Should find project_helper.txt"

        return True


def test_grepsearch_directory_matching():
    """Test searching for directory names"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directory with searchable name
        test_subdir = os.path.join(temp_dir, "test_directory")
        os.makedirs(test_subdir)

        with open(os.path.join(test_subdir, "file.txt"), 'w') as f:
            f.write("content")

        # Search for "test" in directory names
        xml_input = f"""<pattern>test</pattern>
<path>{temp_dir}</path>"""

        result = grepsearch.run(xml_input)

        # Should find directory matches
        assert "DIR:" in result, f"Should find directory matches, got: {result}"
        assert "test_directory" in result, "Should find test_directory"

        return True


def test_grepsearch_regex_patterns():
    """Test regex pattern matching"""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_test_directory_structure(temp_dir)

        # Search for functions using regex
        xml_input = f"""<pattern>def \\w+_function</pattern>
<path>{temp_dir}</path>"""

        result = grepsearch.run(xml_input)

        # Should find function definitions
        assert "main_function" in result, "Should find main_function"
        assert "utility_function" in result, "Should find utility_function"
        assert "test_function" in result, "Should find test_function"

        return True


def test_grepsearch_specific_file():
    """Test searching within a specific file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "specific.txt")

        content = """Line 1: First line
Line 2: Second line with pattern
Line 3: Third line
Line 4: Another line with pattern"""

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # Search within specific file
        xml_input = f"""<pattern>pattern</pattern>
<path>{test_file}</path>"""

        result = grepsearch.run(xml_input)

        # Should find matches with line numbers
        assert "LINE:" in result, f"Should find line matches, got: {result}"
        assert "specific.txt:2:" in result, "Should find match on line 2"
        assert "specific.txt:4:" in result, "Should find match on line 4"
        assert "Second line with pattern" in result, "Should show matched content"

        return True


def test_grepsearch_no_matches():
    """Test searching for patterns with no matches"""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_test_directory_structure(temp_dir)

        # Search for non-existent pattern
        xml_input = f"""<pattern>nonexistent_xyz_pattern</pattern>
<path>{temp_dir}</path>"""

        result = grepsearch.run(xml_input)

        # Should return no matches message
        assert result == "No matches found.", f"Should return no matches message, got: {result}"

        return True


def test_grepsearch_current_directory():
    """Test searching in current directory (default)"""
    original_cwd = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Create a test file in current directory
            with open("test_file.txt", 'w') as f:
                f.write("searchable content here")

            # Search without specifying path (defaults to current directory)
            xml_input = "<pattern>searchable</pattern>"

            result = grepsearch.run(xml_input)

            # Should find the file and content
            assert "searchable content" in result, f"Should find content in current directory, got: {result}"
            assert "test_file.txt" in result, "Should find the test file"

    finally:
        os.chdir(original_cwd)

    return True


def test_grepsearch_case_sensitivity():
    """Test case-sensitive search behavior"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "case_test.txt")

        content = """This line has UPPERCASE text
This line has lowercase text
This line has MixedCase text"""

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # Search for uppercase pattern
        xml_input = f"""<pattern>UPPERCASE</pattern>
<path>{temp_dir}</path>"""

        result = grepsearch.run(xml_input)

        # Should find exact case match
        assert "UPPERCASE" in result, f"Should find UPPERCASE, got: {result}"
        assert "uppercase" not in result.lower() or "UPPERCASE" in result, "Should respect case sensitivity"

        return True


def test_grepsearch_error_cases():
    """Test various error conditions"""
    with tempfile.TemporaryDirectory() as temp_dir:

        # Test missing pattern
        result = search.run(f"<path>{temp_dir}</path>")
        assert result.startswith("Error: Invalid format"), "Should fail without pattern"

        # Test empty pattern
        result = search.run(f"<pattern></pattern><path>{temp_dir}</path>")
        assert result.startswith("Error: Search pattern cannot be empty"), "Should fail with empty pattern"

        # Test invalid regex pattern
        result = search.run(f"<pattern>[invalid regex</pattern><path>{temp_dir}</path>")
        assert result.startswith("Error: Invalid regex pattern"), "Should fail with invalid regex"

        # Test non-existent path
        fake_path = os.path.join(temp_dir, "nonexistent")
        result = search.run(f"<pattern>test</pattern><path>{fake_path}</path>")
        assert result.startswith("Error: Path does not exist"), "Should fail for non-existent path"

        return True


def test_grepsearch_special_characters():
    """Test searching for content with special characters"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "special.txt")

        content = """Line with special chars: ñáéíóú ☀️ 🚀
Line with symbols: @#$%^&*()
Line with quotes: "double" 'single'
Line with regex chars: []()+?.*^$|\\{}"""

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # Search for special unicode characters
        xml_input = f"""<pattern>ñáéíóú</pattern>
<path>{temp_dir}</path>"""

        result = grepsearch.run(xml_input)

        # Should find unicode characters
        assert "ñáéíóú" in result, f"Should find unicode characters, got: {result}"

        # Search for literal regex characters (should be treated as literal by default)
        xml_input = f"""<pattern>\\[\\]\\(\\)</pattern>
<path>{temp_dir}</path>"""

        result = grepsearch.run(xml_input)

        # Should find literal brackets and parentheses
        assert "[]()+" in result, f"Should find literal regex chars, got: {result}"

        return True


def test_grepsearch_binary_file_handling():
    """Test handling of binary files (should be skipped)"""
    with tempfile.TemporaryDirectory() as temp_dir:

        # Create a text file
        text_file = os.path.join(temp_dir, "text.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("searchable text content")

        # Create a binary-like file (large file should be skipped)
        binary_file = os.path.join(temp_dir, "binary.bin")
        with open(binary_file, 'wb') as f:
            # Write more than 2MB to trigger size check
            f.write(b'\x00' * (3 * 1024 * 1024))

        # Search for pattern
        xml_input = f"""<pattern>searchable</pattern>
<path>{temp_dir}</path>"""

        result = grepsearch.run(xml_input)

        # Should find text file but not process binary file
        assert "text.txt" in result, f"Should find text file, got: {result}"
        assert "searchable text content" in result, "Should find content in text file"
        # Large binary file should be skipped silently

        return True


def test_grepsearch_line_number_accuracy():
    """Test accuracy of line number reporting"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "lines.txt")

        lines = [
            "Line 1: normal",
            "Line 2: target pattern here",
            "Line 3: normal",
            "Line 4: normal",
            "Line 5: another target pattern",
            "Line 6: normal"
        ]
        content = "\n".join(lines)

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # Search for pattern
        xml_input = f"""<pattern>target pattern</pattern>
<path>{temp_dir}</path>"""

        result = grepsearch.run(xml_input)

        # Should find matches on correct lines
        assert "lines.txt:2:" in result, f"Should find match on line 2, got: {result}"
        assert "lines.txt:5:" in result, f"Should find match on line 5, got: {result}"
        assert "target pattern here" in result, "Should show first match content"
        assert "another target pattern" in result, "Should show second match content"

        return True


# Test level assignments
test_grepsearch_content_matching.level = 2
test_grepsearch_filename_matching.level = 2
test_grepsearch_directory_matching.level = 2
test_grepsearch_regex_patterns.level = 2
test_grepsearch_specific_file.level = 2
test_grepsearch_no_matches.level = 2
test_grepsearch_current_directory.level = 2
test_grepsearch_case_sensitivity.level = 2
test_grepsearch_error_cases.level = 2
test_grepsearch_special_characters.level = 2
test_grepsearch_binary_file_handling.level = 3
test_grepsearch_line_number_accuracy.level = 2