"""
WriteFile Tool Tests

Tests for the writefile tool including file creation,
overwriting, and error handling capabilities.
"""

import os
import sys
import tempfile
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.writefile import writefile


def test_writefile_basic_creation():
    """Test basic file creation functionality"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "new_file.txt")

        # Test file creation
        content = "Hello, World!\nThis is a test file."
        xml_input = f"""<filepath>{test_file}</filepath>
<content>{content}</content>"""

        result = writefile.run(xml_input)

        # Verify success message
        assert "Successfully wrote" in result and str(len(content)) in result, f"Expected success message, got: {result}"
        assert test_file in result, "Should mention the file path"

        # Verify file was created and has correct content
        assert os.path.exists(test_file), "File should be created"

        with open(test_file, encoding='utf-8') as f:
            file_content = f.read()

        assert file_content == content, f"Content mismatch. Expected: {repr(content)}, Got: {repr(file_content)}"

        return True


def test_writefile_overwrite_existing():
    """Test overwriting existing files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "existing_file.txt")

        # Create initial file
        initial_content = "Original content"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(initial_content)

        # Verify initial content
        assert os.path.exists(test_file), "Initial file should exist"

        # Overwrite with new content
        new_content = "New content that replaces the old"
        xml_input = f"""<filepath>{test_file}</filepath>
<content>{new_content}</content>"""

        result = writefile.run(xml_input)

        # Verify success
        assert "Successfully wrote" in result, f"Expected success, got: {result}"

        # Verify file was overwritten
        with open(test_file, encoding='utf-8') as f:
            file_content = f.read()

        assert file_content == new_content, f"File should be overwritten. Expected: {repr(new_content)}, Got: {repr(file_content)}"
        assert file_content != initial_content, "Content should be different from original"

        return True


def test_writefile_directory_creation():
    """Test automatic parent directory creation"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested path that doesn't exist
        nested_file = os.path.join(temp_dir, "subdir", "nested", "file.txt")

        content = "Content in nested directory"
        xml_input = f"""<filepath>{nested_file}</filepath>
<content>{content}</content>"""

        result = writefile.run(xml_input)

        # Verify success
        assert "Successfully wrote" in result, f"Expected success with directory creation, got: {result}"

        # Verify directories were created
        assert os.path.exists(os.path.dirname(nested_file)), "Parent directories should be created"
        assert os.path.exists(nested_file), "File should be created"

        # Verify content
        with open(nested_file, encoding='utf-8') as f:
            file_content = f.read()

        assert file_content == content, f"Content mismatch in nested file. Expected: {repr(content)}, Got: {repr(file_content)}"

        return True


def test_writefile_special_characters():
    """Test writing files with special characters and encoding"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "special_chars.txt")

        # Content with various special characters
        content = """Special characters test:
- Unicode: ñáéíóú ☀️ 🚀
- Quotes: "double" 'single' "smart" 'smart'
- Symbols: @#$%^&*()_+-=[]{}|;:,.<>?
- Newlines and tabs:
\tTabbed line
\tAnother tab

Empty lines above
"""

        xml_input = f"""<filepath>{test_file}</filepath>
<content>{content}</content>"""

        result = writefile.run(xml_input)

        # Verify success
        assert "Successfully wrote" in result, f"Expected success with special characters, got: {result}"

        # Verify content preservation
        with open(test_file, encoding='utf-8') as f:
            file_content = f.read()

        assert file_content == content, f"Special characters not preserved. Expected: {repr(content)}, Got: {repr(file_content)}"

        return True


def test_writefile_empty_content():
    """Test writing empty files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "empty_file.txt")

        xml_input = f"""<filepath>{test_file}</filepath>
<content></content>"""

        result = writefile.run(xml_input)

        # Verify success
        assert "Successfully wrote 0 bytes" in result, f"Should handle empty content, got: {result}"

        # Verify empty file created
        assert os.path.exists(test_file), "Empty file should be created"

        with open(test_file, encoding='utf-8') as f:
            content = f.read()

        assert content == "", f"File should be empty, got: {repr(content)}"

        return True


def test_writefile_error_cases():
    """Test various error conditions"""
    with tempfile.TemporaryDirectory() as temp_dir:

        # Test missing filepath tag
        result = writefile.run("<content>test content</content>")
        assert result.startswith("Error: Invalid format"), "Should fail without filepath tag"

        # Test missing content tag
        result = writefile.run(f"<filepath>{temp_dir}/test.txt</filepath>")
        assert result.startswith("Error: Invalid format"), "Should fail without content tag"

        # Test empty filepath
        result = writefile.run("<filepath></filepath><content>test</content>")
        assert result.startswith("Error: The <filepath> tag cannot be empty"), "Should fail with empty filepath"

        # Test malformed XML
        result = writefile.run("<filepath>test.txt<content>test content</content>")
        assert result.startswith("Error: Invalid format"), "Should fail with malformed XML"

        return True


def test_writefile_relative_paths():
    """Test handling of relative paths"""
    # Change to temporary directory to test relative paths
    original_cwd = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Test relative path
            content = "Relative path test"
            xml_input = f"""<filepath>./relative_file.txt</filepath>
<content>{content}</content>"""

            result = writefile.run(xml_input)

            # Verify success
            assert "Successfully wrote" in result, f"Should handle relative paths, got: {result}"

            # Check file exists at expected location
            expected_path = os.path.join(temp_dir, "relative_file.txt")
            assert os.path.exists(expected_path), "File should exist at resolved path"

            # Verify content
            with open(expected_path, encoding='utf-8') as f:
                file_content = f.read()

            assert file_content == content, f"Content should match. Expected: {repr(content)}, Got: {repr(file_content)}"

    finally:
        os.chdir(original_cwd)

    return True


def test_writefile_large_content():
    """Test writing large files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "large_file.txt")

        # Create large content (about 1MB)
        line = "This is line number with some content to make it longer.\n"
        large_content = line * 20000  # Approximately 1MB

        xml_input = f"""<filepath>{test_file}</filepath>
<content>{large_content}</content>"""

        result = writefile.run(xml_input)

        # Verify success
        assert "Successfully wrote" in result, f"Should handle large files, got: {result}"
        expected_bytes = str(len(large_content))
        assert expected_bytes in result, f"Should report correct byte count, expected {expected_bytes} in result"

        # Verify file size
        assert os.path.exists(test_file), "Large file should be created"
        file_size = os.path.getsize(test_file)
        assert file_size == len(large_content), f"File size should match content length. Expected: {len(large_content)}, Got: {file_size}"

        # Sample verification (don't read entire file to keep test fast)
        with open(test_file, encoding='utf-8') as f:
            first_line = f.readline()
            assert first_line == line, f"First line should match template. Expected: {repr(line)}, Got: {repr(first_line)}"

        return True


# Test level assignments
test_writefile_basic_creation.level = 2
test_writefile_overwrite_existing.level = 2
test_writefile_directory_creation.level = 2
test_writefile_special_characters.level = 2
test_writefile_empty_content.level = 2
test_writefile_error_cases.level = 2
test_writefile_relative_paths.level = 2
test_writefile_large_content.level = 3