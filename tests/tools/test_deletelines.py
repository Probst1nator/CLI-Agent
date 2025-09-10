"""
DeleteLines Tool Tests

Tests for the deletelines tool including single line deletion,
range deletion, and error handling capabilities.
"""

import os
import sys
import tempfile
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.deletelines import deletelines


def test_deletelines_single_line():
    """Test deleting a single line"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file with numbered lines
        original_lines = ["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Delete line 3
        xml_input = f"""<filepath>{test_file}</filepath>
<line>3</line>"""
        
        result = deletelines.run(xml_input)
        
        # Verify success message
        assert "Successfully deleted line 3" in result, f"Expected success message, got: {result}"
        assert "3→Line 3" in result, "Should show deleted content with line number"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_lines = ["Line 1", "Line 2", "Line 4", "Line 5"]
        expected_content = "\n".join(expected_lines) + "\n"
        
        assert new_content == expected_content, f"Content mismatch. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


def test_deletelines_range():
    """Test deleting a range of lines"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file with numbered lines
        original_lines = ["Line 1", "Line 2", "Line 3", "Line 4", "Line 5", "Line 6"]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Delete lines 2-4
        xml_input = f"""<filepath>{test_file}</filepath>
<start>2</start>
<end>4</end>"""
        
        result = deletelines.run(xml_input)
        
        # Verify success message
        assert "Successfully deleted lines 2-4" in result, f"Expected success message, got: {result}"
        assert "2→Line 2" in result, "Should show deleted content"
        assert "3→Line 3" in result, "Should show deleted content"
        assert "4→Line 4" in result, "Should show deleted content"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_lines = ["Line 1", "Line 5", "Line 6"]
        expected_content = "\n".join(expected_lines) + "\n"
        
        assert new_content == expected_content, f"Content mismatch. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


def test_deletelines_first_line():
    """Test deleting the first line"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file
        original_lines = ["First line", "Second line", "Third line"]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Delete first line
        xml_input = f"""<filepath>{test_file}</filepath>
<line>1</line>"""
        
        result = deletelines.run(xml_input)
        
        # Verify success message
        assert "Successfully deleted line 1" in result, f"Expected success message, got: {result}"
        assert "First line" in result, "Should show deleted content"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_lines = ["Second line", "Third line"]
        expected_content = "\n".join(expected_lines) + "\n"
        
        assert new_content == expected_content, f"Content mismatch. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


def test_deletelines_last_line():
    """Test deleting the last line"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file
        original_lines = ["First line", "Second line", "Last line"]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Delete last line
        xml_input = f"""<filepath>{test_file}</filepath>
<line>3</line>"""
        
        result = deletelines.run(xml_input)
        
        # Verify success message
        assert "Successfully deleted line 3" in result, f"Expected success message, got: {result}"
        assert "Last line" in result, "Should show deleted content"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_lines = ["First line", "Second line"]
        expected_content = "\n".join(expected_lines) + "\n"
        
        assert new_content == expected_content, f"Content mismatch. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


def test_deletelines_entire_file():
    """Test deleting all lines in a file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file with 3 lines
        original_lines = ["Line 1", "Line 2", "Line 3"]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Delete all lines
        xml_input = f"""<filepath>{test_file}</filepath>
<start>1</start>
<end>3</end>"""
        
        result = deletelines.run(xml_input)
        
        # Verify success message
        assert "Successfully deleted lines 1-3" in result, f"Expected success message, got: {result}"
        
        # Verify file is now empty
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        assert new_content == "", f"File should be empty, got: {repr(new_content)}"
        
        return True


def test_deletelines_single_line_file():
    """Test deleting from a single-line file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "single.txt")
        
        # Create single-line file
        original_content = "Only line"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Delete the only line
        xml_input = f"""<filepath>{test_file}</filepath>
<line>1</line>"""
        
        result = deletelines.run(xml_input)
        
        # Verify success
        assert "Successfully deleted line 1" in result, f"Expected success, got: {result}"
        assert "Only line" in result, "Should show deleted content"
        
        # Verify file is empty
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        assert new_content == "", f"File should be empty, got: {repr(new_content)}"
        
        return True


def test_deletelines_same_start_end():
    """Test deleting when start and end are the same (single line via range)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file
        original_lines = ["Line 1", "Line 2", "Line 3"]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Delete line 2 using range syntax
        xml_input = f"""<filepath>{test_file}</filepath>
<start>2</start>
<end>2</end>"""
        
        result = deletelines.run(xml_input)
        
        # Verify success message
        assert "Successfully deleted lines 2-2" in result, f"Expected success message, got: {result}"
        assert "2→Line 2" in result, "Should show deleted content"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_lines = ["Line 1", "Line 3"]
        expected_content = "\n".join(expected_lines) + "\n"
        
        assert new_content == expected_content, f"Content mismatch. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


def test_deletelines_error_cases():
    """Test various error conditions"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file with 3 lines
        original_lines = ["Line 1", "Line 2", "Line 3"]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Test missing filepath
        result = deletelines.run("<line>1</line>")
        assert result.startswith("Error: Invalid format"), "Should fail without filepath"
        
        # Test missing line specification
        result = deletelines.run(f"<filepath>{test_file}</filepath>")
        assert result.startswith("Error: Invalid format"), "Should fail without line specification"
        
        # Test non-existent file
        fake_file = os.path.join(temp_dir, "nonexistent.txt")
        xml_input = f"""<filepath>{fake_file}</filepath>
<line>1</line>"""
        result = deletelines.run(xml_input)
        assert "Error: File not found" in result, "Should fail for non-existent file"
        
        # Test line number too high
        xml_input = f"""<filepath>{test_file}</filepath>
<line>10</line>"""
        result = deletelines.run(xml_input)
        assert "Error: Line 10 exceeds file length" in result, "Should fail for line number too high"
        
        # Test line number too low
        xml_input = f"""<filepath>{test_file}</filepath>
<line>0</line>"""
        result = deletelines.run(xml_input)
        assert "Error: Line number must be 1 or greater" in result, "Should fail for line number 0"
        
        # Test negative line number
        xml_input = f"""<filepath>{test_file}</filepath>
<line>-1</line>"""
        result = deletelines.run(xml_input)
        assert "Error: Line number must be 1 or greater" in result, "Should fail for negative line number"
        
        # Test invalid line number
        xml_input = f"""<filepath>{test_file}</filepath>
<line>invalid</line>"""
        result = deletelines.run(xml_input)
        assert "Error: Invalid line number" in result, "Should fail for non-integer line number"
        
        # Test start line too high
        xml_input = f"""<filepath>{test_file}</filepath>
<start>10</start>
<end>12</end>"""
        result = deletelines.run(xml_input)
        assert "Error: Start line 10 exceeds file length" in result, "Should fail for start line too high"
        
        # Test end line too high
        xml_input = f"""<filepath>{test_file}</filepath>
<start>1</start>
<end>10</end>"""
        result = deletelines.run(xml_input)
        assert "Error: End line 10 exceeds file length" in result, "Should fail for end line too high"
        
        # Test start > end
        xml_input = f"""<filepath>{test_file}</filepath>
<start>3</start>
<end>1</end>"""
        result = deletelines.run(xml_input)
        assert "Error: Start line must be less than or equal to end line" in result, "Should fail when start > end"
        
        # Test invalid range numbers
        xml_input = f"""<filepath>{test_file}</filepath>
<start>invalid</start>
<end>2</end>"""
        result = deletelines.run(xml_input)
        assert "Error: Invalid line numbers" in result, "Should fail for invalid start number"
        
        # Test missing end for range
        xml_input = f"""<filepath>{test_file}</filepath>
<start>1</start>"""
        result = deletelines.run(xml_input)
        assert result.startswith("Error: Invalid format"), "Should fail with start but no end"
        
        return True


def test_deletelines_special_characters():
    """Test deleting lines with special characters"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "special.txt")
        
        # Create file with special characters
        original_lines = [
            "Normal line",
            "Line with special chars: ñáéíóú ☀️ 🚀",
            "Quotes: \"double\" 'single' \"smart\" 'smart'",
            "Symbols: @#$%^&*()_+-=[]{}|;:,.<>?",
            "Final line"
        ]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Delete the line with special characters
        xml_input = f"""<filepath>{test_file}</filepath>
<line>2</line>"""
        
        result = deletelines.run(xml_input)
        
        # Verify success and content display
        assert "Successfully deleted line 2" in result, f"Expected success, got: {result}"
        assert "ñáéíóú ☀️ 🚀" in result, "Should preserve special characters in output"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_lines = [
            "Normal line",
            "Quotes: \"double\" 'single' \"smart\" 'smart'",
            "Symbols: @#$%^&*()_+-=[]{}|;:,.<>?",
            "Final line"
        ]
        expected_content = "\n".join(expected_lines) + "\n"
        
        assert new_content == expected_content, f"Content mismatch. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


def test_deletelines_whitespace_lines():
    """Test deleting lines with various whitespace patterns"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "whitespace.txt")
        
        # Create file with different whitespace patterns
        original_lines = [
            "Line 1",
            "  Line with leading spaces",
            "\tLine with tab",
            "Line with trailing spaces  ",
            "  \t Mixed whitespace \t  ",
            "",  # Empty line
            "Final line"
        ]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Delete the empty line (line 6)
        xml_input = f"""<filepath>{test_file}</filepath>
<line>6</line>"""
        
        result = deletelines.run(xml_input)
        
        # Verify success
        assert "Successfully deleted line 6" in result, f"Expected success, got: {result}"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_lines = [
            "Line 1",
            "  Line with leading spaces", 
            "\tLine with tab",
            "Line with trailing spaces  ",
            "  \t Mixed whitespace \t  ",
            "Final line"
        ]
        expected_content = "\n".join(expected_lines) + "\n"
        
        assert new_content == expected_content, f"Content mismatch. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


# Test level assignments
test_deletelines_single_line.level = 2
test_deletelines_range.level = 2
test_deletelines_first_line.level = 2
test_deletelines_last_line.level = 2
test_deletelines_entire_file.level = 2
test_deletelines_single_line_file.level = 2
test_deletelines_same_start_end.level = 2
test_deletelines_error_cases.level = 2
test_deletelines_special_characters.level = 2
test_deletelines_whitespace_lines.level = 2