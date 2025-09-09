"""
InsertLine Tool Tests

Tests for the insertline tool including line insertion
at various positions and error handling.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.insertline import insertline


def test_insertline_after_specific_line():
    """Test inserting after a specific line number"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file with numbered lines
        original_lines = ["Line 1", "Line 2", "Line 3", "Line 4"]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Insert after line 2
        new_line = "Inserted after line 2"
        xml_input = f"""<filepath>{test_file}</filepath>
<after>2</after>
<content>{new_line}</content>"""
        
        result = insertline.run(xml_input)
        
        # Verify success message
        assert "Successfully inserted line after line 2" in result, f"Expected success message, got: {result}"
        assert new_line in result, "Should show the inserted content"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_lines = ["Line 1", "Line 2", new_line, "Line 3", "Line 4"]
        expected_content = "\n".join(expected_lines) + "\n"
        
        assert new_content == expected_content, f"Content mismatch. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


def test_insertline_before_specific_line():
    """Test inserting before a specific line number"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file
        original_lines = ["Line 1", "Line 2", "Line 3", "Line 4"]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Insert before line 3
        new_line = "Inserted before line 3"
        xml_input = f"""<filepath>{test_file}</filepath>
<before>3</before>
<content>{new_line}</content>"""
        
        result = insertline.run(xml_input)
        
        # Verify success message
        assert "Successfully inserted line before line 3" in result, f"Expected success message, got: {result}"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_lines = ["Line 1", "Line 2", new_line, "Line 3", "Line 4"]
        expected_content = "\n".join(expected_lines) + "\n"
        
        assert new_content == expected_content, f"Content mismatch. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


def test_insertline_at_beginning():
    """Test inserting at the beginning of file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file
        original_lines = ["Line 1", "Line 2", "Line 3"]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Insert before line 1 (at beginning)
        new_line = "First line"
        xml_input = f"""<filepath>{test_file}</filepath>
<before>1</before>
<content>{new_line}</content>"""
        
        result = insertline.run(xml_input)
        
        # Verify success message
        assert "Successfully inserted line before line 1" in result, f"Expected success message, got: {result}"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_lines = [new_line, "Line 1", "Line 2", "Line 3"]
        expected_content = "\n".join(expected_lines) + "\n"
        
        assert new_content == expected_content, f"Content mismatch. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


def test_insertline_at_end():
    """Test inserting at the end of file (default behavior)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file
        original_lines = ["Line 1", "Line 2", "Line 3"]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Insert without position specification (defaults to end)
        new_line = "Last line"
        xml_input = f"""<filepath>{test_file}</filepath>
<content>{new_line}</content>"""
        
        result = insertline.run(xml_input)
        
        # Verify success message
        assert "Successfully inserted line at end of file" in result, f"Expected success message, got: {result}"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_lines = ["Line 1", "Line 2", "Line 3", new_line]
        expected_content = "\n".join(expected_lines) + "\n"
        
        assert new_content == expected_content, f"Content mismatch. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


def test_insertline_after_zero():
    """Test inserting after line 0 (beginning of file)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file
        original_lines = ["Line 1", "Line 2"]
        original_content = "\n".join(original_lines) + "\n"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Insert after line 0 (at beginning)
        new_line = "New first line"
        xml_input = f"""<filepath>{test_file}</filepath>
<after>0</after>
<content>{new_line}</content>"""
        
        result = insertline.run(xml_input)
        
        # Verify success message
        assert "Successfully inserted line after line 0" in result, f"Expected success message, got: {result}"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_lines = [new_line, "Line 1", "Line 2"]
        expected_content = "\n".join(expected_lines) + "\n"
        
        assert new_content == expected_content, f"Content mismatch. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


def test_insertline_newline_handling():
    """Test automatic newline handling"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file
        original_content = "Line 1\nLine 2"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Test content without newline (should be added automatically)
        new_line = "Line without newline"
        xml_input = f"""<filepath>{test_file}</filepath>
<after>1</after>
<content>{new_line}</content>"""
        
        result = insertline.run(xml_input)
        
        # Verify file content has proper newlines
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_content = "Line 1\nLine without newline\nLine 2"
        
        assert new_content == expected_content, f"Newline handling failed. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        # Test content with existing newline (should not double up)
        test_file2 = os.path.join(temp_dir, "test2.txt")
        with open(test_file2, 'w', encoding='utf-8') as f:
            f.write("Line 1\nLine 2")
        
        new_line_with_newline = "Line with newline\n"
        xml_input = f"""<filepath>{test_file2}</filepath>
<after>1</after>
<content>{new_line_with_newline}</content>"""
        
        insertline.run(xml_input)
        
        with open(test_file2, 'r', encoding='utf-8') as f:
            content2 = f.read()
        
        # Should not have double newlines
        expected_content2 = "Line 1\nLine with newline\nLine 2"
        assert content2 == expected_content2, f"Double newline issue. Expected: {repr(expected_content2)}, Got: {repr(content2)}"
        
        return True


def test_insertline_empty_file():
    """Test inserting into empty file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "empty.txt")
        
        # Create empty file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("")
        
        # Insert into empty file (should go at end)
        new_line = "First line in empty file"
        xml_input = f"""<filepath>{test_file}</filepath>
<content>{new_line}</content>"""
        
        result = insertline.run(xml_input)
        
        # Verify success
        assert "Successfully inserted line at end of file" in result, f"Expected success message, got: {result}"
        
        # Verify file content
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_content = new_line + "\n"
        assert new_content == expected_content, f"Empty file insertion failed. Expected: {repr(expected_content)}, Got: {repr(new_content)}"
        
        return True


def test_insertline_error_cases():
    """Test various error conditions"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file with 3 lines
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Line 1\nLine 2\nLine 3")
        
        # Test missing filepath
        result = insertline.run("<content>test</content>")
        assert result.startswith("Error: Invalid format"), "Should fail without filepath"
        
        # Test missing content
        result = insertline.run(f"<filepath>{test_file}</filepath><after>1</after>")
        assert result.startswith("Error: Invalid format"), "Should fail without content"
        
        # Test non-existent file
        fake_file = os.path.join(temp_dir, "nonexistent.txt")
        xml_input = f"""<filepath>{fake_file}</filepath>
<content>test</content>"""
        result = insertline.run(xml_input)
        assert "Error: File not found" in result, "Should fail for non-existent file"
        
        # Test invalid after line number (too high)
        xml_input = f"""<filepath>{test_file}</filepath>
<after>10</after>
<content>test</content>"""
        result = insertline.run(xml_input)
        assert "Error: 'after' line 10 exceeds file length" in result, "Should fail for after line too high"
        
        # Test negative after line number
        xml_input = f"""<filepath>{test_file}</filepath>
<after>-1</after>
<content>test</content>"""
        result = insertline.run(xml_input)
        assert "Error: 'after' line number must be 0 or greater" in result, "Should fail for negative after line"
        
        # Test invalid before line number (too high)
        xml_input = f"""<filepath>{test_file}</filepath>
<before>10</before>
<content>test</content>"""
        result = insertline.run(xml_input)
        assert "Error: 'before' line 10 exceeds file length" in result, "Should fail for before line too high"
        
        # Test before line 0
        xml_input = f"""<filepath>{test_file}</filepath>
<before>0</before>
<content>test</content>"""
        result = insertline.run(xml_input)
        assert "Error: 'before' line number must be 1 or greater" in result, "Should fail for before line 0"
        
        # Test invalid line number (non-integer)
        xml_input = f"""<filepath>{test_file}</filepath>
<after>invalid</after>
<content>test</content>"""
        result = insertline.run(xml_input)
        assert "Error: Invalid 'after' line number" in result, "Should fail for non-integer line number"
        
        return True


def test_insertline_special_content():
    """Test inserting content with special characters"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Line 1\nLine 2")
        
        # Insert content with special characters
        special_content = """Special line: "quotes" 'single' ñáéíóú ☀️ @#$%^&*()"""
        xml_input = f"""<filepath>{test_file}</filepath>
<after>1</after>
<content>{special_content}</content>"""
        
        result = insertline.run(xml_input)
        
        # Verify success
        assert "Successfully inserted line" in result, f"Should handle special characters, got: {result}"
        
        # Verify content preservation
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        assert special_content in new_content, f"Special characters not preserved: {repr(new_content)}"
        
        return True


# Test level assignments
test_insertline_after_specific_line.level = 2
test_insertline_before_specific_line.level = 2
test_insertline_at_beginning.level = 2
test_insertline_at_end.level = 2
test_insertline_after_zero.level = 2
test_insertline_newline_handling.level = 2
test_insertline_empty_file.level = 2
test_insertline_error_cases.level = 2
test_insertline_special_content.level = 2