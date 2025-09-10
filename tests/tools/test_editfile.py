"""
EditFile Tool Tests

Tests for the editfile tool including find/replace functionality
and line-specific editing capabilities.
"""

import os
import sys
import tempfile
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.editfile import editfile


def test_editfile_find_and_replace():
    """Test basic find and replace functionality"""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file with known content
        original_content = """line 1
old_variable = 42
line 3
old_variable = 100
line 5"""
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Test find and replace
        xml_input = f"""<filepath>{test_file}</filepath>
<search>old_variable</search>
<replace>new_variable</replace>"""
        
        result = editfile.run(xml_input)
        
        # Verify success message
        assert result.startswith("Successfully edited file:"), f"Expected success, got: {result}"
        
        # Verify file content was changed
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        expected_content = """line 1
new_variable = 42
line 3
new_variable = 100
line 5"""
        
        assert new_content == expected_content, f"Content mismatch. Expected:\n{expected_content}\nGot:\n{new_content}"
        
        return True


def test_editfile_line_specific_editing():
    """Test line-specific editing functionality"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.py")
        
        # Create test file with numbered lines
        original_content = """def function1():
    pass

def function2():
    return 42

def function3():
    print("hello")"""
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Test line-specific editing (replace line 5)
        xml_input = f"""<filepath>{test_file}</filepath>
<line>5</line>
<replace>    return "modified"</replace>"""
        
        result = editfile.run(xml_input)
        
        # Verify success message
        assert "Successfully edited line 5" in result, f"Expected line edit success, got: {result}"
        
        # Verify file content was changed
        with open(test_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        lines = new_content.split('\n')
        assert lines[4] == '    return "modified"', f"Line 5 not edited correctly. Got: {lines[4]}"
        
        # Verify other lines unchanged
        assert lines[0] == "def function1():", "Line 1 should be unchanged"
        assert lines[1] == "    pass", "Line 2 should be unchanged"
        
        return True


def test_editfile_error_cases():
    """Test various error conditions"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        
        # Create test file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("line 1\nline 2\nline 3\n")
        
        # Test missing filepath
        result = editfile.run("<search>test</search><replace>new</replace>")
        assert result.startswith("Error: Invalid format"), "Should fail without filepath"
        
        # Test missing search/replace for find-replace mode
        result = editfile.run(f"<filepath>{test_file}</filepath><search>test</search>")
        assert result.startswith("Error: Invalid format"), "Should fail without replace tag"
        
        # Test non-existent file
        fake_file = os.path.join(temp_dir, "nonexistent.txt")
        xml_input = f"""<filepath>{fake_file}</filepath>
<search>test</search>
<replace>new</replace>"""
        result = editfile.run(xml_input)
        assert "Error: File not found" in result, "Should fail for non-existent file"
        
        # Test search string not found
        xml_input = f"""<filepath>{test_file}</filepath>
<search>nonexistent_string</search>
<replace>new</replace>"""
        result = editfile.run(xml_input)
        assert "Error: The string to find was not found" in result, "Should fail when search string not found"
        
        # Test invalid line number (too high)
        xml_input = f"""<filepath>{test_file}</filepath>
<line>10</line>
<replace>new line</replace>"""
        result = editfile.run(xml_input)
        assert "Error: Line 10 exceeds file length" in result, "Should fail for line number too high"
        
        # Test invalid line number (negative)
        xml_input = f"""<filepath>{test_file}</filepath>
<line>-1</line>
<replace>new line</replace>"""
        result = editfile.run(xml_input)
        assert "Error: Line number must be 1 or greater" in result, "Should fail for negative line number"
        
        return True


def test_editfile_edge_cases():
    """Test edge cases and boundary conditions"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "edge_test.txt")
        
        # Test with empty file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("")
        
        # Try to edit line 1 of empty file
        xml_input = f"""<filepath>{test_file}</filepath>
<line>1</line>
<replace>first line</replace>"""
        result = editfile.run(xml_input)
        assert "Error: Line 1 exceeds file length" in result, "Should fail on empty file"
        
        # Test with single line file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("single line")
        
        # Edit the single line
        xml_input = f"""<filepath>{test_file}</filepath>
<line>1</line>
<replace>modified line</replace>"""
        result = editfile.run(xml_input)
        assert "Successfully edited line 1" in result, "Should succeed on single line file"
        
        # Verify content
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == "modified line\n", f"Content should be 'modified line\\n', got: {repr(content)}"
        
        # Test find/replace with special characters
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("print('hello world')\nprint(\"goodbye world\")")
        
        xml_input = f"""<filepath>{test_file}</filepath>
<search>print('hello world')</search>
<replace>print('hello universe')</replace>"""
        result = editfile.run(xml_input)
        assert "Successfully edited file" in result, "Should handle special characters"
        
        return True


# Test level assignment (following existing pattern)
test_editfile_find_and_replace.level = 2
test_editfile_line_specific_editing.level = 2  
test_editfile_error_cases.level = 2
test_editfile_edge_cases.level = 2