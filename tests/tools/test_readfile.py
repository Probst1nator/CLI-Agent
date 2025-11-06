"""
ReadFile Tool Tests

Tests for the readfile tool including basic reading,
truncation handling, line-indexed access, and error handling.
"""

import os
import sys
import tempfile
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.readfile import readfile


def test_readfile_basic_reading():
    """Test basic file reading functionality"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")

        # Create test file with known content
        content = """Line 1: Hello World
Line 2: This is a test
Line 3: With multiple lines
Line 4: End of file"""

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # Test basic reading
        xml_input = f"<filepath>{test_file}</filepath>"
        result = readfile.run(xml_input)

        # Should return the exact content
        assert result == content, f"Content mismatch. Expected: {repr(content)}, Got: {repr(result)}"

        return True


def test_readfile_line_indexed_reading():
    """Test line-indexed reading functionality"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "indexed_test.txt")

        # Create file with numbered lines
        lines = [f"Line {i+1}: Content for line {i+1}" for i in range(20)]
        content = "\n".join(lines)

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)

        # Test single line reading
        xml_input = f"<filepath>{test_file}</filepath><lines>5</lines>"
        result = readfile.run(xml_input)
        expected = "Line 5: Content for line 5"
        assert result == expected, f"Single line read failed. Expected: {repr(expected)}, Got: {repr(result)}"

        # Test line range reading
        xml_input = f"<filepath>{test_file}</filepath><lines>3-6</lines>"
        result = readfile.run(xml_input)
        expected = "\n".join(lines[2:6])  # Lines 3-6 (0-indexed 2-5)
        assert result == expected, f"Line range read failed. Expected: {repr(expected)}, Got: {repr(result)}"

        # Test reading from line to end
        xml_input = f"<filepath>{test_file}</filepath><lines>18-</lines>"
        result = readfile.run(xml_input)
        expected = "\n".join(lines[17:])  # Lines 18 to end
        assert result == expected, f"Read to end failed. Expected: {repr(expected)}, Got: {repr(result)}"

        # Test reading from start to line
        xml_input = f"<filepath>{test_file}</filepath><lines>-5</lines>"
        result = readfile.run(xml_input)
        expected = "\n".join(lines[:5])  # Lines 1-5 (0-indexed 0-4)
        assert result == expected, f"Read from start failed. Expected: {repr(expected)}, Got: {repr(result)}"

        return True


def test_readfile_large_file_truncation():
    """Test truncation behavior for large files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "large_test.txt")

        # Create a file larger than CHUNK_LIMIT
        line = "This is a line with some content to make the file large.\n"
        # Make file larger than the 50000 character CHUNK_LIMIT
        large_content = line * 2000  # Should be well over 50k characters

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(large_content)

        # First read should be truncated
        xml_input = f"<filepath>{test_file}</filepath>"
        result = readfile.run(xml_input)

        # Should contain truncation message
        assert "File is too large" in result, "Should indicate truncation"
        assert "Truncating to show start and end" in result, "Should explain truncation"
        assert len(result) < len(large_content), "Result should be shorter than original"

        # Should contain content from start and end
        assert large_content[:100] in result, "Should contain content from start"
        assert large_content[-100:] in result, "Should contain content from end"

        return True


def test_readfile_error_cases():
    """Test various error conditions"""
    with tempfile.TemporaryDirectory() as temp_dir:

        # Test non-existent file
        fake_file = os.path.join(temp_dir, "nonexistent.txt")
        xml_input = f"<filepath>{fake_file}</filepath>"
        result = readfile.run(xml_input)
        assert "Error: File not found" in result, "Should fail for non-existent file"

        # Test invalid line specification
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("line 1\nline 2\nline 3")

        xml_input = f"<filepath>{test_file}</filepath><lines>invalid</lines>"
        result = readfile.run(xml_input)
        assert "Error: Invalid line specification" in result, "Should fail with invalid line spec"

        # Test line number exceeding file length
        xml_input = f"<filepath>{test_file}</filepath><lines>10</lines>"
        result = readfile.run(xml_input)
        assert "Error extracting lines" in result or "Error: Invalid line specification" in result, "Should fail for line beyond file length"

        # Test invalid line range
        xml_input = f"<filepath>{test_file}</filepath><lines>5-2</lines>"
        result = readfile.run(xml_input)
        assert "Error extracting lines" in result or "Error: Invalid line specification" in result, "Should fail for invalid range"

        return True


def test_readfile_empty_file():
    """Test reading empty files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "empty.txt")

        # Create empty file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("")

        xml_input = f"<filepath>{test_file}</filepath>"
        result = readfile.run(xml_input)

        assert result == "", f"Empty file should return empty string, got: {repr(result)}"

        return True


def test_readfile_special_characters():
    """Test reading files with special characters"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "special.txt")

        # Content with various encodings and special characters
        content = """Unicode test: ñáéíóú ☀️ 🚀
Quotes: "double" 'single' "smart" 'smart'
Symbols: @#$%^&*()_+-=[]{}|;:,.<>?
Tabs:\tand\tnewlines
"""

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)

        xml_input = f"<filepath>{test_file}</filepath>"
        result = readfile.run(xml_input)

        assert result == content, f"Special characters not preserved. Expected: {repr(content)}, Got: {repr(result)}"

        return True


def test_readfile_different_line_endings():
    """Test reading files with different line endings"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "line_endings.txt")

        # Create file with Unix line endings
        content = "Line 1\nLine 2\nLine 3"

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)

        xml_input = f"<filepath>{test_file}</filepath>"
        result = readfile.run(xml_input)

        assert result == content, f"Line endings not preserved. Expected: {repr(content)}, Got: {repr(result)}"

        return True


def test_readfile_relative_paths():
    """Test reading with relative paths"""
    original_cwd = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Create file in current directory
            test_content = "Relative path test content"
            with open("relative_test.txt", 'w', encoding='utf-8') as f:
                f.write(test_content)

            # Test relative path reading
            xml_input = "<filepath>./relative_test.txt</filepath>"
            result = readfile.run(xml_input)

            assert result == test_content, f"Relative path reading failed. Expected: {repr(test_content)}, Got: {repr(result)}"

    finally:
        os.chdir(original_cwd)

    return True


def test_readfile_whitespace_handling():
    """Test reading files with various whitespace patterns"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "whitespace.txt")

        # Content with leading/trailing whitespace and empty lines
        content = """  Line with leading spaces
\tLine with leading tab
Line with trailing spaces
\t\tMultiple tabs and spaces  \t

\t
Line after empty lines"""

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)

        xml_input = f"<filepath>{test_file}</filepath>"
        result = readfile.run(xml_input)

        assert result == content, f"Whitespace not preserved. Expected: {repr(content)}, Got: {repr(result)}"

        return True


def test_readfile_office_documents():
    """Test reading office document formats (.docx, .xlsx, .pptx)"""
    try:
        import docx
        import openpyxl
        import pptx
    except ImportError:
        # Skip test if libraries not installed
        print("Skipping office document tests - required libraries not installed")
        return True

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test DOCX
        docx_file = os.path.join(temp_dir, "test.docx")
        doc = docx.Document()
        doc.add_heading('Test Document', 0)
        doc.add_paragraph('This is test content.')
        doc.save(docx_file)

        result = readfile.run(docx_file)
        assert "Test Document" in result, "DOCX: Should extract heading"
        assert "This is test content" in result, "DOCX: Should extract paragraph"
        assert "Word document" in result, "DOCX: Should identify as Word document"

        # Test XLSX
        xlsx_file = os.path.join(temp_dir, "test.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws['A1'] = 'Header1'
        ws['B1'] = 'Header2'
        ws['A2'] = 'Data1'
        ws['B2'] = 'Data2'
        wb.save(xlsx_file)

        result = readfile.run(xlsx_file)
        assert "Header1" in result, "XLSX: Should extract cell content"
        assert "Data1" in result, "XLSX: Should extract data"
        assert "Excel spreadsheet" in result, "XLSX: Should identify as Excel"

        # Test PPTX
        pptx_file = os.path.join(temp_dir, "test.pptx")
        prs = pptx.Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        title = slide.shapes.title
        title.text = "Test Slide Title"
        prs.save(pptx_file)

        result = readfile.run(pptx_file)
        assert "Test Slide Title" in result, "PPTX: Should extract slide title"
        assert "PowerPoint presentation" in result, "PPTX: Should identify as PowerPoint"

        return True


def test_readfile_office_document_detection():
    """Test office document detection without file extensions"""
    try:
        import docx
    except ImportError:
        print("Skipping office document detection test - python-docx not installed")
        return True

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create DOCX without extension
        docx_file = os.path.join(temp_dir, "no_extension")
        doc = docx.Document()
        doc.add_paragraph('Test content for detection')
        doc.save(docx_file)

        # Should still detect and process it
        is_office, format_type = readfile.is_office_document(docx_file)
        assert is_office, "Should detect office document without extension"
        assert format_type == 'docx', f"Should detect as docx, got {format_type}"

        # Should be able to read it
        result = readfile.run(docx_file)
        assert "Test content for detection" in result, "Should extract content from file without extension"

        return True


# Test level assignments
test_readfile_basic_reading.level = 2
test_readfile_line_indexed_reading.level = 2
test_readfile_large_file_truncation.level = 3
test_readfile_error_cases.level = 2
test_readfile_empty_file.level = 2
test_readfile_special_characters.level = 2
test_readfile_different_line_endings.level = 2
test_readfile_relative_paths.level = 2
test_readfile_whitespace_handling.level = 2
test_readfile_office_documents.level = 2
test_readfile_office_document_detection.level = 2