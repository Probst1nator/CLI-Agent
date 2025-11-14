"""
Bug Report Tool Tests

Tests for the bug_report tool including report creation,
file handling, and error conditions.
"""

import os
import sys
import tempfile
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.bug_report import bug_report


def test_bug_report_basic_creation():
    """Test basic bug report creation"""
    original_cwd = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Test basic bug report
            title = "Test Tool Not Working"
            report = "The test tool returns incorrect results when given valid input."
            xml_input = f"""<title>{title}</title>
<report>{report}</report>"""

            result = bug_report.run(xml_input)

            # Verify success message
            assert "successfully added" in result.lower(), f"Expected success message, got: {result}"
            assert title in result, "Should mention the title in response"

            # Verify file was created
            bug_file = Path(temp_dir) / "BUG_REPORTS.md"
            assert bug_file.exists(), "BUG_REPORTS.md should be created"

            # Verify content
            content = bug_file.read_text()
            assert "# Bug Reports" in content, "Should have header"
            assert title in content, "Should contain the title"
            assert report in content, "Should contain the report"
            assert "Reported:" in content, "Should have timestamp"

            return True

    finally:
        os.chdir(original_cwd)


def test_bug_report_multiple_entries():
    """Test appending multiple bug reports"""
    original_cwd = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # First bug report
            xml_input1 = """<title>First Bug</title>
<report>This is the first bug report.</report>"""
            result1 = bug_report.run(xml_input1)
            assert "successfully added" in result1.lower(), "First report should succeed"

            # Second bug report
            xml_input2 = """<title>Second Bug</title>
<report>This is the second bug report.</report>"""
            result2 = bug_report.run(xml_input2)
            assert "successfully added" in result2.lower(), "Second report should succeed"

            # Verify both reports in file
            bug_file = Path(temp_dir) / "BUG_REPORTS.md"
            content = bug_file.read_text()

            assert "First Bug" in content, "Should contain first bug"
            assert "Second Bug" in content, "Should contain second bug"
            assert "first bug report" in content, "Should contain first report text"
            assert "second bug report" in content, "Should contain second report text"

            # Verify ordering (first should appear before second)
            first_pos = content.find("First Bug")
            second_pos = content.find("Second Bug")
            assert first_pos < second_pos, "First bug should appear before second"

            return True

    finally:
        os.chdir(original_cwd)


def test_bug_report_special_characters():
    """Test bug reports with special characters"""
    original_cwd = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Bug report with special characters
            title = "Tool fails with symbols: @#$%^&*()"
            report = """The tool crashes when processing:
- Unicode: ñáéíóú ☀️ 🚀
- Quotes: "double" 'single'
- Code: `function test() { return true; }`
- Symbols: @#$%^&*()_+-=[]{}|;:,.<>?
"""
            xml_input = f"""<title>{title}</title>
<report>{report}</report>"""

            result = bug_report.run(xml_input)
            assert "successfully added" in result.lower(), f"Should handle special characters, got: {result}"

            # Verify content preservation
            bug_file = Path(temp_dir) / "BUG_REPORTS.md"
            content = bug_file.read_text()

            assert title in content, "Title with special chars should be preserved"
            assert "Unicode: ñáéíóú" in content, "Unicode should be preserved"
            assert "@#$%^&*()" in content, "Symbols should be preserved"

            return True

    finally:
        os.chdir(original_cwd)


def test_bug_report_multiline_content():
    """Test bug reports with multiline content"""
    original_cwd = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Multiline bug report
            report = """Steps to reproduce:
1. Execute the tool with parameter X
2. Observe output
3. Note the discrepancy

Expected behavior:
The tool should return formatted JSON

Actual behavior:
The tool returns malformed JSON with missing brackets

Error message:
JSONDecodeError: Expecting property name enclosed in double quotes
"""
            xml_input = f"""<title>JSON Formatting Issue</title>
<report>{report}</report>"""

            result = bug_report.run(xml_input)
            assert "successfully added" in result.lower(), "Should handle multiline content"

            # Verify multiline preservation
            bug_file = Path(temp_dir) / "BUG_REPORTS.md"
            content = bug_file.read_text()

            assert "Steps to reproduce:" in content, "Should preserve section headers"
            assert "Expected behavior:" in content, "Should preserve all sections"
            assert "JSONDecodeError" in content, "Should preserve error messages"

            return True

    finally:
        os.chdir(original_cwd)


def test_bug_report_missing_title():
    """Test error handling for missing title tag"""
    original_cwd = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Missing title tag
            xml_input = "<report>This report has no title</report>"
            result = bug_report.run(xml_input)

            assert "Error" in result or "Missing" in result, "Should report error for missing title"
            assert "title" in result.lower(), "Error should mention title"

            # Verify file not created or remains empty
            bug_file = Path(temp_dir) / "BUG_REPORTS.md"
            if bug_file.exists():
                content = bug_file.read_text()
                # Should only have header, no bug entries
                assert "This report has no title" not in content, "Invalid report should not be added"

            return True

    finally:
        os.chdir(original_cwd)


def test_bug_report_missing_report():
    """Test error handling for missing report tag"""
    original_cwd = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Missing report tag
            xml_input = "<title>A Title Without Report</title>"
            result = bug_report.run(xml_input)

            assert "Error" in result or "Missing" in result, "Should report error for missing report"
            assert "report" in result.lower(), "Error should mention report"

            # Verify file not created or remains empty
            bug_file = Path(temp_dir) / "BUG_REPORTS.md"
            if bug_file.exists():
                content = bug_file.read_text()
                assert "A Title Without Report" not in content, "Invalid report should not be added"

            return True

    finally:
        os.chdir(original_cwd)


def test_bug_report_empty_title():
    """Test error handling for empty title"""
    original_cwd = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Empty title
            xml_input = "<title></title><report>Report with empty title</report>"
            result = bug_report.run(xml_input)

            assert "Error" in result or "empty" in result.lower(), "Should report error for empty title"

            return True

    finally:
        os.chdir(original_cwd)


def test_bug_report_empty_report():
    """Test error handling for empty report"""
    original_cwd = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Empty report
            xml_input = "<title>Title with empty report</title><report></report>"
            result = bug_report.run(xml_input)

            assert "Error" in result or "empty" in result.lower(), "Should report error for empty report"

            return True

    finally:
        os.chdir(original_cwd)


def test_bug_report_file_format():
    """Test that the bug report file follows expected markdown format"""
    original_cwd = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            # Create a bug report
            xml_input = """<title>Format Test Bug</title>
<report>Testing the output format of the bug report file.</report>"""
            bug_report.run(xml_input)

            # Read and check format
            bug_file = Path(temp_dir) / "BUG_REPORTS.md"
            content = bug_file.read_text()

            # Check structure
            assert content.startswith("# Bug Reports"), "Should start with main header"
            assert "## Format Test Bug" in content, "Should have bug title as h2"
            assert "**Reported:**" in content, "Should have timestamp label"
            assert "---" in content, "Should have separators"

            # Check it's valid markdown structure
            lines = content.split('\n')
            assert any(line.startswith('# ') for line in lines), "Should have h1 header"
            assert any(line.startswith('## ') for line in lines), "Should have h2 headers"

            return True

    finally:
        os.chdir(original_cwd)


# Test level assignments
test_bug_report_basic_creation.level = 2
test_bug_report_multiple_entries.level = 2
test_bug_report_special_characters.level = 2
test_bug_report_multiline_content.level = 2
test_bug_report_missing_title.level = 2
test_bug_report_missing_report.level = 2
test_bug_report_empty_title.level = 2
test_bug_report_empty_report.level = 2
test_bug_report_file_format.level = 2
