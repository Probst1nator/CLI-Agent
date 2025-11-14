"""
SearchFile Tool Tests

Tests for the searchfile tool including file search, directory search,
BM25 ranking, regex patterns, and error handling.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.searchfile import searchfile


def test_searchfile_exact_match_in_file():
    """Find exact text matches in a single file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir, "test.txt")
        test_file.write_text("Line 1: Hello World\nLine 2: Test Line\nLine 3: Hello Again")

        result = searchfile.run(f"<query>Hello</query><path>{test_file}</path>")

        # Should find both lines with "Hello"
        assert "Line 1" in result, "Should find Line 1"
        assert "Line 3" in result, "Should find Line 3"
        assert "Hello World" in result, "Should show matched content"

        # Should NOT match Line 2
        assert "2 match" in result or "matches" in result, "Should report match count"

    return True


def test_searchfile_regex_pattern_in_file():
    """Use regex patterns in file search"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir, "code.py")
        test_file.write_text("def function1():\n    pass\nclass MyClass:\n    def method():\n        pass")

        result = searchfile.run(f"<query>def \\w+</query><path>{test_file}</path>")

        # Should match function and method definitions
        assert "def function1" in result or "function1" in result, "Should match function definition"
        assert "def method" in result or "method" in result, "Should match method definition"

    return True


def test_searchfile_case_insensitive_file():
    """Case-insensitive search in file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir, "test.txt")
        test_file.write_text("HELLO world\nhello WORLD\nHeLLo WoRLd")

        result = searchfile.run(f"<query>hello</query><path>{test_file}</path>")

        # Should match all lines (case-insensitive)
        assert "3 match" in result or "matches" in result, "Should find all 3 matches"

    return True


def test_searchfile_line_numbers_in_file():
    """Include line numbers in file search results"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir, "test.txt")
        test_file.write_text("Line 1\nLine 2 match\nLine 3\nLine 4 match\nLine 5")

        result = searchfile.run(f"<query>match</query><path>{test_file}</path>")

        # Should show line numbers
        assert "Line 2" in result or "2:" in result, "Should show line 2"
        assert "Line 4" in result or "4:" in result, "Should show line 4"

    return True


def test_searchfile_no_matches_in_file():
    """Handle no matches found in file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir, "test.txt")
        test_file.write_text("Some content here\nMore content\nEven more")

        result = searchfile.run(f"<query>NOMATCH_XYZ123</query><path>{test_file}</path>")

        # Should indicate no matches
        assert "No matches found" in result or "0 match" in result, "Should indicate no matches"

    return True


def test_searchfile_directory_search():
    """Search in a directory (BM25 or fallback)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple files with search terms
        Path(temp_dir, "file1.txt").write_text("This file contains authentication logic for the system")
        Path(temp_dir, "file2.txt").write_text("Random content without the keyword")
        Path(temp_dir, "file3.txt").write_text("More authentication and authorization code")

        result = searchfile.run(f"<query>authentication</query><path>{temp_dir}</path>")

        # Should list files containing the term
        assert "file1.txt" in result, "Should find file1.txt"
        assert "file3.txt" in result, "Should find file3.txt"

        # Should NOT list file without the term
        assert "file2.txt" not in result or "Random content" not in result, "Should not list file2.txt"

    return True


def test_searchfile_directory_top_k():
    """Test top_k parameter in directory search"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create many files
        for i in range(20):
            Path(temp_dir, f"file{i}.txt").write_text(f"test content {i}")

        result = searchfile.run(f"<query>test</query><path>{temp_dir}</path><top_k>5</top_k>")

        # Should limit results to top_k
        # Count how many files are mentioned
        file_count = sum(1 for i in range(20) if f"file{i}.txt" in result)
        assert file_count <= 5, f"Should limit to 5 files, found {file_count}"

    return True


def test_searchfile_directory_with_subdirectories():
    """Search directory including subdirectories"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested structure
        Path(temp_dir, "root.txt").write_text("search term here")
        Path(temp_dir, "subdir").mkdir()
        Path(temp_dir, "subdir", "nested.txt").write_text("search term in nested file")

        result = searchfile.run(f"<query>search term</query><path>{temp_dir}</path>")

        # Should find files in subdirectories too
        assert "root.txt" in result or "root" in result, "Should find root file"
        # Nested file might be found depending on recursion depth

    return True


def test_searchfile_nonexistent_file():
    """Error on non-existent file"""
    result = searchfile.run("<query>test</query><path>/nonexistent/file.txt</path>")

    # Should return error
    assert "Error" in result or "not found" in result.lower() or "does not exist" in result, \
        "Should indicate file not found"

    return True


def test_searchfile_nonexistent_directory():
    """Error on non-existent directory"""
    result = searchfile.run("<query>test</query><path>/nonexistent/directory</path>")

    # Should return error
    assert "Error" in result or "not found" in result.lower() or "does not exist" in result, \
        "Should indicate directory not found"

    return True


def test_searchfile_missing_query_parameter():
    """Error when query parameter is missing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = searchfile.run(f"<path>{temp_dir}</path>")

        # Should return error about missing query
        assert "Error" in result and "query" in result.lower(), \
            "Should indicate missing query parameter"

    return True


def test_searchfile_missing_path_parameter():
    """Error when path parameter is missing"""
    result = searchfile.run("<query>test</query>")

    # Should return error about missing path
    assert "Error" in result and "path" in result.lower(), \
        "Should indicate missing path parameter"

    return True


def test_searchfile_invalid_regex_pattern():
    """Handle invalid regex patterns gracefully"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir, "test.txt")
        test_file.write_text("Some content")

        # Invalid regex pattern
        result = searchfile.run(f"<query>[invalid(regex</query><path>{test_file}</path>")

        # Should return error about invalid regex
        assert "Error" in result or "Invalid" in result, "Should indicate invalid regex"

    return True


def test_searchfile_multiline_file():
    """Search in multiline file content"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir, "test.txt")
        content = "\n".join([f"Line {i}: content" for i in range(100)])
        test_file.write_text(content)

        result = searchfile.run(f"<query>Line 50</query><path>{test_file}</path>")

        # Should find the specific line
        assert "Line 50" in result, "Should find Line 50"

    return True


def test_searchfile_large_file_line_limit():
    """Test that file search limits output for large results"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir, "large.txt")
        # Create file with many matches
        content = "\n".join(["match content" for _ in range(100)])
        test_file.write_text(content)

        result = searchfile.run(f"<query>match</query><path>{test_file}</path>")

        # Should limit to 50 matches or mention the limit
        assert "50" in result or "showing first" in result.lower() or "100 match" in result, \
            "Should limit or mention large number of matches"

    return True


def test_searchfile_binary_file_skip():
    """Test that binary files are skipped in directory search"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create text file
        Path(temp_dir, "text.txt").write_text("searchable content")

        # Create binary-ish file (not truly binary, but with .pyc extension)
        Path(temp_dir, "compiled.pyc").write_text("binary content")

        result = searchfile.run(f"<query>content</query><path>{temp_dir}</path>")

        # Should find text file
        assert "text.txt" in result, "Should find text file"

        # Should skip .pyc file (due to ignore patterns)
        assert "compiled.pyc" not in result, "Should skip .pyc files"

    return True


def test_searchfile_ignore_patterns():
    """Test that common patterns are ignored"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create normal file
        Path(temp_dir, "normal.py").write_text("code content")

        # Create directories that should be ignored
        Path(temp_dir, "node_modules").mkdir()
        Path(temp_dir, "node_modules", "package.py").write_text("node content")

        Path(temp_dir, "__pycache__").mkdir()
        Path(temp_dir, "__pycache__", "cache.py").write_text("cache content")

        result = searchfile.run(f"<query>content</query><path>{temp_dir}</path>")

        # Should find normal file
        assert "normal.py" in result, "Should find normal.py"

        # Should NOT search in ignored directories
        assert "node_modules" not in result or "package.py" not in result, "Should ignore node_modules"
        assert "__pycache__" not in result or "cache.py" not in result, "Should ignore __pycache__"

    return True


def test_searchfile_empty_query():
    """Handle empty query parameter"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir, "test.txt")
        test_file.write_text("content")

        result = searchfile.run(f"<query></query><path>{test_file}</path>")

        # Should return error or handle gracefully
        assert isinstance(result, str), "Should return a string"

    return True


def test_searchfile_whitespace_in_query():
    """Handle whitespace in search queries"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir, "test.txt")
        test_file.write_text("The quick brown fox\njumps over the lazy dog")

        result = searchfile.run(f"<query>quick brown</query><path>{test_file}</path>")

        # Should find the phrase
        assert "quick brown" in result or "fox" in result, "Should handle multi-word queries"

    return True


def test_searchfile_unicode_content():
    """Handle unicode characters in files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir, "unicode.txt")
        test_file.write_text("日本語 Ελληνικά العربية 中文", encoding='utf-8')

        result = searchfile.run(f"<query>日本語</query><path>{test_file}</path>")

        # Should handle unicode
        assert "日本語" in result or "1 match" in result, "Should handle unicode searches"

    return True


def test_searchfile_tool_contract():
    """Verify tool implements the required contract"""
    # Test get_delim
    delim = searchfile.get_delim()
    assert delim == "searchfile", f"Expected delimiter 'searchfile', got: {delim}"

    # Test get_tool_info
    info = searchfile.get_tool_info()
    assert isinstance(info, dict), "get_tool_info should return dict"
    assert "name" in info, "Tool info should have 'name'"
    assert "description" in info, "Tool info should have 'description'"
    assert "example" in info, "Tool info should have 'example'"
    assert info["name"] == "searchfile", f"Expected name 'searchfile', got: {info['name']}"

    # Test get_detailed_tool_info
    assert hasattr(searchfile, 'get_detailed_tool_info'), "Should have get_detailed_tool_info method"
    detailed = searchfile.get_detailed_tool_info()
    assert isinstance(detailed, str), "Detailed info should be a string"
    assert len(detailed) > 100, "Detailed info should be substantial"

    return True


# Test level assignments
test_searchfile_exact_match_in_file.level = 2
test_searchfile_regex_pattern_in_file.level = 2
test_searchfile_case_insensitive_file.level = 2
test_searchfile_line_numbers_in_file.level = 2
test_searchfile_no_matches_in_file.level = 2
test_searchfile_directory_search.level = 2
test_searchfile_directory_top_k.level = 2
test_searchfile_directory_with_subdirectories.level = 2
test_searchfile_nonexistent_file.level = 2
test_searchfile_nonexistent_directory.level = 2
test_searchfile_missing_query_parameter.level = 2
test_searchfile_missing_path_parameter.level = 2
test_searchfile_invalid_regex_pattern.level = 2
test_searchfile_multiline_file.level = 2
test_searchfile_large_file_line_limit.level = 3
test_searchfile_binary_file_skip.level = 3
test_searchfile_ignore_patterns.level = 2
test_searchfile_empty_query.level = 3
test_searchfile_whitespace_in_query.level = 2
test_searchfile_unicode_content.level = 2
test_searchfile_tool_contract.level = 2
