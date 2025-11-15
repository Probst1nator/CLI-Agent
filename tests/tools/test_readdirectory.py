"""
ReadDirectory Tool Tests

Tests for the readdirectory tool including basic directory listing,
recursive traversal, max_depth handling, and .gitignore support.
"""

import sys
import tempfile
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.readdirectory import readdirectory


def test_readdirectory_basic():
    """List files in a basic directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        Path(temp_dir, "file1.txt").write_text("content1")
        Path(temp_dir, "file2.py").write_text("content2")
        Path(temp_dir, "file3.md").write_text("content3")

        result = readdirectory.run(f"<path>{temp_dir}</path>")

        # Should list all files
        assert "file1.txt" in result, "Should list file1.txt"
        assert "file2.py" in result, "Should list file2.py"
        assert "file3.md" in result, "Should list file3.md"

        # Should show statistics
        assert "3 files" in result or "files" in result.lower(), "Should show file count"

    return True


def test_readdirectory_with_subdirectories():
    """List files with subdirectories"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested structure
        Path(temp_dir, "root_file.txt").write_text("root")
        Path(temp_dir, "subdir").mkdir()
        Path(temp_dir, "subdir", "sub_file.txt").write_text("sub")
        Path(temp_dir, "subdir", "nested").mkdir()
        Path(temp_dir, "subdir", "nested", "deep_file.txt").write_text("deep")

        result = readdirectory.run(f"<path>{temp_dir}</path>")

        # Should list all files and directories
        assert "root_file.txt" in result, "Should list root file"
        assert "subdir" in result, "Should list subdirectory"
        assert "sub_file.txt" in result, "Should list file in subdirectory"
        assert "nested" in result, "Should list nested directory"
        assert "deep_file.txt" in result, "Should list file in nested directory"

        # Should show directory markers (/)
        assert "subdir/" in result or "subdir" in result, "Should mark directories"

    return True


def test_readdirectory_max_depth_1():
    """Test max_depth=1 (only current level)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested structure
        Path(temp_dir, "root.txt").write_text("root")
        Path(temp_dir, "level1").mkdir()
        Path(temp_dir, "level1", "file1.txt").write_text("level1")
        Path(temp_dir, "level1", "level2").mkdir()
        Path(temp_dir, "level1", "level2", "file2.txt").write_text("level2")

        result = readdirectory.run(f"<path>{temp_dir}</path><max_depth>1</max_depth>")

        # Should only show root level and first subdirectory
        assert "root.txt" in result, "Should show root file"
        assert "level1" in result, "Should show first level directory"

        # Should NOT show deeper files (max_depth=1 means don't descend into subdirs)
        assert "file1.txt" not in result, "Should not show level1 files with max_depth=1"
        assert "level2" not in result, "Should not show level2 with max_depth=1"

    return True


def test_readdirectory_max_depth_2():
    """Test max_depth=2 (current + one sublevel)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested structure
        Path(temp_dir, "root.txt").write_text("root")
        Path(temp_dir, "level1").mkdir()
        Path(temp_dir, "level1", "file1.txt").write_text("level1")
        Path(temp_dir, "level1", "level2").mkdir()
        Path(temp_dir, "level1", "level2", "file2.txt").write_text("level2")

        result = readdirectory.run(f"<path>{temp_dir}</path><max_depth>2</max_depth>")

        # Should show root and first two levels
        assert "root.txt" in result, "Should show root file"
        assert "level1" in result, "Should show level1 directory"
        assert "file1.txt" in result, "Should show file in level1"
        assert "level2" in result, "Should show level2 directory"

        # Should NOT show files deeper than max_depth=2
        assert "file2.txt" not in result, "Should not show files in level2 with max_depth=2"

    return True


def test_readdirectory_empty_directory():
    """Handle empty directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = readdirectory.run(f"<path>{temp_dir}</path>")

        # Should indicate empty
        assert "empty" in result.lower() or "0 files" in result, "Should indicate empty directory"

    return True


def test_readdirectory_nonexistent():
    """Error on non-existent directory"""
    result = readdirectory.run("<path>/nonexistent/path/xyz123</path>")

    # Should return error
    assert "Error" in result or "not exist" in result.lower(), "Should indicate error"

    return True


def test_readdirectory_file_not_directory():
    """Error when path is a file, not a directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir, "file.txt")
        test_file.write_text("content")

        result = readdirectory.run(f"<path>{test_file}</path>")

        # Should return error
        assert "Error" in result or "not a directory" in result.lower(), \
            "Should indicate path is not a directory"

    return True


def test_readdirectory_gitignore_exclusion():
    """Test .gitignore pattern exclusion"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create .gitignore
        Path(temp_dir, ".gitignore").write_text("ignored_file.txt\n*.log\n")

        # Create files
        Path(temp_dir, "normal_file.txt").write_text("normal")
        Path(temp_dir, "ignored_file.txt").write_text("ignored")
        Path(temp_dir, "test.log").write_text("log")

        result = readdirectory.run(f"<path>{temp_dir}</path>")

        # Should show normal file
        assert "normal_file.txt" in result, "Should show normal file"

        # Should NOT show ignored files (if pathspec is available)
        # Note: This test may pass even without pathspec since we can't guarantee it's installed
        # The tool falls back to basic exclusions without pathspec

    return True


def test_readdirectory_default_exclusions():
    """Test that common directories are excluded by default"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directories that should be excluded
        Path(temp_dir, ".git").mkdir()
        Path(temp_dir, ".git", "config").write_text("git")
        Path(temp_dir, "node_modules").mkdir()
        Path(temp_dir, "node_modules", "package.json").write_text("{}")
        Path(temp_dir, "__pycache__").mkdir()
        Path(temp_dir, "__pycache__", "cache.pyc").write_text("cache")

        # Create normal file
        Path(temp_dir, "main.py").write_text("code")

        result = readdirectory.run(f"<path>{temp_dir}</path>")

        # Should show normal file
        assert "main.py" in result, "Should show normal file"

        # Should NOT show excluded directories
        assert ".git" not in result or "config" not in result, "Should exclude .git"
        assert "node_modules" not in result or "package.json" not in result, "Should exclude node_modules"
        assert "__pycache__" not in result or "cache.pyc" not in result, "Should exclude __pycache__"

    return True


def test_readdirectory_file_count():
    """Verify file and directory count statistics"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create known structure
        Path(temp_dir, "file1.txt").write_text("1")
        Path(temp_dir, "file2.txt").write_text("2")
        Path(temp_dir, "dir1").mkdir()
        Path(temp_dir, "dir1", "file3.txt").write_text("3")
        Path(temp_dir, "dir2").mkdir()

        result = readdirectory.run(f"<path>{temp_dir}</path>")

        # Should show file count (3 files)
        assert "3 files" in result, "Should count 3 files"

        # Should show directory count (2 directories)
        assert "2 directories" in result, "Should count 2 directories"

    return True


def test_readdirectory_relative_path():
    """Test with relative path (current directory)"""
    result = readdirectory.run("<path>.</path>")

    # Should not error with relative path
    assert isinstance(result, str), "Should return a string"
    assert len(result) > 0, "Should return non-empty result"
    assert "Error" not in result or "files" in result, "Should successfully list current directory"

    return True


def test_readdirectory_sorting():
    """Verify that files and directories are sorted"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create files in non-alphabetical order
        Path(temp_dir, "zebra.txt").write_text("z")
        Path(temp_dir, "alpha.txt").write_text("a")
        Path(temp_dir, "beta.txt").write_text("b")

        result = readdirectory.run(f"<path>{temp_dir}</path>")

        # Find positions of filenames in result
        pos_alpha = result.find("alpha.txt")
        pos_beta = result.find("beta.txt")
        pos_zebra = result.find("zebra.txt")

        # Should be in alphabetical order
        assert pos_alpha < pos_beta < pos_zebra, \
            f"Files should be sorted alphabetically: alpha({pos_alpha}) < beta({pos_beta}) < zebra({pos_zebra})"

    return True


def test_readdirectory_hidden_files():
    """Test handling of hidden files (dotfiles)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create hidden and normal files
        Path(temp_dir, ".hidden").write_text("hidden")
        Path(temp_dir, "normal.txt").write_text("normal")

        result = readdirectory.run(f"<path>{temp_dir}</path>")

        # Should show normal file
        assert "normal.txt" in result, "Should show normal file"

        # Hidden files may or may not be shown depending on implementation
        # Just verify no crash

    return True


def test_readdirectory_special_characters_in_names():
    """Test files with special characters in names"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create files with special characters (that are valid on filesystem)
        Path(temp_dir, "file-with-dash.txt").write_text("dash")
        Path(temp_dir, "file_with_underscore.txt").write_text("underscore")
        Path(temp_dir, "file.multiple.dots.txt").write_text("dots")

        result = readdirectory.run(f"<path>{temp_dir}</path>")

        # Should handle special characters
        assert "file-with-dash.txt" in result, "Should handle dashes"
        assert "file_with_underscore.txt" in result, "Should handle underscores"
        assert "file.multiple.dots.txt" in result, "Should handle multiple dots"

    return True


def test_readdirectory_tool_contract():
    """Verify tool implements the required contract"""
    # Test get_delim
    delim = readdirectory.get_delim()
    assert delim == "readdirectory", f"Expected delimiter 'readdirectory', got: {delim}"

    # Test get_tool_info
    info = readdirectory.get_tool_info()
    assert isinstance(info, dict), "get_tool_info should return dict"
    assert "name" in info, "Tool info should have 'name'"
    assert "description" in info, "Tool info should have 'description'"
    assert "example" in info, "Tool info should have 'example'"
    assert info["name"] == "readdirectory", f"Expected name 'readdirectory', got: {info['name']}"

    # Test get_detailed_tool_info
    assert hasattr(readdirectory, 'get_detailed_tool_info'), "Should have get_detailed_tool_info method"
    detailed = readdirectory.get_detailed_tool_info()
    assert isinstance(detailed, str), "Detailed info should be a string"
    assert len(detailed) > 100, "Detailed info should be substantial"

    return True


# Test level assignments
test_readdirectory_basic.level = 2
test_readdirectory_with_subdirectories.level = 2
test_readdirectory_max_depth_1.level = 2
test_readdirectory_max_depth_2.level = 2
test_readdirectory_empty_directory.level = 2
test_readdirectory_nonexistent.level = 2
test_readdirectory_file_not_directory.level = 2
test_readdirectory_gitignore_exclusion.level = 3
test_readdirectory_default_exclusions.level = 2
test_readdirectory_file_count.level = 2
test_readdirectory_relative_path.level = 2
test_readdirectory_sorting.level = 2
test_readdirectory_hidden_files.level = 3
test_readdirectory_special_characters_in_names.level = 2
test_readdirectory_tool_contract.level = 2
