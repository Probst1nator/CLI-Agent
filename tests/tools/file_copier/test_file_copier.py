"""
File Copier Tool Tests

Tests for the file copier tool including smart pasting,
file discovery, and GUI functionality.
"""

import sys
import os
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(test_root))

from shared.path_resolver import setup_cli_agent_imports
setup_cli_agent_imports()


def test_file_copier_structure():
    """Test file copier has expected structure"""
    from shared.path_resolver import PathResolver
    
    tool_path = PathResolver.get_tool_path("file_copier")
    
    # Check essential files exist
    expected_files = ["main.py", "gui.py", "smart_paster.py", "__init__.py"]
    for file_name in expected_files:
        file_path = tool_path / file_name
        assert file_path.exists(), f"Missing file: {file_path}"
    
    return True


def test_smart_paster_imports():
    """Test smart paster can be imported"""
    try:
        # Import from tool directory
        tool_path = Path(__file__).parent.parent.parent.parent / "tools" / "file_copier"
        sys.path.insert(0, str(tool_path))
        
        from smart_paster import build_clipboard_content, apply_changes_to_files
        
        # Test basic function availability
        assert callable(build_clipboard_content), "build_clipboard_content not callable"
        assert callable(apply_changes_to_files), "apply_changes_to_files not callable"
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False


def test_ai_path_finder_imports():
    """Test AI path finder can be imported"""
    try:
        tool_path = Path(__file__).parent.parent.parent.parent / "tools" / "file_copier"
        sys.path.insert(0, str(tool_path))
        
        from ai_path_finder import AIFixPath
        
        assert AIFixPath is not None, "AIFixPath is None"
        
        return True
    except ImportError as e:
        print(f"AI path finder import error: {e}")
        return False


def test_file_copier_constants():
    """Test file copier has expected constants"""
    try:
        tool_path = Path(__file__).parent.parent.parent.parent / "tools" / "file_copier"
        sys.path.insert(0, str(tool_path))
        
        from smart_paster import IGNORE_DIRS, IGNORE_FILES
        
        # Check constants are sets with expected values
        assert isinstance(IGNORE_DIRS, set), "IGNORE_DIRS not a set"
        assert isinstance(IGNORE_FILES, set), "IGNORE_FILES not a set"
        assert "__pycache__" in IGNORE_DIRS, "Missing __pycache__ in IGNORE_DIRS"
        assert ".DS_Store" in IGNORE_FILES, "Missing .DS_Store in IGNORE_FILES"
        
        return True
    except ImportError as e:
        print(f"Constants import error: {e}")
        return False


if __name__ == "__main__":
    tests = [
        test_file_copier_structure,
        test_smart_paster_imports,
        test_ai_path_finder_imports,
        test_file_copier_constants
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            print(f"✅ {test.__name__}: {'PASS' if result else 'FAIL'}")
            results.append(result)
        except Exception as e:
            print(f"❌ {test.__name__}: ERROR - {e}")
            results.append(False)
    
    print(f"\nFile Copier Tests: {sum(results)}/{len(results)} passed")