"""
PathResolver Unit Tests

Detailed tests for the PathResolver utility that handles
import path resolution across all CLI-Agent components.
"""

import sys
import os
from pathlib import Path

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

from shared.path_resolver import PathResolver, setup_cli_agent_imports


def test_find_cli_agent_root():
    """Test finding CLI-Agent root directory"""
    root = PathResolver.find_cli_agent_root()
    
    # Should be a valid directory
    assert root.exists(), f"Root directory does not exist: {root}"
    assert root.is_dir(), f"Root is not a directory: {root}"
    
    # Should contain expected markers
    markers = ["py_classes", "tools"]  # At least one should exist
    has_markers = any((root / marker).exists() for marker in markers)
    assert has_markers, f"Root directory missing expected markers: {root}"
    
    return True


def test_get_tool_paths():
    """Test getting tool-specific paths"""
    tools = ["file_copier", "podcast_generator", "main_cli_agent"]
    
    for tool in tools:
        tool_path = PathResolver.get_tool_path(tool)
        # Path should be under tools directory
        assert "tools" in str(tool_path), f"Tool path not under tools/: {tool_path}"
        # Directory should exist (we created it in setup)
        assert tool_path.exists(), f"Tool directory missing: {tool_path}"
    
    return True


def test_get_shared_path():
    """Test getting shared utilities path"""
    shared_path = PathResolver.get_shared_path()
    
    assert shared_path.exists(), f"Shared directory missing: {shared_path}"
    assert shared_path.name == "shared", f"Wrong shared directory name: {shared_path}"
    
    # Should have __init__.py
    init_file = shared_path / "__init__.py"
    assert init_file.exists(), f"Shared __init__.py missing: {init_file}"
    
    return True


def test_get_core_path():
    """Test getting core infrastructure path"""
    core_path = PathResolver.get_core_path()
    
    assert core_path.exists(), f"Core directory missing: {core_path}"
    assert core_path.name == "core", f"Wrong core directory name: {core_path}"
    
    # Should have __init__.py
    init_file = core_path / "__init__.py"
    assert init_file.exists(), f"Core __init__.py missing: {init_file}"
    
    return True


def test_setup_imports():
    """Test the setup_cli_agent_imports convenience function"""
    root = setup_cli_agent_imports()
    
    assert root.exists(), "setup_cli_agent_imports returned invalid path"
    
    # Should have added to sys.path
    root_str = str(root)
    assert root_str in sys.path, "CLI-Agent root not added to sys.path"
    
    return True


if __name__ == "__main__":
    tests = [
        test_find_cli_agent_root,
        test_get_tool_paths,
        test_get_shared_path,
        test_get_core_path,
        test_setup_imports
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            print(f"✅ {test.__name__}: PASS")
            results.append(True)
        except Exception as e:
            print(f"❌ {test.__name__}: FAIL - {e}")
            results.append(False)
    
    print(f"\nPathResolver Tests: {sum(results)}/{len(results)} passed")