"""
Permission System Tests

Core tests for the CLI-Agent permission system, focusing on bug fixes
and general functionality validation.
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

from shared.path_resolver import setup_cli_agent_imports
setup_cli_agent_imports()

from core.permissions.permissions_manager import PermissionsManager


class MockPermissionsManager(PermissionsManager):
    """Mock permissions manager for isolated testing"""
    
    def __init__(self, initial_permissions=None):
        # Don't call super().__init__ to avoid file I/O
        self.config_path = None
        self.permissions = initial_permissions or {
            "always_allow_tools": {},
            "always_deny_tools": {},
            "always_allow_commands": {},
            "always_deny_commands": {},
            "always_allow_groups": {},
            "always_deny_groups": {},
            "always_allow_dev_tools": {},
            "pattern_permissions": {},
            "global_always_allow": False
        }
    
    def _save_permissions(self):
        # Override to prevent file I/O during tests
        pass


def test_permission_redundancy_basic():
    """Test that already-permitted actions don't show prompts"""
    
    # Create mock permissions with a dev tool category already allowed
    mock_permissions = MockPermissionsManager({
        "always_allow_tools": {},
        "always_deny_tools": {},
        "always_allow_commands": {},
        "always_deny_commands": {},
        "always_allow_groups": {},
        "always_deny_groups": {},
        "always_allow_dev_tools": {
            "linting": ["/test/project/*"]  # Linting tools already allowed
        },
        "pattern_permissions": {},
        "global_always_allow": False
    })
    
    # Test smart action for linting in the allowed directory
    smart_action = {
        'category': 'linting',
        'quick_key': 'r',
        'display_name': 'linting tools',
        'scope_path': '/test/project/*',
        'scope_description': 'this project',
        'permission_scope': 'project'
    }
    
    # Should return True (already permitted) when in the allowed directory
    with patch('os.getcwd', return_value='/test/project'):
        result = mock_permissions.is_smart_action_already_permitted(smart_action, 'bash', '/test/project/src/file.py')
        if not result:
            return False
    
    # Should return False when in a different directory
    with patch('os.getcwd', return_value='/other/project'):
        result = mock_permissions.is_smart_action_already_permitted(smart_action, 'bash', '/other/project/src/file.py')
        if result:
            return False
    
    return True


def test_pattern_scope_context():
    """Test pattern permissions work with file paths, not just CWD"""
    
    mock_permissions = MockPermissionsManager({
        "always_allow_tools": {},
        "always_deny_tools": {},
        "always_allow_commands": {},
        "always_deny_commands": {},
        "always_allow_groups": {},
        "always_deny_groups": {},
        "always_allow_dev_tools": {},
        "pattern_permissions": {
            "bash:ls*": "/test/project/*"  # ls commands allowed in /test/project
        },
        "global_always_allow": False
    })
    
    # Test that pattern permission works with file path context
    # even when CWD is different
    with patch('os.getcwd', return_value='/different/directory'):
        # Should be allowed because file path is in allowed directory
        result = mock_permissions._is_pattern_allowed('bash', 'bash: ls -la', '/test/project/src/file.py')
        if not result:
            return False
        
        # Should not be allowed for files outside the pattern
        result = mock_permissions._is_pattern_allowed('bash', 'bash: ls -la', '/other/project/file.py')
        if result:
            return False
    
    return True


def test_smart_action_filtering():
    """Test smart actions aren't shown when already permitted"""
    
    mock_permissions = MockPermissionsManager({
        "always_allow_tools": {},
        "always_deny_tools": {},
        "always_allow_commands": {},
        "always_deny_commands": {},
        "always_allow_groups": {},
        "always_deny_groups": {},
        "always_allow_dev_tools": {
            "testing": ["/test/project/*"],  # Testing already allowed
            "git": ["git"]  # Git globally allowed
        },
        "pattern_permissions": {},
        "global_always_allow": False
    })
    
    # Test that testing tools smart action is filtered out in allowed directory
    testing_action = {
        'category': 'testing',
        'quick_key': 't',
        'display_name': 'testing commands',
        'scope_path': '/test/project/*',
        'scope_description': 'this project',
        'permission_scope': 'project'
    }
    
    with patch('os.getcwd', return_value='/test/project'):
        result = mock_permissions.is_smart_action_already_permitted(testing_action, 'bash', '/test/project/test_file.py')
        if not result:
            return False
    
    # Test that git smart action is filtered out globally
    git_action = {
        'category': 'git',
        'quick_key': 'g', 
        'display_name': 'git commands',
        'scope_path': 'git',
        'scope_description': 'globally',
        'permission_scope': 'global'
    }
    
    result = mock_permissions.is_smart_action_already_permitted(git_action, 'bash', '/any/directory/file.py')
    if not result:
        return False
    
    return True


def test_permission_precedence_order():
    """Test exact → pattern → dev_tool → group precedence"""
    
    mock_permissions = MockPermissionsManager({
        "always_allow_tools": {
            "bash": ["/test/project/specific_file.py"]  # Exact file permission
        },
        "always_deny_tools": {},
        "always_allow_commands": {},
        "always_deny_commands": {},
        "always_allow_groups": {},
        "always_deny_groups": {},
        "always_allow_dev_tools": {
            "linting": ["/test/project/*"]  # Dev tool category permission
        },
        "pattern_permissions": {
            "bash:ls*": "/test/project/*"  # Pattern permission
        },
        "global_always_allow": False
    })
    
    # Test exact file permission (should be highest precedence)
    result = mock_permissions.is_always_allowed('bash', '/test/project/specific_file.py')
    if not result:
        return False
    
    # Test pattern permission for ls command
    with patch('os.getcwd', return_value='/test/project'):
        xml_content = '<bash>ls -la</bash>'
        result = mock_permissions.is_always_allowed('bash', '/test/project/other_file.py', xml_content=xml_content)
        if not result:
            return False
    
    # Test dev tool permission for linting
    with patch('os.getcwd', return_value='/test/project'):
        xml_content = '<bash>ruff check .</bash>'
        result = mock_permissions.is_always_allowed('bash', '/test/project/src/code.py', xml_content=xml_content)
        if not result:
            return False
    
    return True


def test_directory_pattern_matching():
    """Test directory patterns work for nested paths"""
    
    mock_permissions = MockPermissionsManager({
        "always_allow_tools": {
            "readfile": ["/project/docs/*"]  # Allow readfile in docs directory
        },
        "always_deny_tools": {},
        "always_allow_commands": {},
        "always_deny_commands": {},
        "always_allow_groups": {},
        "always_deny_groups": {},
        "always_allow_dev_tools": {},
        "pattern_permissions": {},
        "global_always_allow": False
    })
    
    # Test that directory pattern matches nested paths
    test_cases = [
        ("/project/docs/readme.md", True),           # Direct child
        ("/project/docs/api/endpoints.md", True),   # Nested path
        ("/project/docs/deep/nested/file.txt", True), # Deeply nested
        ("/project/src/code.py", False),            # Different directory
        ("/other/project/docs/file.md", False),     # Different root
    ]
    
    for file_path, should_be_allowed in test_cases:
        result = mock_permissions.is_always_allowed('readfile', file_path)
        if result != should_be_allowed:
            return False
    
    return True


def test_dev_tool_category_context():
    """Test dev tool categories work with file path context"""
    
    mock_permissions = MockPermissionsManager({
        "always_allow_tools": {},
        "always_deny_tools": {},
        "always_allow_commands": {},
        "always_deny_commands": {},
        "always_allow_groups": {},
        "always_deny_groups": {},
        "always_allow_dev_tools": {
            "linting": ["/project/src/*"]  # Linting allowed only in src directory
        },
        "pattern_permissions": {},
        "global_always_allow": False
    })
    
    # Test with different CWD but file in allowed directory
    with patch('os.getcwd', return_value='/different/directory'):
        # Should be allowed because file is in /project/src/
        result = mock_permissions._is_dev_tool_allowed('bash: ruff check file.py', '/project/src/main.py')
        if not result:
            return False
        
        # Should not be allowed for files outside src
        result = mock_permissions._is_dev_tool_allowed('bash: ruff check file.py', '/project/tests/test.py')
        if result:
            return False
    
    return True


if __name__ == "__main__":
    # Run all tests when executed directly
    
    tests = [
        test_permission_redundancy_basic,
        test_pattern_scope_context,
        test_smart_action_filtering,
        test_permission_precedence_order,
        test_directory_pattern_matching,
        test_dev_tool_category_context,
    ]
    
    print("Running permission system tests...")
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        test_name = test_func.__name__
        print(f"\n  Running {test_name}...")
        
        try:
            result = test_func()
            if result:
                print(f"    ✅ {test_name} passed")
                passed += 1
            else:
                print(f"    ❌ {test_name} failed")
        except Exception as e:
            print(f"    ❌ {test_name} error: {e}")
            import traceback
            print(f"      {traceback.format_exc()}")
    
    print(f"\nPermission System Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All permission system tests passed!")
    else:
        print("⚠️  Some permission system tests failed")