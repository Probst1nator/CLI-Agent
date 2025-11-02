#!/usr/bin/env python3
"""
Test suite for the viewpermissions tool.

This module tests the user-friendly tool that displays saved permissions.
"""
import os

# Add project root to path for imports
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Updated import to use the new viewpermissions tool
from tools.viewpermissions import viewpermissions


class TestViewPermissionsTool:
    """Test viewpermissions tool functionality."""

    def test_get_delim(self):
        """Test that get_delim returns correct delimiter."""
        assert viewpermissions.get_delim() == 'viewpermissions'

    def test_get_tool_info(self):
        """Test that get_tool_info returns proper structure."""
        info = viewpermissions.get_tool_info()
        assert info['name'] == 'viewpermissions'
        assert 'description' in info
        assert 'example' in info
        assert '<viewpermissions>' in info['example']

    # Updated patch path
    @patch('tools.viewpermissions.get_permissions_manager')
    def test_run_with_permissions(self, mock_get_manager):
        """Test that run lists permissions correctly when they exist."""
        # Mock permissions manager
        mock_manager = Mock()
        mock_manager.permissions = {
            "always_allow_dev_tools": {"git": ["git"]},
            "always_allow_tools": {"readfile": ["*.py"]},
            "always_allow_groups": {},
            "always_allow_commands": {}
        }
        mock_get_manager.return_value = mock_manager

        # Test that any input results in the formatted list
        result = viewpermissions.run('anything')

        # Assertions updated to match the new, user-friendly output
        assert "PERMISSIONS - Actions I Can Execute Without Asking" in result
        assert "DEVELOPMENT TOOLS" in result
        assert "Git Commands" in result
        assert "FILE OPERATIONS" in result
        assert "Read Files" in result

    # Updated patch path
    @patch('tools.viewpermissions.get_permissions_manager')
    def test_run_empty_permissions(self, mock_get_manager):
        """Test behavior with no permissions."""
        # Mock empty permissions
        mock_manager = Mock()
        mock_manager.permissions = {
            "always_allow_dev_tools": {},
            "always_allow_tools": {},
            "always_allow_groups": {},
            "always_allow_commands": {}
        }
        mock_get_manager.return_value = mock_manager

        result = viewpermissions.run()
        # This assertion is now correct for the new tool's output
        assert "No automatic permissions set" in result

    # Updated patch path
    @patch('tools.viewpermissions.get_permissions_manager')
    def test_error_handling(self, mock_get_manager):
        """Test error handling in permissions operations."""
        # Mock permissions manager that raises an exception
        mock_get_manager.side_effect = Exception("Test error")

        result = viewpermissions.run()
        # Updated error message to match the new tool
        assert "Error retrieving permissions" in result
        assert "Test error" in result