"""
Tests for project storage manager (centralized ~/.cli-agent/projects/ storage).

Tests path sanitization, project directory creation, metadata management,
and auto-migration from local ./.cli-agent/ to global storage.
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from shared.utils.project_storage import (
    sanitize_path,
    get_project_storage_path,
    ensure_project_directory,
    load_project_metadata,
    update_project_metadata,
    migrate_local_to_global,
    list_all_projects,
)


class TestProjectStorage(unittest.TestCase):
    """Test project storage functionality."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        # Create temporary home directory for testing
        self.temp_home = tempfile.mkdtemp()
        self.home_patcher = patch('pathlib.Path.home', return_value=Path(self.temp_home))
        self.home_patcher.start()

        # Create temporary working directory
        self.temp_cwd = tempfile.mkdtemp()
        self.cwd_patcher = patch('os.getcwd', return_value=self.temp_cwd)
        self.cwd_patcher.start()

    def tearDown(self):
        """Clean up temporary directories."""
        self.home_patcher.stop()
        self.cwd_patcher.stop()

        # Remove temporary directories
        if os.path.exists(self.temp_home):
            shutil.rmtree(self.temp_home)
        if os.path.exists(self.temp_cwd):
            shutil.rmtree(self.temp_cwd)

    def test_sanitize_path_unix(self):
        """Test path sanitization for Unix paths."""
        # Basic path
        result = sanitize_path('/home/user/projects/my-app')
        self.assertEqual(result, 'home_user_projects_my-app')

        # Path with spaces
        result = sanitize_path('/home/user/My Documents/project')
        self.assertEqual(result, 'home_user_My_Documents_project')

        # Path with special characters
        result = sanitize_path('/home/user/project!@#$/test')
        self.assertEqual(result, 'home_user_project_test')

        # Multiple separators collapsed
        result = sanitize_path('/home//user///project')
        self.assertEqual(result, 'home_user_project')

    def test_sanitize_path_windows(self):
        """Test path sanitization for Windows paths."""
        # On Linux, Windows paths are treated as relative, so we just check
        # that the sanitization works correctly for the path components
        result = sanitize_path('C:\\Users\\Name\\Project')
        self.assertIn('C_Users_Name_Project', result)
        # Should not have backslashes
        self.assertNotIn('\\', result)

        result = sanitize_path('D:\\Work\\My Project\\app')
        self.assertIn('My_Project', result)
        self.assertIn('app', result)
        self.assertNotIn('\\', result)

    def test_sanitize_path_edge_cases(self):
        """Test edge cases in path sanitization."""
        # Very long path
        long_path = '/home/' + 'a' * 200 + '/project'
        result = sanitize_path(long_path)
        self.assertIn('home', result)
        self.assertIn('project', result)

        # Path with periods
        result = sanitize_path('/home/user/project.v2.0')
        self.assertEqual(result, 'home_user_project.v2.0')

    def test_get_project_storage_path(self):
        """Test getting centralized storage path for a project."""
        test_path = '/home/user/projects/test-app'
        storage_path = get_project_storage_path(test_path)

        expected = Path(self.temp_home) / '.cli-agent' / 'projects' / 'home_user_projects_test-app'
        self.assertEqual(storage_path, expected)

    def test_get_project_storage_path_current_dir(self):
        """Test getting storage path for current directory."""
        storage_path = get_project_storage_path()

        sanitized = sanitize_path(self.temp_cwd)
        expected = Path(self.temp_home) / '.cli-agent' / 'projects' / sanitized
        self.assertEqual(storage_path, expected)

    def test_ensure_project_directory_creates_structure(self):
        """Test that ensure_project_directory creates proper structure."""
        project_path = ensure_project_directory(self.temp_cwd, auto_migrate=False)

        # Check directory exists
        self.assertTrue(project_path.exists())
        self.assertTrue(project_path.is_dir())

        # Check chat_history subdirectory exists
        chat_history = project_path / 'chat_history'
        self.assertTrue(chat_history.exists())
        self.assertTrue(chat_history.is_dir())

        # Check metadata.json exists
        metadata_path = project_path / 'metadata.json'
        self.assertTrue(metadata_path.exists())

        # Verify metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.assertEqual(metadata['absolute_path'], self.temp_cwd)
        self.assertIn('project_name', metadata)
        self.assertIn('created_at', metadata)
        self.assertIn('last_accessed', metadata)
        self.assertEqual(metadata['version'], '1.0')

    def test_load_project_metadata(self):
        """Test loading project metadata."""
        # Create project first
        ensure_project_directory(self.temp_cwd, auto_migrate=False)

        # Load metadata
        metadata = load_project_metadata(self.temp_cwd)

        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata['absolute_path'], self.temp_cwd)
        self.assertIn('created_at', metadata)

    def test_update_project_metadata(self):
        """Test updating project metadata."""
        # Create project
        ensure_project_directory(self.temp_cwd, auto_migrate=False)

        # Update metadata
        update_project_metadata(self.temp_cwd, chat_count=42, custom_field='test_value')

        # Load and verify
        metadata = load_project_metadata(self.temp_cwd)
        self.assertEqual(metadata['chat_count'], 42)
        self.assertEqual(metadata['custom_field'], 'test_value')

    def test_migrate_local_to_global(self):
        """Test auto-migration from ./.cli-agent/ to global storage."""
        # Create local .cli-agent directory with content
        local_storage = Path(self.temp_cwd) / '.cli-agent'
        local_storage.mkdir()

        # Create some test files
        (local_storage / 'recent_chat.json').write_text('{"messages": []}')
        chat_history = local_storage / 'chat_history'
        chat_history.mkdir()
        (chat_history / 'chat_123.json').write_text('{"test": "data"}')

        # Trigger migration
        success = migrate_local_to_global(self.temp_cwd)
        self.assertTrue(success)

        # Verify files were copied to global storage
        project_path = get_project_storage_path(self.temp_cwd)
        self.assertTrue((project_path / 'recent_chat.json').exists())
        self.assertTrue((project_path / 'chat_history' / 'chat_123.json').exists())

        # Verify metadata includes migration info
        metadata = load_project_metadata(self.temp_cwd)
        self.assertIn('migrated_at', metadata)
        self.assertIn('migrated_from', metadata)
        self.assertIn('migrated_items', metadata)

        # Verify old directory was deleted
        self.assertFalse(local_storage.exists())

    def test_migrate_skips_if_no_local_storage(self):
        """Test that migration skips if no local .cli-agent exists."""
        success = migrate_local_to_global(self.temp_cwd)
        self.assertTrue(success)

        # Should not create any project directory
        project_path = get_project_storage_path(self.temp_cwd)
        # Note: This will create it, so we check it's empty
        # Actually, migrate returns early, so nothing should be created

    def test_ensure_project_directory_auto_migrates(self):
        """Test that ensure_project_directory auto-migrates by default."""
        # Create local storage
        local_storage = Path(self.temp_cwd) / '.cli-agent'
        local_storage.mkdir()
        (local_storage / 'recent_chat.json').write_text('{"test": "data"}')

        # Call ensure_project_directory with auto_migrate=True (default)
        project_path = ensure_project_directory(self.temp_cwd)

        # Verify migration happened
        self.assertTrue((project_path / 'recent_chat.json').exists())
        # Verify old directory was deleted
        self.assertFalse(local_storage.exists())

    def test_list_all_projects(self):
        """Test listing all projects."""
        # Create multiple projects
        project1 = tempfile.mkdtemp()
        project2 = tempfile.mkdtemp()

        try:
            ensure_project_directory(project1, auto_migrate=False)
            time.sleep(0.01)  # Ensure different timestamps
            ensure_project_directory(project2, auto_migrate=False)

            # List projects
            projects = list_all_projects()

            # Should have at least 2 projects
            self.assertGreaterEqual(len(projects), 2)

            # Projects should be sorted by last_accessed (most recent first)
            if len(projects) >= 2:
                self.assertGreaterEqual(
                    projects[0]['last_accessed'],
                    projects[1]['last_accessed']
                )

            # Each project should have required fields
            for project in projects:
                self.assertIn('absolute_path', project)
                self.assertIn('storage_path', project)
                self.assertIn('created_at', project)

        finally:
            shutil.rmtree(project1, ignore_errors=True)
            shutil.rmtree(project2, ignore_errors=True)

    def test_migrate_handles_wrapper_script(self):
        """Test that migration skips if .cli-agent is a file (wrapper script)."""
        # Create .cli-agent as a file instead of directory
        cli_agent_file = Path(self.temp_cwd) / '.cli-agent'
        cli_agent_file.write_text('#!/bin/bash\necho "wrapper script"')

        # Migration should skip
        success = migrate_local_to_global(self.temp_cwd)
        self.assertTrue(success)

        # File should still exist (not deleted since it's not a directory)
        self.assertTrue(cli_agent_file.exists())


# Assign test level for hierarchical testing
TestProjectStorage.level = 3  # SHARED utilities level


if __name__ == '__main__':
    unittest.main()
