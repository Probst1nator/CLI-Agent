"""
Todos Tool Tests

Tests for the todos tool including persistent markdown task list
management and parsing functionality.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Setup test imports
test_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(test_root))

# Import the tool
from tools.primitives.todos import todos


def test_todos_save_list():
    """Test saving a todo list to persistent storage"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the storage path to use temp directory
        with patch('tools.primitives.todos.g.CLIAGENT_PERSISTENT_STORAGE_PATH', temp_dir):
            
            todo_content = """- [x] Completed task 1
- [ ] Incomplete task 2
- [x] Another completed task
- [ ] Another incomplete task"""
            
            result = todos.run(todo_content)
            
            # Verify success message
            assert result == "Todos list updated successfully.", f"Expected success message, got: {result}"
            
            # Verify file was created
            expected_path = os.path.join(temp_dir, "todos.md")
            assert os.path.exists(expected_path), "Todo file should be created"
            
            # Verify content was saved correctly
            with open(expected_path, 'r', encoding='utf-8') as f:
                saved_content = f.read()
            
            assert saved_content == todo_content, f"Content mismatch. Expected: {repr(todo_content)}, Got: {repr(saved_content)}"
            
            return True


def test_todos_update_existing_list():
    """Test updating an existing todo list"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('tools.primitives.todos.g.CLIAGENT_PERSISTENT_STORAGE_PATH', temp_dir):
            
            # Create initial todo list
            initial_content = """- [x] Task 1
- [ ] Task 2"""
            todos.run(initial_content)
            
            # Update with new content
            updated_content = """- [x] Task 1
- [x] Task 2
- [ ] Task 3
- [ ] Task 4"""
            
            result = todos.run(updated_content)
            
            # Verify success
            assert result == "Todos list updated successfully.", f"Expected success message, got: {result}"
            
            # Verify updated content
            expected_path = os.path.join(temp_dir, "todos.md")
            with open(expected_path, 'r', encoding='utf-8') as f:
                saved_content = f.read()
            
            assert saved_content == updated_content, f"Updated content mismatch. Expected: {repr(updated_content)}, Got: {repr(saved_content)}"
            
            return True


def test_todos_get_next_unchecked_task():
    """Test retrieving the next unchecked task"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('tools.primitives.todos.g.CLIAGENT_PERSISTENT_STORAGE_PATH', temp_dir):
            
            # Create todo list with mixed completed/incomplete tasks
            todo_content = """- [x] Completed task 1
- [ ] First incomplete task
- [x] Completed task 2
- [ ] Second incomplete task
- [x] Another completed task"""
            
            todos.run(todo_content)
            
            # Get next unchecked task
            next_task = todos.get_next_unchecked_task()
            
            # Should return the first unchecked task
            assert next_task == "First incomplete task", f"Expected 'First incomplete task', got: {repr(next_task)}"
            
            return True


def test_todos_get_next_unchecked_no_file():
    """Test getting next unchecked task when no file exists"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('tools.primitives.todos.g.CLIAGENT_PERSISTENT_STORAGE_PATH', temp_dir):
            
            # Try to get next task when no file exists
            next_task = todos.get_next_unchecked_task()
            
            # Should return None
            assert next_task is None, f"Expected None when no file exists, got: {repr(next_task)}"
            
            return True


def test_todos_get_next_unchecked_all_completed():
    """Test getting next unchecked task when all tasks are completed"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('tools.primitives.todos.g.CLIAGENT_PERSISTENT_STORAGE_PATH', temp_dir):
            
            # Create todo list with all tasks completed
            todo_content = """- [x] Completed task 1
- [x] Completed task 2
- [x] Completed task 3"""
            
            todos.run(todo_content)
            
            # Get next unchecked task
            next_task = todos.get_next_unchecked_task()
            
            # Should return None since all are completed
            assert next_task is None, f"Expected None when all tasks completed, got: {repr(next_task)}"
            
            return True


def test_todos_get_next_unchecked_complex_content():
    """Test getting next unchecked task with complex content"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('tools.primitives.todos.g.CLIAGENT_PERSISTENT_STORAGE_PATH', temp_dir):
            
            # Create todo list with complex task descriptions
            todo_content = """# My Todo List

- [x] Task with "quotes" and special chars: @#$%
- [ ] Task with 🚀 emojis and ñáéíóú unicode
- [x] Multi-word task with lots of details and explanations
- [ ] Simple task
- [x] Task with [brackets] and (parentheses)

## Notes
Some additional notes here.
"""
            
            todos.run(todo_content)
            
            # Get next unchecked task
            next_task = todos.get_next_unchecked_task()
            
            # Should return the first unchecked task with special characters preserved
            expected = 'Task with 🚀 emojis and ñáéíóú unicode'
            assert next_task == expected, f"Expected {repr(expected)}, got: {repr(next_task)}"
            
            return True


def test_todos_empty_content():
    """Test saving empty todo content"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('tools.primitives.todos.g.CLIAGENT_PERSISTENT_STORAGE_PATH', temp_dir):
            
            result = todos.run("")
            
            # Should still succeed
            assert result == "Todos list updated successfully.", f"Expected success with empty content, got: {result}"
            
            # Verify empty file was created
            expected_path = os.path.join(temp_dir, "todos.md")
            assert os.path.exists(expected_path), "Empty todo file should be created"
            
            with open(expected_path, 'r', encoding='utf-8') as f:
                saved_content = f.read()
            
            assert saved_content == "", f"Expected empty content, got: {repr(saved_content)}"
            
            return True


def test_todos_whitespace_handling():
    """Test handling of various whitespace patterns"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('tools.primitives.todos.g.CLIAGENT_PERSISTENT_STORAGE_PATH', temp_dir):
            
            # Content with various whitespace patterns
            todo_content = """  - [x] Task with leading spaces
\t- [ ] Task with leading tab
- [x]   Task with spaces after checkbox
- [ ]\tTask with tab after checkbox
  
- [ ] Task after empty lines
\t\t
- [x] Final task"""
            
            todos.run(todo_content)
            
            # Get next unchecked task
            next_task = todos.get_next_unchecked_task()
            
            # Should find the first unchecked task, preserving whitespace correctly
            expected = "Task with leading tab"
            assert next_task == expected, f"Expected {repr(expected)}, got: {repr(next_task)}"
            
            return True


def test_todos_malformed_checkboxes():
    """Test handling of malformed checkbox patterns"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('tools.primitives.todos.g.CLIAGENT_PERSISTENT_STORAGE_PATH', temp_dir):
            
            # Content with various checkbox formats (some malformed)
            todo_content = """- [x] Properly completed task
- [ ] Properly unchecked task
- [X] Uppercase X (should not match)
- [ ] Missing space in checkbox
-[ ] Missing space before bracket
- [x] Another properly completed task
- [ ] Another properly unchecked task"""
            
            todos.run(todo_content)
            
            # Get next unchecked task
            next_task = todos.get_next_unchecked_task()
            
            # Should find the first properly formatted unchecked task
            expected = "Properly unchecked task"
            assert next_task == expected, f"Expected {repr(expected)}, got: {repr(next_task)}"
            
            return True


def test_todos_directory_creation():
    """Test automatic directory creation for storage path"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use nested directory that doesn't exist yet
        nested_dir = os.path.join(temp_dir, "nested", "storage", "path")
        
        with patch('tools.primitives.todos.g.CLIAGENT_PERSISTENT_STORAGE_PATH', nested_dir):
            
            todo_content = "- [ ] Test task"
            result = todos.run(todo_content)
            
            # Should succeed and create directories
            assert result == "Todos list updated successfully.", f"Expected success, got: {result}"
            
            # Verify directories were created
            assert os.path.exists(nested_dir), "Storage directory should be created"
            
            # Verify file exists
            expected_path = os.path.join(nested_dir, "todos.md")
            assert os.path.exists(expected_path), "Todo file should be created in nested directory"
            
            return True


def test_todos_file_permissions_error():
    """Test handling of file permission errors"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a read-only directory (simulate permission error)
        readonly_dir = os.path.join(temp_dir, "readonly")
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, 0o444)  # Read-only
        
        try:
            with patch('tools.primitives.todos.g.CLIAGENT_PERSISTENT_STORAGE_PATH', readonly_dir):
                
                todo_content = "- [ ] Test task"
                result = todos.run(todo_content)
                
                # Should return error message
                assert result.startswith("Error saving todos:"), f"Expected error message, got: {result}"
                
        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)
        
        return True


def test_todos_large_content():
    """Test handling of large todo lists"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('tools.primitives.todos.g.CLIAGENT_PERSISTENT_STORAGE_PATH', temp_dir):
            
            # Create large todo list
            lines = []
            for i in range(1000):
                status = "[x]" if i % 3 == 0 else "[ ]"
                lines.append(f"- {status} Task number {i+1} with some description")
            
            todo_content = "\n".join(lines)
            result = todos.run(todo_content)
            
            # Should succeed with large content
            assert result == "Todos list updated successfully.", f"Expected success with large content, got: {result}"
            
            # Get next unchecked task (should find first unchecked one)
            next_task = todos.get_next_unchecked_task()
            
            # First unchecked should be task 2 (since task 1 is completed: 1 % 3 == 1, not 0)
            expected = "Task number 2 with some description"
            assert next_task == expected, f"Expected {repr(expected)}, got: {repr(next_task)}"
            
            return True


# Test level assignments
test_todos_save_list.level = 2
test_todos_update_existing_list.level = 2
test_todos_get_next_unchecked_task.level = 2
test_todos_get_next_unchecked_no_file.level = 2
test_todos_get_next_unchecked_all_completed.level = 2
test_todos_get_next_unchecked_complex_content.level = 2
test_todos_empty_content.level = 2
test_todos_whitespace_handling.level = 2
test_todos_malformed_checkboxes.level = 2
test_todos_directory_creation.level = 2
test_todos_file_permissions_error.level = 3
test_todos_large_content.level = 3