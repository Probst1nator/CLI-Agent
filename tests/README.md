# CLI-Agent Testing Framework

This directory contains the unified testing framework for CLI-Agent, including comprehensive tool testing capabilities.

## Quick Start

From the project root, run:

```bash
# Essential tests (recommended)
python test.py

# Quick tests only (< 30s)
python test.py --quick

# Complete test suite
python test.py --full

# Autonomous end-to-end tests
python test.py --autonomous
```

## Test Structure

- **Integration Tests (Level 0)**: End-to-end autonomous testing where CLI-Agent actually creates files and runs code
- **Tool Tests (Level 1)**: Individual tool functionality validation
- **Core Tests (Level 2)**: Core infrastructure and imports (default level)
- **Shared Tests (Level 3)**: Shared utilities validation
- **Unit Tests (Level 4)**: Detailed component testing

## Tool Testing Framework

### Overview

The CLI-Agent tool testing framework provides comprehensive validation for all file manipulation and utility tools. Each tool is tested for functionality, error handling, edge cases, and integration capabilities.

### Tested Tools

Located in `tests/tools/`, the framework includes comprehensive test suites for:

1. **`test_editfile.py`** - Find/replace and line-specific editing
2. **`test_writefile.py`** - File creation and overwriting capabilities  
3. **`test_readfile.py`** - File reading with truncation and line indexing
4. **`test_insertline.py`** - Line insertion at specific positions
5. **`test_deletelines.py`** - Single line and range deletion
6. **`test_search.py`** - Recursive file/content search with regex
7. **`test_todos.py`** - Persistent markdown task list management

### Test Coverage

Each tool test suite covers:

- **Core Functionality**: Basic operations work as expected
- **Error Handling**: Invalid inputs, missing files, malformed XML
- **Edge Cases**: Empty files, single lines, large files, boundary conditions
- **Special Characters**: Unicode, quotes, regex characters, whitespace preservation
- **File System Operations**: Path resolution, directory creation, permissions
- **Content Validation**: Exact content matching and data preservation

### Test Architecture

```python
# Standard test function structure
def test_tool_functionality():
    """Test description"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup test files and data
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # Execute tool with XML input
        xml_input = f"<filepath>{test_file}</filepath><content>new content</content>"
        result = tool.run(xml_input)
        
        # Verify results
        assert "Success" in result, "Expected success message"
        # Verify file changes, content preservation, etc.
        
        return True

# Test level assignment
test_tool_functionality.level = 2
```

### Test Patterns

**File Operations Testing**:
- Create temporary directories for isolation
- Test with various file sizes and content types
- Verify exact content matching before/after operations
- Test path resolution (relative/absolute)

**Error Condition Testing**:
- Missing required XML tags
- Non-existent files and directories
- Invalid line numbers and ranges
- File permission issues

**Content Preservation Testing**:
- Unicode characters and special symbols
- Whitespace patterns (tabs, spaces, newlines)
- Large files and truncation behavior
- Binary file handling

### Running Tool Tests

```bash
# Run all tool tests
python test.py --level 2

# Run specific tool tests
python -c "
from tests.tools.test_editfile import test_editfile_find_and_replace
result = test_editfile_find_and_replace()
print(f'Test result: {result}')
"

# Run tests with verbose output
python test.py --verbose --level 2
```

### Adding New Tool Tests

1. **Create test file**: `tests/tools/test_newtool.py`
2. **Import the tool**: 
   ```python
   from tools.newtool import newtool
   ```
3. **Write test functions** following the established patterns
4. **Assign test levels** (typically level 2 for tool tests)
5. **Test all scenarios**: functionality, errors, edge cases, special chars

### Test Utilities

The framework provides utilities for common testing patterns:

```python
# Temporary file creation
with tempfile.TemporaryDirectory() as temp_dir:
    test_file = os.path.join(temp_dir, "test.txt")

# Content verification
assert new_content == expected_content, f"Mismatch: {repr(expected)} vs {repr(actual)}"

# XML input formatting
xml_input = f"""<filepath>{test_file}</filepath>
<content>{content}</content>"""
```

## Files

- `test.py` - Main unified test runner (use this!)
- `run_tests.py` - Advanced test runner with hierarchical levels
- `tests/tools/` - **Comprehensive tool test suites**
- `tests/core/` - Core infrastructure tests
- `tests/shared/` - Shared utility tests
