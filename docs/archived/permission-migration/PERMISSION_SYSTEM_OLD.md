# CLI-Agent Permission System Documentation

## Overview

The CLI-Agent permission system is a comprehensive security framework that controls access to system operations, file modifications, and command executions. It provides persistent "always allow" and "always deny" functionality with intelligent permission suggestions and security protections.

## Architecture

The permission system consists of three main components:

### 1. Enhanced Permissions (`core/permissions/enhanced_permissions.py`)
- **Primary Interface**: Main user-facing permission prompts and UI
- **Interactive Prompts**: Rich terminal UI with colored output and multiple permission options
- **Smart Actions**: Intelligent quick-action suggestions based on command context
- **Visual Formatting**: Styled boxes, icons, and path formatting for better UX

### 2. Permissions Manager (`core/permissions/permissions_manager.py`)
- **Data Persistence**: JSON-based storage in `~/.cli_agent_permissions.json`
- **Permission Logic**: Core allow/deny logic with pattern matching
- **Security Controls**: Sensitive content detection and automatic blocking
- **Multiple Permission Types**: File-based, command-based, group-based, and pattern-based permissions

### 3. Path Detector (`core/permissions/path_detector.py`)
- **XML Analysis**: Extracts file paths and commands from tool XML blocks
- **Command Classification**: Categorizes commands into development tool types
- **Security Scanning**: Detects sensitive keywords and risky operations
- **Smart Suggestions**: Generates intelligent permission scope recommendations

## Permission Types

### 1. Tool-Based Permissions
```json
"always_allow_tools": {
  "readfile": ["/home/user/project/*", "/home/user/docs/file.txt"],
  "bash": ["/home/user/scripts/*"]
}
```
- Grants access to specific tools for specific paths or globally
- Supports directory patterns with `/*` wildcard
- Can be tool-wide (all operations) or path-specific

### 2. Command-Based Permissions
```json
"always_allow_commands": {
  "bash": ["abc123def456"]  // MD5 hashes of exact commands
}
```
- Stores exact command hashes for precise matching
- Useful for allowing specific complex commands
- Prevents similar-but-different commands from being auto-approved

### 3. Pattern-Based Permissions
```json
"pattern_permissions": {
  "bash:ls*": "/home/user/project/*",
  "bash:git*": "*"  // Global permission
}
```
- Allows command patterns (e.g., all `ls` commands, all `git` commands)
- Scope can be directory-specific or global (`*`)
- Supports wildcard matching for flexible command approval

### 4. Group-Based Permissions
```json
"always_allow_groups": {
  "file_reading": ["/home/user/docs/*"],
  "safe_commands": ["*"]
}
```
- Groups related tools together (e.g., `readfile`, `viewfiles` in `file_reading`)
- Enables bulk permission grants for related functionality
- Predefined groups: `file_editing`, `file_reading`, `code_execution`, `file_management`

### 5. Development Tool Categories
```json
"always_allow_dev_tools": {
  "linting": ["/home/user/project/*"],
  "testing": ["/home/user/project/*"],
  "safe_commands": ["*"]
}
```
- Specialized categories for common development tasks
- Categories: `linting`, `testing`, `building`, `formatting`, `git`, `safe_commands`, `search`
- Each category has predefined command patterns and recommended permission scopes

## Security Features

### 1. Sensitive Content Detection
The system automatically blocks access to sensitive content:

```python
SENSITIVE_KEYWORDS = {
    'sudo', '.env', 'password', 'api_key', 'secret', 
    'token', 'private_key', 'id_rsa', 'id_dsa'
}
```

- **Automatic Blocking**: `.env` files are always blocked
- **Context-Aware**: Other sensitive keywords trigger security warnings
- **Tool-Specific**: Only applies to tools that can actually access files (`bash`, `readfile`, `editfile`, `writefile`, `viewfiles`)

### 2. Path Relevance Filtering
The system filters out irrelevant paths to reduce permission prompt noise:

- **Cache Files**: `.pyc`, `__pycache__`, `.cache/`
- **System Directories**: `site-packages/`, `lib/python`, `/tmp/`
- **Version Control**: `.git/` internals
- **Dependencies**: `node_modules/`, `.venv/`

### 3. Safe Command Recognition
Predefined patterns for read-only/safe commands that can be auto-approved:

```python
safe_patterns = [
    r'^ls(\s+(-[lartRhAFSG1]*\s*)*(\S+\s*)*)?$',  # ls with flags
    r'^pwd\s*$', r'^whoami\s*$', r'^date(\s+.*)?$',
    # ... many more safe patterns
]
```

## Permission Storage Format

The permissions are stored in `~/.cli_agent_permissions.json`:

```json
{
  "always_allow_tools": {},
  "always_deny_tools": {},
  "always_allow_commands": {},
  "always_deny_commands": {},
  "always_allow_groups": {},
  "always_deny_groups": {},
  "always_allow_dev_tools": {
    "grepsearch": ["grepsearch"]  // Pre-approved safe search
  },
  "pattern_permissions": {},
  "global_always_allow": false
}
```

## User Interface

### 1. Permission Prompt Structure
When a tool requires permission, users see:

1. **Tool Operation Box**: Shows tool name, file path, arguments, and action description
2. **Permission Options Box**: Lists available permission choices with colored keys
3. **Context-Aware Options**: Smart suggestions based on command type and file location

### 2. Available Options
- **`Enter/y`**: Allow once
- **`n`**: Deny once
- **`r`**: Allow read-only operations in current directory (for bash)
- **`f`**: Allow specific command pattern globally (for bash)
- **Smart Keys** (`t`, `g`, `b`, etc.): Quick actions based on command category
- **`a/s/p/t/c/e`**: Choose from suggested permission scopes
- **`d/D`**: Always deny (file-specific or tool-wide)
- **`m`**: Manage existing permissions
- **`?`**: Show help

### 3. Voice Support
All permission prompts support voice commands when voice mode is enabled:
- Say "yes" or "no" for basic allow/deny
- Say category names like "testing", "linting", "git" for smart actions
- Say "always1", "always2", etc. for numbered suggestions

## Development Tool Categories

### Smart Quick Actions
The system recognizes common development tools and provides smart permissions:

| Category | Tools | Quick Key | Scope |
|----------|-------|-----------|-------|
| `linting` | ruff, flake8, pylint, black, mypy | `r` | Project |
| `testing` | pytest, unittest, nose, tox | `t` | Project |
| `building` | npm, pip, poetry, make, cargo | `b` | Project |
| `formatting` | black, prettier, autopep8 | `f` | Directory |
| `git` | git commands | `g` | Project |
| `safe_commands` | ls, pwd, whoami, grep, find | `r` | Global |
| `search` | grep, rg, find, ack | `s` | Global |

### Project Root Detection
The system automatically detects project roots by looking for:
- `.git` directory
- `package.json`, `pyproject.toml`, `requirements.txt`
- `Cargo.toml`, `pom.xml`, `build.gradle`
- `Makefile`, `.gitignore`

## Command Analysis

### 1. XML Path Extraction
The system analyzes tool XML blocks to extract:
- **File Paths**: From tags containing "path" substring
- **Command Content**: From bash/python tool content
- **Action Context**: What operation will be performed

### 2. Command Pattern Generation
Each command gets a readable pattern for permission matching:
- **Bash**: `"bash: ls -la"` or `"bash: git status"`
- **Python**: `"python: import os"` (first meaningful line)
- **File Tools**: `"editfile: Replace 'old' with 'new'"`

### 3. Security Analysis
Before any permission checking:
1. **Sensitive Content Scan**: Check for dangerous keywords
2. **Tool Risk Assessment**: Evaluate if tool can actually cause harm
3. **Context Analysis**: Understand what the command will do

## Permission Management

### 1. Interactive Management (`m` option)
Users can view and manage existing permissions:
- **List Permissions**: View all saved permissions with IDs
- **Remove Specific**: `r <id>` to remove specific permission
- **Clear All**: Remove all saved permissions
- **Return**: Go back to permission prompt

### 2. Programmatic Management
The PermissionsManager class provides methods for:
```python
# Add permissions
add_always_allow(tool_name, file_path=None, pattern=None, 
                command_hash=None, group_name=None, 
                dev_tool_category=None, command_pattern=None, 
                scope_path=None)

# Check permissions
is_always_allowed(tool_name, file_path=None, command_hash=None, 
                 xml_content=None)
is_always_denied(tool_name, file_path=None)

# Manage permissions
remove_permission(tool_name, file_path=None, permission_type="allow")
clear_all_permissions()
```

## Integration Points

### 1. Tool Execution Flow
1. **Tool Called**: System intercepts tool execution request
2. **XML Analysis**: PathDetector analyzes the tool's XML content
3. **Permission Check**: PermissionsManager checks existing permissions
4. **User Prompt**: EnhancedPermissions shows interactive prompt if needed
5. **Permission Storage**: User choice is stored persistently

### 2. Security Checkpoints
- **Pre-Execution**: Sensitive content scanning before permission prompts
- **Pattern Matching**: Multiple layers of permission matching
- **Context Awareness**: Different rules for different tool types
- **User Override**: Manual permission management always available

## Configuration

### Default Permissions
The system ships with minimal safe defaults:
```json
"always_allow_dev_tools": {
  "grepsearch": ["grepsearch"]  // Safe search operations
}
```

### Customization
Users can customize through:
1. **Interactive Prompts**: Build permissions through normal usage
2. **Direct File Edit**: Modify `~/.cli_agent_permissions.json`
3. **Management Interface**: Use the `/permissions` command
4. **API Calls**: Programmatic access through PermissionsManager

## Error Handling

### 1. Permission File Issues
- **Missing File**: Creates default permissions structure
- **Corrupt JSON**: Falls back to defaults with warning
- **IO Errors**: Warns user but continues with in-memory permissions

### 2. Security Blocks
- **Sensitive Content**: Clear blocking messages with explanations
- **Automatic Denial**: No user prompt for dangerous operations
- **Context Information**: Shows what triggered the security block

### 3. User Experience
- **Keyboard Interrupts**: Graceful handling of Ctrl+C
- **Invalid Options**: Clear error messages with valid option lists
- **Voice Command Errors**: Fallback to keyboard input

## Best Practices

### 1. For Users
- **Start Restrictive**: Don't grant broad permissions initially
- **Use Smart Actions**: Leverage category-based permissions
- **Review Regularly**: Use permission management to audit saved rules
- **Project-Scoped**: Prefer project/directory scoped over global permissions

### 2. For Developers
- **Context Rich**: Provide detailed action descriptions in tools
- **Path Clarity**: Use clear, specific file path tags in XML
- **Security First**: Consider permission implications in tool design
- **User Feedback**: Test permission prompts for clarity and usability

## Troubleshooting

### Common Issues
1. **Permission Denied**: Check if file/command matches saved deny rules
2. **Unexpected Prompts**: May indicate new file paths or command variations
3. **Security Blocks**: Review command for sensitive keywords
4. **Permission Not Saved**: Check file permissions on config directory

### Debugging
- **Enable Verbose Mode**: See detailed permission matching logic
- **Check Config File**: Review `~/.cli_agent_permissions.json` structure
- **Test Patterns**: Use management interface to understand matching
- **Clear and Restart**: Sometimes clearing all permissions helps isolate issues