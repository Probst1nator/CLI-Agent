# CLI-Agent Permission System - Comprehensive Analysis

## 1. Directory Structure & Files

### Core Permission Files
Located in: `/home/prob/repos/CLI-Agent/core/permissions/`

| File | Size | Purpose |
|------|------|---------|
| `enhanced_permissions.py` | 52 KB | User-facing permission prompt system with GUI support |
| `permissions_manager.py` | 30 KB | Core permission storage, checking, and management |
| `path_detector.py` | 32 KB | Path extraction, tool analysis, dev tool categorization |
| `__init__.py` | 26 B | Package initialization |

### Integration Points
- **Main Loop**: `/home/prob/repos/CLI-Agent/main.py` (lines 2606-2671)
- **Bash Tool**: `/home/prob/repos/CLI-Agent/tools/bash.py`
- **Permission Command**: `/home/prob/repos/CLI-Agent/agent/commands/simple_commands.py` (lines 350-749)
- **View Tool**: `/home/prob/repos/CLI-Agent/tools/viewpermissions.py`
- **Tests**: `/home/prob/repos/CLI-Agent/tests/core/test_permissions_system.py`

### Configuration Storage
```
~/.cli-agent/permissions.json     # XDG-compliant config directory
                                  # Fallback: ~/.config/cli-agent/permissions.json
```

---

## 2. Permission System Architecture

### High-Level Flow

```
User Command (bash, editfile, etc.)
    ↓
main.py: PathDetector.analyze_tool_block() 
    ↓
Extract paths, command patterns, dev tool categories
    ↓
Check if tool needs permission checking
    ↓
PermissionsManager.is_always_allowed()
    ├─→ Check global permission
    ├─→ Check command-specific permissions
    ├─→ Check tool-specific permissions
    ├─→ Check group-based permissions
    ├─→ Check dev tool category permissions
    ├─→ Check pattern-based permissions
    └─→ Return (is_allowed, rule_description)
    ↓
If NOT allowed → enhanced_permission_prompt()
    ├─→ Show permission UI
    ├─→ Get user choice (y/n/always_allow/always_deny/smart_actions)
    ├─→ Store decision in PermissionsManager
    └─→ Return True/False
    ↓
Execute tool or reject
```

---

## 3. Core Classes & Methods

### A. PermissionsManager (permissions_manager.py)

**Purpose**: Persistent storage and retrieval of permission rules

**Key Methods**:

```python
class PermissionsManager:
    def __init__(self, config_path: Optional[str] = None)
        # Initialize from ~/.cli-agent/permissions.json
        
    def _load_permissions(self) -> Dict[str, Any]
        # Load from JSON file or create default structure
        # Default structure includes:
        # - always_allow_tools
        # - always_deny_tools
        # - always_allow_commands
        # - always_allow_groups
        # - always_allow_dev_tools
        # - pattern_permissions
        # - global_always_allow
        
    def _save_permissions(self)
        # Persist to JSON file in XDG config directory
        
    def is_always_allowed(tool_name, file_path, command_hash, xml_content)
        # Returns: (is_allowed: bool, rule_description: str)
        # Checks in order:
        # 1. Global permission
        # 2. Exact command matches
        # 3. Tool-specific rules
        # 4. Group-based rules
        # 5. Dev tool category rules (bash only)
        # 6. Pattern-based rules (command wildcards)
        # 7. Directory/path patterns
        
    def is_always_denied(tool_name, file_path) -> bool
        # Returns True if explicitly denied
        
    def add_always_allow(tool_name, file_path, pattern, command_hash, 
                         group_name, dev_tool_category, command_pattern, scope_path)
        # Add new allow rule
        # Supports multiple rule types:
        # - Exact file paths
        # - Directory patterns (with /*)
        # - Command hashes (exact commands)
        # - Group-based (file_editing, file_reading, code_execution)
        # - Dev tool categories (linting, testing, building, git, etc.)
        # - Command patterns (bash:ls*, git:commit*, etc.)
        
    def add_always_deny(tool_name, file_path)
        # Add deny rule for tool/path combination
        
    def remove_permission(tool_name, file_path, permission_type)
        # Remove existing permission
        
    def get_permission_suggestions(tool_name, file_path, command_pattern, has_paths)
        # Generate suggested permission scopes based on context
        
    def clear_all_permissions()
        # Wipe all stored permissions
```

### B. Enhanced Permission Prompt System (enhanced_permissions.py)

**Purpose**: Interactive user permission interface with visual UI support

**Key Functions**:

```python
async def enhanced_permission_prompt(
    args,
    tool_name: str,
    action_description: str,
    file_path: Optional[str] = None,
    command_hash: Optional[str] = None,
    command_pattern: Optional[str] = None,
    has_paths: bool = False,
    xml_content: Optional[str] = None,
    input_event: Optional[asyncio.Event] = None,
    input_lock: Optional[threading.Lock] = None,
    shared_input: Optional[Dict[str, Any]] = None,
    get_user_input_func = None
) -> bool:
    """
    Main permission prompt function.
    
    Returns: bool (True = allow, False = deny)
    
    Execution flow:
    1. Check if always_denied → show denied box, return False
    2. Check if YOLO mode (auto_execute) → return True
    3. Check if always_allowed → show granted box, return True
    4. If GUI mode active → send permission_request, wait for response
    5. Fallback to CLI prompt with options:
       - Enter/y: Allow once
       - n: Deny once
       - r: Allow read-only operations
       - f: Always allow bash command
       - a/s/p/t/c/e: Smart actions (suggested permissions)
       - d: Always deny this file
       - D: Always deny all tool operations
       - m: Manage permissions
       - ?: Show help
    """
```

**Display Functions**:

```python
def format_xml_args(xml_content: Optional[str]) -> str
    # Extract readable arguments from XML for display
    
def get_path_type_info(file_path: str) -> Dict[str, Any]
    # Analyze file type (python, javascript, config, script, etc.)
    
def format_file_path_for_display(file_path: str, max_length: int) -> str
    # Smart path truncation (relative paths, parent/filename, etc.)
    
def get_enhanced_file_display(file_path: str, max_length: int) -> str
    # Enhanced path display with type indicators
    
def get_terminal_width() -> int
    # Dynamic terminal width detection for responsive UI
```

**Helper Functions**:

```python
async def handle_direct_always_allow(
    suggestion: str, tool_name: str, file_path, permissions_manager, command_hash
)
    # Process user selection of a suggested permission scope
    
async def handle_smart_quick_action(
    smart_action: Dict, tool_name: str, file_path, permissions_manager, command_pattern
)
    # Handle smart action (linting, testing, git, etc.)
    
async def handle_permission_management(
    get_user_input_func, args, input_event, input_lock, shared_input
)
    # Interactive permission management UI
    
def show_permission_help(tool_name: str, file_path: Optional[str] = None)
    # Display help for permission options
```

### C. PathDetector (path_detector.py)

**Purpose**: Analyze tool blocks, extract paths, detect dev tools, categorize commands

**Key Constants**:

```python
PERMISSION_GROUPS = {
    'file_editing': ['editfile', 'writefile', 'appendfile'],
    'file_reading': ['readfile', 'viewfiles'],
    'code_execution': ['bash', 'python'],
    'file_management': ['bash', 'python']
}

COMMON_DEV_TOOLS = {
    'linting': {
        'patterns': ['ruff', 'flake8', 'pylint', 'black', 'isort', 'mypy'],
        'quick_key': 'r',
        'display_name': 'linting tools (ruff, black, etc.)',
        'permission_scope': 'project'
    },
    'testing': {
        'patterns': ['pytest', 'python -m pytest', 'nose', 'tox'],
        'quick_key': 't',
        'display_name': 'testing commands (pytest, etc.)',
        'permission_scope': 'project'
    },
    'building': {
        'patterns': ['npm install', 'pip install', 'make', 'cargo build'],
        'quick_key': 'b',
        'display_name': 'build tools (npm, pip, etc.)',
        'permission_scope': 'project'
    },
    'formatting': {
        'patterns': ['black', 'prettier', 'autopep8'],
        'quick_key': 'f',
        'display_name': 'code formatters',
        'permission_scope': 'directory'
    },
    'git': {
        'patterns': ['git add', 'git commit', 'git status'],
        'quick_key': 'g',
        'display_name': 'git commands',
        'permission_scope': 'project'
    },
    'safe_commands': {
        'patterns': ['ls', 'pwd', 'cat', 'grep', 'find'],
        'quick_key': 'r',
        'display_name': 'read-only/safe commands',
        'permission_scope': 'project'
    },
    'search': {
        'patterns': ['grep', 'rg', 'find', 'ack'],
        'quick_key': 's',
        'display_name': 'search tools',
        'permission_scope': 'global'
    }
}

SENSITIVE_KEYWORDS = {
    'sudo', '.env', 'password', 'api_key', 'secret', 'token',
    'private_key', 'id_rsa', 'id_dsa'
}
```

**Key Methods**:

```python
@staticmethod
def get_tool_groups(tool_name: str) -> List[str]
    # Get permission groups a tool belongs to
    
@staticmethod
def get_group_tools(group_name: str) -> List[str]
    # Get all tools in a group
    
@staticmethod
def detect_dev_tool_category(command_pattern: str) -> Tuple[Optional[str], Optional[Dict]]
    # Detect dev tool category from command string
    # Returns: (category_name, category_config) or (None, None)
    
@staticmethod
def _is_safe_command(command_lower: str) -> bool
    # Regex-based detection of read-only commands
    # Safe patterns: ls, pwd, cat, grep, find with file arguments
    
@staticmethod
def get_smart_quick_action(tool_name: str, command_pattern: str, file_path) -> Optional[Dict]
    # Generate smart quick action suggestion
    # Returns: {category, quick_key, display_name, scope_path, scope_description, permission_scope}
    
@staticmethod
def _find_project_root(start_path: str) -> Optional[str]
    # Find project root by looking for: .git, package.json, pyproject.toml, etc.
    
@staticmethod
def extract_paths_from_xml(xml_content: str, tool_name: str) -> List[Tuple[str, str]]
    # Extract file paths from XML tool blocks
    # Returns: [(tag_name, absolute_path), ...]
    
@staticmethod
def should_check_permissions(tool_name: str, paths: List[Tuple[str, str]]) -> bool
    # Determine if permission check is needed
    
@staticmethod
def contains_sensitive_keywords(content: str) -> Tuple[bool, List[str]]
    # Check for sensitive keywords (env files, secrets, etc.)
    
@staticmethod
def check_sensitive_content_in_xml(xml_content: str) -> Tuple[bool, List[str]]
    # Check XML content for sensitive keywords
    
@staticmethod
def get_primary_path(paths: List[Tuple[str, str]]) -> Optional[str]
    # Select most relevant path from list
    # Priorities: user files > cache, current dir > system, source code > config
    
@staticmethod
def get_command_hash(xml_content: str) -> str
    # Generate MD5 hash of normalized command for exact matching
    
@staticmethod
def get_command_pattern(tool_name: str, xml_content: str) -> str
    # Extract readable command pattern: "bash: ls -la", "python: import os", etc.
    
@staticmethod
def analyze_tool_block(tool_name: str, xml_content: str) -> Dict[str, Any]
    # Complete analysis of tool block
    # Returns: {
    #     'tool_name': str,
    #     'paths': [(tag, path), ...],
    #     'primary_path': str,
    #     'action_description': str,
    #     'needs_permission': bool,
    #     'xml_content': str,
    #     'command_hash': str,
    #     'command_pattern': str,
    #     'has_paths': bool
    # }
```

---

## 4. Permission Rule Types & Storage Structure

### JSON Storage Format (~/.cli-agent/permissions.json)

```json
{
  "always_allow_tools": {
    "bash": ["/home/user/*", "/tmp/*"],
    "readfile": ["/home/user/docs/*"],
    "editfile": ["/home/user/docs/*"]
  },
  
  "always_deny_tools": {
    "bash": ["/etc/passwd", "/root/*"],
    "writefile": ["/system/*"]
  },
  
  "always_allow_commands": {
    "bash": ["abc123def456...", "xyz789uvw..."]  // MD5 hashes
  },
  
  "always_allow_groups": {
    "file_editing": ["/home/user/project/*"],
    "code_execution": ["/home/user/project/*"]
  },
  
  "always_allow_dev_tools": {
    "linting": ["/home/user/project/*", "linting"],
    "testing": ["/home/user/project/*"],
    "git": ["git"],  // Global category-level permission
    "grepsearch": ["grepsearch"]  // Pre-configured search tools
  },
  
  "pattern_permissions": {
    "bash:ls*": "/home/user/*",
    "bash:git*": "/home/user/project/*",
    "bash:pytest*": "*"  // Global scope
  },
  
  "global_always_allow": false
}
```

### Rule Precedence Order

When `is_always_allowed()` is called:

1. **Sensitive Content Check** - Blocks access to .env, passwords, secrets
2. **Global Permission** - If `global_always_allow` is True
3. **Exact Command Match** - By command hash
4. **Tool Permissions** - Exact tool + path match
5. **Group Permissions** - Tool's parent group permissions
6. **Dev Tool Category** - Category of bash command (linting, testing, git)
7. **Pattern Permissions** - Wildcard command patterns with scope
8. **Directory Patterns** - Paths ending with `/*`

---

## 5. Path-Aware Rules - Examples

### Exact File Permission
```python
pm.add_always_allow("readfile", file_path="/home/user/important.txt")
# Stored as: always_allow_tools["readfile"] = ["/home/user/important.txt"]
# Matches ONLY that exact file
```

### Directory Permission
```python
pm.add_always_allow("bash", file_path="/home/user/project")
# Stored as: always_allow_tools["bash"] = ["/home/user/project"]
# Also matches all files under the directory
```

### Pattern Permission (Directory with Wildcard)
```python
pm.add_always_allow("editfile", pattern="/home/user/docs/*")
# Stored as: always_allow_tools["editfile"] = ["/home/user/docs/*"]
# Matches any file under /home/user/docs/
```

### Command Pattern Permission
```python
pm.add_always_allow("bash", command_pattern="ls*", scope_path="/home/user/*")
# Stored as: pattern_permissions["bash:ls*"] = "/home/user/*"
# Matches: ls, ls -la, lsof, etc. in /home/user/ directory

pm.add_always_allow("bash", command_pattern="git*", scope_path="*")
# Global git command permission
```

### Dev Tool Category Permission
```python
pm.add_always_allow("bash", dev_tool_category="linting", pattern="/home/user/project/*")
# Stored as: always_allow_dev_tools["linting"] = ["/home/user/project/*"]
# Matches: ruff, black, flake8, pylint, etc. in project directory
```

### Group Permission
```python
pm.add_always_allow(tool_name="bash", group_name="file_editing", pattern="/home/user/*")
# Stored as: always_allow_groups["file_editing"] = ["/home/user/*"]
# Applies to: editfile, writefile, appendfile in /home/user/
```

---

## 6. Permission Checking in Tool Execution (main.py)

### Execution Flow (lines 2606-2671)

```python
# Step 1: Analyze all tool blocks
for tool_class in tool_tags_found:
    delim = tool_class.get_delim()
    blocks = get_extract_blocks()(assistant_response, [delim])
    
    for block in blocks:
        # PathDetector analyzes tool block completely
        analysis = PathDetector.analyze_tool_block(delim, block)
        tool_analyses.append(analysis)
        
        if analysis['needs_permission']:
            permission_needed = True

# Step 2: Check write path restrictions (LAYER 1)
allowed_write_paths = getattr(g, 'ALLOWED_WRITE_PATHS', [])
if allowed_write_paths:
    for analysis in tool_analyses:
        if tool_name in ['writefile', 'editfile', 'insertline', 'deletelines']:
            if primary_path not in allowed_write_paths:
                # INSTANT DENIAL - No permission prompt
                should_execute = False
                break

# Step 3: Enhanced permission prompts (LAYER 2)
for analysis in tool_analyses:
    permission_granted = await enhanced_permission_prompt(
        args,
        tool_name=analysis['tool_name'],
        action_description=analysis['action_description'],
        file_path=analysis['primary_path'],
        command_hash=analysis['command_hash'],
        command_pattern=analysis['command_pattern'],
        has_paths=analysis['has_paths'],
        xml_content=analysis.get('xml_content', ''),
        input_event=input_event,
        input_lock=input_lock,
        shared_input=shared_input,
        get_user_input_func=get_user_input_with_bindings
    )
    
    if not permission_granted:
        should_execute = False
        break
```

---

## 7. Permission Command System (/perms)

### Command Structure
Implemented in: `/home/prob/repos/CLI-Agent/agent/commands/simple_commands.py`

```bash
/perms list              # List all permissions with IDs (default)
/perms view              # Pretty-printed user-friendly view
/perms add               # Interactive wizard for adding permissions
/perms add dev <cat>     # Add dev tool permission
/perms add tool <name> <target>  # Add tool permission
/perms add group <name> <target> # Add group permission
/perms add command <pattern> <scope> # Add command pattern
/perms remove <id>       # Remove specific permission
/perms clear             # Clear all permissions
```

### Adding Permissions - Interactive Wizard

```python
async def _add_permission_wizard(self, parts: List[str]) -> None:
    """
    Interactive wizard with options:
    1. Development tools (grep, find, etc.)
    2. Specific tools on files/directories
    3. Groups of related tools
    4. Command patterns
    """
```

---

## 8. YOLO Mode (Auto-Execute)

```python
# Check if YOLO mode is active (auto_execute flag)
from core.globals import g
if hasattr(g, 'auto_execute') and g.auto_execute:
    # In yolo mode, auto-approve without showing prompt
    # (unless already denied by always-deny rules)
    return True
```

**Enabled with**: `/yolo` command or `--yolo` flag

---

## 9. Sensitive Content Protection

### Automatic Blocking
The system automatically blocks execution when sensitive content is detected:

```python
SENSITIVE_KEYWORDS = {
    'sudo',           # Privilege escalation
    '.env',           # Environment files with secrets
    'password',       # Passwords
    'api_key',        # API credentials
    'secret',         # Generic secrets
    'token',          # Auth tokens
    'private_key',    # SSH keys
    'id_rsa', 'id_dsa' # SSH key files
}

# Check in permissions_manager.is_sensitive_content_blocked()
if tool_name in ['bash', 'readfile', 'editfile', 'writefile', 'viewfiles']:
    has_sensitive, keywords = check_sensitive_content_in_xml(xml_content)
    if has_sensitive:
        # Block execution, require additional confirmation
        return False
```

---

## 10. GUI Permission Request Integration

### WebSocket Flow
When GUI mode is active and a permission prompt is needed:

```python
# Create permission request payload
permission_payload = {
    "id": f"permission-{int(time.time() * 1000)}",
    "action": "create",
    "type": "permission_request",
    "permission_data": {
        "tool_name": tool_name,
        "action_description": action_description,
        "file_path": file_path,
        "command_pattern": command_pattern,
        "xml_content": xml_content,
        "args_display": formatted_args,
        "smart_action": smart_action,
        "read_only_option": read_only_option,
        "suggestions": suggestions
    }
}

# Send via WebSocket to frontend
session.send_update(permission_payload)

# Wait for response from visual interface (60s timeout)
response_data = await asyncio.wait_for(
    g.visual_input_queue.get(), 
    timeout=60
)

# Process permission response
if response.get('result') == 'allow':
    return True
elif response.get('result') == 'always_allow':
    # Add persistent rule
    scope = permission_action.get('scope', 'exact')
    if scope == 'directory':
        pm.add_always_allow(tool_name, pattern=f"{os.path.dirname(file_path)}/*")
    elif scope == 'project':
        # Find project root and add rule
    return True
```

---

## 11. Testing Infrastructure

### Test Levels
- **Level 2 (Core)**: `test_permissions_system.py` - Core permission logic
- Test key functionality without file I/O

### Example Tests
```python
def test_permission_redundancy_basic():
    # Verify already-permitted actions don't show prompts
    
def test_pattern_scope_context():
    # Pattern permissions work with file paths, not just CWD
    
def test_smart_action_filtering():
    # Smart actions hidden when already permitted
```

---

## 12. Key Integration Points

### 1. Main.py Permission Check Loop (lines 2606-2671)
- Analyzes each tool block
- Checks ALLOWED_WRITE_PATHS restrictions
- Calls enhanced_permission_prompt()
- Returns True/False for execution

### 2. Bash Tool Integration
- No direct permission checks in bash.py
- All checks happen in main.py before tool invocation
- Tool analysis detects bash commands and extracts command patterns

### 3. File Tools (readfile, writefile, editfile)
- Path extraction in PathDetector
- File paths passed to permission prompt
- Pattern matching against stored rules

### 4. Permission Management Command
- `/perms` command provides user interface
- Add/remove/list/clear permissions
- Interactive wizard for adding rules

---

## 13. Configuration File Location

### Primary Location (XDG-Compliant)
```
~/.cli-agent/permissions.json
```

### Fallback (Legacy)
```
~/.config/cli-agent/permissions.json
```

### Initialization
From `core/globals.py`:
```python
CLIAGENT_CONFIG_DIR = _get_xdg_config_dir()  # ~/.cli-agent/

# Auto-migration from legacy location if present
if legacy_path and os.path.exists(legacy_permissions_file):
    # Migrate permissions.json to new location
```

---

## 14. Permission Display & User Experience

### Permission Box Format
```
╭─ TOOL OPERATION PERMISSION - bash ────────────────────────────────────────╮
│ 💻 user@hostname:/current/dir$ ▶️  ls -la /path/to/file                  │
│                                                                             │
├─ Permission Options ──────────────────────────────────────────────────────┤
│  [Enter] or y → Confirm this operation                                    │
│  n → Abort this operation                                                 │
│  r → Allow read-only operations in current_dir                           │
│  f → Always allow 'bash: ls' commands                                     │
│  a/s/p/t → Always allow with scope                                       │
│  d → Always deny this file                                               │
│  D → Always deny all bash operations                                     │
│  m → Manage all saved permissions                                        │
│  ? → Show detailed help                                                  │
╰──────────────────────────────────────────────────────────────────────────╯
```

### Dynamic Terminal Width
- Responsive UI that adapts to terminal size
- Intelligent path truncation
- Smart formatting of command arguments

---

## 15. Summary: Complete Permission Flow

```
User Issues Command
    ↓
Main loop detects tool XML blocks
    ↓
PathDetector.analyze_tool_block()
    - Extract paths
    - Identify tool type
    - Detect dev tool category
    - Generate command pattern/hash
    - Determine if permission needed
    ↓
Is permission needed?
    NO  → Execute tool immediately
    YES → Check PermissionsManager
        ↓
        Is always_denied?
            YES → Show denied box, return False
            NO  → Continue
        ↓
        Is YOLO mode?
            YES → Auto-approve
            NO  → Continue
        ↓
        Is always_allowed?
            YES → Show granted box, return True
            NO  → Show permission prompt
        ↓
        User makes choice (Allow/Deny/Always Allow/Smart Action)
            ↓
            Add rule to PermissionsManager if persistent choice
            ↓
            Return decision (True/False)
    ↓
Execute tool or reject
    ↓
Store in chat history
```

