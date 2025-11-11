# CLI-Agent Permission System Documentation

## Overview

The CLI-Agent permission system uses a **simplified Claude Code-style config-based approach** that controls access to system operations, file modifications, and command executions through explicit allow/deny/ask lists.

**Key Characteristics:**
- **Config-driven**: Static JSON configuration, not runtime learning
- **3 simple lists**: allow, deny, ask (vs old 7 rule types)
- **Transparent**: Explicit declarations, easy to audit
- **Team-friendly**: Shareable config files, version controllable
- **Minimal code**: 825 lines vs old 2,400 lines (66% reduction)

## Architecture

The permission system consists of three main components:

### 1. Settings Manager (`core/permissions/settings_manager.py` - 331 lines)
- **Configuration Loading**: Loads `~/.cli-agent/settings.json`
- **Permission Checking**: Simple 3-level precedence (deny → ask → allow)
- **Pattern Matching**: Uses fnmatch for file paths and prefix matching for commands
- **XML Parsing**: Extracts paths and commands from tool blocks

### 2. Permission Prompt (`core/permissions/permission_prompt.py` - 384 lines)
- **User Prompts**: Simple allow/deny/always prompts
- **GUI Integration**: WebSocket-based prompts for web interface
- **Optional Memory**: Can save decisions to config on request

### 3. Dev Tool Hints (`core/permissions/dev_tool_hints.py` - 110 lines)
- **Tool Detection**: Recognizes common dev tools (linting, testing, git, etc.)
- **Suggestions**: Provides suggested rules when tools are detected
- **Help Text Only**: Does NOT automatically grant permissions

## Configuration Format

Permissions are stored in `~/.cli-agent/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "bash(npm:*)",
      "bash(pytest:*)",
      "bash(git:*)",
      "readfile(./src/**)",
      "editfile(./tests/**)"
    ],
    "deny": [
      "readfile(.env)",
      "readfile(.env.*)",
      "bash(sudo:*)",
      "bash(rm -rf:*)"
    ],
    "ask": [
      "bash(git push:*)",
      "bash(docker:*)",
      "writefile(./package.json)"
    ]
  }
}
```

## Permission Lists

### 1. Allow List
Tools and commands that execute **without asking for permission**.

**Format:** `toolname(specifier)` or `toolname`

**Examples:**
```json
"allow": [
  "bash(npm:*)",           // All npm commands
  "bash(pytest:*)",        // All pytest commands
  "readfile(./src/**)",    // Read any file in src/ (glob pattern)
  "editfile(./tests/**)",  // Edit any file in tests/
  "grepsearch"             // All grep searches (no specifier)
]
```

### 2. Deny List
**Blocked** tools and commands. These cannot be executed.

**Examples:**
```json
"deny": [
  "readfile(.env)",        // Block reading .env file
  "readfile(.env.*)",      // Block .env.local, .env.production, etc.
  "readfile(./secrets/**)", // Block entire secrets directory
  "bash(sudo:*)",          // Block all sudo commands
  "bash(rm -rf:*)"         // Block dangerous rm commands
]
```

### 3. Ask List
Tools and commands that **require user confirmation** before execution.

**Examples:**
```json
"ask": [
  "bash(git push:*)",      // Confirm before git push
  "bash(docker:*)",        // Confirm before docker commands
  "writefile(.env)",       // Confirm before modifying .env
  "writefile(./package.json)" // Confirm before modifying package.json
]
```

## Pattern Syntax

### Tool Names
Map to actual tool files using these conventions:

| Pattern | Maps To | Notes |
|---------|---------|-------|
| `bash` | `bash.py` | Raw tool name |
| `readfile` | `readfile.py`, `viewfiles.py` | No aliases, use exact names |
| `writefile` | `writefile.py` | Single tool only |
| `editfile` | `editfile.py`, `insertline.py`, `deletelines.py` | Edit operations |
| `grepsearch` | `grepsearch.py` | Search tool |

**Important:** Pattern matching uses **raw tool names**, not aliases. Use the exact tool name from `tools/` directory.

### Command Patterns (for bash/python)
```
bash(command:*)          # Prefix match: "npm:*" matches "npm install", "npm run", etc.
bash(git push:*)         # Matches all git push commands
bash(sudo:*)             # Matches all sudo commands
```

**Matching Logic:**
- `command:*` uses **prefix matching**
- `:` separates command from wildcard
- `*` means "and anything after"

### File Path Patterns
```
readfile(./src/**)       # Glob: any file in src/ at any depth
readfile(./.env)         # Exact file
readfile(./tests/*.py)   # Glob: Python files in tests/
readfile(~/.ssh/id_rsa)  # Absolute path with home expansion
```

**Matching Logic:**
- Uses `fnmatch` for glob patterns
- Supports `*` (any characters), `**` (recursive), `?` (single char)
- Paths can be relative (`./`) or absolute

## Precedence Rules

When checking permissions, the system follows this order:

1. **Deny** - Check deny list first → **Blocks immediately**
2. **Ask** - Check ask list → **Prompts user**
3. **Allow** - Check allow list → **Executes without prompt**
4. **Default** - No match → **Uses default behavior** (prompt)

**Example:**
```json
{
  "allow": ["bash(npm:*)"],
  "deny": ["bash(npm install:*)"],
  "ask": ["bash(npm publish:*)"]
}
```

- `bash(npm install package)` → **DENIED** (deny list wins)
- `bash(npm publish)` → **ASK** (not denied, matches ask)
- `bash(npm run test)` → **ALLOW** (not denied/asked, matches allow)
- `bash(yarn install)` → **DEFAULT PROMPT** (no match)

## Commands

### `/perms` Command

View and manage permission settings:

```bash
/perms view                           # Show current settings
/perms add allow bash(npm:*)          # Add to allow list
/perms add deny readfile(.env)        # Add to deny list
/perms add ask bash(git push:*)       # Add to ask list
/perms remove bash(npm:*)             # Remove a rule
/perms reset                          # Clear all settings
/perms template                       # Copy default template
```

### Template Usage

Copy the default template with sensible defaults:

```bash
/perms template
```

This copies `/templates/default_settings.json` to `~/.cli-agent/settings.json` with:
- Common safe commands pre-allowed (ls, pwd, grep, npm, pytest, etc.)
- Dangerous operations pre-denied (sudo, rm -rf, .env files)
- Risky operations requiring confirmation (git push, docker, npm install)

## Dev Tool Hints

The system provides **optional hints** when it detects common development tools:

**Detected Categories:**
- **Linting:** ruff, flake8, pylint, black, mypy
- **Testing:** pytest, unittest, nose, tox
- **Building:** npm, pip, poetry, make, cargo, docker build
- **Git:** git commands
- **Search:** grep, rg, find, ack

**What Hints Do:**
- Display suggested rules in prompt
- Help users understand what to allow
- **Do NOT automatically grant permissions**

**Example:**
```
Tool: bash(pytest tests/)
Hint: This is a testing tool. Consider adding:
  /perms add allow bash(pytest:*)
```

## Best Practices

### 1. Start with Template
Use `/perms template` to get sensible defaults, then customize.

### 2. Explicit Over Automatic
Unlike the old system, **declare what you allow** instead of learning from prompts.

### 3. Use Deny for Security
Always deny sensitive operations:
```json
"deny": [
  "readfile(.env)",
  "readfile(.env.*)",
  "readfile(./secrets/**)",
  "bash(sudo:*)",
  "bash(rm -rf:*)"
]
```

### 4. Use Ask for Risky Operations
Require confirmation for operations with side effects:
```json
"ask": [
  "bash(git push:*)",
  "bash(npm install:*)",
  "bash(docker:*)",
  "writefile(./package.json)"
]
```

### 5. Team Collaboration
Commit `~/.cli-agent/settings.json` to your project repository as `.cli-agent/settings.json` to share permissions across team.

### 6. Project-Specific Patterns
Use relative paths for project-specific permissions:
```json
"allow": [
  "readfile(./src/**)",    // Project src directory
  "editfile(./tests/**)"   // Project tests directory
]
```

## Migration from Old System

The old permission system (2,400 lines, 7 rule types) has been replaced with this simplified approach (825 lines, 3 lists).

**Old Format** (`~/.cli-agent/permissions.json`):
```json
{
  "always_allow_tools": {...},
  "always_deny_tools": {...},
  "always_allow_commands": {...},
  "always_allow_groups": {...},
  "always_allow_dev_tools": {...},
  "pattern_permissions": {...},
  "global_always_allow": false
}
```

**New Format** (`~/.cli-agent/settings.json`):
```json
{
  "permissions": {
    "allow": [...],
    "deny": [...],
    "ask": [...]
  }
}
```

**Migration:** No automatic migration. Users start fresh with template or manually configure.

**Archive:** Old system documentation and analysis preserved in `docs/archived/permission-migration/`

## Troubleshooting

### Permission Not Working

1. **Check precedence**: Deny overrides allow
2. **Verify pattern**: Use exact tool names (no aliases)
3. **Test path matching**: Use absolute paths or `./` prefix
4. **Check file**: Verify `~/.cli-agent/settings.json` exists and is valid JSON

### Too Many Prompts

1. Use `/perms add allow <pattern>` for frequently used commands
2. Use glob patterns for directories: `readfile(./src/**)`
3. Use command prefixes: `bash(npm:*)` for all npm commands

### Blocked Unexpectedly

1. Check deny list: `/perms view`
2. Remove rule: `/perms remove <rule>`
3. Verify no typos in allowed patterns

## Comparison with Claude Code

This system is **inspired by Claude Code's permission model** but adapted for CLI-Agent:

| Feature | Claude Code | CLI-Agent |
|---------|-------------|-----------|
| **Config Format** | `.claude/settings.json` | `~/.cli-agent/settings.json` |
| **Lists** | allow, deny, ask | allow, deny, ask ✅ |
| **Pattern Syntax** | `Bash(npm:*)` | `bash(npm:*)` (lowercase) |
| **Tool Names** | Aliases supported | Raw tool names only |
| **Hints** | No hints | Optional dev tool hints |
| **Learning** | No | No ✅ |
| **Team Config** | `.claude/settings.json` | `.cli-agent/settings.json` |

**Key Difference:** CLI-Agent uses lowercase tool names and no aliases for simplicity.

## Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `core/permissions/settings_manager.py` | 331 | Core permission checking |
| `core/permissions/permission_prompt.py` | 384 | User prompts |
| `core/permissions/dev_tool_hints.py` | 110 | Optional hints |
| **Total** | **825** | **66% reduction from old 2,400 lines** |

## Related Documentation

- `CLAUDE.md` (lines 323-353) - Quick reference
- `docs/PERMISSION_SYSTEM_GUIDE.md` - User guide
- `docs/archived/permission-migration/` - Historical migration docs
- `templates/default_settings.json` - Default configuration template

---

**Last Updated:** November 2025 (Simplified Permission System)
**Migration Date:** November 1, 2025
**Old System:** Archived in `docs/archived/permission-migration/`
