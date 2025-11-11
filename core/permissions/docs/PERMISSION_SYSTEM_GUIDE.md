# CLI-Agent Permission System - User Guide

## Quick Start

### 1. Copy the default template
```bash
/perms template
```

This creates `~/.cli-agent/settings.json` with sensible defaults.

### 2. Customize for your workflow
```bash
/perms view                            # See current settings
/perms add allow bash(npm:*)           # Allow all npm commands
/perms add deny readfile(./secrets/**) # Block secrets directory
/perms add ask bash(git push:*)        # Confirm before git push
```

### 3. Test your permissions
Run any command and observe permission behavior.

---

## Understanding the System

### Three Simple Lists

The permission system uses three lists: **allow**, **deny**, and **ask**.

```json
{
  "permissions": {
    "allow": [...],  // Execute without asking
    "deny": [...],   // Block completely
    "ask": [...]     // Prompt for confirmation
  }
}
```

### Precedence (What Wins)

1. **Deny** - Checked first, always blocks
2. **Ask** - Checked second, prompts user
3. **Allow** - Checked third, executes immediately
4. **Default** - No match, shows prompt

**Example:**
- If `bash(npm:*)` is in both allow and deny → **Deny wins**
- If only in ask → **Prompts every time**
- If only in allow → **Executes without asking**

---

## Pattern Syntax

### Basic Format

```
toolname(specifier)
```

**Examples:**
```json
"bash(npm:*)"           // Tool: bash, Command: npm*, Wildcard: *
"readfile(./src/**)"    // Tool: readfile, Path: ./src/**, Glob: **
"grepsearch"            // Tool only, no specifier (matches all uses)
```

### Tool Names

Use the **exact tool file name** (lowercase, no .py extension):

| Tool File | Permission Pattern |
|-----------|--------------------|
| `bash.py` | `bash(...)` |
| `readfile.py` | `readfile(...)` |
| `writefile.py` | `writefile(...)` |
| `editfile.py` | `editfile(...)` |
| `grepsearch.py` | `grepsearch(...)` |

**Important:** No aliases! Use the actual file name from `tools/` directory.

### Command Patterns (for bash/python)

**Syntax:** `bash(command:*)`

The `:*` means "and anything after this command".

**Examples:**
```json
"bash(npm:*)"           // Matches: npm install, npm run, npm test
"bash(git push:*)"      // Matches: git push, git push origin, git push --force
"bash(pytest:*)"        // Matches: pytest, pytest tests/, pytest -v
"bash(sudo:*)"          // Matches: sudo anything (to block)
```

**How Matching Works:**
- Split on first `:` → command part and wildcard part
- Check if actual command starts with pattern before `:`
- Example: `npm install` starts with `npm` → matches `bash(npm:*)`

### File Path Patterns

**Syntax:** `readfile(path/pattern)`

Uses **glob patterns** (fnmatch):

```json
"readfile(./src/**)"         // Any file in src/, any depth
"readfile(./.env)"           // Exact file
"readfile(./.env.*)"         // .env.local, .env.production, etc.
"readfile(./tests/*.py)"     // Python files in tests/ directory
"readfile(~/.ssh/id_rsa)"    // Absolute path with home expansion
```

**Glob Wildcards:**
- `*` - Matches any characters (within one directory level)
- `**` - Matches any characters (recursive, across directories)
- `?` - Matches exactly one character
- `[abc]` - Matches one of a, b, or c

---

## Common Use Cases

### 1. Allow Development Tools

```json
"allow": [
  "bash(npm:*)",
  "bash(pip:*)",
  "bash(pytest:*)",
  "bash(ruff:*)",
  "bash(black:*)",
  "bash(git:*)"
]
```

### 2. Block Sensitive Files

```json
"deny": [
  "readfile(.env)",
  "readfile(.env.*)",
  "readfile(./secrets/**)",
  "readfile(./credentials/**)",
  "writefile(.env)"
]
```

### 3. Block Dangerous Commands

```json
"deny": [
  "bash(sudo:*)",
  "bash(rm -rf:*)",
  "bash(dd:*)",
  "bash(mkfs:*)"
]
```

### 4. Require Confirmation for Risky Operations

```json
"ask": [
  "bash(git push:*)",
  "bash(git force-push:*)",
  "bash(docker:*)",
  "bash(npm install:*)",
  "writefile(./package.json)",
  "writefile(./requirements.txt)"
]
```

### 5. Project-Specific Permissions

```json
"allow": [
  "readfile(./src/**)",
  "readfile(./tests/**)",
  "readfile(./docs/**)",
  "editfile(./src/**)",
  "editfile(./tests/**)"
]
```

---

## Commands Reference

### `/perms view`
Show current permission settings.

```bash
/perms view
```

**Output:**
```
📋 Permission Settings
========================================

✅ ALLOW (executed without asking):
  • bash(npm:*)
  • readfile(./src/**)

❓ ASK (requires confirmation):
  • bash(git push:*)

❌ DENY (blocked):
  • readfile(.env)
  • bash(sudo:*)

📄 Config file: ~/.cli-agent/settings.json
```

### `/perms add`
Add a new permission rule.

**Syntax:**
```bash
/perms add <list> <rule>
```

**Examples:**
```bash
/perms add allow bash(npm:*)
/perms add deny readfile(.env)
/perms add ask bash(git push:*)
```

### `/perms remove`
Remove a permission rule.

**Syntax:**
```bash
/perms remove <rule>
```

**Examples:**
```bash
/perms remove bash(npm:*)
/perms remove readfile(.env)
```

Removes the rule from all lists (allow, deny, ask).

### `/perms reset`
Clear all permission settings.

```bash
/perms reset
```

**Warning:** This removes all custom permissions. Use with caution.

### `/perms template`
Copy default template to `~/.cli-agent/settings.json`.

```bash
/perms template
```

**What it includes:**
- Safe commands pre-allowed (ls, pwd, grep, npm, pip, pytest, git, etc.)
- Sensitive files pre-denied (.env, secrets/, credentials/)
- Dangerous commands pre-denied (sudo, rm -rf)
- Risky operations require confirmation (git push, docker, npm install)

---

## Interactive Prompts

When a command requires permission and is not in allow list, you'll see:

```
🔒 Permission Required

Tool: bash
Command: npm install

[y] Allow once
[n] Deny once
[a] Always allow
[d] Always deny
```

**Options:**
- **`y`** or **`Enter`**: Allow this once
- **`n`**: Deny this once
- **`a`**: Add to allow list and save
- **`d`**: Add to deny list and save

**After choosing "always":**
- Settings automatically updated in `~/.cli-agent/settings.json`
- Future uses won't prompt again

---

## Dev Tool Hints

The system provides optional hints when it detects common tools:

```
Tool: bash(pytest tests/)

💡 Hint: This is a testing tool.
Suggested rule: /perms add allow bash(pytest:*)
```

**Categories Detected:**
- **Linting:** ruff, flake8, pylint, black, mypy
- **Testing:** pytest, unittest, nose, tox
- **Building:** npm, pip, poetry, make, cargo
- **Git:** git commands
- **Search:** grep, rg, find, ack

**Important:** Hints are suggestions only. They **do NOT** automatically grant permissions.

---

## Best Practices

### 1. Start with Template
```bash
/perms template
```
Get sensible defaults, then customize.

### 2. Explicit Permissions
**Don't wait for prompts.** Declare what you allow upfront:
```bash
/perms add allow bash(npm:*)
/perms add allow bash(pytest:*)
```

### 3. Deny Sensitive Operations
Always block dangerous operations:
```json
"deny": [
  "readfile(.env)",
  "bash(sudo:*)",
  "bash(rm -rf:*)"
]
```

### 4. Use Ask for Side Effects
Require confirmation for operations that modify external state:
```json
"ask": [
  "bash(git push:*)",
  "bash(docker:*)",
  "bash(npm install:*)"
]
```

### 5. Project-Specific Config
For team projects, commit `.cli-agent/settings.json` to repository root.

### 6. Glob Patterns for Directories
Instead of individual files:
```json
"allow": [
  "readfile(./src/**)",      // All files in src/
  "editfile(./tests/**)"     // All files in tests/
]
```

### 7. Review Periodically
```bash
/perms view
```
Check what permissions are granted and remove unused ones.

---

## Troubleshooting

### "Permission denied" for allowed command

**Possible Causes:**
1. **Pattern doesn't match**: Check if pattern is correct
   ```bash
   /perms view  # Verify pattern
   ```

2. **Deny overrides allow**: Deny list is checked first
   ```bash
   /perms remove <rule>  # Remove blocking deny rule
   ```

3. **Tool name wrong**: Use exact tool file name (lowercase, no .py)
   ```json
   "readfile(...)"  // ✅ Correct
   "ReadFile(...)"  // ❌ Wrong (capitalization)
   "read(...)"      // ❌ Wrong (not the actual tool name)
   ```

### "Too many prompts"

**Solution:** Add frequently used patterns to allow list:

```bash
# Common development commands
/perms add allow bash(npm:*)
/perms add allow bash(pytest:*)
/perms add allow bash(git:*)
/perms add allow readfile(./src/**)
/perms add allow editfile(./src/**)
```

### "Blocked unexpectedly"

**Check deny list:**
```bash
/perms view
```

Look for overly broad deny patterns that might be catching your command.

**Remove specific deny rule:**
```bash
/perms remove <blocking-rule>
```

### "Pattern not working"

**Common Issues:**

1. **Missing colon for commands:**
   ```json
   "bash(npm*)"     // ❌ Wrong
   "bash(npm:*)"    // ✅ Correct
   ```

2. **Wrong glob syntax:**
   ```json
   "readfile(./src/*)"    // ✅ One level only
   "readfile(./src/**)"   // ✅ Recursive (all levels)
   ```

3. **Relative vs absolute paths:**
   ```json
   "readfile(src/**)"     // ❌ Might not match
   "readfile(./src/**)"   // ✅ Relative to working directory
   "/home/user/project/src/**"  // ✅ Absolute path
   ```

### "Can't edit settings.json"

**Manual Editing:**
```bash
# Open in editor
nano ~/.cli-agent/settings.json
# or
code ~/.cli-agent/settings.json
```

**Verify JSON syntax:**
```bash
python3 -m json.tool ~/.cli-agent/settings.json
```

If JSON is invalid, you'll see an error message.

---

## Configuration File Format

Full example with all options:

```json
{
  "permissions": {
    "allow": [
      "bash(npm:*)",
      "bash(pip:*)",
      "bash(pytest:*)",
      "bash(ruff:*)",
      "bash(black:*)",
      "bash(git:*)",
      "bash(ls:*)",
      "bash(pwd:*)",
      "bash(grep:*)",
      "readfile(./src/**)",
      "readfile(./tests/**)",
      "readfile(./docs/**)",
      "editfile(./src/**)",
      "editfile(./tests/**)",
      "grepsearch"
    ],
    "deny": [
      "readfile(.env)",
      "readfile(.env.*)",
      "readfile(./secrets/**)",
      "readfile(./credentials/**)",
      "writefile(.env)",
      "bash(sudo:*)",
      "bash(rm -rf:*)",
      "bash(dd:*)"
    ],
    "ask": [
      "bash(git push:*)",
      "bash(git force-push:*)",
      "bash(docker:*)",
      "bash(npm install:*)",
      "writefile(./package.json)",
      "writefile(./requirements.txt)"
    ]
  }
}
```

---

## Team Collaboration

### Project-Level Settings

1. **Create** `.cli-agent/settings.json` in project root
2. **Commit** to version control
3. **Team members** copy to `~/.cli-agent/settings.json` or symlink:
   ```bash
   ln -s $(pwd)/.cli-agent/settings.json ~/.cli-agent/settings.json
   ```

### Example Team Config

```json
{
  "permissions": {
    "allow": [
      "bash(npm run:*)",
      "bash(npm test:*)",
      "bash(pytest:*)",
      "bash(ruff:*)",
      "bash(git:*)",
      "readfile(./src/**)",
      "readfile(./tests/**)",
      "editfile(./src/**)",
      "editfile(./tests/**)"
    ],
    "deny": [
      "readfile(.env)",
      "readfile(.env.*)",
      "bash(npm publish:*)",
      "bash(sudo:*)"
    ],
    "ask": [
      "bash(git push:*)",
      "bash(npm install:*)",
      "writefile(./package.json)"
    ]
  }
}
```

---

## Migration from Old System

If you used the old permission system (before November 2025):

### What Changed

**Old system** (2,400 lines, 7 rule types):
- Complex interactive learning
- Dev tool categories
- Command hashing
- Group permissions
- Pattern permissions

**New system** (825 lines, 3 lists):
- Simple config-based
- Explicit declarations
- Allow/deny/ask only

### Migration Steps

1. **No automatic migration** - Start fresh
2. **Copy template:** `/perms template`
3. **Customize** for your workflow
4. **Old config** at `~/.cli-agent/permissions.json` is ignored

### Manual Migration

If you want to preserve old permissions:

1. **View old config:**
   ```bash
   cat ~/.cli-agent/permissions.json
   ```

2. **Convert to new format** manually:
   - `always_allow_tools.readfile` → `allow: ["readfile(...)"]`
   - `always_deny_tools.bash` → `deny: ["bash(...)"]`
   - Dev tool categories → Explicit command patterns

3. **Test new config:**
   ```bash
   /perms view
   ```

---

## Advanced Patterns

### Wildcard Combinations

```json
"bash(git *:*)"          // All git commands with space (git commit, git push, etc.)
"readfile(./**/*.py)"    // All Python files recursively
"editfile(./src/**/*test*.py)"  // Test files in src/
```

### Multiple Patterns for Same Tool

```json
"allow": [
  "bash(npm run:*)",     // npm run commands
  "bash(npm test:*)",    // npm test commands
  "bash(npm list:*)"     // npm list commands
]
```

This gives more granular control than `bash(npm:*)` which matches everything.

### Project-Specific Deny

```json
"deny": [
  "readfile(./project-a/.env)",
  "readfile(./project-b/secrets/**)"
]
```

### Environment-Specific Permissions

**.env files:**
```json
"deny": [
  "readfile(.env)",
  "readfile(.env.local)",
  "readfile(.env.production)",
  "readfile(.env.*.local)"
]
```

Or use a pattern:
```json
"deny": [
  "readfile(.env*)"
]
```

---

## Related Documentation

- `docs/PERMISSION_SYSTEM.md` - Technical documentation
- `CLAUDE.md` (lines 323-353) - Quick reference
- `docs/archived/permission-migration/` - Historical migration docs
- `templates/default_settings.json` - Default template

---

**Last Updated:** November 2025 (Simplified Permission System)
