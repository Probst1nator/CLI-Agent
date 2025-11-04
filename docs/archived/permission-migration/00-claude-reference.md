# Claude Code Permissions Overview

## What is the Permissions System?

Claude Code uses a permissions system to control which tools and commands can be executed. Permissions can be configured through settings files at multiple levels (user, project, or local), allowing fine-grained control over tool access while respecting organizational policies.

## Settings File Locations

Permissions are configured in `settings.json` files stored at these locations (in order of precedence):

| Level | Path | Scope | Notes |
|-------|------|-------|-------|
| User | `~/.claude/settings.json` | Global | Applies to all projects |
| Project | `.claude/settings.json` | Team-shared | Checked into source control |
| Local | `.claude/settings.local.json` | Personal | Ignored by git, overrides project settings |
| Enterprise | Platform-specific | Organization | Managed by administrators |

**Precedence Order**: Local > Project > User > Enterprise defaults

## Permission Configuration Structure

Permissions are defined using three lists: `allow`, `deny`, and `ask`.

### Basic Format

```json
{
  "permissions": {
    "allow": [],
    "deny": [],
    "ask": []
  }
}
```

## Permission Lists

### Allow List
Tools and commands that Claude can use **without asking for permission**.

```json
"allow": [
  "Bash(npm run lint)",
  "Read(~/.zshrc)",
  "Write(./src/**/*.js)"
]
```

**Use cases:**
- Frequently-used commands
- Safe read operations
- Development workflows

### Deny List
**Blocked** tools and commands. These cannot be executed.

```json
"deny": [
  "Bash(curl:*)",
  "Read(./.env)",
  "Read(./secrets/**)",
  "Bash(rm -rf:*)"
]
```

**Common patterns:**
- Sensitive files (`.env`, credentials, keys)
- Destructive commands
- External network requests
- Specific command patterns

### Ask List
Tools and commands that **require user confirmation** before execution.

```json
"ask": [
  "Bash(git push:*)",
  "Bash(git force-push:*)",
  "Write(.env)"
]
```

**Use cases:**
- Operations with side effects
- Repository-altering commands
- Writes to sensitive files

## Pattern Syntax

### Bash Patterns
Bash patterns use **prefix matching** and support wildcards:

```
Bash(npm:*)              # Matches: npm install, npm run, npm test, etc.
Bash(curl:*)             # Matches: curl http://..., curl -X POST, etc.
Bash(git push:*)         # Matches: git push, git push origin, etc.
Bash(rm:*)               # Matches: rm file, rm -rf /, etc.
```

**Note:** Bash patterns are prefix matches. To prevent accidental exposure of sensitive data, use the `deny` list for files rather than relying solely on command restrictions.

### File Patterns
File paths support glob patterns:

```
Read(./.env)              # Single file
Read(~/.ssh/id_rsa)       # User home expansion
Read(./secrets/**)        # Glob patterns (any depth)
Write(./build/**)         # Directory and subdirectories
```

### Tool Patterns
Tool specifications follow the format: `Tool` or `Tool(optional-specifier)`

```
Read                      # All read operations
Read(./src/**)            # Reads in src directory
Bash                      # All bash commands
Bash(npm:*)               # npm commands only
Edit(./config.json)       # Edit specific file
```

## Complete Example

```json
{
  "permissions": {
    "allow": [
      "Bash(npm:*)",
      "Bash(npm run test)",
      "Bash(npm run build)",
      "Read(./src/**)",
      "Read(./package.json)",
      "Edit(./src/**)",
      "Glob(**/*.js)",
      "Grep(pattern)"
    ],
    "deny": [
      "Bash(curl:*)",
      "Bash(git push:*)",
      "Bash(rm:*)",
      "Read(./.env)",
      "Read(./.env.local)",
      "Read(./secrets/**)",
      "Write(./.env)",
      "Write(./.env.local)"
    ],
    "ask": [
      "Bash(git push origin main)",
      "Bash(docker:*)",
      "Write(./package.json)"
    ]
  }
}
```

## Best Practices

### Security
1. **Block sensitive files** in the `deny` list:
   - `.env`, `.env.local`, `.env.*.local`
   - `secrets/`, `credentials/`, `.ssh/`
   - API keys, tokens, private keys

2. **Restrict destructive operations** in the `deny` list:
   - `git push --force`
   - `rm -rf`
   - Database operations without confirmation

### Development Workflow
1. **Allow common dev operations**:
   - `npm run`, `npm install`, `npm test`
   - `git commit`, `git add`
   - Build and test commands

2. **Ask for confirmation** on:
   - Repository-altering commands (`git push`)
   - Deployment operations
   - Writes to configuration files

### Organization
1. Define **project-level settings** (`.claude/settings.json`) for team consistency
2. Use **local settings** (`.claude/settings.local.json`) for personal overrides
3. Document permission rationale in comments or adjacent documentation

## Precedence and Conflicts

If a permission appears in multiple lists, the precedence is:

1. **Deny** - Most restrictive, always blocks
2. **Ask** - Requires confirmation
3. **Allow** - Least restrictive, always permits

**Example:**
If `Read(.env)` is in both `deny` and `allow`, the `deny` takes precedence and the read is blocked.

## Testing Permissions

When Claude Code encounters a tool or command:

1. Checks if it matches a `deny` rule → **Blocks execution**
2. Checks if it matches an `ask` rule → **Requests confirmation**
3. Checks if it matches an `allow` rule → **Executes without asking**
4. No match → **Uses default behavior** (typically asks)

## Additional Notes

- **Hooks** can enforce additional permission policies programmatically
- **Enterprise managed policies** can override user/project settings
- Bash patterns are **prefix matches** and not foolproof - use `deny` lists for sensitive files
- Settings files are JSON format and support standard JSON comments

## References

- [Claude Code IAM Documentation](https://docs.claude.com/en/docs/claude-code/iam.md)
- [Claude Code Settings Documentation](https://docs.claude.com/en/docs/claude-code/settings.md)
- [Claude Code Hooks Documentation](https://docs.claude.com/en/docs/claude-code/hooks.md)
