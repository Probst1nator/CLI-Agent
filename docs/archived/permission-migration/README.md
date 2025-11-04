# Permission System Migration Archive

**Status:** ✅ Completed (November 2025)

This directory contains historical documentation from the permission system simplification project that migrated CLI-Agent from a complex 2,400-line permission system to a Claude Code-style config-based approach.

## Migration Results

- **Code Reduction:** 2,400 lines → 825 lines (66% reduction)
- **Complexity:** 7 rule types → 3 simple lists (allow/deny/ask)
- **Implementation:** `core/permissions/` directory
  - `settings_manager.py` (331 lines)
  - `permission_prompt.py` (384 lines)
  - `dev_tool_hints.py` (110 lines)

## Archive Contents

| File | Purpose |
|------|---------|
| `00-claude-reference.md` | Educational reference on Claude Code permissions |
| `01-proposal.md` | Initial complexity reduction analysis and proposal |
| `02-comparison.md` | Detailed scenarios comparing old vs new approaches |
| `03-code-examples.md` | Code examples showing the simplified implementation |
| `04-implementation-summary.md` | Final implementation summary and metrics |

## Current Documentation

For current permission system documentation, see:
- `docs/PERMISSION_SYSTEM.md` - Technical documentation
- `docs/PERMISSION_SYSTEM_GUIDE.md` - User guide
- `CLAUDE.md` - Quick reference (lines 323-353)

## Key Changes

### Before (Old System)
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

### After (New System)
```json
{
  "permissions": {
    "allow": ["bash(npm:*)", "readfile(./src/**)"],
    "deny": ["readfile(.env)", "bash(sudo:*)"],
    "ask": ["bash(git push:*)", "bash(docker:*)"]
  }
}
```

## Historical Context

This migration was driven by:
1. **Excessive complexity** - 8-level precedence, 7 rule types, intelligent detection
2. **Hard to audit** - Learned permissions were opaque
3. **Team unfriendly** - No shared config, interactive prompts only
4. **Misalignment** - Different from Claude Code's approach

The new system prioritizes simplicity, transparency, and team collaboration over automatic detection and learning.

---

*This archive preserved for historical reference and to document the migration decision-making process.*
