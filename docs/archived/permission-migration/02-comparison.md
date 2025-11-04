# Permission System: CLI-Agent vs Claude Code - Side-by-Side Scenarios

## Scenario 1: "I want to allow Python linting (ruff, black) in my project"

### **CLI-Agent Current Approach**

```
User runs: python main.py
LLM outputs: <bash><command>ruff check /project</command></bash>
System: ✅ PathDetector.analyze_tool_block()
  ├─ Extract path: /project
  ├─ Detect command: ruff check
  ├─ Detect category: LINTING (matches 'ruff' in COMMON_DEV_TOOLS)
  └─ needs_permission: True

System: Show permission prompt:
┌─────────────────────────────────────────────────────┐
│ 💻 /project$ ▶️ ruff check /project               │
│                                                     │
│ [Enter] or y → Allow once                          │
│ r → Allow read-only operations                     │
│ a → Always allow in this directory                 │
│ s → Always allow for this scope                    │
│ t → Always allow for testing tools ← SMART ACTION  │
│ ? → Show help                                      │
└─────────────────────────────────────────────────────┘

User presses: 's'
System: 🧠 Detects scope = PROJECT
System: Saves rule:
  "always_allow_dev_tools": {
    "linting": ["/home/user/project/*"]
  }
Result: ✅ Ruff executes

Future runs: Automatic approval (no prompt)
```

**Workflow: Interactive → Learn → Remember**

---

### **Claude Code Approach**

```
User edits: ~/.claude/settings.json
  "permissions": {
    "allow": [
      "Bash(ruff:*)",
      "Bash(black:*)",
      "Bash(flake8:*)"
    ]
  }

User runs: python main.py
LLM outputs: <bash><command>ruff check /project</command></bash>
System: Check permission
  ├─ Matches deny? No
  ├─ Matches ask? No
  ├─ Matches allow? "Bash(ruff:*)" → YES
  └─ Result: ✅ Approve

Result: ✅ Ruff executes immediately
```

**Workflow: Configure → Run → Execute**

---

### **Comparison**

| Aspect | CLI-Agent | Claude Code |
|--------|-----------|------------|
| **Time to first execution** | Instant (run and choose) | Slightly slower (edit config) |
| **Number of prompts** | 1 | 0 |
| **Config lines** | Auto-saved JSON | Manual: 3 lines in config |
| **Code executed** | 400+ lines (prompt, detect, save) | ~20 lines (regex match) |
| **User mental model** | "System learns from my choices" | "I declare what I allow" |
| **Discoverability** | User discovers options via prompt | User must know what to write |
| **Flexibility** | Learns exact scope (file/dir/project) | User specifies scope explicitly |

---

## Scenario 2: "Block access to .env file"

### **CLI-Agent Current Approach**

```
User adds rule via `/perms add` wizard:
  /perms add tool readfile /home/user/.env

System: Shows prompt:
  "Add this file to deny list? (y/n)"

User presses: 'd' (deny)
System: Saves rule:
  "always_deny_tools": {
    "readfile": ["/home/user/.env"]
  }

Future: LLM tries <readfile><path>/home/user/.env</path></readfile>
System: Checks is_always_denied() → TRUE
Result: ❌ Blocked automatically (no prompt)
```

---

### **Claude Code Approach**

```
User edits: ~/.claude/settings.json
  "permissions": {
    "deny": [
      "Read(./.env)",
      "Write(.env)"
    ]
  }

Future: LLM tries <readfile><path>.env</path></readfile>
System: Check permission
  ├─ Matches deny? "Read(./.env)" → YES
  └─ Result: ❌ Block immediately

Result: ❌ Blocked
```

---

## Scenario 3: "Ask me before pushing to git"

### **CLI-Agent Current Approach**

```
User wants: Require confirmation before git push

Current system: No direct "ask" mechanism
Workaround: Don't add to "always_allow", let permission prompt appear

Result: Permission prompt every time
  (But no way to make it "ask once per session")
```

---

### **Claude Code Approach**

```
User edits: ~/.claude/settings.json
  "permissions": {
    "ask": [
      "Bash(git push:*)",
      "Bash(git force-push:*)"
    ]
  }

LLM tries: <bash><command>git push origin main</command></bash>
System: Check permission
  ├─ Matches deny? No
  ├─ Matches ask? "Bash(git push:*)" → YES
  └─ Result: ❓ Prompt user for confirmation

User choice: [Allow] / [Deny] / [Always Allow]
Result: Executes or blocks per user choice
```

**Difference:** Claude Code has explicit "ask" state; CLI-Agent doesn't.

---

## Scenario 4: "Different rules for different directories"

### **CLI-Agent Current Approach**

```
User wants:
  - Allow linting in /project/
  - Block linting in /tmp/

User adds rules:
  pm.add_always_allow("bash",
    dev_tool_category="linting",
    pattern="/project/*")

  pm.add_always_deny("bash",
    file_path="/tmp")

Storage:
  "always_allow_dev_tools": {
    "linting": ["/project/*"]
  },
  "always_deny_tools": {
    "bash": ["/tmp"]
  }

Checking: When /tmp/test.py:ruff called
  → Checks deny rules FIRST
  → Finds /tmp in deny list
  → Result: ❌ Blocked

Complexity: 7 rule types with precedence order
```

---

### **Claude Code Approach**

```
User edits: ~/.claude/settings.json
  "permissions": {
    "allow": [
      "Bash(ruff:*)"  # Global - matches everywhere
    ],
    "deny": [
      "Bash(ruff:*:*/tmp/*)"  # Deny ruff in /tmp
    ]
  }

Precedence:
  1. Check deny → /tmp matches → DENY
  2. Check ask → No match
  3. Check allow → Doesn't matter (deny won)

Result: ❌ Blocked in /tmp

Note: This is awkward; Claude Code doesn't have path-scoped rules
Workaround: Either block tool globally or allow globally
```

**Issue:** Claude Code's glob patterns are file-path based, not scope-aware like CLI-Agent.

---

## Scenario 5: "Allow npm commands but not npm install"

### **CLI-Agent Current Approach**

```
User wants: npm run lint, npm test OK; npm install BLOCKED

Possible solutions:
  1. Command pattern matching:
     pm.add_always_allow("bash",
       command_pattern="npm run*",  # Not "npm install*"
       scope_path="/project/*")

  2. Command hash (exact):
     pm.add_always_allow("bash",
       command_hash="abc123...",  # Hash of exact "npm test" command
       )

Complexity: Requires command pattern or hashing
```

---

### **Claude Code Approach**

```
User edits: ~/.claude/settings.json
  "permissions": {
    "allow": [
      "Bash(npm run:*)",
      "Bash(npm test:*)",
      "Bash(npm list:*)"
    ],
    "deny": [
      "Bash(npm install:*)",
      "Bash(npm uninstall:*)",
      "Bash(npm publish:*)"
    ]
  }

Result: Explicit allow for safe npm commands, deny for risky ones
```

**Simpler:** Claude Code forces explicit allowlists; CLI-Agent tries to be smart about patterns.

---

## Scenario 6: "File operation permissions"

### **CLI-Agent Current Approach**

```
User wants: Allow editing in /project/src/ only

pm.add_always_allow("editfile", pattern="/project/src/*")

Storage:
  "always_allow_tools": {
    "editfile": ["/project/src/*"]
  }

Result: Can edit any file in /project/src/
```

---

### **Claude Code Approach**

```
User edits: ~/.claude/settings.json
  "permissions": {
    "allow": [
      "Edit(./src/**)"
    ]
  }

Result: Can edit any file in ./src/ (with full glob support)

Features:
  - Glob patterns: **, *, ?, [abc]
  - Path expansion: ~, ./
  - Can be very specific: Edit(./src/**/*.js)
```

**Claude Code wins:** Better glob support, cleaner syntax.

---

## Scenario 7: "Sensitive content automatic blocking"

### **CLI-Agent Current Approach**

```
LLM tries: <bash><command>cat /home/user/.env</command></bash>
System: enhanced_permission_prompt()
  ├─ PathDetector checks for sensitive keywords
  ├─ Finds: ".env" in path
  ├─ Checks: Tool is 'bash' (risky tool)
  └─ Result: 🔒 SECURITY: Access to .env files blocked

User cannot override this even with "always allow"
Automatic blocking prevents accidental secret leaks
```

---

### **Claude Code Approach**

```
User edits: ~/.claude/settings.json
  "permissions": {
    "deny": [
      "Read(.env)",
      "Read(.env.*)",
      "Read(./secrets/**)"
    ]
  }

Result: 🔒 Access blocked

If user removes these from deny list and adds to allow:
  "allow": ["Read(.env)"]
  → .env becomes READABLE

⚠️  RISK: User can accidentally enable access to secrets
```

**CLI-Agent wins on security:** Automatic blocking of .env regardless of config.

---

## Summary Table: What Each System Excels At

| Feature | CLI-Agent | Claude Code | Winner |
|---------|-----------|------------|--------|
| **Simplicity** | Moderate | Very High | 🏆 Claude |
| **Code Size** | 2,400 lines | ~200 lines | 🏆 Claude |
| **Automatic Detection** | Yes (dev tools) | No | 🏆 CLI-Agent |
| **Interactive Learning** | Yes | No | 🏆 CLI-Agent |
| **Team Collaboration** | No | Yes (shared config) | 🏆 Claude |
| **Sensitive Content Blocking** | Automatic | Manual | 🏆 CLI-Agent |
| **Path-Scoped Rules** | Yes | Partial (glob only) | 🏆 CLI-Agent |
| **Glob Pattern Support** | Basic (`/*`) | Advanced (`**/*.js`) | 🏆 Claude |
| **CI/CD Friendly** | No (interactive) | Yes (static config) | 🏆 Claude |
| **User Onboarding** | Easy (prompted) | Moderate (config) | 🏆 CLI-Agent |
| **Permission Auditability** | Hard (learned rules) | Easy (explicit config) | 🏆 Claude |
| **Precedence Clarity** | Complex (8 levels) | Simple (3 levels) | 🏆 Claude |

---

## Hybrid Approach (Best of Both Worlds)

If you want to keep beneficial features while reducing complexity:

```python
# Simplified config (Claude style)
{
  "permissions": {
    "allow": ["Bash(npm:*)", "Read(./src/**)", "Edit(./src/**)"],
    "deny": ["Read(.env)"],
    "ask": ["Bash(git push:*)"]
  }
}

# With optional smart detection (CLI-Agent feature)
# If LLM uses "ruff check" and config says allow "Bash(ruff:*)",
# System could suggest: "This is a linting tool, detected for /project"
```

**Benefits:**
- Simple config format (Claude style)
- Optional smart suggestions (CLI-Agent style)
- No interactive prompts
- Reduced code (~1,200 lines instead of 2,400)

---

## Conclusion

**Choose Claude style if you want:**
- ✅ Simpler, cleaner code
- ✅ Better team collaboration
- ✅ CI/CD compatibility
- ✅ Lower maintenance burden
- ❌ No interactive learning
- ❌ Users must pre-configure permissions

**Keep CLI-Agent style if you want:**
- ✅ Interactive permission prompts
- ✅ Automatic tool detection
- ✅ Learning from user choices
- ✅ Path-scoped rule granularity
- ✅ Automatic .env protection
- ❌ 2,400 lines of complex code
- ❌ Hard to audit permission decisions
- ❌ Not suitable for teams/CI

**Recommendation:** Adopt Claude style with optional suggestion system (~1,200 lines total).
