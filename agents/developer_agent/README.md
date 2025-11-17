# Developer Agent

A minimal, focused AI agent for software development tasks with essential tools for coding, version control, and GitHub operations.

## Overview

The Developer Agent is built on the minimal agent architecture (~250 lines of core code) and provides a curated set of 7 essential development tools:

1. **readfile** - Read source code with line numbers and ranges
2. **writefile** - Create new files with content
3. **editfile** - Modify existing files using find-replace
4. **grepsearch** - Search for patterns in files and directories
5. **bash** - Execute shell commands (npm, pip, pytest, etc.)
6. **git** - Version control operations (status, diff, commit, push, etc.)
7. **github** - Safe GitHub operations (PR, issues, status checks)

## Features

- **Multi-Provider LLM Support**: Automatically fails over between Google Gemini, Ollama (local), and Anthropic Claude
- **Auto-Discovery**: Tools are automatically loaded from the `tools/` directory
- **Safety Built-In**: Blocks destructive operations and warns before dangerous git commands
- **Simple Interface**: All tools use XML-based communication
- **Minimal Dependencies**: Only 4 external libraries needed

## Installation

1. **Install Python 3.9+**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure API keys**:
```bash
cp .env.example .env
# Edit .env and add at least one API key
```

4. **Install GitHub CLI** (optional, for GitHub operations):
```bash
# macOS
brew install gh

# Linux
# See: https://github.com/cli/cli/blob/trunk/docs/install_linux.md

# Windows
# See: https://github.com/cli/cli#windows

# Authenticate
gh auth login
```

## Quick Start

```bash
# Run the agent
python main.py
```

Example conversation:
```
👤 You: Find all Python files in the src directory

🤖 Agent: I'll search for Python files in the src directory.
<grepsearch><pattern>\.py$</pattern><path>./src</path></grepsearch>

[Results shown...]

👤 You: Read the main.py file

🤖 Agent: <readfile>./src/main.py</readfile>

[File contents with line numbers...]

👤 You: Add a new function to greet users

🤖 Agent: <editfile>
<filepath>./src/main.py</filepath>
<find>if __name__ == "__main__":</find>
<replace>def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":</replace>
</editfile>

Successfully edited the file.

👤 You: Commit this change

🤖 Agent: Let me check the current status first.
<git><command>status</command></git>
<git><command>diff</command></git>

[Shows changes...]

<git><command>add</command><files>src/main.py</files></git>
<git><command>commit</command><message>Add greet function for user greetings</message></git>

Successfully committed!
```

## Tool Reference

### readfile
Read files with automatic encoding detection and line number support.

```xml
<!-- Simple format -->
<readfile>./src/app.py</readfile>

<!-- With line ranges -->
<readfile>
  <filepath>./src/app.py</filepath>
  <lines>50-75</lines>
</readfile>
```

### writefile
Create or overwrite files.

```xml
<writefile>
  <filepath>./src/new_module.py</filepath>
  <content>
def hello():
    print("Hello, world!")
  </content>
</writefile>
```

### editfile
Find and replace text in files.

```xml
<editfile>
  <filepath>./src/app.py</filepath>
  <find>old_function_name</find>
  <replace>new_function_name</replace>
  <all>true</all>  <!-- Optional: replace all occurrences -->
</editfile>
```

### grepsearch
Search for patterns in files.

```xml
<grepsearch>
  <pattern>TODO</pattern>
  <path>./src</path>
  <max_results>50</max_results>  <!-- Optional -->
</grepsearch>
```

### bash
Execute shell commands.

```xml
<bash>npm install</bash>
<bash>pytest tests/</bash>
<bash>python -m pip list</bash>
```

**Safety**: Blocks potentially destructive commands like `rm -rf /`, `mkfs`, etc.

### git
Version control operations.

```xml
<!-- Check status -->
<git><command>status</command></git>

<!-- View diff -->
<git><command>diff</command></git>

<!-- Add files -->
<git><command>add</command><files>src/main.py</files></git>

<!-- Commit -->
<git><command>commit</command><message>Fix bug in auth module</message></git>

<!-- Push -->
<git><command>push</command></git>

<!-- View log -->
<git><command>log</command><args>--oneline -10</args></git>

<!-- Create branch -->
<git><command>branch</command><branch>feature/new-feature</branch></git>

<!-- Checkout -->
<git><command>checkout</command><branch>main</branch></git>
```

**Safety**: Warns before force operations, recommends showing status/diff before commits.

### github
GitHub operations using gh CLI.

```xml
<!-- Create PR -->
<github>
  <operation>pr-create</operation>
  <title>Fix authentication bug</title>
  <body>This PR fixes the authentication issue in the login module</body>
</github>

<!-- List PRs -->
<github><operation>pr-list</operation></github>

<!-- Check PR status -->
<github>
  <operation>pr-status</operation>
  <pr_number>123</pr_number>
</github>

<!-- View PR checks -->
<github>
  <operation>pr-checks</operation>
  <pr_number>123</pr_number>
</github>

<!-- Create issue -->
<github>
  <operation>issue-create</operation>
  <title>Bug: Login fails on mobile</title>
  <body>Detailed description...</body>
</github>

<!-- List issues -->
<github><operation>issue-list</operation></github>

<!-- Repo info -->
<github><operation>repo-info</operation></github>

<!-- Workflow status -->
<github><operation>workflow-view</operation></github>
```

**Safety**: All operations are read-only except PR/issue creation. Requires gh CLI to be installed and authenticated.

## Configuration

### Environment Variables

Create a `.env` file (from `.env.example`) with at least one API key:

```bash
# Recommended: Google Gemini (fast, cost-effective)
GOOGLE_API_KEY=your-key-here

# Alternative: Anthropic Claude
ANTHROPIC_API_KEY=your-key-here

# Ollama works without API key (just install Ollama)
```

### LLM Fallback Chain

The agent tries providers in this order:
1. Google Gemini Flash Lite (fastest, cheapest)
2. Google Gemini Flash (balanced)
3. Ollama qwen3:1.7b (local, private)
4. Anthropic Claude Haiku (most capable)

If one provider fails, it automatically tries the next.

## Development Workflow Examples

### Bug Fix Workflow
```
1. Search for the bug: grepsearch
2. Read relevant files: readfile
3. Edit the code: editfile
4. Run tests: bash pytest tests/
5. Check git status: git status
6. Review changes: git diff
7. Commit: git add + git commit
8. Push: git push
9. Create PR: github pr-create
```

### Feature Development
```
1. Create branch: git branch feature/new-feature
2. Checkout: git checkout feature/new-feature
3. Create new files: writefile
4. Edit existing: editfile
5. Test: bash npm test
6. Commit iteratively: git add + git commit
7. Push: git push
8. Create PR: github pr-create
9. Check status: github pr-status
```

### Code Review
```
1. List PRs: github pr-list
2. View PR: github pr-view 123
3. Check CI status: github pr-checks 123
4. Read changed files: readfile
5. Search for patterns: grepsearch
6. Run tests locally: bash pytest
```

## Architecture

```
developer_agent/
├── main.py              # Agent loop (250 lines)
├── chat.py              # Message management (60 lines)
├── llm_router.py        # Multi-provider LLM (320 lines)
├── tools/               # Auto-discovered tools
│   ├── readfile.py      # Read files with line numbers
│   ├── writefile.py     # Create files
│   ├── editfile.py      # Edit files (find-replace)
│   ├── grepsearch.py    # Pattern search
│   ├── bash.py          # Shell command execution
│   ├── git.py           # Git operations
│   └── github.py        # GitHub operations (gh CLI)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── README.md            # This file
└── CLAUDE.md            # Developer guide

Total: ~1200 lines of code
```

### Tool Contract

All tools follow a 3-method interface:
```python
class toolname:
    @staticmethod
    def get_delim() -> str:
        return 'toolname'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "toolname",
            "description": "What it does",
            "example": "<toolname>example</toolname>"
        }

    @staticmethod
    def run(content: str) -> str:
        # Parse XML, execute, return result
        pass
```

Tools are automatically discovered and loaded at startup.

## Limitations

By design, this agent is minimal and focused:

- ❌ No session persistence (restart = fresh state)
- ❌ No permission system (use in trusted environments)
- ❌ No caching or vector DB
- ❌ No async tool execution
- ❌ No web UI (console only)
- ❌ No audio/TTS support
- ❌ No multi-user support

For advanced features, see the main CLI-Agent project.

## Extending

### Add a New Tool

1. Create `tools/newtool.py`:
```python
class newtool:
    @staticmethod
    def get_delim() -> str:
        return 'newtool'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "newtool",
            "description": "What it does",
            "example": "<newtool>example</newtool>"
        }

    @staticmethod
    def run(content: str) -> str:
        # Your implementation
        return "result"
```

2. Restart the agent - tool is auto-discovered!

## Troubleshooting

### No providers configured
- Add at least one API key to `.env`
- Or install Ollama for local models: https://ollama.ai/

### GitHub operations fail
- Install gh CLI: https://cli.github.com/
- Authenticate: `gh auth login`

### Import errors
- Run: `pip install -r requirements.txt`

### Encoding errors when reading files
- The tool auto-detects encoding using chardet
- Override with: `<readfile><filepath>file.txt</filepath><encoding>latin-1</encoding></readfile>`

## License

Part of the CLI-Agent project. See main repository for license information.

## Related

- **minimal_agent**: Reference implementation with 4 basic tools
- **CLI-Agent**: Full-featured agent with 20+ tools, web UI, permissions, etc.

## Contributing

This is a reference implementation. For contributions, see the main CLI-Agent repository.
