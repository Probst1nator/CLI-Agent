# CLAUDE.md - Developer Agent Guide

Instructions for Claude Code when working with the Developer Agent.

## Overview

The Developer Agent is a **minimal, focused agent for software development** with 7 essential tools. It demonstrates best practices for building specialized agents while keeping the codebase clean and maintainable (~1200 lines total).

## Quick Reference

### Running
```bash
cd agents/developer_agent
python main.py
```

### Testing Tools
```python
# From Python REPL
from tools.readfile import readfile
result = readfile.run("./main.py")
print(result)
```

### Adding a Tool
1. Create `tools/newtool.py` with 3-method contract
2. Restart agent (auto-discovered)
3. Test with LLM

## Architecture Principles

### 1. Minimal by Design

This agent intentionally limits complexity:
- **7 tools** (vs 20+ in main CLI-Agent)
- **No permissions system** (trusted environment)
- **No persistence** (stateless sessions)
- **No caching** (simple operation)
- **Console-only** (no web UI)

**Why**: Easier to understand, modify, and extend. Use as a template for specialized agents.

### 2. Tool-Centric Design

Everything happens through tools:
- **File operations**: readfile, writefile, editfile
- **Search**: grepsearch
- **Execution**: bash
- **Version control**: git
- **Collaboration**: github

**Pattern**: Tools are self-contained, XML-based, auto-discovered.

### 3. Safety First

Built-in safety checks:
- **bash**: Blocks `rm -rf /`, `mkfs`, `dd if=`, format commands
- **git**: Warns on `--force`, `--hard`, `--delete` operations
- **github**: Read-only operations (except PR/issue creation)

**Philosophy**: Better to ask permission than recover from disasters.

## Core Components

### main.py (~250 lines)
- **DeveloperAgent class**: Agent loop
- **_load_tools()**: Dynamic tool discovery
- **_initialize_system_prompt()**: Inject tool docs into LLM context
- **_parse_tool_calls()**: Extract `<toolname>...</toolname>` from responses
- **_execute_tool()**: Route to tool.run()
- **process_message()**: Main agent loop (max 10 iterations)

### chat.py (~60 lines)
- **Role enum**: SYSTEM, USER, ASSISTANT
- **Message dataclass**: Simple message structure
- **Chat class**: Message list management

### llm_router.py (~320 lines)
- **ModelSpec**: Provider + model key
- **FALLBACK_MODELS**: Chain of models to try
- **LLMRouter**: Multi-provider abstraction with auto-failover
- **Providers**: Google Gemini, Ollama (local), Anthropic Claude

## Tool Implementation Guide

### The 3-Method Contract

Every tool must implement:

```python
class toolname:
    @staticmethod
    def get_delim() -> str:
        """XML tag for this tool"""
        return 'toolname'

    @staticmethod
    def get_tool_info() -> dict:
        """Documentation for LLM system prompt"""
        return {
            "name": "toolname",
            "description": "What it does and when to use it",
            "example": "<toolname>usage example</toolname>"
        }

    @staticmethod
    def run(content: str) -> str:
        """Execute the tool with XML content"""
        # Parse XML
        # Execute logic
        # Return result or error string
        pass
```

### XML Parsing Pattern

All tools use regex for XML extraction:

```python
import re

# Single tag
pattern_match = re.search(r'<tag>(.*?)</tag>', content, re.DOTALL)
if not pattern_match:
    return "Error: Invalid format. Input must contain a <tag> tag."

value = pattern_match.group(1).strip()
```

### Error Handling Pattern

Return error strings, don't raise exceptions:

```python
@staticmethod
def run(content: str) -> str:
    try:
        # Parse and validate
        if not something:
            return "Error: Validation failed"

        # Execute
        result = do_work()

        return result

    except PermissionError:
        return f"Error: Permission denied"
    except Exception as e:
        return f"Error: {str(e)}"
```

### Tool Examples

#### Simple Read Tool (readfile.py)
- Parses filepath from XML
- Auto-detects encoding (UTF-8 → chardet → latin-1)
- Supports line ranges: `50-75`, `10-`, `-20`
- Returns content with line numbers

#### Write Tool (writefile.py)
- Parses filepath + content from XML
- Creates parent directories if needed
- Overwrites existing files (with warning in description)

#### Complex Tool (git.py)
- Routes based on `<command>` tag
- Validates git repository
- Blocks dangerous operations
- Formats output consistently

## Development Workflows

### Typical Development Task
```
User: "Fix the authentication bug in auth.py"

Agent Process:
1. grepsearch: Find auth.py and related files
2. readfile: Read auth.py with line numbers
3. editfile: Fix the bug
4. bash: Run tests (pytest tests/test_auth.py)
5. git: Check status, diff, add, commit
6. github: Create PR
```

### Code Review Task
```
User: "Review PR #123"

Agent Process:
1. github: View PR details
2. github: Check CI status
3. readfile: Read changed files
4. grepsearch: Search for anti-patterns
5. bash: Run tests locally
```

## Safety Guidelines

### When to Block Operations

**Always block**:
- `rm -rf /` or similar destructive file operations
- `mkfs`, `dd if=`, format commands
- Force push to main/master branch

**Warn and ask**:
- `git push --force` (any branch)
- `git reset --hard`
- `git rebase` (on shared branches)

**Allow**:
- Normal git operations (commit, push, pull)
- File reads/writes in project directory
- Package installs (npm, pip)
- Test execution

### Error Message Pattern

Be specific and helpful:

```python
# Bad
return "Error: Invalid input"

# Good
return "Error: 'add' command requires <files> tag (e.g., '.', 'file.py', 'src/*.py')"
```

## Testing

### Manual Tool Testing

```bash
cd agents/developer_agent
python3 -c "
from tools.readfile import readfile
print(readfile.run('./README.md'))
"
```

### Agent Testing

```bash
python main.py

# Test sequence
👤 You: List all Python files
# Expect: grepsearch tool usage

👤 You: Read main.py
# Expect: readfile tool usage

👤 You: Check git status
# Expect: git tool usage
```

## Common Modifications

### Add a New Tool

Example: Add a `format` tool for code formatting:

```python
# tools/format.py
import subprocess
import re

class format:
    @staticmethod
    def get_delim() -> str:
        return 'format'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "format",
            "description": "Format code using black (Python) or prettier (JS/TS)",
            "example": "<format><files>src/main.py</files></format>"
        }

    @staticmethod
    def run(content: str) -> str:
        files_match = re.search(r'<files>(.*?)</files>', content, re.DOTALL)
        if not files_match:
            return "Error: Requires <files> tag"

        files = files_match.group(1).strip()

        # Detect language and format
        if files.endswith('.py'):
            cmd = ['black', files]
        elif files.endswith(('.js', '.ts', '.tsx')):
            cmd = ['prettier', '--write', files]
        else:
            return "Error: Unsupported file type"

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
```

Restart agent → tool auto-loaded!

### Modify System Prompt

Edit `main.py` → `_initialize_system_prompt()`:

```python
system_prompt = f"""You are a helpful software development AI assistant...

IMPORTANT INSTRUCTIONS:
1. Use tools by wrapping them in XML tags
2. Always verify file paths before writing
3. Use git to track all changes
4. Test before committing
5. Write descriptive commit messages
6. CUSTOM INSTRUCTION: Always use black for Python formatting

{chr(10).join(tool_docs)}
"""
```

### Change LLM Fallback Order

Edit `llm_router.py` → `FALLBACK_MODELS`:

```python
FALLBACK_MODELS: list[ModelSpec] = [
    ModelSpec("anthropic", "claude-4-5-haiku"),  # Try Claude first
    ModelSpec("google", "gemini-flash-latest"),
    ModelSpec("ollama", "qwen3:1.7b"),
]
```

## Debugging

### Enable Verbose Output

The agent streams responses to console by default. To see full message history:

```python
# In main.py, add after process_message():
print("\n=== Message History ===")
for msg in self.chat.messages:
    print(f"{msg.role.value}: {msg.content[:200]}...")
```

### Test Tool Directly

```python
from tools.git import git
result = git.run("<command>status</command>")
print(result)
```

### Check LLM Provider

```python
from llm_router import LLMRouter
router = LLMRouter()
print(router.list_available_providers())
# Output: ['Google (Gemini)', 'Ollama (Local)']
```

## Performance Tips

### Reduce Token Usage

1. Use line ranges when reading large files:
   ```xml
   <readfile><filepath>main.py</filepath><lines>100-150</lines></readfile>
   ```

2. Limit grepsearch results:
   ```xml
   <grepsearch><pattern>TODO</pattern><path>./src</path><max_results>20</max_results></grepsearch>
   ```

3. Use faster models (Gemini Flash Lite) for simple tasks

### Speed Up Agent

1. **Reduce max_iterations**: Default 10 → Lower for simple tasks
2. **Use Gemini Flash**: Faster than Claude, similar quality
3. **Local models**: Ollama for offline/private work (no API latency)

## Limitations and Workarounds

### No Session Persistence
- **Limitation**: Agent forgets everything on restart
- **Workaround**: Use git commits to track state between sessions

### No Permission System
- **Limitation**: Agent can write any file
- **Workaround**: Run in isolated environment or use read-only mode (remove writefile/editfile)

### No Async Tools
- **Limitation**: Tools block during execution
- **Workaround**: Keep tasks short, use timeout in subprocess calls

### No Vector Search
- **Limitation**: Can't do semantic search across codebase
- **Workaround**: Use grepsearch with good regex patterns

## Best Practices

### 1. Keep Tools Focused
Each tool should do one thing well. Don't create a "do_everything" tool.

### 2. Validate Input Early
Check XML tags exist before processing:
```python
if not tag_match:
    return "Error: Missing <tag>"
```

### 3. Provide Good Examples
In `get_tool_info()`, show realistic examples:
```python
"example": (
    "<git><command>add</command><files>src/main.py tests/</files></git>\n"
    "<git><command>commit</command><message>Fix auth bug</message></git>"
)
```

### 4. Return Useful Errors
Include actionable information:
```python
return f"Error: File not found at '{full_path}'. Did you mean './src/main.py'?"
```

### 5. Use Timeouts
All subprocess calls should have timeouts:
```python
subprocess.run(cmd, timeout=30)
```

## Comparison with Main CLI-Agent

| Feature | Developer Agent | Main CLI-Agent |
|---------|----------------|----------------|
| Lines of code | ~1200 | ~15,000 |
| Tools | 7 (focused) | 20+ (comprehensive) |
| Permission system | No | Yes |
| Session persistence | No | Yes |
| Web UI | No | Yes |
| Caching | No | Yes |
| Vector DB | No | Yes |
| Use case | Learning, prototyping, simple tasks | Production, complex workflows |

**When to use Developer Agent**:
- Learning agent architecture
- Building custom specialized agent
- Simple scripting tasks
- Trusted environments

**When to use Main CLI-Agent**:
- Production workflows
- Multi-user environments
- Complex tool orchestration
- Need web UI, persistence, permissions

## Troubleshooting Common Issues

### Tool Not Loading
- Check filename matches class name: `readfile.py` → `class readfile`
- Verify 3 methods present: `get_delim()`, `get_tool_info()`, `run()`
- No syntax errors in tool file

### XML Parsing Fails
- Use `re.DOTALL` flag for multiline content
- Strip whitespace: `match.group(1).strip()`
- Check tag names match exactly

### Git Operations Fail
- Verify in git repository: `git rev-parse --git-dir`
- Check git installed: `git --version`
- Ensure proper permissions on `.git/` directory

### GitHub Operations Fail
- Install gh CLI: https://cli.github.com/
- Authenticate: `gh auth login`
- Test: `gh auth status`

## Advanced Patterns

### Stateful Tool (Singleton)

```python
class stateful_tool:
    _state = {}  # Class variable for state

    @staticmethod
    def run(content: str) -> str:
        # Access shared state
        stateful_tool._state['count'] = stateful_tool._state.get('count', 0) + 1
        return f"Execution count: {stateful_tool._state['count']}"
```

### Multi-Format Output

```python
@staticmethod
def run(content: str) -> str:
    # Detect desired format
    format_match = re.search(r'<format>(.*?)</format>', content, re.DOTALL)
    output_format = format_match.group(1).strip() if format_match else 'text'

    result = get_data()

    if output_format == 'json':
        return json.dumps(result, indent=2)
    elif output_format == 'table':
        return format_as_table(result)
    else:
        return str(result)
```

## Contributing

To contribute improvements:

1. Keep it minimal (don't add features that belong in main CLI-Agent)
2. Follow the 3-method tool contract
3. Add examples to documentation
4. Test with multiple LLM providers
5. Maintain safety checks

## Related Documentation

- `README.md` - User guide and tool reference
- `agents/minimal_agent/` - Even simpler reference (4 tools)
- Main CLI-Agent `CLAUDE.md` - Full-featured agent patterns
