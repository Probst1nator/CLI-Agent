# Agents Directory - Developer Guide

Guidance for working with the agents directory and creating specialized agents.

## Overview

The `agents/` directory contains **standalone, purpose-built agents** that are separate from the main CLI-Agent. Each agent is a self-contained implementation focused on a specific domain or use case.

**Key Distinction**:
- **Main CLI-Agent** (`/`): Full-featured, production-ready agent with 15+ tools, permissions, logging, web UI, audio support
- **Agents** (`/agents/`): Minimal, specialized implementations for specific tasks or educational purposes

## Philosophy

### Why Separate Agents?

1. **Simplicity**: Focused tools for specific domains (code review, documentation, testing)
2. **Education**: Learn agent patterns without production complexity
3. **Experimentation**: Rapid prototyping of new agent architectures
4. **Portability**: Easy to extract and use independently
5. **Specialization**: Domain-specific tool sets without bloat

### Design Principles

- **Self-contained**: Each agent is a complete, runnable system
- **Minimal dependencies**: Only what's needed for the specific purpose
- **No shared state**: Agents don't interact with each other or main CLI-Agent
- **Educational first**: Code clarity > feature completeness
- **Read-only preferred**: Unless the agent's purpose requires writes

## Current Agents

### minimal_agent/

**Purpose**: Reference implementation of core agent patterns

**Features**:
- 4 read-only tools: think, todos, readfile, grepsearch
- 3 LLM providers: Google (Gemini), Ollama, Anthropic
- ~250 lines of core code
- XML-based tool invocation
- Streaming responses

**Use cases**:
- Learning agent architecture
- Base for specialized agents
- Teaching agentic systems
- Rapid prototyping

**Documentation**: See `minimal_agent/CLAUDE.md` for detailed developer guide

## Creating a New Agent

### Strategy: Start from minimal_agent

```bash
# Copy the minimal agent as a template
cp -r agents/minimal_agent agents/your_agent

# Navigate and customize
cd agents/your_agent

# Update dependencies
edit requirements.txt  # Add domain-specific packages

# Add your tools
create tools/your_tool.py  # Follow 3-method contract

# Update documentation
edit README.md        # User-facing docs
edit CLAUDE.md        # Developer guidance
```

### Example: developer_agent

**Concept**: Agent specialized for code development tasks

**Tools to add**:
- `ast_analyze`: Parse code structure (Python AST)
- `run_tests`: Execute pytest/unittest
- `git_diff`: Show uncommitted changes
- `lint_check`: Run ruff/black/mypy (read-only)
- `find_definition`: Locate function/class definitions
- `trace_imports`: Map import dependencies

**Keep it focused**: 8-12 tools max, all development-related

**Sample tool** (`tools/ast_analyze.py`):
```python
import ast

class ast_analyze:
    @staticmethod
    def get_delim() -> str:
        return 'ast_analyze'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "ast_analyze",
            "description": "Parse Python file and extract structure: classes, functions, imports. Use to understand code organization.",
            "example": "<ast_analyze>./src/main.py</ast_analyze>"
        }

    @staticmethod
    def run(content: str) -> str:
        filepath = content.strip()
        with open(filepath) as f:
            tree = ast.parse(f.read())

        # Extract classes and functions
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        return f"Classes: {', '.join(classes)}\nFunctions: {', '.join(functions)}"
```

### Example: code_reviewer

**Concept**: Agent focused on code review tasks

**Tools**:
- `read_pr`: Fetch PR contents via GitHub API
- `check_patterns`: Look for anti-patterns
- `suggest_tests`: Identify untested code paths
- `check_docs`: Verify docstring coverage
- `security_scan`: Basic security checks

### Example: doc_writer

**Concept**: Agent for documentation generation

**Tools**:
- `scan_functions`: Extract function signatures
- `check_docstrings`: Find missing documentation
- `generate_examples`: Create usage examples
- `check_links`: Validate markdown links
- `suggest_diagrams`: Identify diagram opportunities

## Agent Structure

### Required Files

Every agent should include:

```
your_agent/
├── main.py              # Agent loop and orchestration
├── chat.py              # Message management (can copy from minimal_agent)
├── llm_router.py        # LLM provider abstraction (can copy from minimal_agent)
├── tools/               # Tool implementations
│   ├── tool1.py
│   ├── tool2.py
│   └── ...
├── requirements.txt     # Dependencies
├── .env.example         # API key template
├── README.md            # User documentation
├── CLAUDE.md            # Developer guidance
└── test_tools.py        # Optional: tool tests
```

### Optional Files

```
├── config.yaml          # If agent needs configuration
├── .gitignore           # Ignore .env, __pycache__, etc.
└── docs/                # Extended documentation
```

## Tool Design Guidelines

### The 3-Method Contract

All tools must implement:

1. **get_delim()** → str: XML tag name
2. **get_tool_info()** → dict: LLM-facing documentation
3. **run(content)** → str | None: Execution logic

### Writing Effective Tool Documentation

**Description field** (for LLM):
- When to use the tool (triggers)
- What it does (capabilities)
- Constraints/limitations
- Expected input format

**Example field**:
- Show realistic usage
- Include XML tags if multi-parameter
- Use actual file paths, not placeholders

**Good example**:
```python
{
    "name": "run_tests",
    "description": "Execute pytest on specified test file or directory. Returns test results with pass/fail counts. Use when validating code changes or debugging test failures.",
    "example": "<run_tests><path>./tests/test_api.py</path><verbose>true</verbose></run_tests>"
}
```

**Bad example**:
```python
{
    "name": "run_tests",
    "description": "Runs tests",
    "example": "<run_tests>tests</run_tests>"
}
```

### Input Parsing Patterns

**Simple string input**:
```python
def run(content: str) -> str:
    filepath = content.strip()
    # Use directly
```

**XML multi-parameter** (see `minimal_agent/tools/readfile.py`):
```python
import re

def run(content: str) -> str:
    filepath_match = re.search(r'<filepath>(.*?)</filepath>', content, re.DOTALL)
    lines_match = re.search(r'<lines>(.*?)</lines>', content, re.DOTALL)

    filepath = filepath_match.group(1).strip() if filepath_match else None
    lines = lines_match.group(1).strip() if lines_match else None
```

**JSON input** (for complex structures):
```python
import json

def run(content: str) -> str:
    data = json.loads(content)
    filepath = data.get('filepath')
    options = data.get('options', {})
```

### Return Values

- **Str**: Normal output, shown to LLM
- **None**: Cognitive tools (think), no output to LLM
- **Error strings**: Start with "Error: " for clarity

### Security Considerations

**For read-only agents**:
- Validate file paths (no arbitrary reads)
- Limit file sizes (prevent OOM)
- Check file types (text only, typically)
- Respect ignore patterns (.git, node_modules, etc.)

**For write-capable agents**:
- Add confirmation prompts
- Log all writes
- Consider adding a permission system (see main CLI-Agent)
- Provide dry-run modes

## Integration with Main CLI-Agent

### When to Keep Separate vs Merge

**Keep as separate agent when**:
- Highly specialized use case (not general-purpose)
- Experimental/educational purpose
- Different LLM requirements (e.g., vision models only)
- Standalone tool makes sense

**Merge into main CLI-Agent when**:
- Tools are generally useful across domains
- Need production features (logging, permissions, audit)
- Multi-user scenarios
- Requires integration with existing tools

### Migration Path

To move tools from an agent to main CLI-Agent:

1. **Add async support** (if needed):
```python
async def run(content: str) -> str:
    # Use async operations
```

2. **Add logging**:
```python
from shared.utils.tool_logging import get_logger_for_tool_module
logger = get_logger_for_tool_module(__name__)

async def run(content: str) -> str:
    logger.info(f"Executing {self.get_delim()}")
    # ...
```

3. **Copy to tools/** directory:
```bash
cp agents/your_agent/tools/your_tool.py tools/
```

4. **Update dependencies**:
```bash
# Add to main requirements.txt
```

5. **Test integration**:
```bash
python run_tests.py --quick
```

6. **Update docs**:
```
# Add to tools/CLAUDE.md or relevant docs
```

## Testing Agents

### Manual Testing

```bash
cd agents/your_agent
python main.py

# Test specific tool
python -c "from tools.your_tool import your_tool; print(your_tool.run('test input'))"
```

### Automated Testing

Create `test_tools.py`:

```python
import pytest
from tools.your_tool import your_tool

def test_your_tool_basic():
    result = your_tool.run("valid input")
    assert "expected" in result

def test_your_tool_error():
    result = your_tool.run("invalid input")
    assert result.startswith("Error:")
```

Run tests:
```bash
pytest test_tools.py -v
```

## Common Pitfalls

### 1. Over-Engineering

**Problem**: Trying to replicate main CLI-Agent features

**Solution**: Keep agents minimal. If you need it, use main CLI-Agent instead.

### 2. Too Many Tools

**Problem**: Agent with 20+ tools becomes unfocused

**Solution**: Limit to 8-12 tools. Create multiple specialized agents if needed.

### 3. Poor Tool Descriptions

**Problem**: LLM doesn't know when to use tools

**Solution**: Write descriptions from LLM perspective. Include triggers and constraints.

### 4. Tight Coupling

**Problem**: Tools depend on each other or shared state

**Solution**: Each tool should be independently usable. Pass data explicitly.

### 5. Missing Error Handling

**Problem**: Tools crash on invalid input

**Solution**: Validate inputs, catch exceptions, return error strings.

## File Organization

```
agents/
├── CLAUDE.md                 # This file (developer guidance)
├── minimal_agent/            # Reference implementation
│   ├── CLAUDE.md
│   ├── README.md
│   ├── main.py
│   ├── chat.py
│   ├── llm_router.py
│   ├── tools/
│   ├── requirements.txt
│   └── .env.example
└── your_agent/               # Your specialized agent
    ├── CLAUDE.md
    ├── README.md
    ├── main.py
    ├── tools/
    └── ...
```

## Naming Conventions

- **Directory**: `lowercase_with_underscores` (e.g., `developer_agent`)
- **Tool files**: `toolname.py` (e.g., `ast_analyze.py`)
- **Tool classes**: Same as filename (e.g., `class ast_analyze`)
- **XML tags**: Same as tool name (e.g., `<ast_analyze>`)

## Documentation Standards

### README.md (User-facing)

Include:
- Quick start instructions
- Tool descriptions with examples
- Environment setup
- Usage examples
- Troubleshooting

### CLAUDE.md (Developer-facing)

Include:
- Architecture overview
- Key patterns and conventions
- Extending the agent
- Testing strategies
- Related documentation links

Follow the structure of `minimal_agent/CLAUDE.md` as a template.

## Example Use Cases

### Code Review Workflow

```python
# Agent: code_reviewer
# User: "Review PR #123 from my-org/my-repo"

<think>I need to fetch the PR, analyze changes, check for issues</think>
<read_pr><repo>my-org/my-repo</repo><pr>123</pr></read_pr>
# [PR contents shown]
<check_patterns><content>[PR diff]</content></check_patterns>
# [Anti-patterns found]
<suggest_tests><files>[changed files]</files></suggest_tests>
# [Test suggestions]
```

### Development Workflow

```python
# Agent: developer_agent
# User: "Find all uses of the User class"

<think>I should first find where User is defined, then search for imports and usage</think>
<find_definition><name>User</name><path>./src</path></find_definition>
# [Definition location shown]
<grepsearch><pattern>from .* import .*User</pattern><path>./src</path></grepsearch>
# [Import locations shown]
```

## Best Practices

1. **Start simple**: Copy minimal_agent, add one tool, test, repeat
2. **Document thoroughly**: Both README.md and CLAUDE.md from the start
3. **Test early**: Add tests as you add tools
4. **Keep focused**: One domain per agent
5. **Use version control**: Commit after each working tool addition

## Related Documentation

- `minimal_agent/CLAUDE.md` - Detailed developer guide for minimal_agent
- `minimal_agent/README.md` - User documentation for minimal_agent
- `../CLAUDE.md` - Main CLI-Agent development guide
- `../tools/CLAUDE.md` - Tool implementation guide for main CLI-Agent
- `../docs/` - Full project documentation

## Questions?

When working with agents, refer to:
- **Architecture questions**: `minimal_agent/CLAUDE.md`
- **Tool patterns**: `minimal_agent/tools/*.py` (reference implementations)
- **LLM routing**: `minimal_agent/llm_router.py`
- **Main CLI-Agent integration**: `../CLAUDE.md`
