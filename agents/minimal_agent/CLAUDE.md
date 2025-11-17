# Minimal Agent - Developer Guide

Guidance for working with the minimal agent implementation.

## Overview

The minimal agent is a **stripped-down, educational implementation** of an LLM-powered agent with only 4 read-only tools. It demonstrates core agent architecture patterns in ~250 lines of code, making it ideal for:
- Understanding agent fundamentals
- Building specialized agents (e.g., developer_agent, code_reviewer, etc.)
- Teaching/learning agentic systems
- Rapid prototyping

**Key Principle**: Simplicity over features. No caching, no permissions, no UI—just the essentials.

## Quick Start

```bash
cd agents/minimal_agent
pip install -r requirements.txt
cp .env.example .env  # Add GOOGLE_API_KEY or ANTHROPIC_API_KEY
python main.py
```

## Architecture

### Three-Component Design

```
┌─────────────┐
│   main.py   │  ← Agent loop, tool orchestration
└─────┬───────┘
      │
      ├─→ chat.py        ← Message management (Role, Message, Chat)
      ├─→ llm_router.py  ← LLM provider abstraction
      └─→ tools/*.py     ← Tool implementations (auto-discovered)
```

### Core Components

**main.py** (MinimalAgent class):
- Agent loop with max 10 iterations
- Tool auto-discovery from `tools/` directory
- System prompt generation from tool metadata
- Streaming response handling
- XML tag parsing: `<toolname>content</toolname>`

**chat.py** (Message management):
- `Role` enum: SYSTEM, USER, ASSISTANT
- `Message` dataclass with role + content
- `Chat` class: message list with add/convert methods
- No persistence—state cleared on restart

**llm_router.py** (LLM abstraction):
- `ModelSpec` dataclass: (provider, model_key)
- `FALLBACK_MODELS` list: ordered provider preference
- Auto-failover: Google → Ollama → Anthropic
- Streaming and non-streaming support
- Provider detection via environment variables

### Tool System

Tools follow a 3-method contract:

```python
class toolname:
    @staticmethod
    def get_delim() -> str:
        return 'toolname'  # XML tag name

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "toolname",
            "description": "What it does and when to use it",
            "example": "<toolname>usage example</toolname>"
        }

    @staticmethod
    def run(content: str) -> str:  # or None for cognitive tools
        # Parse content, execute, return result
        pass
```

**Tool Discovery**:
- Scans `tools/*.py` at startup
- Loads modules dynamically
- Expects class name == module name
- Auto-generates system prompt from `get_tool_info()`

**Current Tools** (all read-only):
1. **think**: Cognitive reasoning (returns None)
2. **todos**: Persistent task list (`~/.minimal-agent/todos.md`)
3. **readfile**: File reading with line numbers/ranges
4. **grepsearch**: Regex search in files and filenames

### Agent Loop Flow

```
1. User input → chat.add_user_message()
2. LLM call → llm_router.generate_completion(stream=True)
3. Stream response to console
4. Parse <toolname>...</toolname> tags
5. Execute tools → collect results
6. If results exist → chat.add_user_message(results) → goto 2
7. If no tool calls → return response to user
8. Max 10 iterations (prevents infinite loops)
```

## Key Patterns

### Clean Breaks Over Compatibility
- No legacy support—change things completely
- Update all references immediately
- Delete unused code rather than commenting out

### Minimal Dependencies
```
anthropic      # Claude API
google-generativeai  # Gemini API
ollama         # Local models
chardet        # Encoding detection
```

### Error Handling Strategy
- Fail fast with clear error messages
- Provider fallback automatic (no retries)
- Tool errors returned as strings to LLM
- Encoding errors handled with 'replace' mode

### Stateless by Design
- No session persistence
- No chat history across restarts
- No configuration beyond .env
- Fresh state every run

## Extending the Agent

### Adding a New Tool

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
            "description": "Clear, specific description for LLM",
            "example": "<newtool>concrete example</newtool>"
        }

    @staticmethod
    def run(content: str) -> str:
        # Parse XML if needed (use regex for tags)
        # Execute operation
        # Return result string or None
        return "result"
```

2. Restart agent—tool auto-discovered

**Tool Design Guidelines**:
- Keep `description` concise but complete
- Show realistic `example` usage
- Return `None` for cognitive tools (no output)
- Use XML tags for complex inputs (see readfile.py)
- Respect security (read-only preferred for minimal agent)

### Creating a Specialized Agent (e.g., developer_agent)

**Strategy**: Copy minimal_agent, add focused tools

```bash
cp -r agents/minimal_agent agents/developer_agent
cd agents/developer_agent
```

**Example tool additions**:
- `ast_analyze`: Parse code structure (AST)
- `run_tests`: Execute test suites
- `git_status`: Check repository state
- `lint_check`: Run linters (read-only)

**Keep it focused**: Don't replicate full CLI-Agent—add 5-10 specialized tools max.

### Adding a New Provider

Edit `llm_router.py`:

1. Add to `FALLBACK_MODELS`:
```python
ModelSpec("newprovider", "model-name"),
```

2. Add provider check in `_is_provider_available()`:
```python
elif provider == "newprovider":
    return self.newprovider_configured
```

3. Add call method `_call_newprovider()`:
```python
def _call_newprovider(self, messages, model, max_tokens, temperature, stream):
    # Implement provider API call
    # Return str (non-streaming) or Iterator[str] (streaming)
```

## Testing

```bash
# Run test suite
python -m pytest test_tools.py -v

# Test specific tool
python -c "from tools.readfile import readfile; print(readfile.run('./main.py'))"

# Test LLM router
python -c "from llm_router import LLMRouter; r = LLMRouter(); print(r.list_available_providers())"
```

## Common Patterns from main.py

### Tool Parsing (Regex-based)
```python
pattern = f'<{tool_name}>(.*?)</{tool_name}>'
matches = re.finditer(pattern, text, re.DOTALL)
```

### Streaming Response Accumulation
```python
full_response = ""
for chunk in response_stream:
    print(chunk, end="", flush=True)
    full_response += chunk
```

### Dynamic Module Loading
```python
spec = importlib.util.spec_from_file_location(module_name, tool_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tool_class = getattr(module, module_name)
```

## Limitations (By Design)

What this agent **intentionally lacks**:
- Permission system (all tools trusted)
- Session persistence (no chat history saved)
- Streaming UI (console-only, basic prints)
- Tool result caching
- Rate limiting
- Vector-based tool discovery
- Async tool execution
- Web UI
- Audio support
- Multi-user support

**Why**: These are educational/architectural concerns handled by the main CLI-Agent. This is a reference implementation for core patterns only.

## Differences from Main CLI-Agent

| Feature | Minimal Agent | Main CLI-Agent |
|---------|---------------|----------------|
| Code size | ~250 lines | ~15,000+ lines |
| Tools | 4 (read-only) | 15+ (full CRUD) |
| Providers | 3 | 5+ with failover |
| Persistence | None | Full session/audit |
| Permissions | None | Pattern-based system |
| UI | CLI only | CLI + Web + Audio |
| Tool discovery | Simple import | Vector DB + embeddings |
| Logging | Print statements | Structured logging framework |
| Dependencies | 4 packages | 30+ packages |

## File Reference

```
minimal_agent/
├── main.py              # 250 lines - Agent loop, tool orchestration
├── chat.py              # 60 lines - Message management (Role, Message, Chat)
├── llm_router.py        # 320 lines - LLM provider abstraction
├── tools/
│   ├── think.py         # 28 lines - Cognitive reasoning (no output)
│   ├── todos.py         # 40 lines - Task list persistence
│   ├── readfile.py      # 192 lines - File reading with line ranges
│   └── grepsearch.py    # 136 lines - Regex search in files
├── test_tools.py        # Test suite
├── requirements.txt     # 4 dependencies
├── .env.example         # API key template
├── README.md            # User documentation
└── CLAUDE.md            # This file (developer guidance)
```

## When to Use This vs Main CLI-Agent

**Use minimal_agent when**:
- Building a specialized agent (code reviewer, doc writer, etc.)
- Learning agent architecture
- Prototyping new agent patterns
- Need predictable, simple behavior
- Educational/teaching context

**Use main CLI-Agent when**:
- Need production features (logging, audit, permissions)
- Multi-user environment
- Require web UI or audio support
- Need tool execution safety
- Building on existing tools ecosystem

## Migration Path

To migrate features from minimal_agent → specialized agent:

1. **Copy the base**: `cp -r minimal_agent developer_agent`
2. **Add tools**: Create `tools/newtool.py` with 3-method contract
3. **Update README.md**: Document new tools and use cases
4. **Test thoroughly**: Add tests to `test_tools.py`
5. **Keep it minimal**: Don't replicate main CLI-Agent features

To migrate to main CLI-Agent:

1. Tool implementations → `tools/` directory (add async support)
2. Add permission patterns to system prompt
3. Use `shared/utils/tool_logging.py` for logging
4. Register in tool vector DB
5. Add to command system if needed

## Best Practices

1. **Tool descriptions**: Write for the LLM, not humans
   - Bad: "This tool reads files"
   - Good: "Read file contents with line numbers. Use for examining code, configs, or logs. Supports line ranges (50-75) for large files."

2. **Error messages**: Include context and solutions
   - Bad: "File not found"
   - Good: "Error: File not found at '/path/to/file'. Check the path and try again."

3. **Examples in tool_info**: Show realistic usage
   - Bad: `<readfile>file.txt</readfile>`
   - Good: `<readfile>./src/main.py</readfile>` and `<readfile><filepath>./app.js</filepath><lines>50-75</lines></readfile>`

4. **System prompt**: Let tools self-document
   - Don't hardcode tool instructions—generate from `get_tool_info()`
   - Keep system prompt short—tools provide their own docs

5. **Keep it simple**: Resist feature creep
   - If you need it, it probably belongs in main CLI-Agent
   - Minimal agent should stay <500 lines total

## Related Documentation

- `README.md` - User-facing documentation
- `../CLAUDE.md` - Agents directory overview
- `../../CLAUDE.md` - Main CLI-Agent development guide
- `../../docs/` - Full project documentation
