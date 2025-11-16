# Minimal Agent

A minimal version of the CLI-Agent project that runs an agent via the LLM router with only 4 essential tools.

## Features

- **LLM Router**: Supports Google (Gemini), Ollama (local models), and Anthropic (Claude)
- **4 Core Tools**:
  - `think`: Cognitive tool for reasoning and planning
  - `todos`: Persistent task list management
  - `readfile`: Read file contents with line numbers
  - `grepsearch`: Search for patterns in files

## Project Structure

```
minimal_agent/
├── main.py              # Agent loop and orchestration
├── llm_router.py        # LLM provider abstraction
├── chat.py              # Message management
├── tools/               # Tool implementations
│   ├── think.py
│   ├── todos.py
│   ├── readfile.py
│   └── grepsearch.py
├── requirements.txt     # Dependencies
├── .env.example         # Environment template
└── README.md           # This file
```

## Setup

### 1. Install Dependencies

```bash
cd minimal_agent
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add at least one API key:

```bash
# For Gemini (Google) - recommended
GOOGLE_API_KEY=your-google-api-key-here

# For Ollama (local models) - no API key needed
# Just install Ollama from https://ollama.ai/
# Then run: ollama pull llama3.2

# For Claude (Anthropic)
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Set default model
DEFAULT_MODEL=gemini-1.5-flash
```

### 3. Run the Agent

```bash
python main.py
```

## Usage

### Interactive Mode

Once started, you can chat with the agent:

```
👤 You: Can you help me find all TODO comments in my Python files?

🤖 Agent: I'll search for TODO comments in Python files using grepsearch.
```

The agent will automatically:
1. Parse your request
2. Decide which tools to use
3. Execute the tools
4. Return results

### Available Tools

#### 1. Think
Internal reasoning tool (produces no output):
```
<think>I should first check what files exist, then read the relevant ones.</think>
```

#### 2. Todos
Manage a persistent task list:
```
<todos>
- [x] Search for config files
- [ ] Read the main config
- [ ] Analyze settings
</todos>
```

The todo list is saved to `~/.minimal-agent/todos.md`

#### 3. ReadFile
Read files with line numbers:
```
<readfile>./src/main.py</readfile>
<readfile><filepath>./app.js</filepath><lines>50-75</lines></readfile>
```

Features:
- Line-numbered output
- Line range support (e.g., `50-75`, `100-`, `-50`)
- Encoding support for non-UTF8 files

#### 4. GrepSearch
Search for patterns in files:
```
<grepsearch><pattern>TODO</pattern><path>./src</path></grepsearch>
<grepsearch><pattern>function \w+</pattern><path>.</path></grepsearch>
```

Features:
- Regex pattern matching
- Search in filenames and content
- Respects common ignore patterns (`.git`, `node_modules`, etc.)
- Mode options: `all`, `files`, `directories`

## Examples

### Example 1: Finding and Reading Files

```
You: Find all Python files in the src directory

Agent: <grepsearch><pattern>\.py$</pattern><path>./src</path></grepsearch>

[Results shown]

You: Read the main.py file

Agent: <readfile>./src/main.py</readfile>
```

### Example 2: Multi-Step Task with Todos

```
You: Help me refactor the authentication code

Agent: <think>I should break this down into steps and track progress.</think>

<todos>
- [ ] Search for authentication-related files
- [ ] Read current implementation
- [ ] Identify refactoring opportunities
- [ ] Suggest improvements
</todos>

<grepsearch><pattern>auth</pattern><path>./src</path></grepsearch>
```

### Example 3: Code Analysis

```
You: Find all functions that start with "get_" in the utils directory

Agent: <grepsearch><pattern>^def get_\w+</pattern><path>./utils</path></grepsearch>
```

## Architecture

### Agent Loop

The agent operates in a loop:

1. **User Input** → Added to chat history
2. **LLM Call** → Get response with potential tool calls
3. **Tool Parsing** → Extract `<toolname>content</toolname>` tags
4. **Tool Execution** → Run tools and collect results
5. **Result Feedback** → Add results to chat history
6. **Repeat** → Continue until no more tool calls (max 10 iterations)

### LLM Router

The router supports three providers (in order of preference):
- **Google**: Gemini models (fast, powerful, recommended)
- **Ollama**: Local models (free, private, no API key needed)
- **Anthropic**: Claude models (fast, efficient)

Models are auto-detected based on name:
- `gemini-*` → Google
- `llama*`, `mistral*`, `codellama*`, etc. → Ollama
- `claude-*` → Anthropic

### Tool System

Tools follow a simple contract:
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
            "example": "<toolname>example usage</toolname>"
        }

    @staticmethod
    def run(content: str) -> str:
        # Execute and return result
        pass
```

## Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Your Google API key (for Gemini models)
- `ANTHROPIC_API_KEY`: Your Anthropic API key (for Claude models)
- `DEFAULT_MODEL`: Default model to use (default: `gemini-1.5-flash`)

### Supported Models

**Google (Gemini):**
- `gemini-1.5-flash` (recommended - fast and cost-effective)
- `gemini-1.5-pro` (more capable)
- `gemini-2.0-flash-exp` (experimental)

**Ollama (Local):**
- `llama3.2` (Meta's latest, efficient)
- `llama3.1` (larger, more capable)
- `mistral` (fast and capable)
- `codellama` (code-specialized)
- `phi3` (Microsoft's small model)
- Any model available in Ollama library

**Anthropic (Claude):**
- `claude-haiku-4-5` (fast and efficient)
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`

## Limitations

This is a minimal implementation focused on core functionality:

- No caching or optimization
- No streaming output (non-streaming mode only)
- No permission system
- No web UI
- No audio support
- No session persistence across restarts
- No advanced features (vector DB, rate limiting, etc.)

For the full-featured version, see the parent CLI-Agent project.

## Troubleshooting

### No providers configured

```
⚠️  No providers configured!
Configure at least one (in order of preference):
  1. GOOGLE_API_KEY for Gemini models (recommended)
  2. Install Ollama for local models
  3. ANTHROPIC_API_KEY for Claude models
```

**Solution** (in order of preference):
1. Create a `.env` file with `GOOGLE_API_KEY` (see Setup section), OR
2. Install Ollama from https://ollama.ai/ and run `ollama pull llama3.2`, OR
3. Create a `.env` file with `ANTHROPIC_API_KEY`

### Tool not found

```
Error: Unknown tool 'toolname'
```

**Solution**: The agent can only use the 4 built-in tools. Check the tool name spelling.

### Maximum iterations reached

```
Error: Maximum iterations reached. Agent may be stuck in a loop.
```

**Solution**: The agent tried to use tools 10 times in a row. Try rephrasing your request or breaking it into smaller steps.

## Extending the Agent

### Adding a New Tool

1. Create a new file in `tools/` (e.g., `tools/mytool.py`)
2. Implement the tool contract:

```python
class mytool:
    @staticmethod
    def get_delim() -> str:
        return 'mytool'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "mytool",
            "description": "Description of what it does",
            "example": "<mytool>example</mytool>"
        }

    @staticmethod
    def run(content: str) -> str:
        # Your implementation
        return "result"
```

3. Restart the agent - tools are auto-loaded

### Using Ollama (Local Models)

Ollama allows you to run models locally without API keys:

1. **Install Ollama**: Download from https://ollama.ai/
2. **Pull a model**: `ollama pull llama3.2`
3. **Run the agent**: The agent will automatically detect Ollama
4. **Use in chat**: Set `DEFAULT_MODEL=llama3.2` or specify in requests

Benefits:
- **Free**: No API costs
- **Private**: Data stays on your machine
- **Fast**: No network latency
- **Offline**: Works without internet

## License

This minimal agent is part of the CLI-Agent project.

## Related

- **Full CLI-Agent**: https://github.com/Probst1nator/CLI-Agent
- **Google AI (Gemini)**: https://ai.google.dev/
- **Ollama**: https://ollama.ai/
- **Anthropic API**: https://docs.anthropic.com/
