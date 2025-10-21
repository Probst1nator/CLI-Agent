Of course. Here is a `CLAUDE.md` file authored in a helpful, technical style, explaining the tool system, how to implement new tools with examples of increasing complexity, and crucially, how to test them.

***

# CLAUDE.md: Agent Tool System Guide

This `CLAUDE.md` serves as the primary developer guide for the agent's tool system. It explains the architecture, provides a practical walkthrough for adding new capabilities, and outlines best practices for building robust and testable tools.

## 1. How the Tool System Works

The agent loop **auto-discovers** every Python file in `tools/`, imports the class that matches the file name, and **registers** the XML-tag returned by `get_delim()`; whenever the LLM later emits that tag, the loop calls `run(inner_text)` and feeds the string result back to the LLM—so adding a new capability is literally “drop a file in the folder”.

### The Tool Contract

This stable interface is the foundation of the system. Each tool module must export **one class** that implements these **three static methods**:

| Method          | Signature                 | Explanation                                                                                |
| --------------- | ------------------------- | ------------------------------------------------------------------------------------------ |
| `get_delim`     | `() -> str`               | The XML tag the agent uses, e.g. `"python"` → `<python>…</python>`                                   |
| `get_tool_info` | `() -> dict`              | `{"name": "…", "description": "…", "example": "…"}` injected into the agent's system prompt. |
| `run`           | `(content: str) -> str` | Receives *everything* between the opening/closing tags, does the work, and returns the result.  |

The power of this design is its simplicity. All internal complexity—state management, dependencies, subprocesses, network calls—is encapsulated within the tool's class, invisible to the agent loop.

## 2. Extending the System: Implementation Examples

Adding a new tool involves creating a file in `tools/` and implementing the contract.

### a. Minimal Example: `hello.py`

This demonstrates the basic structure.

`tools/hello.py`
```python
import re
class hello:
    @staticmethod
    def get_delim(): return "hello"
    @staticmethod
    def get_tool_info():
        return {"name": "hello", "description": "Greets a user", "example": "<hello>Alice</hello>"}
    @staticmethod
    def run(content: str) -> str:
        return f"Hello, {content.strip()}!"
```
Once this file is added, the agent can immediately use `<hello>World</hello>` and will receive `Hello, World!` as the result.

### b. Medium Complexity Example: Stateful Python REPL

This pattern shows how to manage persistent state (a running subprocess) behind the simple 3-method interface.

`tools/python.py` (abridged)
```python
import pexpect, base64, logging, os
class python:
    _session = None                       # Singleton keeps REPL alive
    @staticmethod
    def get_delim(): return "python"
    @staticmethod
    def get_tool_info():
        return {"name": "python", "description": "Persistent Python interpreter", "example": "<python>print(2+2)</python>"}
    @staticmethod
    def _get_session():
        if python._session is None:
            python._session = PythonSession()   # Spawns 'python3 -i'
        return python._session
    @staticmethod
    def run(content: str) -> str:
        session = python._get_session()
        # Escaping is handled internally to ensure robustness
        encoded = base64.b64encode(content.encode()).decode()
        session.execute(f"import base64; exec(base64.b64decode('{encoded}'))")
        return "Executed (output streamed to console)"
```
The class fully encapsulates the `pexpect` lifecycle, base64 escaping, timeouts, and crash recovery. The agent and the main loop only interact with the standard contract.

### c. Arbitrary Complexity Example: Kubernetes Client

This pattern demonstrates how to integrate heavy-weight, asynchronous, or external clients.

`tools/k8s.py`
```python
import asyncio, re
# In a real implementation, you would import your K8s client library
# from kubernetes import client, config

class k8s:
    _api_client = None # Singleton for the K8s API client
    @staticmethod
    def get_delim(): return "k8s"
    @staticmethod
    def get_tool_info():
        return {"name": "k8s", "description": "Read/write Kubernetes resources", "example": "<k8s><cmd>get pods</cmd><ns>prod</ns></k8s>"}
    @staticmethod
    def _get_api_client():
        if k8s._api_client is None:
            # config.load_kube_config()
            # k8s._api_client = client.CoreV1Api()
            pass # Placeholder for real K8s client initialization
        return k8s._api_client
    @staticmethod
    def run(content: str) -> str:
        # The run method remains synchronous and simple
        cmd = re.search("<cmd>(.*?)</cmd>", content, re.S).group(1).strip()
        ns  = (re.search("<ns>(.*?)</ns>", content, re.S) or "default").group(1).strip()
        
        api = k8s._get_api_client()
        if cmd == "get pods":
            # pods = api.list_namespaced_pod(namespace=ns)
            # return "\n".join(p.metadata.name for p in pods.items)
            return f"Simulated: Got pods from namespace '{ns}'"
        return f"Error: Unknown k8s command '{cmd}'"
```
The agent loop continues to call the same three synchronous methods. All complexity of managing credentials, network connections, and API client lifecycles is hidden within the tool.

### d. Async Tools Pattern

Tools can be fully asynchronous to prevent blocking operations. The calling system must handle async tools appropriately.

`tools/searchweb.py` (simplified example)
```python
import asyncio
from core.llm_router import LlmRouter

class searchweb:
    @staticmethod
    def get_delim(): return "searchweb"
    @staticmethod
    def get_tool_info():
        return {"name": "searchweb", "description": "Async web search with AI summaries", "example": "<searchweb><query>python</query></searchweb>"}

    # --- ASYNC PATTERN ---
    @staticmethod
    async def run(content: str) -> str:
        query = extract_query(content)

        # Non-blocking API call
        results = await asyncio.to_thread(api_search, query)

        # Non-blocking LLM generation
        summary = await LlmRouter.generate_completion(build_chat(results))

        return summary
```

**Important**: Async tools require the calling system to use `await tool.run(content)` instead of `tool.run(content)`.

## 3. Testing New Tools

Tools should be unit-tested to ensure reliability. The stateless nature of the `run` method makes this straightforward.

**Structure:** Create a parallel test file in a `tests/` directory. For `tools/hello.py`, create `tests/tools/test_hello.py`.

**Example Test using `pytest`:**

`tests/tools/test_hello.py`
```python
import pytest
from tools.hello import hello

def test_hello_tool_basic():
    """Tests the basic functionality of the hello tool."""
    content = "World"
    expected_output = "Hello, World!"
    assert hello.run(content) == expected_output

def test_hello_tool_with_whitespace():
    """Tests that the tool correctly handles leading/trailing whitespace."""
    content = "  Universe  "
    expected_output = "Hello, Universe!"
    assert hello.run(content) == expected_output
```
**Testing Complex Tools:** For stateful or I/O-bound tools, use mocking libraries (`unittest.mock`) to isolate the logic. For `k8s.py`, you would mock the Kubernetes client to test the `run` method's parsing and command dispatching without making real API calls.

## 4. Advanced Patterns & Best Practices

As tools grow, they can be treated as miniature services.
*   **Configuration:** Use dataclasses for typed config and libraries like Pydantic for validation. Load secrets and settings from a central config or environment variables, never hardcode them.
*   **Dependencies:** Ship auxiliary data files in `tools/MY_TOOL_DATA/` and load them with `importlib.resources`.
*   **Caching:** For expensive operations, cache results to disk in a standard location like `~/.cache/agent_toolname/`.
*   **Resilience:** Use libraries like `tenacity` to add automatic retries for network-dependent tools.
*   **Structured Output:** Have `run()` return a JSON string instead of plain text. This allows for richer data to be passed back and can be used to build UIs or debuggers on top of the agent's actions.

## 5. Essential Implementation Patterns

### a. XML Parsing with Error Handling

Robust XML parsing ensures your tool handles malformed input gracefully:

```python
import re

@staticmethod
def run(content: str) -> str:
    try:
        # Required parameter with validation
        title_match = re.search(r'<title>(.*?)</title>', content, re.DOTALL)
        if not title_match:
            return "Error: Missing required <title> tag"
        title = title_match.group(1).strip()

        if not title:
            return "Error: <title> cannot be empty"

        # Optional parameter with default
        timeout_match = re.search(r'<timeout>(\d+)</timeout>', content)
        timeout = int(timeout_match.group(1)) if timeout_match else 30

        # Boolean flag parsing
        use_cache = True
        cache_match = re.search(r'<cache>(.*?)</cache>', content, re.DOTALL)
        if cache_match and cache_match.group(1).strip().lower() in ['false', 'no', '0']:
            use_cache = False

        # Process the request...

    except Exception as e:
        logger.error(f"Error in tool execution: {e}", exc_info=True)
        return f"Error: {str(e)}"
```

### b. Environment Variables and Configuration

Use environment variables for API keys and configuration:

```python
import os
from dotenv import load_dotenv

load_dotenv()

class my_api_tool:
    @staticmethod
    def _get_api_key() -> Optional[str]:
        api_key = os.getenv('MY_API_KEY')
        if not api_key:
            logger.error("MY_API_KEY environment variable not set")
            return None
        return api_key

    @staticmethod
    def run(content: str) -> str:
        api_key = my_api_tool._get_api_key()
        if not api_key:
            return "Error: MY_API_KEY not configured. Set it in .env file."

        # Use the API key...
```

**Don't forget to add configuration to `.env.example`:**

```bash
# My API Tool Configuration
MY_API_KEY=your_api_key_here
MY_API_ENDPOINT=https://api.example.com
```

### c. Tool-Specific Logging

Every tool should use dedicated logging for debugging:

```python
from shared.utils.tool_logging import get_logger_for_tool_module

logger = get_logger_for_tool_module(__name__)

class my_tool:
    @staticmethod
    async def run(content: str) -> str:
        logger.debug(f"Tool called with content: {content[:100]}...")
        logger.info("Processing request...")

        try:
            result = await process_request(content)
            logger.info(f"Successfully processed request, result length: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Request failed: {e}", exc_info=True)
            return f"Error: {str(e)}"
```

Logs automatically go to `~/.cli-agent/logs/tools_<toolname>.log` with:
- 14-day retention
- 20MB max size
- Automatic rotation and cleanup

### d. Singleton Pattern for Stateful Resources

Use class-level variables for persistent connections or sessions:

```python
class database_tool:
    _connection = None  # Singleton connection

    @staticmethod
    def _get_connection():
        """Gets or creates database connection."""
        if database_tool._connection is None:
            logger.info("Creating new database connection")
            database_tool._connection = create_connection()
        return database_tool._connection

    @staticmethod
    def run(content: str) -> str:
        conn = database_tool._get_connection()
        # Use connection...
```

### e. Integration with LLM Router

Tools can call LLMs for processing (e.g., summarization):

```python
from core.llm_router import LlmRouter
from core.chat import Chat, Role

class summarize_tool:
    @staticmethod
    async def run(content: str) -> str:
        # Extract text to summarize
        text = extract_text(content)

        # Create chat for LLM
        chat = Chat()
        chat.add_message(Role.USER, f"Summarize this text: {text}")

        # Use LLM router (automatically selects available model)
        summary = await LlmRouter.generate_completion(
            chat,
            force_preferred_model=False  # Allow any free model
        )

        logger.info(f"Generated summary using {LlmRouter.get_last_used_model()}")

        return f"Summary: {summary}"
```

### f. Permission System Integration

For tools that access files or sensitive operations:

```python
from core.permissions.enhanced_permissions import EnhancedPermissions

class file_tool:
    @staticmethod
    def run(content: str) -> str:
        filepath = extract_filepath(content)

        # Check permissions before accessing file
        perms = EnhancedPermissions()
        if not perms.check_path_permission(filepath, 'read'):
            return f"Error: Permission denied to read {filepath}"

        # Proceed with file access...
```

### g. Async Operations with Threading

For blocking operations in async tools, use `asyncio.to_thread`:

```python
import asyncio

class blocking_api_tool:
    @staticmethod
    async def run(content: str) -> str:
        # Run blocking API call in thread pool to avoid blocking
        result = await asyncio.to_thread(
            blocking_api_call,
            param1,
            param2
        )
        return result
```

## 7. Dependencies Management

When adding external dependencies:

1.  **Add to `requirements.txt`** with version constraints.
2.  **Update `.env.example`** with configuration.
3.  **Document in tool README** or docstrings how to obtain credentials/API keys.

## 8. Checklist for Adding a New Tool

1.  [ ] Create `tools/your_tool.py` with the 3-method contract
2.  [ ] Add tool-specific logging with `get_logger_for_tool_module(__name__)`
3.  [ ] Implement robust XML parsing with error handling
4.  [ ] Add dependencies to `requirements.txt`
5.  [ ] Add environment variables to `.env.example`
6.  [ ] Use singleton pattern for stateful resources (connections, sessions)
7.  [ ] Create comprehensive tests in `tests/tools/test_your_tool.py` with mocking
8.  [ ] Assign test levels (2 for core, 3 for shared utilities)
9.  [ ] Run tests: `python -m pytest tests/tools/test_your_tool.py -v`
10. [ ] Optional: Create `tools/your_tool_README.md` for setup documentation
11. [ ] No manual registration needed - tool is auto-discovered!

## 9. Common Pitfalls and Solutions

| Pitfall | Solution |
|---------|----------|
| Blocking I/O in async tools | Use `await asyncio.to_thread(blocking_call, args)` |
| Missing error handling | Wrap `run()` in try/except, return error strings |
| Hardcoded credentials | Use environment variables via `os.getenv()` |
| No logging | Always add tool-specific logger |
| Regex match failures | Check if match exists before calling `.group()` |
| State not persisting | Use class-level variables (`_session`, `_connection`) |
| Tests fail in CI | Mock external dependencies (APIs, file system) |
| Token refresh issues | Check for expired tokens and implement refresh logic |

## 10. Quick Reference

**Minimal tool skeleton:**
```python
from shared.utils.tool_logging import get_logger_for_tool_module
logger = get_logger_for_tool_module(__name__)

class my_tool:
    @staticmethod
    def get_delim():
        return "my_tool"

    @staticmethod
    def get_tool_info():
        return {
            "name": "my_tool",
            "description": "What it does",
            "example": "<my_tool>example</my_tool>"
        }

    @staticmethod
    async def run(content: str) -> str:
        try:
            logger.info("Tool invoked")
            # Your logic here
            return "Success"
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return f"Error: {str(e)}"
```

**See existing tools for more examples:**
- Simple sync tool: `writefile.py`
- Async with API calls: `searchweb.py`
- Persistent session: `bash.py`