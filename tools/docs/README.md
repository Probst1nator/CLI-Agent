# Tools Documentation

The tools system provides auto-discovered, modular capabilities for CLI-Agent, including file operations, web search, visual response, and task management.

## Architecture Overview

Tools follow an auto-discovery pattern with a standardized interface:
- **Auto-discovery**: All Python files in `tools/` automatically registered
- **Unified interface**: 3-method contract for all tools
- **Vector-based discovery**: ChromaDB for semantic tool search
- **Async support**: Auto-detected sync/async execution
- **Per-tool logging**: Dedicated logs for each tool

## Tool Interface

Every tool must implement:

```python
class tool_name:
    @staticmethod
    def get_delim() -> str:
        """Return the XML delimiter for this tool"""
        return 'tool_name'

    @staticmethod
    def get_tool_info() -> dict:
        """Return tool metadata for discovery"""
        return {
            "name": "tool_name",
            "description": "What this tool does",
            "example": "<tool_name>usage example</tool_name>"
        }

    @staticmethod
    def run(content: str) -> str:  # or async def
        """Execute the tool with XML content"""
        # Parse XML, execute, return result
        pass
```

See: `tools/CLAUDE.md` for complete implementation guide.

## Directory Structure

```
tools/
├── readfile.py          # Enhanced file reading with line ranges
├── writefile.py         # File creation and overwriting
├── editfile.py          # Find-and-replace and line editing
├── insertline.py        # Line insertion at specific positions
├── deletelines.py       # Line deletion with range support
├── bash.py              # Command execution
├── grepsearch.py        # Code searching
├── searchweb.py         # Web search with vector caching
├── visualresponse.py    # Web UI integration
├── taskmanager.py       # Background process management
├── todos.py             # Task tracking
├── permissions.py       # Permission management
├── takescreenshot.py    # Screen capture
├── tts.py               # Text-to-speech
├── configure_tts.py     # TTS configuration
├── viewimage.py         # Image viewing and analysis
└── file_copier/         # Intelligent file copying
    ├── main.py
    ├── smart_paster.py
    ├── ai_path_finder.py
    └── gui.py
```

## File Operation Tools

### ReadFile (`readfile.py`)

Enhanced file reading with line-indexed access:

**Features**:
- **Full file read**: Read entire file
- **Line ranges**: `readfile(file.txt, lines=10-20)`
- **Single line**: `readfile(file.txt, line=15)`
- **Precise inspection**: Surgical file viewing

**Usage**:
```xml
<readfile>
<file_path>/path/to/file.py</file_path>
<start_line>10</start_line>
<end_line>20</end_line>
</readfile>
```

### WriteFile (`writefile.py`)

File creation and complete overwrite:

**Features**:
- Create new files
- Overwrite existing files
- Automatic directory creation
- Content validation

### EditFile (`editfile.py`)

File editing with multiple modes:

**Features**:
- **Find-and-replace**: Traditional search and replace
- **Line-specific editing**: Edit specific line numbers
- **Replace all**: Global replacements
- **Surgical modifications**: Precise changes

**Usage**:
```xml
<editfile>
<file_path>/path/to/file.py</file_path>
<old_string>old code</old_string>
<new_string>new code</new_string>
</editfile>
```

### InsertLine (`insertline.py`)

Line insertion at specific positions:

**Features**:
- Insert before/after line number
- Insert at beginning/end of file
- Detailed feedback

### DeleteLines (`deletelines.py`)

Line deletion with range support:

**Features**:
- Delete specific lines
- Delete line ranges
- Precise control
- Detailed feedback

## Command Execution

### Bash (`bash.py`)

Command execution with permissions:

**Features**:
- Shell command execution
- Background processes (with `<background>true</background>`)
- Permission checking
- Output capture

**Usage**:
```xml
<bash>
<command>ls -la</command>
</bash>
```

### TaskManager (`taskmanager.py`)

Background process management:

**Features**:
- List background tasks
- Kill processes
- Monitor output
- Process lifecycle management

**Commands**:
- `<taskmanager><action>list</action></taskmanager>`
- `<taskmanager><action>kill</action><pid>1234</pid></taskmanager>`

## Search Tools

### GrepSearch (`grepsearch.py`)

Code searching:

**Features**:
- Pattern matching in files
- Directory recursion
- Regex support
- Result limiting

### SearchWeb (`searchweb.py`)

Web search with intelligent caching:

**Features**:
- **Brave Search API** integration
- **Vector database caching**: Semantic search result caching
- **Progressive knowledge**: Builds understanding over time
- **Reduced API calls**: Reuses similar search results

**Vector DB**:
- Uses `SearchResultsVectorDB` from shared/utils/search/
- Two collections: web_search_results + search_summaries
- Embeddings via embeddinggemma:latest

See: `tools/docs/SEARCHWEB_VECTOR_DB.md` or `shared/docs/SEARCHWEB_VECTOR_DB.md`

## Web Interface

### VisualResponse (`visualresponse.py`)

**Refactored** (January 2025):
- Now delegates to `api/server.py`
- Tool handler for `<visualresponse>` tags only
- Uses ServerAdapter pattern
- Auto-starts web server

**Legacy functionality moved to**:
- `api/server.py` - Web server
- `frontend/` - Frontend assets
- `api/chat_api.py`, `api/audio_api.py`, etc. - API endpoints

See: `/api/docs/README.md` and `/frontend/docs/README.md`

## Media Tools

### TakeScreenshot (`takescreenshot.py`)

Screen capture:

**Features**:
- Desktop screenshot capture
- Window selection
- Area selection
- Format options

### ViewImage (`viewimage.py`)

Image viewing and analysis:

**Features**:
- Display images
- Extract metadata
- Image analysis capabilities
- Integration with vision models

## Audio Tools

### TTS (`tts.py`)

Text-to-speech:

**Features**:
- Multiple TTS engines (Piper, Kokoro, Google)
- Voice selection
- Audio routing (CLI vs Web)
- Integration with `shared/audio_router.py`

### ConfigureTTS (`configure_tts.py`)

TTS configuration:

**Features**:
- Interactive TTS setup
- Engine selection
- Voice preferences
- Audio output settings

## Task Management

### Todos (`todos.py`)

Task tracking:

**Features**:
- Task creation
- Status tracking
- Reminders
- Workflow organization

## Permission Management

### Permissions (`permissions.py`)

Advanced permission management:

**Features**:
- View saved permissions
- Edit specific permissions
- Remove permissions
- Interactive management
- Detailed listing

**Commands**:
- `<permissions><action>view</action></permissions>`
- `<permissions><action>remove</action><pattern>bash:npm*</pattern></permissions>`

## File Copier Suite

### Main (`file_copier/main.py`)

Intelligent file copying:

**Features**:
- Path resolution
- Duplicate detection
- Smart paste functionality
- AI-powered path finding

### Smart Paster (`file_copier/smart_paster.py`)

Clipboard content analysis:

**Features**:
- Detect file types
- Content analysis
- Intelligent placement suggestions
- Processing recommendations

### AI Path Finder (`file_copier/ai_path_finder.py`)

LLM-powered path resolution:

**Features**:
- Context-aware path suggestions
- Optimal location recommendations
- Language model integration

### GUI (`file_copier/gui.py`)

Graphical interface for file operations.

## Tool Discovery

### Auto-Discovery

Tools are automatically discovered:
1. Scan `tools/` directory for Python files
2. Import each module
3. Check for required methods (get_delim, get_tool_info, run)
4. Register in tool registry

### Vector Database Discovery

**ToolVectorDB** (in `infrastructure/vector_db/`):
- Semantic tool search using ChromaDB
- Embed tool descriptions
- Find relevant tools based on query
- Context-aware tool suggestions

## Tool Logging

Per-tool logging via `shared/utils/tool_logging.py`:

**Features**:
- Separate log file per tool
- Location: `~/.cli-agent/logs/tools_<name>.log`
- Retention: 14 days, 20MB max
- Automatic rotation

**Usage in tools**:
```python
from shared.utils.tool_logging import get_logger_for_tool_module

logger = get_logger_for_tool_module(__name__)
logger.info("Tool operation")
```

## Permission Integration

All file and command tools integrate with permission system:
- Check `enhanced_permissions.py` before execution
- Support pattern-based permissions
- Interactive prompts when needed
- Persistent permission storage

## Testing

Tool-specific tests in `tests/tools/`:
- Unit tests per tool
- Integration tests for tool chains
- Mock external dependencies

See: `tests/tools/INDEX.md`

## Creating New Tools

1. Create `tools/your_tool.py`
2. Implement 3-method interface (get_delim, get_tool_info, run)
3. Add per-tool logging
4. Add dependencies to `requirements.txt`
5. Add config to `.env.example` if needed
6. Create tests in `tests/tools/test_your_tool.py`
7. Tool automatically discovered on next startup

**Complete guide**: `tools/CLAUDE.md`

## Tool Patterns

### Sync vs Async

Tools can be sync or async:

```python
# Sync tool
@staticmethod
def run(content: str) -> str:
    result = do_work(content)
    return result

# Async tool
@staticmethod
async def run(content: str) -> str:
    result = await async_work(content)
    return result
```

Detection is automatic.

### XML Parsing

Standard pattern for parsing XML content:

```python
import xml.etree.ElementTree as ET

@staticmethod
def run(content: str) -> str:
    try:
        root = ET.fromstring(f"<root>{content}</root>")
        param = root.find('param').text
        # Use param
    except Exception as e:
        return f"Error: {e}"
```

### Error Handling

Always return error strings, don't raise exceptions:

```python
@staticmethod
def run(content: str) -> str:
    try:
        # Tool logic
        return "Success"
    except Exception as e:
        logger.error(f"Tool failed: {e}")
        return f"Error: {e}"
```

## Related Documentation

- Tool implementation guide: `tools/CLAUDE.md`
- SearchWeb vector DB: `tools/docs/SEARCHWEB_VECTOR_DB.md`
- Tool testing: `tests/tools/INDEX.md`
- Permission system: `/core/permissions/docs/PERMISSION_SYSTEM.md`
- File format support: `/shared/docs/FILE_FORMAT_SUPPORT.md`
