# Tools Index

This document provides a comprehensive reference for all available tools in the CLI-Agent system, including their required and optional parameters, defaults, and implementation notes.

## Usage Format

All tools use XML-style tags: `<toolname>...parameters...</toolname>`

**Legend:**
- ✅ **Required** - Must be provided
- 🔧 **Optional** - Can be omitted (uses defaults)
- 📝 **Content** - Direct text content (no tags)
- ⚡ **Default** - Default value when parameter omitted
- ⚠️ **Issue** - Known limitation or improvement opportunity
- 💡 **Improvement** - Suggested enhancement

---

## File Operations

### 📖 readfile
**Purpose**: Reads file contents with line range support and image handling

**Parameters:**
- ✅ **filepath** - Path to file (simple format) OR use XML format below
- 🔧 **lines** - Line range (e.g., "10-20", "15", "5-") - XML format only
  - ⚡ **Default**: Read entire file (no line filtering)

**Formats:**
```xml
<!-- Simple format -->
<readfile>/path/to/file.txt</readfile>

<!-- XML format with line range -->
<readfile>
<filepath>/path/to/file.txt</filepath>
<lines>10-20</lines>
</readfile>
```

**Features:**
- Auto-truncates large files (⚡ 50,000 char limit)
- Supports images (converts to base64 for vision models)
- Recursive file search if not found at specified path
- Line indexing support
- State tracking for truncated reads in `~/.readfile_state.json`

**Issues & Improvements:**
- ⚠️ **Hardcoded limit**: 50K character limit not configurable
- ⚠️ **Mixed format support**: Supporting both simple and XML formats adds complexity
- 💡 **Improvement**: Add configurable chunk size via environment variable
- 💡 **Improvement**: Better error messages for binary files vs text files
- ⚠️ **Security**: Recursive search could be exploited to scan entire filesystem

---

### ✏️ editfile
**Purpose**: Find and replace text or edit specific lines in files

**Parameters:**
- ✅ **filepath** - Path to target file
- **Mode 1 - Find/Replace:**
  - ✅ **search** - Text to find 
  - ✅ **replace** - Replacement text
- **Mode 2 - Line Specific:**
  - ✅ **line** - Line number to edit
  - ✅ **replace** - New content for the line

**Examples:**
```xml
<!-- Find and replace mode -->
<editfile>
<filepath>./src/app.js</filepath>
<grepsearch>const old_variable = 1;</search>
<replace>const new_variable = 2;</replace>
</editfile>

<!-- Line-specific editing -->
<editfile>
<filepath>./src/main.py</filepath>
<line>15</line>
<replace>print('new line')</replace>
</editfile>
```

**Features:**
- Replaces ALL occurrences in find/replace mode
- 1-based line numbering for line-specific mode
- UTF-8 encoding support

**Issues & Improvements:**
- ⚠️ **No backup**: No automatic backup before editing
- ⚠️ **No undo**: No way to revert changes
- ⚠️ **All occurrences**: Find/replace changes ALL matches, not just first
- 💡 **Improvement**: Add optional backup creation
- 💡 **Improvement**: Add preview mode to show changes before applying
- 💡 **Improvement**: Add option for "replace first N occurrences"

---

### 📝 writefile
**Purpose**: Creates or overwrites files with specified content

**Parameters:**
- ✅ **filepath** - Path where file will be created/overwritten
- ✅ **content** - File content to write

**Example:**
```xml
<writefile>
<filepath>./my_file.txt</filepath>
<content>This is the file content.</content>
</writefile>
```

**Special Behavior:**
- Automatically creates parent directories if needed
- Special handling for memory compaction when filepath contains "MEMORY"

---

### ➕ insertline
**Purpose**: Inserts new lines at specific positions in files

**Parameters:**
- ✅ **filepath** - Path to target file
- ✅ **content** - Content to insert
- **Position (choose one):**
  - 🔧 **after** - Insert after specified line number (0 = beginning)
  - 🔧 **before** - Insert before specified line number (1-based)
  - ⚡ **Default**: Append to end of file if no position specified

**Examples:**
```xml
<!-- Insert after line 10 -->
<insertline>
<filepath>./src/main.py</filepath>
<after>10</after>
<content>print('inserted line')</content>
</insertline>

<!-- Insert before line 1 (prepend) -->
<insertline>
<filepath>./src/main.py</filepath>
<before>1</before>
<content># New header comment</content>
</insertline>

<!-- Append to end -->
<insertline>
<filepath>./src/main.py</filepath>
<content>print('appended line')</content>
</insertline>
```

**Features:**
- Automatically adds newline if content doesn't end with one
- Validates line numbers against file length
- ⚡ Default position: End of file

**Issues & Improvements:**
- ⚠️ **Inconsistent indexing**: `after` uses 0-based, `before` uses 1-based
- ⚠️ **Confusing semantics**: `after: 0` means "insert at beginning"
- 💡 **Improvement**: Standardize on 1-based line numbering for both
- 💡 **Improvement**: Add `position: "start"|"end"|line_number` parameter instead

---

### ❌ deletelines
**Purpose**: Deletes specific lines or line ranges from files

**Parameters:**
- ✅ **filepath** - Path to target file
- **Delete Mode (choose one):**
  - **Single Line:** ✅ **line** - Line number to delete
  - **Range:** ✅ **start** + ✅ **end** - Line range to delete

**Examples:**
```xml
<!-- Delete single line -->
<deletelines>
<filepath>./src/main.py</filepath>
<line>15</line>
</deletelines>

<!-- Delete line range -->
<deletelines>
<filepath>./src/main.py</filepath>
<start>5</start>
<end>8</end>
</deletelines>
```

---

## Search & Discovery

### 🔍 grepsearch  
**Purpose**: Recursively searches for regex patterns in file contents and names

**Parameters:**
- ✅ **pattern** - Regex pattern to search for
- 🔧 **path** - Starting directory 
  - ⚡ **Default**: Current directory (`.`)

**Example:**
```xml
<grepsearch>
<pattern>my_function</pattern>
<path>./src</path>
</grepsearch>
```

**Features:**
- ⚡ **Result limit**: 100 matches maximum
- Searches both file contents and filenames
- Validates regex patterns before execution
- Works on both files and directories

**Issues & Improvements:**
- ⚠️ **Hardcoded limit**: 100-match limit not configurable
- ⚠️ **No file type filtering**: Searches all files including binaries
- ⚠️ **Performance**: No optimization for large directories
- 💡 **Improvement**: Add file extension filtering (e.g., `--include="*.py"`)
- 💡 **Improvement**: Add case-insensitive search option
- 💡 **Improvement**: Add context lines around matches

---

### 🌐 searchweb
**Purpose**: Performs web search using Brave Search API

**Parameters:**
- 📝 **Direct Content** - Search query (no tags needed)

**Example:**
```xml
<searchweb>latest Python frameworks 2024</searchweb>
```

**Features:**
- ⚡ **Result limit**: 20 search results maximum
- HTML tag removal and text cleaning
- URL and formatting cleanup

**Requirements & Issues:**
- ✅ **Required**: `BRAVE_API_KEY` environment variable
- ⚠️ **API dependency**: Fails completely if API key missing or invalid
- ⚠️ **No fallback**: No alternative search engines
- ⚠️ **Hardcoded limit**: 20-result limit not configurable
- 💡 **Improvement**: Add fallback to other search APIs (DuckDuckGo, etc.)
- 💡 **Improvement**: Add search filters (date, site, etc.)
- 💡 **Improvement**: Add configuration for result count

---

## Code Execution

### 🐍 python
**Purpose**: Executes Python code in persistent REPL session

**Parameters:**
- 📝 **Direct Content** - Python code to execute

**Example:**
```xml
<python>
print("Hello World")
x = 2 + 2
print(f"Result: {x}")
</python>
```

**Features:**
- Persistent session (variables retained between calls)
- Output streamed to console
- Base64 encoding for safe execution

---

### 🔧 bash
**Purpose**: Executes shell commands with timeout control

**Parameters:**
- 📝 **Command Content** - Shell command(s) to execute
- 🔧 **timeout** - Timeout in seconds
  - ⚡ **Default**: 10 seconds

**Examples:**
```xml
<!-- With explicit timeout -->
<bash>
<timeout>30</timeout>
ls -la /home
</bash>

<!-- Using default 10-second timeout -->
<bash>ls -F</bash>
```

**Features:**
- Persistent bash session across calls
- Live output streaming to console
- ⚡ **Default timeout**: 10 seconds (recently fixed!)
- Automatic session recovery on crashes

**Issues & Improvements:**
- ⚠️ **Session persistence**: Shared state between calls could cause conflicts
- ⚠️ **No working directory control**: Can't easily change or reset working directory
- ⚠️ **No environment isolation**: Environment variables persist between calls
- 💡 **Improvement**: Add session isolation options
- 💡 **Improvement**: Add working directory parameter
- 💡 **Improvement**: Add environment variable control

---

## User Interface & Display

### 🖥️ showuser (OpenGL)
**Purpose**: Creates and updates live OpenGL window with Python/pyglet scripts

**Parameters:**
- 📝 **Direct Content** - Complete Python script using pyglet

**Example:**
```xml
<showuser>
import pyglet

window = pyglet.window.Window(800, 600)

@window.event
def on_draw():
    window.clear()
    # Your OpenGL/pyglet code here

pyglet.app.run()
</showuser>
```

**Requirements:** `pip install pyglet`
**Features:**
- Hot-reloading of scripts
- Separate process for rendering
- Full pyglet API access

---

### 📁 getuserfiles
**Purpose**: Retrieves current file selection from File Copier GUI via WebSocket

**Parameters:**
- 📝 **No Parameters** - Tool takes no arguments

**Example:**
```xml
<getuserfiles></getuserfiles>
```

**Features:**
- Automatic WebSocket discovery (ports 8765-8769)
- Returns formatted file paths and contents
- Integrates with File Copier GUI tool

---

## Memory & State Management

### 🧠 memory
**Purpose**: Searches agent's long-term memory using vector database

**Parameters:**
- 📝 **Direct Content** - Natural language query with optional SQL-like syntax

**Advanced Query Examples:**
```xml
<!-- Simple semantic search -->
<memory>What did I learn about Python async programming?</memory>

<!-- With limit -->
<memory>Python frameworks LIMIT 10</memory>

<!-- With metadata filtering -->
<memory>databases WHERE topic CONTAINS 'performance'</memory>
```

**Features:**
- ⚡ **Default limit**: 5 results
- ⚡ **Query multiplication**: Fetches 3x limit for better sorting
- Semantic search using ChromaDB embeddings
- SQL-like syntax for filtering and limits
- Results sorted by timestamp (latest first)

**Issues & Improvements:**
- ⚠️ **Hardcoded defaults**: 5-result limit, 3x multiplication not configurable
- ⚠️ **Limited filtering**: Only supports simple CONTAINS operations
- ⚠️ **No relevance sorting**: Always sorts by date, ignoring relevance scores
- 💡 **Improvement**: Add relevance-based sorting options
- 💡 **Improvement**: Add more metadata filter operators
- 💡 **Improvement**: Add date range filtering

---

### ✅ todos
**Purpose**: Manages persistent task list for tracking progress

**Parameters:**
- 📝 **Direct Content** - Complete todo list in markdown format

**Example:**
```xml
<todos>
- [x] Step 1: List files.
- [ ] Step 2: Read the relevant file.
- [ ] Step 3: Analyze its content.
</todos>
```

**Features:**
- Persistent storage in `todos.md`
- Markdown checkbox format
- Always provide complete updated list

---

## Utilities

### 🤔 think
**Purpose**: Cognitive tool for reasoning and planning (no operational output)

**Parameters:**
- 📝 **Direct Content** - Thoughts, plans, reflections

**Example:**
```xml
<think>
The previous command failed. I need to check the file permissions 
before trying again. Let me use readfile first to see the current state.
</think>
```

**Features:**
- No-op execution (pure reasoning)
- Visible in chat history
- Helps structure agent's thought process

---

### 🔐 viewpermissions
**Purpose**: Displays saved permission rules for automatic tool execution

**Parameters:**
- 📝 **No Parameters** - Tool takes no arguments

**Example:**
```xml
<viewpermissions></viewpermissions>
```

**Features:**
- Shows tool-specific permissions
- Path-specific rules
- Command-specific rules

---

### 🔧 permissions
**Purpose**: Advanced permission management tool for editing and removing specific saved permissions

**Parameters:**
- 📝 **Direct Content** - Management command (list, remove <id>, clear)

**Examples:**
```xml
<!-- List all permissions with IDs -->
<permissions>list</permissions>

<!-- Remove specific permission by ID -->
<permissions>remove 5</permissions>

<!-- Clear all saved permissions -->
<permissions>clear</permissions>
```

**Features:**
- Interactive permission management with numbered IDs
- Selective permission removal by ID
- Bulk permission clearing
- Detailed permission information display
- Safe error handling for invalid operations

---

## Configuration Tools

### 🎤 configure_tts
**Purpose**: Standalone TTS configuration utility (not a runtime tool)

**Usage:** Run directly as Python script, not via XML tags
```bash
python configure_tts.py --show
python configure_tts.py --set-voice nova
```

### 🧪 tts_tester
**Purpose**: TTS testing and benchmarking utility (not a runtime tool)

**Usage:** Run directly as Python script for TTS engine evaluation

---

## Tool Development Notes

### Parameter Parsing Patterns

Most tools follow these common patterns:

1. **Simple Format**: Direct content without XML tags
   ```xml
   <toolname>direct content here</toolname>
   ```

2. **XML Format**: Structured parameters with nested tags
   ```xml
   <toolname>
   <param1>value1</param1>
   <param2>value2</param2>
   </toolname>
   ```

3. **Hybrid Format**: Support both simple and XML based on content detection

### Error Handling

All tools return descriptive error messages when:
- Required parameters are missing
- File paths don't exist
- Invalid formats are provided
- External services are unavailable

### Path Resolution

File operation tools resolve paths relative to the current working directory unless absolute paths are provided.

### Security Considerations

- Tools validate and sanitize inputs
- File operations restricted to accessible paths
- Command execution uses proper timeout controls
- Permission system governs automatic execution

---

## 🔍 Analysis Summary

### Common Default Patterns
- **File operations**: Default to current working directory and UTF-8 encoding
- **Search tools**: Default limits (grepsearch: 100, searchweb: 20, memory: 5)
- **Timeout operations**: bash defaults to 10 seconds
- **Line operations**: insertline defaults to end-of-file

### Major Issues Identified

#### 🔒 Security Concerns
- **readfile**: Recursive search could scan entire filesystem
- **All file tools**: No access control beyond filesystem permissions
- **bash/python**: Persistent sessions share state between calls

#### ⚙️ Configuration Limitations
- **Hardcoded limits**: Most tools have non-configurable limits
- **No environment configuration**: Limits and timeouts not configurable via env vars
- **Mixed parameter formats**: Some tools support both simple and XML formats

#### 🏗️ Architecture Issues
- **No consistent backup strategy**: File modification tools don't create backups
- **Error handling inconsistency**: Different error message formats across tools
- **Session management**: bash/python persistent sessions lack isolation

### Priority Improvements

#### High Priority
1. **Add backup functionality** to all file modification tools
2. **Standardize parameter formats** (prefer XML for consistency)
3. **Add configuration support** for limits and timeouts
4. **Improve error messages** with consistent formatting

#### Medium Priority
1. **Add preview modes** for destructive operations
2. **Implement session isolation** for code execution tools
3. **Add file type filtering** for search operations
4. **Improve security boundaries** for file access

#### Low Priority
1. **Add undo functionality** for file operations
2. **Implement relevance-based sorting** for memory tool
3. **Add context lines** for search results
4. **Expand metadata filtering** capabilities

### Tool Reliability Matrix

| Tool | Reliability | Improvement Needed |
|------|-------------|-------------------|
| readfile | 🟢 High | Security boundaries |
| editfile | 🟡 Medium | Backup & preview |
| writefile | 🟢 High | Backup functionality |
| insertline | 🟡 Medium | Consistent indexing |
| deletelines | 🟡 Medium | Backup & preview |
| grepsearch | 🟢 High | Performance & filtering |
| searchweb | 🟡 Medium | API fallbacks |
| python | 🟡 Medium | Session isolation |
| bash | 🟡 Medium | Session isolation |
| memory | 🟢 High | Query capabilities |
| todos | 🟢 High | Minimal changes needed |
| think | 🟢 High | Working as designed |
| viewpermissions | 🟢 High | Working as designed |
| permissions | 🟢 High | Interactive management |

---

*Last updated: Generated with comprehensive analysis of tool implementations*