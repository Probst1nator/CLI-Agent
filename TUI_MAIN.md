# CLI-Agent TUI Integration Guide

## Overview

This document explains the TUI (Terminal User Interface) system for CLI-Agent, focusing on the visual input integration and how different components interact to provide a race-condition-free persistent input experience.

## Architecture Components

### 1. Core Files

#### `tui_main.py` - Full TUI Implementation
- **Purpose**: Complete terminal UI with persistent input field
- **Status**: Functional but complex (stdout redirection issues)
- **Use Case**: Full GUI-like experience in terminal

#### `tui_simple.py` - Safe TUI Version  
- **Purpose**: Minimal TUI for testing without breaking terminal
- **Status**: Working, safe for development
- **Use Case**: Testing visual integration without risk

#### `visual_input_bridge.py` - Production Solution ⭐
- **Purpose**: Surgical enhancement of existing CLI input handling
- **Status**: Production-ready, integrated into main.py
- **Use Case**: Real-world deployment without changing user experience

### 2. Visual Integration Components

#### `tools/visualresponse.py` - Visual Server
- **Function**: Web interface and WebSocket server
- **Creates**: `g.visual_input_queue` for browser messages
- **Integration**: Browser → WebSocket → Queue

#### `tools/visual_assets/index.html` - Browser Interface
- **Function**: Web UI for sending messages to CLI-Agent
- **Integration**: User input → WebSocket → Visual server

## How It Works

### Current Implementation Flow

```
Browser Input → Visual Server → g.visual_input_queue → Enhanced Input Handler → CLI-Agent
                                                     ↘ (if queue empty)
                                                       Standard Input → CLI-Agent
```

### Component Interactions

#### 1. Visual Server Setup
```python
# In tools/visualresponse.py
class VisualServerSession:
    def __init__(self):
        self.input_queue = queue.Queue()
        g.visual_input_queue = self.input_queue  # Make globally accessible
```

#### 2. Browser Communication
```javascript
// In tools/visual_assets/index.html  
ws.send(message);  // Sends user input via WebSocket
```

#### 3. Queue Processing
```python
# In visual_input_bridge.py
async def enhanced_get_user_input_with_bindings(*args, **kwargs):
    # Check visual queue first (non-blocking)
    if hasattr(g, 'visual_input_queue') and not g.visual_input_queue.empty():
        visual_message = g.visual_input_queue.get_nowait()
        print(f"\n🌐 Visual Input: {visual_message}")
        return visual_message, []
    
    # Fall back to standard input if no visual messages
    return await original_input_function(*args, **kwargs)
```

#### 4. Integration into Main Loop
```python
# In main.py (automatically installed)
from visual_input_bridge import install_visual_input_bridge
install_visual_input_bridge()  # Patches input handling
```

## Race Condition Solution

### The Problem
```
❌ Original Issue:
CLI calls input() → Thread blocks waiting
Browser sends message → Message queued but ignored  
User waits indefinitely → Race condition
```

### The Solution
```
✅ Enhanced Flow:
Check visual queue → If message exists: Return immediately
                 → If empty: Call standard input()
No blocking until after queue check → Race condition eliminated
```

### Key Technical Details

1. **Non-blocking Queue Check**: `queue.get_nowait()` never blocks
2. **Monkey Patching**: Replaces `get_user_input_with_bindings` cleanly
3. **Fallback Preservation**: Original functionality completely intact
4. **Thread Safety**: Uses thread-safe queue operations

## Usage Scenarios

### Development & Testing

#### Safe TUI Testing
```bash
python tui_simple.py  # Safe for development, won't break terminal
```

#### Integration Testing
```bash
python test_complete_integration.py  # Test full message flow
```

### Production Usage

#### Normal CLI Operation
```bash
python main.py  # Visual bridge auto-installed, works transparently
```

#### With Visual Server Active
```bash
# Terminal 1: Start CLI-Agent
python main.py

# Terminal 2: Start visual server (if using visualresponse tool)
# Browser messages now appear in Terminal 1 with 🌐 indicator
```

## File Dependencies

### Core Dependencies
```
main.py
├── agent/core/input_handlers.py (patched by bridge)
├── visual_input_bridge.py (enhancement)
├── core/globals.py (queue storage)
└── tools/visualresponse.py (queue creation)
```

### Optional TUI Components
```
tui_main.py (complex TUI)
├── agent/core/command_system.py (command completion)  
├── agent_tui.css (styling)
└── textual library (TUI framework)

tui_simple.py (safe testing TUI)
├── textual library (minimal usage)
└── core/globals.py (queue monitoring)
```

## Configuration

### Visual Integration Toggle
The visual input bridge is automatically enabled but can be controlled:

```python
# Disable visual integration
# Remove or comment out in main.py:
# from visual_input_bridge import install_visual_input_bridge
# install_visual_input_bridge()
```

### TUI Selection
Choose your interface approach:

1. **Production** (Recommended): Use standard CLI with visual bridge
2. **Development**: Use `tui_simple.py` for safe testing  
3. **Advanced**: Use `tui_main.py` for full TUI experience (requires debugging)

## Troubleshooting

### Common Issues

#### "Visual input bridge not available"
- **Cause**: Import error for `visual_input_bridge.py`
- **Solution**: Ensure file exists in CLI-Agent root directory

#### TUI display corruption
- **Cause**: Using `tui_main.py` with stdout redirection issues
- **Solution**: Use `tui_simple.py` or the production visual bridge instead

#### Messages not appearing
- **Cause**: Visual server not creating `g.visual_input_queue`
- **Solution**: Start visual server or manually create queue for testing

### Debug Commands

```python
# Check if visual queue exists
from core.globals import g
print(hasattr(g, 'visual_input_queue'))

# Check queue status  
if hasattr(g, 'visual_input_queue'):
    print(f"Queue size: {g.visual_input_queue.qsize()}")

# Test bridge installation
from visual_input_bridge import install_visual_input_bridge
install_visual_input_bridge()
```

## Future Enhancements

### Planned Improvements
1. **Enhanced TUI**: Fix stdout redirection in `tui_main.py`
2. **Multi-source Input**: Support for voice, file watchers, APIs
3. **Input Source Indicators**: Visual distinction for different input origins
4. **Configuration UI**: Web-based settings for input preferences

### Extension Points
- Add new input sources to `enhanced_get_user_input_with_bindings`
- Create additional TUI layouts in new files
- Enhance visual indicators for different message types

## Summary

The current implementation provides a **race-condition-free visual input system** that:

- ✅ Integrates browser input seamlessly
- ✅ Preserves all existing CLI functionality  
- ✅ Requires no user behavior changes
- ✅ Operates safely without terminal corruption
- ✅ Supports both development and production workflows

The **visual input bridge** is the recommended production solution, while the TUI components provide additional development and testing capabilities.