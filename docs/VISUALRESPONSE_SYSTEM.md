# Visual Response System Documentation

## Overview

The Visual Response System is a sophisticated web-based interface that enables real-time visualization and interaction between users and the CLI-Agent. It provides a rich, interactive environment for displaying various types of content including charts, diagrams, code, and multimedia.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    WebSocket    ┌──────────────────┐    IPC Queue    ┌─────────────────┐
│   Browser       │ ←────────────→  │  Visual Server   │ ←─────────────→ │  Main Agent     │
│   Client        │                 │  (Port 7861)     │                 │  Process        │
└─────────────────┘                 └──────────────────┘                 └─────────────────┘
```

### Component Structure

```
Visual Response Interface
├── Logger System (logger.js)
│   ├── Multi-level logging (DEBUG, INFO, WARN, ERROR)
│   ├── Session persistence
│   ├── Performance timing
│   ├── Export capabilities
│   └── Browser console integration
│
├── Content Renderer (renderers.js)
│   ├── Canvas rendering
│   ├── SVG processing
│   ├── Mermaid diagram generation
│   ├── Plotly chart integration
│   ├── 3D molecular visualization
│   ├── HTML/Markdown rendering
│   └── JavaScript code execution
│
├── WebSocket Manager (websocket.js)
│   ├── Connection lifecycle management
│   ├── Automatic reconnection with exponential backoff
│   ├── Message parsing and routing
│   ├── Error handling and recovery
│   └── User input transmission
│
├── Main Application (main.js)
│   ├── Component coordination
│   ├── Initialization sequence
│   ├── Theme management
│   ├── Chat interface
│   └── Library dependency management
│
└── Static Assets
    ├── CSS styling (main.css)
    ├── HTML structure (index.html)
    └── JavaScript libraries (/libs/)
```

## Key Features

### 1. Comprehensive Logging System

The logging system provides detailed insights into system operation:

- **Multiple Log Levels**: DEBUG, INFO, WARN, ERROR with configurable filtering
- **Performance Timing**: Track operation durations with startTiming()/endTiming()
- **Session Persistence**: Logs saved to sessionStorage for debugging across page reloads
- **Export Capabilities**: Download logs as JSON files for analysis
- **Error Tracking**: Detailed error information with stack traces
- **Browser Integration**: Styled console output with grouping

#### Usage Examples

```javascript
// Basic logging
window.logInfo('User action completed', { actionType: 'click', element: 'button' });
window.logError('Operation failed', { operation: 'data_load' }, error);

// Performance timing
window.visualLogger.startTiming('chart_render');
// ... rendering logic ...
window.visualLogger.endTiming('chart_render', { chartType: 'bar', dataPoints: 100 });

// Export logs for analysis
window.downloadLogs('debug-session.json', { level: 'WARN', since: Date.now() - 3600000 });
```

## Debug Console Commands

The enhanced system provides several debugging commands available in the browser console:

- `window.visualLogger.getStats()` - View logging statistics
- `window.downloadLogs()` - Download logs for analysis
- `window.logDebug(msg, data)` - Quick debug logging
- `window.logInfo(msg, data)` - Quick info logging
- `window.logWarn(msg, data)` - Quick warning logging
- `window.logError(msg, data, error)` - Quick error logging

## Troubleshooting Guide

### Common Issues

1. **"Connecting to agent..." never changes**
   - Check browser console for WebSocket errors
   - Verify server is running on port 7861
   - Use: `window.downloadLogs('connection-debug.json')`

2. **JavaScript errors in console**
   - Check for library loading failures
   - Review initialization sequence
   - Use: `window.visualLogger.getStats()`

3. **Visualizations not appearing**
   - Check content rendering errors
   - Verify data format
   - Use: `window.logInfo('Debug', { renderer: 'test' })`