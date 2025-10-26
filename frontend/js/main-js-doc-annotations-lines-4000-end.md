# JSDoc Annotations for main.js (Lines 4000-End)

This document provides comprehensive JSDoc type annotations for all methods from line 4000 to the end of `/home/prob/repos/CLI-Agent/frontend/js/main.js`.

## VisualResponseApp Methods (Lines 4000-4046)

### scheduleAutoSave()
```javascript
/**
 * Schedule auto-save of current LLM configuration to active profile slot
 * Debounces saves to prevent excessive writes (2 second delay)
 * @private
 */
scheduleAutoSave() {
    if (!this.activeProfileSlot) return;

    if (this.autoSaveTimeout) {
        clearTimeout(this.autoSaveTimeout);
    }

    this.autoSaveTimeout = setTimeout(() => {
        this.saveCurrentConfigToSlot(this.activeProfileSlot);
    }, 2000);
}
```

### showAutoplayPrompt()
```javascript
/**
 * Display autoplay prompt banner to enable audio after user interaction
 * Shows a banner that enables pending TTS playback when user clicks
 * Required due to browser autoplay policies
 * @private
 */
showAutoplayPrompt() {
    // Show a subtle banner prompting user to click to enable audio
    let banner = document.getElementById('autoplay-prompt');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'autoplay-prompt';
        banner.className = 'autoplay-prompt';
        banner.innerHTML = '🔊 Click anywhere to enable audio';
        banner.style.cssText = 'position: fixed; top: 10px; left: 50%; transform: translateX(-50%); background: #4a9eff; color: white; padding: 10px 20px; border-radius: 5px; z-index: 10000; cursor: pointer; box-shadow: 0 2px 10px rgba(0,0,0,0.3);';
        document.body.appendChild(banner);

        // Remove banner and play pending TTS on any click
        const playPending = async () => {
            banner.remove();
            if (this.pendingTTS) {
                console.log('🔊 Playing queued TTS after user interaction');
                try {
                    await this.audioManager.playTTS(this.pendingTTS, 'auto');
                    this.pendingTTS = null;
                } catch (error) {
                    console.error('Failed to play pending TTS:', error);
                }
            }
            document.removeEventListener('click', playPending);
        };
        document.addEventListener('click', playPending, { once: true });
    }
}
```

## Global Functions (Lines 4048-4121)

### initializeApp()
```javascript
/**
 * Initialize the application when DOM is ready and all dependencies are loaded
 * Waits for ContentRenderer and WebSocketManager to be available before initializing
 * Sets up global window references for cross-module communication
 */
function initializeApp() {
    // Check if all required classes are available
    if (typeof ContentRenderer === 'undefined') {
        console.log('Waiting for ContentRenderer...');
        setTimeout(initializeApp, 100);
        return;
    }

    if (typeof WebSocketManager === 'undefined') {
        console.log('Waiting for WebSocketManager...');
        setTimeout(initializeApp, 100);
        return;
    }

    console.log('All dependencies loaded, initializing app...');
    const app = new VisualResponseApp();
    window.app = app;  // Global reference for optimistic message tracking
    window.VisualResponseApp = app;
    window.agentInterface = app;  // For message edit/delete buttons
}
```

### window.sendPermissionResponse()
```javascript
/**
 * Global permission response function
 * Sends user's permission decision to backend via WebSocket
 * @param {string} permissionId - Permission request ID
 * @param {'allow'|'deny'|'always_allow'|'always_deny'} result - Permission result
 * @param {Object} action - Additional action parameters
 * @returns {boolean} Whether the response was sent successfully
 */
window.sendPermissionResponse = function(permissionId, result, action = {}) {
    console.log('sendPermissionResponse called', { permissionId, result, action });

    if (window.wsManager && window.wsManager.sendPermissionResponse) {
        return window.wsManager.sendPermissionResponse(permissionId, result, action);
    } else {
        console.error('WebSocket manager not available for permission response');
        return false;
    }
};
```

### window.sendInterruptSignal()
```javascript
/**
 * Global interrupt signal function
 * Sends interrupt signal to backend to cancel current agent operation
 * @returns {boolean} Whether the interrupt signal was sent successfully
 */
window.sendInterruptSignal = function() {
    console.log('sendInterruptSignal called');

    if (window.wsManager && window.wsManager.sendInterruptSignal) {
        return window.wsManager.sendInterruptSignal();
    } else {
        console.error('WebSocket manager not available for interrupt signal');
        return false;
    }
};
```

### window.toggleToolDetail()
```javascript
/**
 * Global tool detail toggle function
 * Toggles visibility of tool execution details in message display
 * @param {string} toolDetailId - ID of the tool detail element to toggle
 */
window.toggleToolDetail = function(toolDetailId) {
    const element = document.getElementById(toolDetailId);
    if (!element) return;

    const isHidden = element.style.display === 'none';
    element.style.display = isHidden ? 'block' : 'none';

    // Find the toggle icon and update it
    const summaryElement = element.parentElement.querySelector('.tool-summary .tool-toggle');
    if (summaryElement) {
        summaryElement.textContent = isHidden ? '▲' : '▼';
    }
};
```

## ConsoleLogManager Class (Lines 4123-4775)

### Constructor
```javascript
/**
 * Console Log Manager
 * Handles the console sidebar functionality for displaying raw CLI-Agent output
 * Supports both file-based log reading and real-time SSE streaming
 * @class
 */
class ConsoleLogManager {
    /**
     * Creates a new ConsoleLogManager instance
     * Initializes console state, filters, and event listeners
     */
    constructor() {
        /** @type {boolean} Console visibility state */
        this.consoleVisible = false;

        /** @type {ConsoleLogEntry[]} Log entries buffer */
        this.consoleLog = [];

        /** @type {number} Maximum log lines to keep in memory */
        this.maxLogLines = 5000;

        // Performance optimization properties
        /** @type {boolean} Pending console update flag */
        this.consoleUpdatePending = false;
        /** @type {number} Last render timestamp */
        this.lastRenderTime = 0;
        /** @type {number} Render throttle interval (60fps max) */
        this.renderThrottleMs = 16; // ~60fps max

        /** @type {number} Console sidebar width in pixels */
        this.consoleWidth = 350; // Default width

        /** @type {boolean} Console resize in progress flag */
        this.isResizing = false;

        this.init();
    }
}
```

### init()
```javascript
/**
 * Initialize console manager components and state
 * Sets up event listeners, filters, and loads saved console state
 * @private
 */
init() {
    this.setupToggleButton();
    this.setupClearButton();
    this.setupResizeHandle();
    this.setupLogSourceSelector();
    this.setupRefreshButton();
    this.setupFilters();
    this.loadConsoleState();

    this.logSource = 'file';
    this.selectedLogFile = 'session.log';
    this.autoRefreshInterval = null;
    this.lastLogUpdate = null;
    this.autoRefreshEnabled = true;
    this.sseConnection = null;

    this.searchFilter = '';
    this.levelFilters = {
        debug: true,
        info: true,
        warning: true,
        error: true
    };
}
```

### setupToggleButton()
```javascript
/**
 * Setup console toggle button event listener
 * Handles showing/hiding console sidebar
 * @private
 */
setupToggleButton() {
    const toggleBtn = document.getElementById('console-toggle-btn');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            this.toggleConsole();
        });
    }
}
```

### setupClearButton()
```javascript
/**
 * Setup console clear button event listener
 * Handles clearing all log entries from display
 * @private
 */
setupClearButton() {
    const clearBtn = document.getElementById('console-clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            this.clearConsole();
        });
    }
}
```

### setupResizeHandle()
```javascript
/**
 * Setup console resize handle for drag-to-resize functionality
 * Allows user to resize console sidebar width via drag handle
 * @private
 */
setupResizeHandle() {
    const resizeHandle = document.getElementById('console-resize-handle');
    if (!resizeHandle) return;

    resizeHandle.addEventListener('mousedown', (e) => {
        this.isResizing = true;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';

        const onMouseMove = (e) => {
            if (!this.isResizing) return;

            const sidebar = document.getElementById('console-sidebar');
            if (!sidebar) return;

            // Calculate new width based on mouse position
            const containerRect = sidebar.parentElement.getBoundingClientRect();
            const newWidth = containerRect.right - e.clientX;

            // Apply constraints
            const minWidth = 200;
            const constrainedWidth = Math.max(minWidth, newWidth);

            // Update width
            this.consoleWidth = constrainedWidth;
            sidebar.style.width = constrainedWidth + 'px';

            // Save to localStorage
            this.saveConsoleState();
        };

        const onMouseUp = () => {
            this.isResizing = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);

        e.preventDefault();
    });
}
```

### setupLogSourceSelector()
```javascript
/**
 * Setup log source selector dropdown event listener
 * Handles switching between different log files
 * @private
 * @async
 */
setupLogSourceSelector() {
    const selector = document.getElementById('console-log-source');
    if (selector) {
        selector.addEventListener('change', async (e) => {
            const value = e.target.value;
            const timestampEl = document.getElementById('console-last-update');

            this.logSource = 'file';
            this.selectedLogFile = value;

            if (timestampEl) {
                timestampEl.textContent = 'Loading...';
            }

            this.stopAutoRefresh();
            await this.fetchLogFile();

            if (value === 'session.log' && this.consoleVisible) {
                await this.connectSSE(value);
            }
        });
    }
}
```

### setupRefreshButton()
```javascript
/**
 * Setup manual refresh button event listener
 * Allows user to manually reload log file contents
 * @private
 */
setupRefreshButton() {
    const refreshBtn = document.getElementById('console-refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', async () => {
            if (this.logSource === 'file') {
                await this.fetchLogFile();
            }
        });
    }
}
```

### setupFilters()
```javascript
/**
 * Setup search filter and log level filter checkboxes
 * Handles text search and log level filtering (debug, info, warning, error)
 * @private
 */
setupFilters() {
    const searchInput = document.getElementById('console-search');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            this.searchFilter = e.target.value.toLowerCase();
            if (this.consoleVisible) {
                this.renderConsoleLog();
            }
        });
    }

    const levelCheckboxes = [
        { id: 'console-level-debug', level: 'debug' },
        { id: 'console-level-info', level: 'info' },
        { id: 'console-level-warning', level: 'warning' },
        { id: 'console-level-error', level: 'error' }
    ];

    levelCheckboxes.forEach(({ id, level }) => {
        const checkbox = document.getElementById(id);
        if (checkbox) {
            checkbox.addEventListener('change', (e) => {
                this.levelFilters[level] = e.target.checked;
                if (this.consoleVisible) {
                    this.renderConsoleLog();
                }
            });
        }
    });
}
```

### shouldShowLogEntry()
```javascript
/**
 * Check if log entry should be displayed based on current filters
 * Applies both level filters and text search filter
 * @param {ConsoleLogEntry} entry - Log entry to check
 * @returns {boolean} True if entry should be shown
 * @private
 */
shouldShowLogEntry(entry) {
    if (!this.levelFilters[entry.level]) {
        return false;
    }

    if (this.searchFilter) {
        const searchableText = (entry.line + ' ' + entry.raw).toLowerCase();
        return searchableText.includes(this.searchFilter);
    }

    return true;
}
```

### fetchAvailableLogs()
```javascript
/**
 * Fetch list of available log files from backend
 * @async
 * @returns {Promise<string[]>} Array of available log file names
 */
async fetchAvailableLogs() {
    try {
        const url = window.backendConnector ? await window.backendConnector.buildURL('/api/logs/list') : '/api/logs/list';
        const response = await fetch(url);
        const result = await response.json();

        if (result.status === 'success') {
            return result.data.logs;
        } else {
            console.error('Failed to fetch log list:', result.error);
            return [];
        }
    } catch (error) {
        console.error('Error fetching log list:', error);
        return [];
    }
}
```

### fetchLogFile()
```javascript
/**
 * Fetch log file contents from backend
 * Retrieves last N lines from specified log file and updates display
 * @async
 * @param {number} [lines=500] - Number of lines to fetch from end of file
 * @returns {Promise<Object|undefined>} Log file data object
 */
async fetchLogFile(lines = 500) {
    if (this.logSource !== 'file') return;

    try {
        const url = window.backendConnector ? await window.backendConnector.buildURL(`/api/logs/tail?log=${encodeURIComponent(this.selectedLogFile)}&lines=${lines}`) : `/api/logs/tail?log=${encodeURIComponent(this.selectedLogFile)}&lines=${lines}`;
        const response = await fetch(url);
        const result = await response.json();

        if (result.status === 'success') {
            const data = result.data;
            this.consoleLog = data.entries.map(entry => ({
                timestamp: entry.timestamp || new Date().toLocaleTimeString(),
                line: entry.message,
                level: this.normalizeLevelFromBackend(entry.level),
                raw: entry.raw
            }));

            this.lastLogUpdate = new Date(data.modified * 1000);

            const timestampEl = document.getElementById('console-last-update');
            if (timestampEl) {
                timestampEl.textContent = `Updated: ${this.lastLogUpdate.toLocaleTimeString()}`;
            }

            if (this.consoleVisible) {
                this.renderConsoleLog();
                this.scrollToBottom();
            }

            return data;
        } else {
            console.error('Failed to fetch log:', result.error);
            this.showLogError(result.error);
        }
    } catch (error) {
        console.error('Error fetching log file:', error);
        this.showLogError('Network error fetching logs');
    }
}
```

### normalizeLevelFromBackend()
```javascript
/**
 * Normalize log level string from backend to standard levels
 * Converts various log level formats to: error, warning, debug, or info
 * @param {string} level - Raw log level from backend
 * @returns {'error'|'warning'|'debug'|'info'} Normalized log level
 * @private
 */
normalizeLevelFromBackend(level) {
    const normalized = (level || 'INFO').toLowerCase().trim();
    if (normalized.includes('error')) return 'error';
    if (normalized.includes('warn')) return 'warning';
    if (normalized.includes('debug')) return 'debug';
    return 'info';
}
```

### showLogError()
```javascript
/**
 * Display error message in console view
 * Shows error message with timestamp and error styling
 * @param {string} message - Error message to display
 * @private
 */
showLogError(message) {
    const consoleContent = document.getElementById('console-content');
    if (consoleContent) {
        consoleContent.innerHTML = `
            <div class="console-line console-error">
                <span class="console-timestamp">[${new Date().toLocaleTimeString()}]</span>
                <span class="console-text">Error: ${this.escapeHtml(message)}</span>
            </div>
        `;
    }
}
```

### connectSSE()
```javascript
/**
 * Connect to Server-Sent Events (SSE) stream for real-time log updates
 * Establishes EventSource connection for live log streaming
 * Falls back to polling on connection error
 * @async
 * @param {string} logFile - Name of log file to stream
 */
async connectSSE(logFile) {
    this.disconnectSSE();

    try {
        const url = window.backendConnector ? await window.backendConnector.buildURL(`/api/logs/tail-stream?log=${encodeURIComponent(logFile)}`) : `/api/logs/tail-stream?log=${encodeURIComponent(logFile)}`;
        this.sseConnection = new EventSource(url);

        this.sseConnection.onmessage = (event) => {
            try {
                const entry = JSON.parse(event.data);

                const logEntry = {
                    timestamp: entry.timestamp || new Date().toLocaleTimeString(),
                    line: entry.message,
                    level: this.normalizeLevelFromBackend(entry.level),
                    raw: entry.raw
                };

                this.consoleLog.push(logEntry);

                if (this.consoleLog.length > this.maxLogLines) {
                    this.consoleLog = this.consoleLog.slice(-this.maxLogLines);
                }

                if (this.consoleVisible) {
                    this.scheduleConsoleUpdate();
                }

                this.lastLogUpdate = new Date();
                const timestampEl = document.getElementById('console-last-update');
                if (timestampEl) {
                    timestampEl.textContent = `${logFile} (Live SSE) - ${this.lastLogUpdate.toLocaleTimeString()}`;
                }
            } catch (error) {
                console.error('Error parsing SSE message:', error);
            }
        };

        this.sseConnection.onerror = (error) => {
            console.warn('SSE connection error, falling back to polling:', error);
            this.disconnectSSE();
            this.startAutoRefresh(2000);
        };

        console.log(`SSE connected for ${logFile}`);
    } catch (error) {
        console.error('Failed to establish SSE connection:', error);
        this.startAutoRefresh(2000);
    }
}
```

### disconnectSSE()
```javascript
/**
 * Disconnect from SSE stream
 * Closes EventSource connection and cleans up
 * @private
 */
disconnectSSE() {
    if (this.sseConnection) {
        this.sseConnection.close();
        this.sseConnection = null;
    }
}
```

### startAutoRefresh()
```javascript
/**
 * Start automatic polling for log file updates
 * Used when SSE is not available or as fallback
 * @param {number} [intervalMs=5000] - Polling interval in milliseconds
 */
startAutoRefresh(intervalMs = 5000) {
    this.stopAutoRefresh();
    if (this.logSource === 'file') {
        this.autoRefreshInterval = setInterval(() => {
            this.fetchLogFile();
        }, intervalMs);
    }
}
```

### stopAutoRefresh()
```javascript
/**
 * Stop automatic refresh polling and disconnect SSE
 * Cleans up both polling interval and SSE connection
 */
stopAutoRefresh() {
    this.disconnectSSE();

    if (this.autoRefreshInterval) {
        clearInterval(this.autoRefreshInterval);
        this.autoRefreshInterval = null;
    }
}
```

### toggleConsole()
```javascript
/**
 * Toggle console sidebar visibility
 * Loads log content when showing, stops updates when hiding
 * @async
 */
async toggleConsole() {
    this.consoleVisible = !this.consoleVisible;
    this.updateConsoleVisibility();
    this.saveConsoleState();

    // Load and render console content when it becomes visible
    if (this.consoleVisible) {
        if (this.consoleLog.length === 0 && this.selectedLogFile) {
            await this.fetchLogFile();
        }

        if (this.selectedLogFile === 'session.log' && !this.sseConnection && !this.autoRefreshInterval) {
            await this.connectSSE('session.log');
        }

        this.renderConsoleLog();
        this.scrollToBottom();
    } else {
        this.stopAutoRefresh();
    }
}
```

### updateConsoleVisibility()
```javascript
/**
 * Update console visibility classes and styles
 * Applies saved width and toggles active state
 * @private
 */
updateConsoleVisibility() {
    const sidebar = document.getElementById('console-sidebar');
    const toggleBtn = document.getElementById('console-toggle-btn');

    if (sidebar) {
        if (this.consoleVisible) {
            sidebar.classList.remove('hidden');
            // Apply saved width when showing
            sidebar.style.width = this.consoleWidth + 'px';
        } else {
            sidebar.classList.add('hidden');
        }
    }

    if (toggleBtn) {
        if (this.consoleVisible) {
            toggleBtn.classList.add('active');
        } else {
            toggleBtn.classList.remove('active');
        }
    }
}
```

### addLogLine()
```javascript
/**
 * Add a log line to the console
 * Used for programmatic log injection (not file-based)
 * @param {string} line - Log line content
 * @param {'info'|'warning'|'error'|'debug'} [level='info'] - Log level
 */
addLogLine(line, level = 'info') {
    if (this.logSource === 'file') {
        return;
    }

    const timestamp = new Date().toLocaleTimeString();

    // Store the raw unfiltered line (preserve original content)
    const rawLine = line;

    const logEntry = {
        timestamp,
        line: rawLine,
        level,
        raw: rawLine
    };

    this.consoleLog.push(logEntry);

    // Keep only the most recent lines
    if (this.consoleLog.length > this.maxLogLines) {
        this.consoleLog = this.consoleLog.slice(-this.maxLogLines);
    }

    // Only render if console is visible - use throttled rendering for better performance
    if (this.consoleVisible) {
        this.scheduleConsoleUpdate();
    }
}
```

### addLogLines()
```javascript
/**
 * Add multiple log lines efficiently (for batch operations)
 * More performant than multiple addLogLine() calls
 * @param {Array<{line: string, level?: string}>} lines - Array of {line, level} objects
 */
addLogLines(lines) {
    const timestamp = new Date().toLocaleTimeString();

    // Add all lines at once
    const logEntries = lines.map(({line, level = 'info'}) => ({
        timestamp,
        line: line,
        level
    }));

    this.consoleLog.push(...logEntries);

    // Keep only the most recent lines
    if (this.consoleLog.length > this.maxLogLines) {
        this.consoleLog = this.consoleLog.slice(-this.maxLogLines);
    }

    // Only render once for all lines if console is visible - use throttled rendering
    if (this.consoleVisible) {
        this.scheduleConsoleUpdate();
    }
}
```

### renderConsoleLogOptimized()
```javascript
/**
 * Optimized console rendering using DocumentFragment for better performance
 * Renders last 500 filtered lines using batch DOM operations
 * @private
 */
renderConsoleLogOptimized() {
    const consoleContent = document.getElementById('console-content');
    if (!consoleContent) return;

    // Show last 500 lines for performance while allowing more content
    let displayLines = this.consoleLog.slice(-500);

    displayLines = displayLines.filter(entry => this.shouldShowLogEntry(entry));

    if (displayLines.length === 0) {
        // Fast path for empty console
        consoleContent.innerHTML = `
            <div class="console-line">
                <span class="console-timestamp">[${new Date().toLocaleTimeString()}]</span>
                <span class="console-text">${this.searchFilter ? 'No logs match filter' : 'Console log will appear here...'}</span>
            </div>
        `;
        return;
    }

    // Use DocumentFragment for efficient DOM manipulation
    const fragment = document.createDocumentFragment();

    // Batch create all elements before DOM insertion
    displayLines.forEach(entry => {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'console-line';

        const timestampSpan = document.createElement('span');
        timestampSpan.className = 'console-timestamp';
        timestampSpan.textContent = `[${entry.timestamp}]`;

        const textSpan = document.createElement('span');
        const levelClass = this.getLevelClass(entry.level);
        textSpan.className = `console-text ${levelClass}`;
        textSpan.textContent = entry.line; // textContent is safer than innerHTML

        lineDiv.appendChild(timestampSpan);
        lineDiv.appendChild(textSpan);
        fragment.appendChild(lineDiv);
    });

    // Single DOM write operation - much faster than multiple innerHTML writes
    consoleContent.textContent = ''; // Clear existing content efficiently
    consoleContent.appendChild(fragment);
}
```

### renderConsoleLog()
```javascript
/**
 * Legacy method for backwards compatibility
 * Delegates to renderConsoleLogOptimized()
 */
renderConsoleLog() {
    this.renderConsoleLogOptimized();
}
```

### getLevelClass()
```javascript
/**
 * Get CSS class for log level
 * Maps log level to corresponding CSS class for styling
 * @param {string} level - Log level
 * @returns {string} CSS class name
 * @private
 */
getLevelClass(level) {
    const normalizedLevel = level.toLowerCase();
    if (normalizedLevel.includes('error') || normalizedLevel.includes('exception')) {
        return 'error';
    } else if (normalizedLevel.includes('warning') || normalizedLevel.includes('warn')) {
        return 'warning';
    } else if (normalizedLevel.includes('info')) {
        return 'info';
    } else if (normalizedLevel.includes('debug')) {
        return 'debug';
    }
    return '';
}
```

### cleanAnsiCodes()
```javascript
/**
 * Remove ANSI escape codes from text
 * Strips color codes, cursor control, and other terminal escape sequences
 * @param {string} text - Text with ANSI codes
 * @returns {string} Cleaned text without ANSI codes
 */
cleanAnsiCodes(text) {
    // Remove ANSI escape sequences for colors, cursor control, etc.
    return text.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '')
              .replace(/\x1b\].*?\x07/g, '')
              .replace(/\x1b\[.*?[@-~]/g, '')
              .replace(/\u001b\[[0-9;]*[a-zA-Z]/g, '');
}
```

### escapeHtml()
```javascript
/**
 * Escape HTML entities in text
 * Prevents XSS by converting HTML special characters to entities
 * @param {string} text - Text to escape
 * @returns {string} HTML-escaped text
 */
escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
```

### scheduleConsoleUpdate()
```javascript
/**
 * Schedule a console update using requestAnimationFrame for optimal performance
 * Throttles updates to 60fps max to prevent browser overload
 * @private
 */
scheduleConsoleUpdate() {
    if (this.consoleUpdatePending) {
        return; // Already scheduled
    }

    this.consoleUpdatePending = true;

    // Use requestAnimationFrame for smooth 60fps updates
    requestAnimationFrame(() => {
        const now = performance.now();

        // Throttle updates to prevent overwhelming the browser
        if (now - this.lastRenderTime >= this.renderThrottleMs) {
            this.performConsoleUpdate();
            this.lastRenderTime = now;
        } else {
            // Schedule for later if we're updating too frequently
            setTimeout(() => {
                this.scheduleConsoleUpdate();
            }, this.renderThrottleMs - (now - this.lastRenderTime));
            return;
        }

        this.consoleUpdatePending = false;
    });
}
```

### performConsoleUpdate()
```javascript
/**
 * Perform the actual console update with optimized DOM manipulation
 * Called by scheduleConsoleUpdate() after throttle check
 * @private
 */
performConsoleUpdate() {
    this.renderConsoleLogOptimized();
    this.scrollToBottomOptimized();
}
```

### scrollToBottomOptimized()
```javascript
/**
 * Optimized scroll to bottom - single efficient operation
 * Scrolls console content to show most recent log entries
 * @private
 */
scrollToBottomOptimized() {
    const consoleContent = document.getElementById('console-content');
    if (consoleContent) {
        // Single efficient scroll operation
        consoleContent.scrollTop = consoleContent.scrollHeight;
    }
}
```

### scrollToBottom()
```javascript
/**
 * Legacy method for backwards compatibility
 * Delegates to scrollToBottomOptimized()
 */
scrollToBottom() {
    this.scrollToBottomOptimized();
}
```

### clearConsole()
```javascript
/**
 * Clear all console log entries
 * Removes all log entries from memory and re-renders empty console
 */
clearConsole() {
    this.consoleLog = [];
    this.renderConsoleLog();
}
```

### saveConsoleState()
```javascript
/**
 * Save console state to localStorage
 * Persists visibility and width settings across page reloads
 * @private
 */
saveConsoleState() {
    try {
        const state = {
            visible: this.consoleVisible,
            width: this.consoleWidth
        };
        localStorage.setItem('console-state', JSON.stringify(state));
    } catch (e) {
        console.warn('Failed to save console state:', e);
    }
}
```

### loadConsoleState()
```javascript
/**
 * Load console state from localStorage
 * Restores visibility and width settings from previous session
 * Supports both new and legacy storage formats
 * @private
 */
loadConsoleState() {
    try {
        // Try new format first
        const stored = localStorage.getItem('console-state');
        if (stored) {
            const state = JSON.parse(stored);
            this.consoleVisible = state.visible || false;
            this.consoleWidth = state.width || 350;
            this.updateConsoleVisibility();
            return;
        }

        // Fallback to old format
        const oldStored = localStorage.getItem('console-visible');
        if (oldStored !== null) {
            this.consoleVisible = JSON.parse(oldStored);
            this.updateConsoleVisibility();
        }
    } catch (e) {
        console.warn('Failed to load console state:', e);
    }
}
```

## DOMContentLoaded Event Handler (Lines 4780-4845)

```javascript
/**
 * DOMContentLoaded event handler
 * Initializes all major components when DOM is ready:
 * - Identity Manager (UUID-based user identification)
 * - Backend Connector (discovers backend container URL)
 * - Container Controls (wipe/reload functionality)
 * - Tool Registry (discovers available tools)
 * - Console Log Manager (log viewing sidebar)
 * - Main Application (Visual Response App)
 *
 * Establishes backend connection and triggers initial data loads
 */
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOM loaded, checking dependencies...');

    // Initialize Identity Manager (UUID-based user identification)
    if (window.IdentityManager) {
        window.identityManager = new window.IdentityManager();
        console.log('Identity Manager initialized');
    }

    // Initialize Backend Connector (discovers backend container)
    if (window.BackendConnector && window.identityManager) {
        window.backendConnector = new window.BackendConnector(window.identityManager);
        console.log('Backend Connector initialized');

        // Start discovery immediately (don't wait for first API call)
        window.backendConnector.waitForReady().then(() => {
            console.log('Backend ready, triggering API reloads...');
            // Reload APIs now that backend is ready
            if (window.ToolRegistry) {
                window.ToolRegistry.initialize(true); // Force reload
            }
            if (window.ToolManager) {
                window.ToolManager.refresh();
            }
            if (window.app) {
                window.app.fetchAgentState();
                window.app.loadProfileSlots();
                // Reload LLM models for all selectors
                [window.app.plannerLLMSelector, window.app.routerLLMSelector,
                 window.app.overrideLLMSelector, window.app.primaryLLMSelector,
                 window.app.fallback1LLMSelector, window.app.fallback2LLMSelector
                ].forEach(selector => {
                    if (selector) selector.loadModelsFromServer();
                });
            }
            if (window.chatState) {
                window.chatState.loadCurrentChat();
            }
            if (window.chatHistory) {
                window.chatHistory.loadChats();
            }
        });

        // Start keep-alive pings (update activity every minute)
        window.backendConnector.startActivityUpdates(60000);
    }

    // Initialize Container Controls (wipe/reload functionality)
    if (window.ContainerControls && window.identityManager && window.backendConnector) {
        window.containerControls = new window.ContainerControls(window.identityManager, window.backendConnector);
        console.log('Container Controls initialized');
    }

    if (window.ToolRegistry) {
        console.log('Initializing ToolRegistry...');
        await window.ToolRegistry.initialize();
    }

    // Initialize console log manager
    consoleLogManager = new ConsoleLogManager();

    // Make it globally accessible for WebSocket messages
    window.consoleLogManager = consoleLogManager;

    initializeApp();
});
```

## Summary

This document provides complete JSDoc type annotations for:
- **2 VisualResponseApp methods** (scheduleAutoSave, showAutoplayPrompt)
- **4 global functions** (initializeApp, sendPermissionResponse, sendInterruptSignal, toggleToolDetail)
- **34 ConsoleLogManager methods** (complete class documentation)
- **1 DOMContentLoaded event handler**

All methods now include:
- `@param` tags with full type information
- `@returns` tags with return types
- `@async` markers for async methods
- `@private` markers for internal methods
- Clear descriptions of functionality
- Implementation notes where relevant

The annotations follow JSDoc best practices and provide comprehensive type information for IDE autocomplete and documentation generation.
