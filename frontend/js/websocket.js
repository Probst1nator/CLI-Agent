/**
 * WebSocket Communication Module
 *
 * Manages real-time bidirectional communication between the browser client
 * and the CLI-Agent visual server.
 *
 * @class WebSocketManager
 * @author CLI-Agent Development Team
 * @version 1.1.0 (Simplified)
 */
class WebSocketManager {
    /**
     * Creates a WebSocketManager instance
     * @param {ContentRenderer} renderer - Content renderer instance for displaying messages
     * @param {HTMLElement} statusElement - DOM element for connection status display
     * @param {HTMLElement} contentWrapper - DOM element for message content display
     */
    constructor(renderer, statusElement, contentWrapper) {
        /** @type {ContentRenderer} */
        this.renderer = renderer;

        /** @type {HTMLElement} */
        this.statusElement = statusElement;

        /** @type {HTMLElement} */
        this.contentWrapper = contentWrapper;

        /** @type {WebSocket|null} WebSocket connection instance */
        this.ws = null;

        /** @type {number} Base delay in milliseconds before reconnection attempt */
        this.reconnectDelay = 3000;

        /** @type {number} Counter for reconnection attempts */
        this.reconnectAttempts = 0;

        /** @type {number} Maximum number of reconnection attempts before giving up */
        this.maxReconnectAttempts = 10;

        /** @type {string|null} Unique identifier for current connection */
        this.connectionId = null;

        /** @type {Logger} Reference to global logger instance */
        this.logger = window.visualLogger;

        /** @type {Object|null} Reference to main app for loading indicator control */
        this.app = null;

        /** @type {Object.<string, Function[]>} Event handlers registry for custom events */
        this.eventHandlers = {};

        /** @type {string|null} Last console log message for rate limiting */
        this._lastConsoleLog = null;

        /** @type {number|null} Timestamp of last console log for rate limiting */
        this._lastConsoleTime = null;

        this.logger.info('WebSocketManager initialized', {
            reconnectDelay: this.reconnectDelay,
            maxReconnectAttempts: this.maxReconnectAttempts
        });

        this.connect();
    }

    /**
     * Register an event handler for a specific event type
     * @param {string} event - Event name to listen for
     * @param {Function} callback - Function to call when event is emitted
     * @event
     */
    on(event, callback) {
        if (!this.eventHandlers[event]) {
            this.eventHandlers[event] = [];
        }
        this.eventHandlers[event].push(callback);
    }

    /**
     * Emit an event to all registered handlers
     * @param {string} event - Event name to emit
     * @param {*} data - Data to pass to event handlers
     * @event
     */
    emit(event, data) {
        if (this.eventHandlers[event]) {
            for (const callback of this.eventHandlers[event]) {
                try {
                    callback(data);
                } catch (error) {
                    this.logger.error(`Error in ${event} handler:`, {}, error);
                }
            }
        }
    }

    /**
     * Set reference to main app for loading indicator control
     * @param {Object} app - Main application instance
     * @param {Function} [app.loadMessageHistory] - Optional method to load message history
     * @param {Function} [app.clearMessageHistory] - Optional method to clear message history
     * @param {Function} [app.hideLoadingIndicator] - Optional method to hide loading indicator
     * @param {Function} [app.hideInterruptButton] - Optional method to hide interrupt button
     * @param {Function} [app.displayUserMessage] - Optional method to display user messages
     * @param {Function} [app.displayAssistantMessage] - Optional method to display assistant messages
     * @param {Function} [app.setScreenshotReady] - Optional method to set screenshot ready state
     * @param {Function} [app.resetScreenshotState] - Optional method to reset screenshot state
     * @param {Function} [app.saveMessageToHistory] - Optional method to save message to history
     * @param {boolean} [app.isRestoring] - Flag indicating if app is restoring state
     * @param {Array} [app.messageHistory] - Message history array
     */
    setApp(app) {
        this.app = app;
    }

    /**
     * Establishes WebSocket connection with automatic URL discovery and reconnection
     * Waits for BackendConnector to be ready, then attempts connection with exponential backoff on failure
     * @async
     * @returns {Promise<void>}
     * @throws {Error} If WebSocket creation fails after max reconnection attempts
     */
    async connect() {
        // Wait for backend connector to be ready (max 10 seconds)
        if (!window.backendConnector) {
            let waitCount = 0;
            while (!window.backendConnector && waitCount < 50) {
                await new Promise(resolve => setTimeout(resolve, 200));
                waitCount++;
            }
        }

        // Get WebSocket URL from BackendConnector if available
        let wsUrl;
        if (window.backendConnector) {
            try {
                wsUrl = await window.backendConnector.getWebSocketURL();
                this.logger.info('Got WebSocket URL from backend connector', { wsUrl });
            } catch (error) {
                this.logger.error('Backend discovery failed, falling back to default URL', {}, error);
                // Fallback to default local URL
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsPort = window.location.port || (wsProtocol === 'wss:' ? 443 : 80);
                wsUrl = `${wsProtocol}//${window.location.hostname}:${wsPort}/ws`;
            }
        } else {
            // Legacy mode: no backend connector (local mode)
            this.logger.warn('No backend connector found, using default WebSocket URL');
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsPort = window.location.port || (wsProtocol === 'wss:' ? 443 : 80);
            wsUrl = `${wsProtocol}//${window.location.hostname}:${wsPort}/ws`;
        }

        this.logger.startTiming('websocket_connection');
        this.logger.info('Attempting WebSocket connection', { fullUrl: wsUrl, attempt: this.reconnectAttempts + 1 });

        try {
            this.ws = new WebSocket(wsUrl);
            this.connectionId = 'conn_' + Date.now();
        } catch (error) {
            this.logger.error('Failed to create WebSocket', { wsUrl }, error);
            this.handleConnectionError(error);
            return;
        }

        // WebSocket connection opened successfully
        this.ws.onopen = () => {
            this.logger.endTiming('websocket_connection', { connectionId: this.connectionId, reconnectAttempts: this.reconnectAttempts });
            this.reconnectAttempts = 0;
            this.statusElement.textContent = '🟢 Connected to Agent';
            this.statusElement.style.color = 'var(--status-connected)';
            this.logger.info('WebSocket connected successfully', { connectionId: this.connectionId });

            // History loading will now be triggered by config message from server
            // This ensures we respect the continue flag from the CLI
        };

        // Handle incoming WebSocket messages
        this.ws.onmessage = (event) => this.handleMessage(JSON.parse(event.data));

        // Handle WebSocket connection closure with exponential backoff reconnection
        this.ws.onclose = (event) => {
            this.logger.warn('WebSocket connection closed', { code: event.code, reason: event.reason });
            this.statusElement.textContent = '🔴 Disconnected. Retrying...';
            this.statusElement.style.color = 'var(--status-disconnected)';

            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                // Exponential backoff: 3s, 6s, 12s, 24s, capped at 30s
                const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
                setTimeout(() => this.connect(), delay);
            } else {
                this.logger.error('Max reconnection attempts reached.');
                this.statusElement.textContent = '🔴 Connection failed';
            }
        };

        // Handle WebSocket errors
        this.ws.onerror = (error) => this.handleConnectionError(error);
    }

    /**
     * Handles WebSocket connection errors
     * @private
     * @param {Event} error - WebSocket error event
     */
    handleConnectionError(error) {
        this.logger.error('WebSocket connection error', {}, error);
        if (this.ws) {
            this.ws.close();
        }
    }

    /**
     * Handles session initialization from server
     * Stores session ID globally and emits session_init event
     * @private
     * @param {Object} data - Session initialization data
     * @param {string} data.session_id - Unique session identifier
     * @event session_init
     */
    handleSessionInit(data) {
        this.logger.info('Session initialized', { session_id: data.session_id });

        window.sessionId = data.session_id;

        this.statusElement.textContent = '🟢 Connected to Agent';
        this.statusElement.style.color = 'var(--status-connected)';

        this.logger.info('Session ID stored for HTTP requests', { session_id: data.session_id });

        this.emit('session_init', data);
    }

    /**
     * Handles connection configuration from server
     * Manages continue mode (load history) vs fresh start (clear history)
     * @private
     * @param {Object} data - Connection configuration
     * @param {boolean} data.continue_mode - Whether to continue previous session
     * @param {boolean} data.clear_history - Whether to clear existing history
     */
    handleConnectionConfig(data) {
        this.logger.info('Received connection configuration', { continue_mode: data.continue_mode, clear_history: data.clear_history });

        // Clear history if requested (fresh chat)
        if (data.clear_history) {
            this.logger.info('Clearing message history (fresh chat detected)');
            try {
                localStorage.removeItem('agent-message-history');
                if (this.app && this.app.messageHistory) {
                    this.app.messageHistory = [];
                }
            } catch (e) {
                this.logger.warn('Failed to clear message history:', e);
            }
        }

        // Update status element to show session type
        if (data.continue_mode) {
            this.statusElement.textContent = '🟢 Reconnected to Agent';
            this.logger.info('Continue mode enabled - will load message history');

            // Load message history since continue mode is enabled
            if (this.app && typeof this.app.loadMessageHistory === 'function') {
                setTimeout(() => this.app.loadMessageHistory(), 500);
            }
        } else {
            this.statusElement.textContent = '🟢 Connected to Agent';
            this.logger.info('Fresh start mode - skipping message history loading');

            // Clear any existing history since we're starting fresh
            if (this.app && typeof this.app.clearMessageHistory === 'function') {
                this.app.clearMessageHistory();
            }

            // Clear the display
            if (this.contentWrapper) {
                this.contentWrapper.innerHTML = '';
            }
        }
    }

    /**
     * Handles permission request from agent
     * Creates interactive UI for user to approve/deny tool actions
     * @private
     * @param {Object} data - Permission request data
     * @param {string} data.id - Unique permission request ID
     * @param {Object} data.permission_data - Permission details
     * @param {string} data.permission_data.tool_name - Name of tool requesting permission
     * @param {string} data.permission_data.action_description - Description of action
     * @param {string} [data.permission_data.file_path] - Optional file path affected
     * @param {string} [data.permission_data.args_display] - Optional command arguments
     */
    handlePermissionRequest(data) {
        this.logger.info('Received permission request', { tool_name: data.permission_data.tool_name });

        const permissionData = data.permission_data;
        const messageId = 'permission-' + data.id;

        // Create permission request UI
        const messageBlock = document.createElement('div');
        messageBlock.className = 'message-block permission-request';
        messageBlock.id = messageId;

        const timestamp = new Date().toLocaleTimeString();

        // Create header
        const headerDiv = document.createElement('div');
        headerDiv.className = 'permission-header';
        headerDiv.innerHTML = `
            <span class="permission-timestamp">${timestamp} - Permission Request</span>
            <span class="permission-tool">🔒 ${permissionData.tool_name}</span>
        `;

        // Create permission content
        const contentDiv = document.createElement('div');
        contentDiv.className = 'permission-content';

        // Tool information
        const toolEmojis = {
            "editfile": "✏️", "readfile": "📖", "writefile": "📝",
            "insertline": "➕", "deletelines": "🗑️", "bash": "🔧",
            "python": "🐍", "searchweb": "🌐", "showuser": "👁️",
            "memory": "🧠", "grepsearch": "🔍"
        };
        const toolEmoji = toolEmojis[permissionData.tool_name] || "⚙️";

        let contentHTML = `
            <div class="permission-info">
                <h3>${toolEmoji} ${permissionData.tool_name} permission required</h3>
                <p><strong>Action:</strong> ${permissionData.action_description}</p>
        `;

        // ML Score Indicator
        if (permissionData.ml_enabled && permissionData.ml_score !== null && permissionData.ml_score !== undefined) {
            const mlScore = permissionData.ml_score;
            const mlConfidence = permissionData.ml_confidence || 0;

            // Determine color, icon, and text based on score
            let scoreColor, scoreIcon, scoreText, confidencePercent;
            if (mlScore > 0.01) {
                scoreColor = '#10b981';  // green
                scoreIcon = '✅';
                scoreText = 'SAFE';
                confidencePercent = Math.min(mlConfidence * 100, 100);
            } else if (mlScore < -0.01) {
                scoreColor = '#ef4444';  // red
                scoreIcon = '⚠️';
                scoreText = 'RISKY';
                confidencePercent = Math.min(mlConfidence * 100, 100);
            } else {
                scoreColor = '#f59e0b';  // amber
                scoreIcon = '❓';
                scoreText = 'UNCERTAIN';
                confidencePercent = 50;
            }

            // Calculate position on score bar (map -1 to +1 → 0% to 100%)
            const scorePosition = ((mlScore + 1) / 2) * 100;

            contentHTML += `
                <div class="ml-score-container">
                    <div class="ml-score-header">
                        <span class="ml-score-label">🤖 ML Safety Score:</span>
                        <span class="ml-score-value" style="color: ${scoreColor}">
                            ${scoreIcon} ${scoreText} (${mlScore.toFixed(3)})
                        </span>
                    </div>
                    <div class="ml-score-bar-container">
                        <div class="ml-score-bar">
                            <div class="ml-score-indicator" style="left: ${scorePosition}%; background-color: ${scoreColor}"></div>
                        </div>
                        <div class="ml-score-markers">
                            <span class="ml-score-marker" style="left: 0%">-1.0</span>
                            <span class="ml-score-marker" style="left: 25%">-0.5</span>
                            <span class="ml-score-marker" style="left: 50%">0</span>
                            <span class="ml-score-marker" style="left: 75%">+0.5</span>
                            <span class="ml-score-marker" style="left: 100%">+1.0</span>
                        </div>
                    </div>
                    <div class="ml-confidence-container">
                        <span class="ml-confidence-label">Confidence:</span>
                        <div class="ml-confidence-bar">
                            <div class="ml-confidence-fill" style="width: ${confidencePercent}%; background-color: ${scoreColor}"></div>
                        </div>
                        <span class="ml-confidence-value">${confidencePercent.toFixed(0)}%</span>
                    </div>
                    ${permissionData.ml_reason ? `
                    <div class="ml-reason">
                        <details>
                            <summary>Why this score?</summary>
                            <p>${this.escapeHtml(permissionData.ml_reason)}</p>
                        </details>
                    </div>
                    ` : ''}
                </div>
            `;
        } else if (permissionData.ml_enabled) {
            contentHTML += `
                <div class="ml-score-container ml-disabled">
                    <p>⚙️ ML classifier enabled but no score available</p>
                </div>
            `;
        }

        if (permissionData.file_path) {
            contentHTML += `<p><strong>File:</strong> ${permissionData.file_path}</p>`;
        }

        // Always show args (truncated to fit the permission box width)
        const argsText = permissionData.args_display || 'No arguments';
        const truncatedArgs = this.truncateText(argsText, 100);
        contentHTML += `<p><strong>Arguments:</strong> <code class="permission-args">${this.escapeHtml(truncatedArgs)}</code></p>`;

        contentHTML += `</div>`;

        // Create action buttons
        contentHTML += `
            <div class="permission-actions">
                <button class="permission-btn allow" onclick="window.sendPermissionResponse('${data.id}', 'allow')">
                    ✅ Allow
                </button>
                <button class="permission-btn deny" onclick="window.sendPermissionResponse('${data.id}', 'deny')">
                    ❌ Deny
                </button>
                <button class="permission-btn configure-autonomy" onclick="window.showAutonomyConfiguration('${data.id}')">
                    ⚙️ Configure Autonomy
                </button>
            </div>
        `;

        // Add autonomy configuration panel (hidden by default)
        contentHTML += `
            <div id="autonomy-config-${data.id}" class="autonomy-config-panel" style="display: none;">
                <div class="autonomy-config-header">
                    <h4>⚙️ Configure Autonomy for ${permissionData.tool_name}</h4>
                    <p class="autonomy-config-description">${permissionData.action_description || `Configure permission for ${permissionData.tool_name}`}</p>
                </div>

                <div class="autonomy-config-body">
                    <!-- General Permission Section -->
                    <div class="autonomy-general-section">
                        <h5>General Permission</h5>
                        <p class="autonomy-hint">Apply to all uses of ${permissionData.tool_name}</p>
                        <div class="tool-permission-controls">
                            <label class="permission-radio">
                                <input type="radio" name="autonomy-${data.id}" value="allow">
                                <span class="permission-label allow">Allow</span>
                            </label>
                            <label class="permission-radio">
                                <input type="radio" name="autonomy-${data.id}" value="ask" checked>
                                <span class="permission-label ask">Ask</span>
                            </label>
                            <label class="permission-radio">
                                <input type="radio" name="autonomy-${data.id}" value="deny">
                                <span class="permission-label deny">Deny</span>
                            </label>
                        </div>
                    </div>`;

        // Add specific scope options if available
        if (permissionData.rule_options && permissionData.rule_options.length > 0) {
            contentHTML += `
                    <!-- Advanced Scope Section -->
                    <div class="autonomy-advanced-section">
                        <details class="autonomy-advanced-details">
                            <summary>Or choose specific scope</summary>
                            <p class="autonomy-hint">Select a specific pattern for fine-grained control</p>
                            <div class="autonomy-scope-options">
            `;

            permissionData.rule_options.forEach((option) => {
                const escapedRule = option.rule.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
                const escapedDescription = option.description.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
                contentHTML += `
                                <div class="autonomy-scope-option" data-rule="${escapedRule}" data-permission-id="${data.id}">
                                    <div class="scope-option-description">${escapedDescription}</div>
                                    <div class="scope-option-rule"><code>${option.rule}</code></div>
                                </div>
                `;
            });

            contentHTML += `
                            </div>
                        </details>
                    </div>`;
        }

        contentHTML += `
                </div>

                <div class="autonomy-config-actions">
                    <button class="permission-btn save-autonomy" onclick="window.saveAutonomyConfiguration('${data.id}')">
                        💾 Save
                    </button>
                    <button class="permission-btn cancel-autonomy" onclick="window.hideAutonomyConfiguration('${data.id}')">
                        ↩️ Cancel
                    </button>
                </div>
            </div>
        `;

        contentDiv.innerHTML = contentHTML;

        messageBlock.appendChild(headerDiv);
        messageBlock.appendChild(contentDiv);

        this.contentWrapper.appendChild(messageBlock);
        this.scrollToBottom();

        // Add event listeners for autonomy scope options
        const scopeOptions = messageBlock.querySelectorAll('.autonomy-scope-option');
        scopeOptions.forEach(option => {
            option.addEventListener('click', () => {
                // Select this option visually
                scopeOptions.forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');
            });
        });

        // Store the permission ID for response handling
        window.pendingPermissionId = data.id;
    }

    /**
     * Handles user message from command line
     * Delegates to app's displayUserMessage or uses fallback rendering
     * @private
     * @param {Object} data - User message data
     * @param {string} data.message - User's message text
     */
    handleUserMessage(data) {
        this.logger.info('Received user message from command line', { message: data.message });

        // Use the app's displayUserMessage function if available
        if (this.app && typeof this.app.displayUserMessage === 'function') {
            this.app.displayUserMessage(data.message);
        } else {
            // Fallback: create user message directly
            this.createUserMessageFallback(data.message);
        }
    }

    /**
     * Handles assistant message from agent
     * Delegates to app's displayAssistantMessage or uses fallback rendering
     * @private
     * @param {Object} data - Assistant message data
     * @param {string} data.message - Assistant's message text
     */
    handleAssistantMessage(data) {
        this.logger.debug('Received assistant message', { message: data.message });

        // Use the app's displayAssistantMessage function if available
        if (this.app && typeof this.app.displayAssistantMessage === 'function') {
            this.app.displayAssistantMessage(data.message);
        } else {
            // Fallback: create assistant message directly
            this.createAssistantMessageFallback(data.message);
        }
    }

    /**
     * Handles console log message from backend
     * Forwards to ConsoleLogManager or browser console with rate limiting
     * @private
     * @param {Object} data - Console log data
     * @param {string} data.line - Log message text
     * @param {string} [data.level='info'] - Log level (info, warn, error, debug)
     */
    handleConsoleLog(data) {
        this.logger.debug('Received console log message', { line: data.line, level: data.level });

        // Forward to console log manager if available
        if (window.consoleLogManager) {
            window.consoleLogManager.addLogLine(data.line, data.level || 'info');
        } else {
            // Fallback: log to browser console with rate limiting
            if (!this._lastConsoleLog || this._lastConsoleLog !== data.line || Date.now() - this._lastConsoleTime > 1000) {
                console.log(`[CLI-Agent Console] ${data.line}`);
                this._lastConsoleLog = data.line;
                this._lastConsoleTime = Date.now();
            }
        }
    }

    /**
     * Handles screenshot event from backend
     * Updates screenshot button state in main app
     * @private
     * @param {Object} data - Screenshot event data
     * @param {string} data.action - Action type (captured, failed)
     * @param {string} data.status - Status (ready, error)
     */
    handleScreenshotEvent(data) {
        this.logger.debug('Received screenshot event', { action: data.action, status: data.status });

        // Forward to main app for screenshot button state management
        if (this.app) {
            if (data.action === 'captured' || data.status === 'ready') {
                this.app.setScreenshotReady();
            } else if (data.action === 'failed' || data.status === 'error') {
                this.app.resetScreenshotState();
            }
        } else {
            this.logger.warn('No app reference available for screenshot state management');
        }
    }


    /**
     * Fallback method to display user message when app.displayUserMessage is unavailable
     * Creates and appends user message block directly to content wrapper
     * @private
     * @param {string} message - User message text
     */
    createUserMessageFallback(message) {
        // Fallback implementation in case app.displayUserMessage is not available
        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper) return;

        const currentTimestamp = Date.now();

        // Create message block
        const messageBlock = document.createElement('div');
        messageBlock.className = 'message-block user-message';

        // Create timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'agent-msg-head';
        timestamp.textContent = new Date(currentTimestamp).toLocaleTimeString();

        // Create content
        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = message;

        // Assemble message
        messageBlock.appendChild(timestamp);
        messageBlock.appendChild(content);

        // Add to content wrapper
        contentWrapper.appendChild(messageBlock);

        // Scroll to bottom
        this.scrollToBottom();

        this.logger.debug('User message displayed via fallback method');
    }

    /**
     * Fallback method to display assistant message when app.displayAssistantMessage is unavailable
     * Creates and appends assistant message block with monospace formatting
     * @private
     * @param {string} message - Assistant message text
     */
    createAssistantMessageFallback(message) {
        // Fallback implementation for assistant messages
        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper) return;

        const currentTimestamp = Date.now();

        // Create message block
        const messageBlock = document.createElement('div');
        messageBlock.className = 'message-block assistant-message';

        // Create timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'agent-msg-head';
        timestamp.textContent = new Date(currentTimestamp).toLocaleTimeString();

        // Create content
        const content = document.createElement('div');
        content.className = 'message-content';

        // For assistant messages, preserve formatting and show XML tags properly
        const pre = document.createElement('pre');
        pre.style.cssText = 'white-space: pre-wrap; font-family: monospace; background: #f5f5f5; padding: 10px; border-radius: 4px; margin: 0;';
        pre.textContent = message;
        content.appendChild(pre);

        // Assemble message
        messageBlock.appendChild(timestamp);
        messageBlock.appendChild(content);

        // Add to content wrapper
        contentWrapper.appendChild(messageBlock);

        // Scroll to bottom
        this.scrollToBottom();

        this.logger.debug('Assistant message displayed via fallback method');
    }

    /**
     * Main WebSocket message handler - routes messages to appropriate handlers
     * Handles various message types including session init, chat events, streaming, permissions, etc.
     * @async
     * @private
     * @param {Object} data - WebSocket message data
     * @param {string} data.type - Message type (session_init, chat_replaced, message_start, etc.)
     * @param {string} [data.action] - Action type (create, update, delete, clear, etc.)
     * @param {string} [data.id] - Message or event unique identifier
     * @param {string} [data.model] - LLM model key used for this message
     * @param {Object} [data.content] - Message content for rendering
     * @param {Object} [data.tool_info] - Tool execution information
     * @event session_init
     * @event chat_replaced
     * @event chat_cleared
     * @event chat_truncated
     * @event message_added
     * @event message_deleted
     * @event state_changed
     * @event message_start
     * @event message_chunk
     * @event message_end
     * @event token_stream
     * @event tool_update
     */
    async handleMessage(data) {
        this.logger.debug('Received WebSocket message', { type: data.type, action: data.action, id: data.id, model: data.model });

        if (data.type === 'session_init') {
            this.handleSessionInit(data);
            return;
        }

        if (data.type === 'chat_replaced') {
            this.emit('chat_replaced', data);
            return;
        }

        if (data.type === 'chat_cleared') {
            this.emit('chat_cleared', data);
            return;
        }

        if (data.type === 'chat_truncated') {
            this.emit('chat_truncated', data);
            return;
        }

        if (data.type === 'message_added') {
            this.emit('message_added', data);
            return;
        }

        if (data.type === 'message_deleted') {
            this.emit('message_deleted', data);
            return;
        }

        if (data.type === 'state_changed') {
            this.emit('state_changed', data);
            return;
        }

        if (data.type === 'message_edited') {
            this.handleMessageEdited(data);
            return;
        }

        if (data.type === 'message_start') {
            this.emit('message_start', data);
            return;
        }

        if (data.type === 'message_chunk') {
            this.emit('message_chunk', data);
            return;
        }

        if (data.type === 'message_end') {
            this.emit('message_end', data);
            return;
        }

        // Handle token stream events (real-time LLM generation)
        if (data.type === 'token_stream') {
            this.emit('token_stream', data);
            return;
        }

        // Handle tool update events (tool execution status)
        if (data.type === 'tool_update') {
            this.emit('tool_update', data);
            return;
        }

        // Handle model failure events (model execution failures)
        if (data.type === 'model_failure') {
            this.emit('model_failure', data);
            return;
        }

        // Handle prompts_regenerated events (suggested prompts updated)
        if (data.type === 'prompts_regenerated') {
            this.emit('prompts_regenerated', data);
            return;
        }

        // Handle voice_transcription events (from CLI voice input)
        if (data.type === 'voice_transcription') {
            this.emit('voice_transcription', data);
            return;
        }

        // Handle configuration messages
        if (data.type === 'config' && data.action === 'connection_init') {
            this.handleConnectionConfig(data);
            return;
        }

        // Handle permission requests
        if (data.type === 'permission_request') {
            this.handlePermissionRequest(data);
            return;
        }

        // Handle user messages from command line
        if (data.type === 'user_message' && data.action === 'display') {
            this.handleUserMessage(data);
            return;
        }

        // Handle assistant messages
        if (data.type === 'assistant_message' && data.action === 'display') {
            this.handleAssistantMessage(data);
            return;
        }

        // Handle console log messages
        if (data.type === 'console_log') {
            this.handleConsoleLog(data);
            return;
        }

        // Handle screenshot events
        if (data.type === 'screenshot') {
            this.handleScreenshotEvent(data);
            return;
        }

        // Handle live tool notifications (legacy)
        if (data.action === 'tool_start') {
            this.handleLiveToolStart(data);
            return;
        }

        if (data.action === 'tool_complete') {
            this.handleLiveToolComplete(data);
            return;
        }


        // Handle loading area tool updates
        if (data.action === 'loading_tool_start') {
            this.handleLoadingToolStart(data);
            return;
        }

        if (data.action === 'loading_tool_complete') {
            this.handleLoadingToolComplete(data);
            return;
        }

        // Handle WebSocket TTS audio streaming
        if (data.action === 'audio_tts' && data.type === 'audio') {
            this.handleWebSocketAudio(data);
            return;
        }

        // Only hide loading indicator when visualresponse/response is received or on error
        const shouldHideLoadingIndicator =
            data.type === 'html' ||
            data.type === 'markdown' ||
            data.action === 'error' ||
            (data.content && (
                data.content.includes('<visualresponse>') ||
                data.content.includes('<response>')
            ));

        if (shouldHideLoadingIndicator && this.app && typeof this.app.hideLoadingIndicator === 'function') {
            this.app.hideLoadingIndicator();
            console.log('✅ Loading indicator hidden - final response received');
        }

        // Hide interrupt button when agent response is complete
        if (shouldHideLoadingIndicator && this.app && typeof this.app.hideInterruptButton === 'function') {
            this.app.hideInterruptButton();
        }

        // Clean up any remaining live tools container when regular message arrives
        this.cleanupLiveToolsContainer();

        // Transfer loading tools to visualresponse header when message arrives
        this.transferLoadingToolsToMessage(data.id, data.tool_info);

        // Clean up any orphaned loading tools header after transfer
        setTimeout(() => {
            this.performMaintenanceCleanup();
        }, 500);

        // If ChatStateManager exists, skip legacy rendering entirely
        // ChatStateManager will handle message rendering via message_added events
        // The legacy renderer is only for fallback when ChatStateManager is not present
        if (window.chatState && typeof window.chatState.addMessage === 'function') {
            this.logger.debug('ChatStateManager present, skipping legacy message renderer');
            return;
        }

        const messageId = 'message-' + data.id;
        const existingBlock = document.getElementById(messageId);

        if (data.action === 'delete') {
            if (existingBlock) {
                existingBlock.remove();
                this.logger.info(`Deleted message block: ${messageId}`);
            }
            return;
        }

        if (data.action === 'clear') {
            // Don't clear during restoration
            if (this.app && this.app.isRestoring) {
                this.logger.info('Skipping clear action during restoration');
                return;
            }
            this.contentWrapper.innerHTML = '';
            // Also clear the message history when clearing the display
            if (this.app && typeof this.app.clearMessageHistory === 'function') {
                this.app.clearMessageHistory();
            }
            this.logger.info('Cleared all visual responses.');
            return;
        }
        
        // ** SIMPLIFIED LOGIC: No more 'update' action **
        // All other actions (like 'create') will generate a new block or replace an existing one.
        
        this.logger.startTiming(`render_message_${messageId}`);
        
        let messageBlock = existingBlock;
        if (!messageBlock) {
            messageBlock = document.createElement('div');
            messageBlock.className = 'message-block agent-message';
            messageBlock.id = messageId;

            // Always append new messages chronologically
            this.contentWrapper.appendChild(messageBlock);
        } else {
            // If block exists, ensure it's positioned correctly (for updates to existing messages)
            // But for new agent responses, we should have unique IDs that prevent this case
            this.logger.debug(`Reusing existing message block: ${messageId}`);
        }
        
        const timestamp = new Date().toLocaleTimeString();
        
        // Create timestamp header with flex layout for potential JS toggle and tool info
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'agent-msg-head';

        // Base timestamp content with model key right after ID
        let timestampContent = `<span class="timestamp-text">${timestamp} - ID: ${data.id}`;

        // Add model key if available, right after the ID
        if (data.model) {
            timestampContent += ` <span class="model-key" title="Model used for this response">[${data.model}]</span>`;
            this.logger.debug('Added model key to timestamp', { model: data.model });
        } else {
            this.logger.debug('No model key found in message data', { dataKeys: Object.keys(data) });
        }

        timestampContent += `</span>`;

        // Tool info is now handled by loading-tools-header, not in timestamp

        timestampDiv.innerHTML = timestampContent;
        messageBlock.appendChild(timestampDiv);

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageBlock.appendChild(messageContent);
        
        await this.renderer.renderContent(messageContent, data, messageId, timestampDiv);

        // Save agent message to history if main app is available (but not during restoration)
        // IMPORTANT: Do not save permission requests or other transient UI elements to history
        if (this.app && typeof this.app.saveMessageToHistory === 'function' && !this.app.isRestoring) {
            // Filter out transient message types that should not be persisted
            // FIXED: Removed 'user_message' from transient types - this was incorrectly preventing user message storage
            // Audio types excluded to prevent replaying all audio on page reload
            const transientTypes = ['permission_request', 'config', 'tts_chunk'];
            const isAudioMessage = data.action === 'audio_tts' && data.type === 'audio';

            if (!transientTypes.includes(data.type) && !isAudioMessage) {
                this.app.saveMessageToHistory('agent', {
                    originalId: data.id,
                    timestamp: Date.now(),
                    messageData: data
                });
            }
        }

        // Auto-scroll to bottom for consistency with user messages
        this.scrollToBottom();
        this.logger.endTiming(`render_message_${messageId}`, { success: true, contentType: data.type });
    }

    /**
     * Unified autoscroll function to ensure consistent chat scrolling behavior
     * Uses a combination of methods for maximum compatibility and smooth scrolling
     * Implements aggressive multi-attempt scrolling to handle dynamic content rendering
     */
    scrollToBottom() {
        if (!this.contentWrapper) return;

        // Aggressive scroll to very bottom
        const forceScrollToEnd = () => {
            // Multiple attempts to ensure we reach the absolute bottom
            this.contentWrapper.scrollTop = this.contentWrapper.scrollHeight + 100;
            this.contentWrapper.scrollTo(0, this.contentWrapper.scrollHeight + 100);

            // Force scroll to actual maximum
            this.contentWrapper.scrollTop = this.contentWrapper.scrollHeight;

            // Use scrollIntoView on last element
            const lastChild = this.contentWrapper.lastElementChild;
            if (lastChild) {
                try {
                    lastChild.scrollIntoView({ behavior: 'instant', block: 'end' });
                } catch (e) {
                    lastChild.scrollIntoView(false);
                }
            }
        };

        // Immediate scroll
        forceScrollToEnd();

        // Use requestAnimationFrame to ensure DOM has been updated
        requestAnimationFrame(() => {
            forceScrollToEnd();
            // Additional delayed scroll to handle async rendering
            setTimeout(forceScrollToEnd, 10);
            setTimeout(forceScrollToEnd, 50);
        });
    }

    /**
     * Sends a message through the WebSocket connection
     * @param {string} message - Message to send (typically JSON stringified)
     * @returns {boolean} True if message was sent, false if connection is not open
     */
    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(message);
            this.logger.info('Message sent to agent', { messageLength: message.length });
            return true;
        }
        this.logger.warn('Cannot send message - WebSocket not connected.');
        return false;
    }

    /**
     * Handles message edit events from backend
     * Updates existing message content and shows visual feedback
     * @private
     * @param {Object} data - Message edit data
     * @param {Object} data.message - Edited message object
     * @param {number} data.message.index - Index of message in chat history
     * @param {string} data.message.content - New message content
     */
    handleMessageEdited(data) {
        const {message} = data;
        if (!message) return;

        const messageBlock = document.querySelector(`[data-message-index="${message.index}"]`);
        if (!messageBlock) {
            console.warn('Message block not found for index:', message.index);
            return;
        }

        const contentDiv = messageBlock.querySelector('.message-content');
        if (contentDiv) {
            contentDiv.textContent = message.content;

            // Visual feedback for edit
            messageBlock.classList.add('message-edited');
            setTimeout(() => messageBlock.classList.remove('message-edited'), 2000);

            this.logger.info('Message edited remotely', {index: message.index});
        }
    }

    /**
     * Checks if WebSocket connection is currently open
     * @returns {boolean} True if connected and ready to send messages
     */
    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }

    /**
     * Sends permission response to backend
     * @param {string} permissionId - Unique permission request ID
     * @param {string} result - Permission decision (allow, deny, always_allow, always_deny)
     * @param {Object} [action={}] - Additional action metadata
     * @returns {boolean} True if response was sent successfully
     */
    sendPermissionResponse(permissionId, result, action = {}) {
        const response = {
            type: 'permission_response',
            permission_id: permissionId,
            result: result,
            action: action,
            timestamp: Date.now()
        };

        if (this.sendMessage(JSON.stringify(response))) {
            this.logger.info('Sent permission response', { permissionId, result });

            // Remove the permission request from UI
            const permissionElement = document.getElementById('permission-' + permissionId);
            if (permissionElement) {
                permissionElement.remove();
            }

            // Clear pending permission ID
            window.pendingPermissionId = null;

            return true;
        }
        return false;
    }

    /**
     * Sends interrupt signal to agent to stop current operation
     * @returns {boolean} True if interrupt signal was sent successfully
     */
    sendInterruptSignal() {
        const interruptMessage = {
            type: 'interrupt_signal',
            timestamp: Date.now(),
            source: 'web_ui'
        };

        if (this.sendMessage(JSON.stringify(interruptMessage))) {
            this.logger.info('Sent interrupt signal to agent');
            return true;
        } else {
            this.logger.error('Failed to send interrupt signal - WebSocket not connected');
            return false;
        }
    }


    /**
     * Handles live tool start event (legacy)
     * Creates visual indicator for tool execution in progress
     * @private
     * @param {Object} data - Tool start data
     * @param {string} data.tool_name - Name of tool being executed
     * @param {string} [data.tool_args] - Tool arguments
     */
    handleLiveToolStart(data) {
        this.logger.info('Live tool started', { tool_name: data.tool_name });

        // Get or create the live tools container
        let liveToolsContainer = document.getElementById('live-tools-container');
        if (!liveToolsContainer) {
            // Create live tools container next to loading indicator
            const loadingIndicator = document.getElementById('agent-loading-indicator');
            if (loadingIndicator) {
                liveToolsContainer = document.createElement('div');
                liveToolsContainer.id = 'live-tools-container';
                liveToolsContainer.className = 'live-tools-container';

                // Insert after loading indicator
                loadingIndicator.parentNode.insertBefore(liveToolsContainer, loadingIndicator.nextSibling);
            }
        }

        if (liveToolsContainer) {
            // Create tool progress item
            const toolItem = document.createElement('div');
            toolItem.id = `live-tool-${data.tool_name}`;
            toolItem.className = 'live-tool-item running';

            const toolEmojis = {
                "editfile": "✏️", "readfile": "📖", "writefile": "📝",
                "insertline": "➕", "deletelines": "🗑️", "bash": "🔧",
                "python": "🐍", "searchweb": "🌐", "showuser": "👁️",
                "memory": "🧠", "grepsearch": "🔍", "visualresponse": "🎨"
            };
            const toolEmoji = toolEmojis[data.tool_name] || "⚙️";

            toolItem.innerHTML = `
                <div class="live-tool-header">
                    <span class="live-tool-emoji">${toolEmoji}</span>
                    <span class="live-tool-name">${data.tool_name}</span>
                    <span class="live-tool-status running">🔄 Running...</span>
                </div>
                <div class="live-tool-args">${this.escapeHtml(data.tool_args || 'No arguments')}</div>
            `;

            liveToolsContainer.appendChild(toolItem);

            // Scroll to show the new tool
            this.scrollToBottom();
        }
    }

    /**
     * Handles live tool completion event (legacy)
     * Updates tool status indicator and auto-hides after delay
     * @private
     * @param {Object} data - Tool completion data
     * @param {string} data.tool_name - Name of completed tool
     * @param {string} data.tool_status - Completion status (success, error)
     * @param {number} data.tool_duration_ms - Execution duration in milliseconds
     */
    handleLiveToolComplete(data) {
        this.logger.info('Live tool completed', { tool_name: data.tool_name, status: data.tool_status });

        const toolItem = document.getElementById(`live-tool-${data.tool_name}`);
        if (toolItem) {
            // Update status
            const statusSpan = toolItem.querySelector('.live-tool-status');
            if (statusSpan) {
                if (data.tool_status === 'success') {
                    statusSpan.innerHTML = `✅ Completed (${data.tool_duration_ms}ms)`;
                    statusSpan.className = 'live-tool-status success';
                } else {
                    statusSpan.innerHTML = `❌ Failed (${data.tool_duration_ms}ms)`;
                    statusSpan.className = 'live-tool-status error';
                }
            }

            // Remove running class and add completed class
            toolItem.classList.remove('running');
            toolItem.classList.add('completed');

            // Auto-hide completed tools after 3 seconds
            setTimeout(() => {
                if (toolItem.parentNode) {
                    toolItem.style.opacity = '0.5';
                    toolItem.style.transform = 'translateX(20px)';
                }
            }, 2000);

            setTimeout(() => {
                if (toolItem.parentNode) {
                    toolItem.remove();

                    // Check if this was the last tool item and clean up container
                    const container = document.getElementById('live-tools-container');
                    if (container && container.children.length === 0) {
                        container.remove();
                        this.logger.info('Removed empty live tools container');
                    }
                }
            }, 5000);
        }
    }

    /**
     * Cleans up live tools container when message arrives
     * Fades out and removes container with animation
     * @private
     */
    cleanupLiveToolsContainer() {
        const container = document.getElementById('live-tools-container');
        if (container) {
            // Fade out and remove the container
            container.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            container.style.opacity = '0';
            container.style.transform = 'translateY(-10px)';

            setTimeout(() => {
                if (container.parentNode) {
                    container.remove();
                    this.logger.info('Cleaned up live tools container on message arrival');
                }
            }, 500);
        }
    }

    /**
     * Escapes HTML special characters to prevent XSS
     * @private
     * @param {string} text - Text to escape
     * @returns {string} HTML-escaped text
     */
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Truncates text to a maximum length and adds ellipsis
     * @private
     * @param {string} text - Text to truncate
     * @param {number} maxLength - Maximum length before truncation
     * @returns {string} Truncated text with "..." if needed
     */
    truncateText(text, maxLength = 100) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }


    /**
     * Handles loading tool start event
     * Creates expandable tool status item in loading tools header
     * @private
     * @param {Object} data - Tool start data
     * @param {string} data.tool_name - Name of tool being executed
     * @param {string} [data.tool_args] - Tool arguments
     * @param {string} [data.model] - LLM model used for tool
     */
    handleLoadingToolStart(data) {
        this.logger.info('Loading tool started', { tool_name: data.tool_name });

        // Find or create the loading tools header
        let loadingToolsHeader = document.getElementById('loading-tools-header');
        if (!loadingToolsHeader) {
            // Create header in the loading indicator area
            const loadingIndicator = document.getElementById('agent-loading-indicator');
            if (loadingIndicator) {
                loadingToolsHeader = document.createElement('div');
                loadingToolsHeader.id = 'loading-tools-header';
                loadingToolsHeader.className = 'loading-tools-header';

                // Insert after loading indicator
                loadingIndicator.parentNode.insertBefore(loadingToolsHeader, loadingIndicator.nextSibling);
            }
        }

        if (!loadingToolsHeader) return;

        // Create individual tool status item
        const toolItem = document.createElement('div');
        toolItem.className = 'loading-tool-item running';
        // REVERT: Use the tool name for the ID, as it's the only common link
        toolItem.id = `loading-tool-${data.tool_name}`;

        const startTime = Date.now();
        toolItem.dataset.startTime = startTime;

        // Add model information to the display
        const modelDisplay = data.model ? `<span class="loading-tool-model">[${this.escapeHtml(data.model)}]</span>` : '';

        toolItem.innerHTML = `
            <div class="loading-tool-main">
                <span class="loading-tool-name">${this.escapeHtml(data.tool_name)}</span>
                ${modelDisplay}
                <span class="loading-tool-status running">
                    <span class="loading-tool-spinner">⏳</span>
                    <span class="loading-tool-stopwatch">0.0s</span>
                </span>
            </div>
            <div class="loading-tool-details collapsed">
                <div class="loading-tool-args">${this.escapeHtml(data.tool_args || 'No arguments')}</div>
                <div class="loading-tool-output">Output will appear here...</div>
            </div>
        `;

        // Add click handler for expandable details with smooth animation
        const mainDiv = toolItem.querySelector('.loading-tool-main');
        mainDiv.addEventListener('click', () => {
            const details = toolItem.querySelector('.loading-tool-details');
            const isCollapsed = details.classList.contains('collapsed');

            if (isCollapsed) {
                // Expand
                details.classList.remove('collapsed');
                details.classList.add('expanded');
            } else {
                // Collapse
                details.classList.remove('expanded');
                details.classList.add('collapsed');
            }
        });

        loadingToolsHeader.appendChild(toolItem);

        // Scroll to show the new tool
        this.scrollToBottom();

        // Start stopwatch
        const stopwatchSpan = toolItem.querySelector('.loading-tool-stopwatch');
        const stopwatchTimer = setInterval(() => {
            const elapsed = (Date.now() - startTime) / 1000;
            if (stopwatchSpan) {
                stopwatchSpan.textContent = elapsed.toFixed(1) + 's';
            }
        }, 100);

        toolItem.dataset.stopwatchTimer = stopwatchTimer;
    }

    /**
     * Handles WebSocket TTS audio streaming
     * Forwards audio data to AudioManager for playback
     * @private
     * @param {Object} data - Audio payload
     * @param {string} data.text - Text that was converted to speech
     * @param {string} data.audio_format - Audio format (wav, mp3, etc.)
     * @param {number} [data.duration] - Audio duration in seconds
     * @param {string} [data.language] - Language code
     * @param {string} data.audio_data - Base64 encoded audio data
     */
    handleWebSocketAudio(data) {
        this.logger.info('Received WebSocket TTS audio', {
            text: data.text?.substring(0, 50) + '...',
            format: data.audio_format,
            duration: data.duration,
            language: data.language
        });

        // Forward to AudioManager if available
        if (window.audioManager && typeof window.audioManager.playWebSocketAudio === 'function') {
            window.audioManager.playWebSocketAudio(data);
        } else {
            this.logger.warn('AudioManager not available for WebSocket audio playback');
        }
    }

    /**
     * Handles loading tool completion event
     * Updates tool status with final result and frozen timer
     * @private
     * @param {Object} data - Tool completion data
     * @param {string} data.tool_name - Name of completed tool
     * @param {string} data.tool_status - Completion status (success, error)
     * @param {number} [data.tool_duration_ms] - Execution duration in milliseconds
     * @param {string} [data.tool_result] - Tool execution result
     * @param {string} [data.tool_error] - Error message if tool failed
     */
    handleLoadingToolComplete(data) {
        console.log('🔧 Received loading tool complete:', data);
        this.logger.info('Loading tool completed', { tool_name: data.tool_name, status: data.tool_status });

        // CORRECT FIX: Find the first element that matches the tool name ID
        // AND currently has the 'running' class. This correctly targets the
        // next sequential tool to be completed.
        const toolItem = document.querySelector(`[id="loading-tool-${data.tool_name}"].running`);

        if (!toolItem) {
            this.logger.warn('No loading tool item found for:', data.tool_name);
            return;
        }

        // Clear the stopwatch timer
        const timerId = toolItem.dataset.stopwatchTimer;
        if (timerId) {
            clearInterval(parseInt(timerId));
        }

        // Update status with success emoji and frozen stopwatch
        const statusSpan = toolItem.querySelector('.loading-tool-status');
        if (statusSpan) {
            const duration = data.tool_duration_ms || 0;
            const frozenTime = (duration / 1000).toFixed(1) + 's';

            if (data.tool_status === 'success') {
                statusSpan.innerHTML = `
                    <span class="loading-tool-indicator">✅</span>
                    <span class="loading-tool-stopwatch frozen">${frozenTime}</span>
                `;
                statusSpan.className = 'loading-tool-status success';
            } else {
                statusSpan.innerHTML = `
                    <span class="loading-tool-indicator">❌</span>
                    <span class="loading-tool-stopwatch frozen">${frozenTime}</span>
                `;
                statusSpan.className = 'loading-tool-status error';
            }
        }

        // Update output section
        const outputDiv = toolItem.querySelector('.loading-tool-output');
        if (outputDiv) {
            if (data.tool_result) {
                outputDiv.innerHTML = `<pre>${this.escapeHtml(data.tool_result)}</pre>`;
            } else if (data.tool_error) {
                outputDiv.innerHTML = `<div class="error">Error: ${this.escapeHtml(data.tool_error)}</div>`;
            } else {
                outputDiv.textContent = 'Completed successfully';
            }
        }

        // Ensure proper state transition
        toolItem.classList.remove('running');
        toolItem.classList.add(data.tool_status === 'success' ? 'completed' : 'failed');

        // Remove any lingering running indicators
        const spinner = toolItem.querySelector('.loading-tool-spinner');
        if (spinner) {
            spinner.remove();
        }

        // Clear any timer attribute
        toolItem.removeAttribute('data-stopwatch-timer');

        // Scroll to show the completed tool
        this.scrollToBottom();
    }

    /**
     * Transfers loading tools from global header to specific message block
     * Called when visualresponse arrives to attach tool status to message
     * @private
     * @param {string} messageId - ID of message to transfer tools to
     * @param {Object} [toolInfo=null] - Optional tool information metadata
     */
    transferLoadingToolsToMessage(messageId, toolInfo = null) {
        const loadingToolsHeader = document.getElementById('loading-tools-header');
        if (!loadingToolsHeader || loadingToolsHeader.children.length === 0) {
            // Still clean up any empty loading headers
            if (loadingToolsHeader) {
                loadingToolsHeader.remove();
            }
            return; // No loading tools to transfer
        }

        // Wait a bit for the message block to be created
        setTimeout(() => {
            const messageBlock = document.getElementById(`message-${messageId}`);
            if (!messageBlock) {
                this.logger.warn('No message block found for transfer:', messageId);
                return;
            }

            // Create loading-tools-header in the message (keep same styling)
            let toolsHeader = messageBlock.querySelector('.loading-tools-header');
            if (!toolsHeader) {
                toolsHeader = document.createElement('div');
                toolsHeader.className = 'loading-tools-header';

                // Insert before the content area
                const contentArea = messageBlock.querySelector('.message-content');
                if (contentArea) {
                    contentArea.parentNode.insertBefore(toolsHeader, contentArea);
                }
            }

            // Transfer all loading tools to message header in execution order
            const toolsArray = Array.from(loadingToolsHeader.children);

            // Sort by order of appearance (document order = execution order)
            toolsArray.forEach(loadingTool => {
                const clonedTool = loadingTool.cloneNode(true);

                // Re-add click handler for expandable details
                const mainDiv = clonedTool.querySelector('.loading-tool-main');
                if (mainDiv) {
                    mainDiv.addEventListener('click', () => {
                        const details = clonedTool.querySelector('.loading-tool-details');
                        const isVisible = details.style.display !== 'none';
                        details.style.display = isVisible ? 'none' : 'block';
                    });
                }

                // Preserve any running timers by checking original state
                const originalStopwatchTimer = loadingTool.dataset.stopwatchTimer;
                if (originalStopwatchTimer && loadingTool.classList.contains('running')) {
                    // Tool is still running, transfer the timer
                    const stopwatchSpan = clonedTool.querySelector('.loading-tool-stopwatch');
                    const startTime = parseInt(loadingTool.dataset.startTime);

                    if (stopwatchSpan && startTime) {
                        const stopwatchTimer = setInterval(() => {
                            const elapsed = (Date.now() - startTime) / 1000;
                            if (stopwatchSpan) {
                                stopwatchSpan.textContent = elapsed.toFixed(1) + 's';
                            }
                        }, 100);
                        clonedTool.dataset.stopwatchTimer = stopwatchTimer;
                        clonedTool.dataset.startTime = startTime;
                    }
                }

                toolsHeader.appendChild(clonedTool);
            });

            // Model information is now displayed in the main tool header
            // No need to add it to individual tool details

            // Remove the loading tools header
            loadingToolsHeader.remove();
            this.logger.info('Transferred loading tools to visualresponse header');
        }, 100);
    }

    /**
     * Comprehensive cleanup function to ensure proper state
     * Removes empty loading tools headers and marks stuck tools as failed
     * Called periodically after message rendering to maintain clean state
     * @private
     */
    performMaintenanceCleanup() {
        // Remove any orphaned loading tools headers (but keep completed ones visible)
        const orphanedHeaders = document.querySelectorAll('.loading-tools-header');
        orphanedHeaders.forEach(header => {
            // Only remove headers that are completely empty
            if (header.children.length === 0) {
                this.logger.info('Removing empty loading tools header');
                header.remove();
            }
            // Keep headers with completed/failed tools visible for persistence
        });

        // Clear any stuck running timers for tools that should be completed
        const stuckTools = document.querySelectorAll('.loading-tool-item.running');
        stuckTools.forEach(tool => {
            const startTime = parseInt(tool.dataset.startTime);
            const currentTime = Date.now();
            const elapsed = currentTime - startTime;

            // If a tool has been running for more than 30 seconds, something is wrong
            if (elapsed > 30000) {
                this.logger.warn('Found stuck tool, marking as failed:', tool.id);

                // Clear timer
                const timerId = tool.dataset.stopwatchTimer;
                if (timerId) {
                    clearInterval(parseInt(timerId));
                }

                // Mark as failed with warning indicator
                const statusSpan = tool.querySelector('.loading-tool-status');
                if (statusSpan) {
                    const frozenTime = (elapsed / 1000).toFixed(1) + 's';
                    statusSpan.innerHTML = `
                        <span class="loading-tool-indicator">⚠️</span>
                        <span class="loading-tool-stopwatch frozen">${frozenTime}</span>
                    `;
                    statusSpan.className = 'loading-tool-status error';
                }

                tool.classList.remove('running');
                tool.classList.add('failed');
                tool.removeAttribute('data-stopwatch-timer');
            }
        });
    }

}

window.WebSocketManager = WebSocketManager;