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
    constructor(renderer, statusElement, contentWrapper) {
        this.renderer = renderer;
        this.statusElement = statusElement;
        this.contentWrapper = contentWrapper;
        this.ws = null;
        this.reconnectDelay = 3000;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.connectionId = null;
        this.logger = window.visualLogger;
        this.app = null; // Reference to main app for loading indicator control

        this.logger.info('WebSocketManager initialized', {
            reconnectDelay: this.reconnectDelay,
            maxReconnectAttempts: this.maxReconnectAttempts
        });

        this.connect();
    }

    /**
     * Set reference to main app for loading indicator control
     */
    setApp(app) {
        this.app = app;
    }

    connect() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsPort = window.location.port || (wsProtocol === 'wss:' ? 443 : 80);
        const wsUrl = `${wsProtocol}//${window.location.hostname}:${wsPort}/ws`;

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

        this.ws.onopen = () => {
            this.logger.endTiming('websocket_connection', { connectionId: this.connectionId, reconnectAttempts: this.reconnectAttempts });
            this.reconnectAttempts = 0;
            this.statusElement.textContent = '🟢 Connected to Agent';
            this.statusElement.style.color = 'var(--status-connected)';
            this.logger.info('WebSocket connected successfully', { connectionId: this.connectionId });

            // History loading will now be triggered by config message from server
            // This ensures we respect the continue flag from the CLI
        };

        this.ws.onmessage = (event) => this.handleMessage(JSON.parse(event.data));

        this.ws.onclose = (event) => {
            this.logger.warn('WebSocket connection closed', { code: event.code, reason: event.reason });
            this.statusElement.textContent = '🔴 Disconnected. Retrying...';
            this.statusElement.style.color = 'var(--status-disconnected)';

            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
                setTimeout(() => this.connect(), delay);
            } else {
                this.logger.error('Max reconnection attempts reached.');
                this.statusElement.textContent = '🔴 Connection failed';
            }
        };

        this.ws.onerror = (error) => this.handleConnectionError(error);
    }

    handleConnectionError(error) {
        this.logger.error('WebSocket connection error', {}, error);
        if (this.ws) {
            this.ws.close();
        }
    }

    handleConnectionConfig(data) {
        this.logger.info('Received connection configuration', { continue_mode: data.continue_mode });

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

        if (permissionData.file_path) {
            contentHTML += `<p><strong>File:</strong> ${permissionData.file_path}</p>`;
        }

        if (permissionData.args_display) {
            contentHTML += `<p><strong>Command:</strong> ${permissionData.args_display}</p>`;
        }

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
                <button class="permission-btn always-allow" onclick="window.sendPermissionResponse('${data.id}', 'always_allow')">
                    ✅ Always Allow
                </button>
                <button class="permission-btn always-deny" onclick="window.sendPermissionResponse('${data.id}', 'always_deny')">
                    ❌ Always Deny
                </button>
            </div>
        `;

        contentDiv.innerHTML = contentHTML;

        messageBlock.appendChild(headerDiv);
        messageBlock.appendChild(contentDiv);

        this.contentWrapper.appendChild(messageBlock);
        this.scrollToBottom();

        // Store the permission ID for response handling
        window.pendingPermissionId = data.id;
    }

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

    handleConsoleLog(data) {
        this.logger.debug('Received console log message', { line: data.line, level: data.level });

        // Forward to console log manager if available
        if (window.consoleLogManager) {
            window.consoleLogManager.addLogLine(data.line, data.level || 'info');
        } else {
            // Fallback: log to browser console
            console.log(`[CLI-Agent Console] ${data.line}`);
        }
    }

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

    async handleMessage(data) {
        this.logger.debug('Received WebSocket message', { type: data.type, action: data.action, id: data.id });

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

        // Handle console log messages
        if (data.type === 'console_log') {
            this.handleConsoleLog(data);
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

        // Hide loading indicator when regular message is received from the agent
        if (this.app && typeof this.app.hideLoadingIndicator === 'function') {
            this.app.hideLoadingIndicator();
        }

        // Hide interrupt button when agent response is complete
        if (this.app && typeof this.app.hideInterruptButton === 'function') {
            this.app.hideInterruptButton();
        }

        // Clean up any remaining live tools container when regular message arrives
        this.cleanupLiveToolsContainer();

        // Transfer loading tools to visualresponse header when message arrives
        this.transferLoadingToolsToMessage(data.id);

        // Clean up any orphaned loading tools header after transfer
        setTimeout(() => {
            this.performMaintenanceCleanup();
        }, 500);

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

        // Base timestamp content
        let timestampContent = `<span class="timestamp-text">${timestamp} - ID: ${data.id}</span>`;

        // Add tool execution info if available
        if (data.tool_info && Array.isArray(data.tool_info) && data.tool_info.length > 0) {
            const toolCount = data.tool_info.length;
            const toolNames = data.tool_info.map(tool => tool.name).join(', ');

            timestampContent += `<span class="tool-info-summary" title="Click to expand tool details">
                🔧 ${toolCount} tool${toolCount > 1 ? 's' : ''}: ${toolNames}
            </span>`;
        }

        timestampDiv.innerHTML = timestampContent;
        messageBlock.appendChild(timestampDiv);

        // Add detailed tool info section (initially hidden)
        if (data.tool_info && Array.isArray(data.tool_info) && data.tool_info.length > 0) {
            const toolDetails = document.createElement('div');
            toolDetails.className = 'tool-details hidden';

            let detailsHTML = '<div class="tool-details-header">Tool Execution Details:</div>';
            data.tool_info.forEach((tool, index) => {
                const statusIcon = tool.status === 'success' ? '✅' : '❌';
                const errorText = tool.error ? ` (${tool.error})` : '';
                const argsPreview = tool.args ? (tool.args.length > 100 ? tool.args.substring(0, 100) + '...' : tool.args) : 'No arguments';
                const contextPreview = tool.context_returned ? (tool.context_returned.length > 100 ? tool.context_returned.substring(0, 100) + '...' : tool.context_returned) : 'No output';
                const contextLength = tool.context_length || 0;

                const toolDetailId = `tool-detail-${data.id}-${index}`;

                detailsHTML += `
                    <div class="tool-detail-item">
                        <div class="tool-summary" onclick="toggleToolDetail('${toolDetailId}')">
                            ${statusIcon} <strong>${tool.name}</strong> - ${tool.duration_ms}ms${errorText}
                            <span class="tool-toggle">▼</span>
                        </div>
                        <div class="tool-detail-expanded" id="${toolDetailId}" style="display: none;">
                            <div class="tool-detail-section">
                                <strong>Arguments:</strong>
                                <pre class="tool-code">${this.escapeHtml(tool.args || 'No arguments')}</pre>
                            </div>
                            <div class="tool-detail-section">
                                <strong>Context Returned (${contextLength} chars):</strong>
                                <pre class="tool-code">${this.escapeHtml(tool.context_returned || 'No output')}</pre>
                            </div>
                        </div>
                    </div>
                `;
            });

            toolDetails.innerHTML = detailsHTML;
            messageBlock.appendChild(toolDetails);

            // Add click handler to toggle tool details
            const toolSummary = timestampDiv.querySelector('.tool-info-summary');
            if (toolSummary) {
                toolSummary.addEventListener('click', () => {
                    toolDetails.classList.toggle('hidden');
                    const isHidden = toolDetails.classList.contains('hidden');
                    toolSummary.style.backgroundColor = isHidden ? '' : 'rgba(100, 149, 237, 0.2)';
                });
                toolSummary.style.cursor = 'pointer';
            }
        }

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageBlock.appendChild(messageContent);
        
        await this.renderer.renderContent(messageContent, data, messageId, timestampDiv);

        // Save agent message to history if main app is available (but not during restoration)
        // IMPORTANT: Do not save permission requests or other transient UI elements to history
        if (this.app && typeof this.app.saveMessageToHistory === 'function' && !this.app.isRestoring) {
            // Filter out transient message types that should not be persisted
            // FIXED: Removed 'user_message' from transient types - this was incorrectly preventing user message storage
            const transientTypes = ['permission_request', 'config'];
            if (!transientTypes.includes(data.type)) {
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
            // Additional delayed scroll
            setTimeout(forceScrollToEnd, 10);
            setTimeout(forceScrollToEnd, 50);
        });
    }
    
    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(message);
            this.logger.info('Message sent to agent', { messageLength: message.length });
            return true;
        }
        this.logger.warn('Cannot send message - WebSocket not connected.');
        return false;
    }

    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }

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

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }


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

        toolItem.innerHTML = `
            <div class="loading-tool-main">
                <span class="loading-tool-name">${this.escapeHtml(data.tool_name)}</span>
                <span class="loading-tool-status running">
                    <span class="loading-tool-spinner">⏳</span>
                    <span class="loading-tool-stopwatch">0.0s</span>
                </span>
            </div>
            <div class="loading-tool-details" style="display: none;">
                <div class="loading-tool-args">${this.escapeHtml(data.tool_args || 'No arguments')}</div>
                <div class="loading-tool-output">Output will appear here...</div>
            </div>
        `;

        // Add click handler for expandable details
        const mainDiv = toolItem.querySelector('.loading-tool-main');
        mainDiv.style.cursor = 'pointer';
        mainDiv.addEventListener('click', () => {
            const details = toolItem.querySelector('.loading-tool-details');
            const isVisible = details.style.display !== 'none';
            details.style.display = isVisible ? 'none' : 'block';
        });

        loadingToolsHeader.appendChild(toolItem);

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
    }

    transferLoadingToolsToMessage(messageId) {
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

            // Remove the loading tools header
            loadingToolsHeader.remove();
            this.logger.info('Transferred loading tools to visualresponse header');
        }, 100);
    }

    // Comprehensive cleanup function to ensure proper state
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

                // Mark as failed
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