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

    async handleMessage(data) {
        this.logger.debug('Received WebSocket message', { type: data.type, action: data.action, id: data.id });

        // Hide loading indicator when any message is received from the agent
        if (this.app && typeof this.app.hideLoadingIndicator === 'function') {
            this.app.hideLoadingIndicator();
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
            this.contentWrapper.innerHTML = '';
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
            this.contentWrapper.appendChild(messageBlock);
        }
        
        const timestamp = new Date().toLocaleTimeString();
        messageBlock.innerHTML = `<div class="message-timestamp">${timestamp} - ID: ${data.id}</div>`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageBlock.appendChild(messageContent);
        
        await this.renderer.renderContent(messageContent, data);
        
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

        // Use requestAnimationFrame to ensure DOM has been updated
        requestAnimationFrame(() => {
            // Method 1: Smooth scroll to bottom (modern browsers)
            try {
                this.contentWrapper.scrollTo({
                    top: this.contentWrapper.scrollHeight,
                    behavior: 'smooth'
                });
            } catch (e) {
                // Fallback for older browsers
                this.contentWrapper.scrollTop = this.contentWrapper.scrollHeight;
            }
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
}

window.WebSocketManager = WebSocketManager;