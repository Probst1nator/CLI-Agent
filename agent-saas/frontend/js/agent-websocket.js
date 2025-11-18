/**
 * Agent WebSocket - WebSocket client for agent communication
 *
 * Data Flow (DAG):
 * - ChatInterface → connect(sessionId) → WebSocket → onOpen() → send(get_history)
 * - WebSocket → onMessage() → handleMessage() → ChatInterface methods
 * - ChatInterface → sendMessage(content) → send() → WebSocket
 * - Reconnect loop: onClose() → connect() (with exponential backoff)
 *
 * Message Types:
 * - Outgoing: chat_message, get_history
 * - Incoming: llm_chunk, tool_execution, tool_result, final_response, error, history
 *
 * Dependencies:
 * - window.chatInterface (ChatInterface): Message display
 * - Backend WebSocket: /api/agents/sessions/{id}/ws
 *
 * @class
 */
class AgentWebSocket {
    /**
     * Initialize WebSocket client
     * @constructor
     */
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        this.isConnecting = false;

        // Connection status element
        this.statusEl = document.getElementById('connection-status');
        this.statusDotEl = this.statusEl.querySelector('.status-dot');
        this.statusTextEl = this.statusEl.querySelector('span:last-child');
    }

    connect(sessionId) {
        if (this.isConnecting) {
            console.log('[AgentWebSocket] Already connecting...');
            return;
        }

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            if (this.sessionId === sessionId) {
                console.log('[AgentWebSocket] Already connected to this session');
                return;
            }
            this.disconnect();
        }

        this.sessionId = sessionId;
        this.isConnecting = true;
        this.reconnectAttempts = 0;

        console.log(`[AgentWebSocket] Connecting to session: ${sessionId}`);
        this.updateStatus('connecting');

        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}${window.location.pathname}api/agents/sessions/${sessionId}/ws`;

            console.log(`[AgentWebSocket] WebSocket URL: ${wsUrl}`);

            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => this.onOpen();
            this.ws.onmessage = (event) => this.onMessage(event);
            this.ws.onerror = (error) => this.onError(error);
            this.ws.onclose = (event) => this.onClose(event);

        } catch (error) {
            console.error('[AgentWebSocket] Connection error:', error);
            this.updateStatus('disconnected');
            this.isConnecting = false;
        }
    }

    disconnect() {
        console.log('[AgentWebSocket] Disconnecting...');

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.sessionId = null;
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.updateStatus('disconnected');
    }

    onOpen() {
        console.log('[AgentWebSocket] Connected');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.updateStatus('connected');

        // Request chat history
        this.send({
            type: 'get_history'
        });
    }

    onMessage(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('[AgentWebSocket] Received:', data.type);

            this.handleMessage(data);
        } catch (error) {
            console.error('[AgentWebSocket] Failed to parse message:', error);
        }
    }

    handleMessage(data) {
        if (!window.chatInterface) return;

        switch (data.type) {
            case 'llm_chunk':
                // Streaming LLM response
                if (data.is_complete) {
                    window.chatInterface.addAgentMessage(data.content);
                } else {
                    window.chatInterface.updateLastAgentMessage(data.content);
                }
                break;

            case 'tool_execution':
                // Tool is being executed
                window.chatInterface.addToolExecution(data.tool_name, data.content || '');
                break;

            case 'tool_result':
                // Tool execution result
                window.chatInterface.addToolResult(data.tool_name, data.content || '');
                break;

            case 'final_response':
                // Final agent response
                window.chatInterface.addAgentMessage(data.content);
                break;

            case 'error':
                // Error from agent
                window.chatInterface.addSystemMessage(`Error: ${data.content}`);
                break;

            case 'history':
                // Chat history
                if (data.messages && Array.isArray(data.messages)) {
                    data.messages.forEach(msg => {
                        if (msg.role === 'user') {
                            window.chatInterface.addUserMessage(msg.content);
                        } else if (msg.role === 'assistant' || msg.role === 'agent') {
                            window.chatInterface.addAgentMessage(msg.content);
                        }
                    });
                }
                break;

            default:
                console.log('[AgentWebSocket] Unknown message type:', data.type);
        }
    }

    onError(error) {
        console.error('[AgentWebSocket] WebSocket error:', error);
        this.updateStatus('error');
    }

    onClose(event) {
        console.log(`[AgentWebSocket] Connection closed (code: ${event.code}, reason: ${event.reason})`);
        this.ws = null;
        this.isConnecting = false;
        this.updateStatus('disconnected');

        // Attempt reconnect if unexpected close
        if (event.code !== 1000 && this.sessionId && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`[AgentWebSocket] Reconnecting (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

            setTimeout(() => {
                if (this.sessionId) {
                    this.connect(this.sessionId);
                }
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }

    sendMessage(content) {
        this.send({
            type: 'chat_message',
            content: content
        });
    }

    send(data) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.error('[AgentWebSocket] WebSocket not connected');
            if (window.chatInterface) {
                window.chatInterface.addSystemMessage('Not connected. Please reconnect.');
            }
            return;
        }

        try {
            this.ws.send(JSON.stringify(data));
            console.log('[AgentWebSocket] Sent:', data.type);
        } catch (error) {
            console.error('[AgentWebSocket] Failed to send message:', error);
        }
    }

    updateStatus(status) {
        console.log(`[AgentWebSocket] Status: ${status}`);

        this.statusDotEl.className = 'status-dot';

        switch (status) {
            case 'connecting':
                this.statusDotEl.classList.add('disconnected');
                this.statusTextEl.textContent = 'Connecting...';
                break;

            case 'connected':
                this.statusDotEl.classList.add('connected');
                this.statusTextEl.textContent = 'Connected';
                break;

            case 'disconnected':
                this.statusDotEl.classList.add('disconnected');
                this.statusTextEl.textContent = 'Disconnected';
                break;

            case 'error':
                this.statusDotEl.classList.add('disconnected');
                this.statusTextEl.textContent = 'Error';
                break;
        }
    }
}

// Export globally
window.AgentWebSocket = AgentWebSocket;
