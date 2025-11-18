/**
 * Chat Interface - Handles chat panel and messaging
 *
 * Data Flow (DAG):
 * - constructor() → init()
 * - SessionManager → setActiveSession() → loadChatHistory() → AgentWebSocket.connect()
 * - User action → sendMessage() → AgentWebSocket.sendMessage()
 * - WebSocket → addUserMessage() / addAgentMessage() / addToolExecution() / addToolResult()
 * - User action → clearChat() → API
 * - User action → terminateSession() → SessionManager.terminateSession()
 *
 * Dependencies:
 * - window.sessionManager (SessionManager): Agent info, session termination
 * - window.agentWebSocket (AgentWebSocket): Real-time communication
 * - Backend API: /api/agents/sessions/{id}/history
 *
 * @class
 */
class ChatInterface {
    /**
     * Initialize ChatInterface with DOM references
     * @constructor
     */
    constructor() {
        this.activeSession = null;
        this.messages = [];

        // DOM elements
        this.sessionInfoEl = document.getElementById('session-info');
        this.chatMessagesEl = document.getElementById('chat-messages');
        this.chatInputEl = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('send-btn');
        this.clearChatBtn = document.getElementById('clear-chat-btn');
        this.terminateSessionBtn = document.getElementById('terminate-session-btn');

        this.init();
    }

    /**
     * Initialize event listeners
     * @returns {void}
     */
    init() {
        console.log('[ChatInterface] Initializing...');

        // Bind events
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.chatInputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.clearChatBtn.addEventListener('click', () => this.clearChat());
        this.terminateSessionBtn.addEventListener('click', () => this.terminateSession());
    }

    /**
     * Set active session and load chat history
     * Data Flow: setActiveSession() → loadChatHistory() → AgentWebSocket.connect()
     * @param {Object} session - Session object from SessionManager
     * @param {string} session.session_id - Session ID
     * @param {string} session.agent_type - Agent type name
     * @param {string} session.status - Session status
     * @returns {void}
     */
    setActiveSession(session) {
        console.log(`[ChatInterface] Setting active session: ${session.session_id}`);

        this.activeSession = session;
        this.messages = [];

        // Update header
        const agentIcon = window.sessionManager ?
            window.sessionManager.getAgentIcon(session.agent_type) : '🤖';
        const agentName = window.sessionManager ?
            window.sessionManager.getAgentDisplayName(session.agent_type) : session.agent_type;

        this.sessionInfoEl.innerHTML = `
            <div class="session-info-details">
                <div class="session-info-agent">
                    <span>${agentIcon}</span>
                    <span>${agentName}</span>
                    <span class="session-status ${session.status === 'running' ? 'running' : 'stopped'}"></span>
                </div>
                <div class="session-info-id">ID: ${session.session_id}</div>
            </div>
        `;

        // Enable inputs
        this.chatInputEl.disabled = false;
        this.chatInputEl.placeholder = 'Type your message...';
        this.sendBtn.disabled = false;
        this.clearChatBtn.disabled = false;
        this.terminateSessionBtn.disabled = false;

        // Clear chat and load history
        this.chatMessagesEl.innerHTML = '';
        this.loadChatHistory();

        // Connect WebSocket
        if (window.agentWebSocket) {
            window.agentWebSocket.connect(session.session_id);
        }
    }

    /**
     * Clear active session and reset UI
     * @returns {void}
     */
    clearActiveSession() {
        console.log('[ChatInterface] Clearing active session');

        this.activeSession = null;
        this.messages = [];

        // Update header
        this.sessionInfoEl.innerHTML = '<span class="no-session-selected">No session selected</span>';

        // Disable inputs
        this.chatInputEl.disabled = true;
        this.chatInputEl.placeholder = 'Select a session to start chatting...';
        this.chatInputEl.value = '';
        this.sendBtn.disabled = true;
        this.clearChatBtn.disabled = true;
        this.terminateSessionBtn.disabled = true;

        // Show welcome message
        this.chatMessagesEl.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to Agent Management</h2>
                <p>Select a session from the sidebar or create a new one to get started.</p>
                <div class="feature-list">
                    <div class="feature">
                        <span class="feature-icon">⚡</span>
                        <span class="feature-text">Multiple concurrent agent sessions</span>
                    </div>
                    <div class="feature">
                        <span class="feature-icon">🔒</span>
                        <span class="feature-text">Fully isolated Docker containers</span>
                    </div>
                    <div class="feature">
                        <span class="feature-icon">🔄</span>
                        <span class="feature-text">Real-time WebSocket streaming</span>
                    </div>
                </div>
            </div>
        `;

        // Disconnect WebSocket
        if (window.agentWebSocket) {
            window.agentWebSocket.disconnect();
        }
    }

    async loadChatHistory() {
        if (!this.activeSession) return;

        try {
            // For now, just add a system message
            // In Phase 3, we'll load actual chat history from agent container
            this.addSystemMessage(`Connected to ${this.activeSession.agent_type}`);
        } catch (error) {
            console.error('[ChatInterface] Failed to load chat history:', error);
        }
    }

    sendMessage() {
        const content = this.chatInputEl.value.trim();
        if (!content || !this.activeSession) return;

        console.log(`[ChatInterface] Sending message: ${content.substring(0, 50)}...`);

        // Add user message to UI
        this.addUserMessage(content);

        // Clear input
        this.chatInputEl.value = '';

        // Send via WebSocket
        if (window.agentWebSocket) {
            window.agentWebSocket.sendMessage(content);
        }
    }

    addUserMessage(content) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message user';
        messageEl.innerHTML = `
            <div class="message-header">
                <span>You</span>
                <span>${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-content">${this.escapeHtml(content)}</div>
        `;

        this.chatMessagesEl.appendChild(messageEl);
        this.scrollToBottom();

        this.messages.push({
            role: 'user',
            content: content,
            timestamp: new Date().toISOString()
        });
    }

    addAgentMessage(content) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message agent';
        messageEl.innerHTML = `
            <div class="message-header">
                <span>Agent</span>
                <span>${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-content">${this.formatContent(content)}</div>
        `;

        this.chatMessagesEl.appendChild(messageEl);
        this.scrollToBottom();

        this.messages.push({
            role: 'agent',
            content: content,
            timestamp: new Date().toISOString()
        });
    }

    addSystemMessage(content) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message system';
        messageEl.innerHTML = `
            <div class="message-content">${this.escapeHtml(content)}</div>
        `;

        this.chatMessagesEl.appendChild(messageEl);
        this.scrollToBottom();
    }

    addToolExecution(toolName, content) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message agent';
        messageEl.innerHTML = `
            <div class="tool-execution">
                🔧 Executing tool: <strong>${toolName}</strong>
            </div>
        `;

        this.chatMessagesEl.appendChild(messageEl);
        this.scrollToBottom();
    }

    addToolResult(toolName, result) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message agent';
        messageEl.innerHTML = `
            <div class="tool-result">
                ✓ Tool result: <strong>${toolName}</strong>
                <pre>${this.escapeHtml(result.substring(0, 200))}${result.length > 200 ? '...' : ''}</pre>
            </div>
        `;

        this.chatMessagesEl.appendChild(messageEl);
        this.scrollToBottom();
    }

    updateLastAgentMessage(content) {
        // Find last agent message
        const messages = this.chatMessagesEl.querySelectorAll('.message.agent');
        if (messages.length === 0) {
            this.addAgentMessage(content);
            return;
        }

        const lastMessage = messages[messages.length - 1];
        const contentEl = lastMessage.querySelector('.message-content');
        if (contentEl) {
            contentEl.innerHTML = this.formatContent(content);
            this.scrollToBottom();
        }
    }

    async clearChat() {
        if (!this.activeSession) return;
        if (!confirm('Clear chat history for this session?')) return;

        console.log('[ChatInterface] Clearing chat history');

        try {
            // In Phase 3, send clear request to agent container
            // For now, just clear UI
            const systemMessages = this.chatMessagesEl.querySelectorAll('.message.system');
            this.chatMessagesEl.innerHTML = '';

            // Re-add system messages
            systemMessages.forEach(msg => this.chatMessagesEl.appendChild(msg.cloneNode(true)));

            this.addSystemMessage('Chat history cleared');
            this.messages = [];
        } catch (error) {
            console.error('[ChatInterface] Failed to clear chat:', error);
        }
    }

    async terminateSession() {
        if (!this.activeSession) return;

        if (window.sessionManager) {
            window.sessionManager.terminateSession(this.activeSession.session_id);
        }
    }

    formatContent(content) {
        // Basic markdown-like formatting
        let formatted = this.escapeHtml(content);

        // Code blocks
        formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

        // Inline code
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Line breaks
        formatted = formatted.replace(/\n/g, '<br>');

        return formatted;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom() {
        this.chatMessagesEl.scrollTop = this.chatMessagesEl.scrollHeight;
    }
}

// Export globally
window.ChatInterface = ChatInterface;
