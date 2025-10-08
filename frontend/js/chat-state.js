/**
 * ChatStateManager - Frontend state management for chat
 *
 * Provides single source of truth for chat state on the client side.
 * Syncs with backend via API calls and WebSocket events.
 * Enables chat loading without page reload.
 */

class ChatStateManager {
    constructor(wsManager, renderer) {
        this.wsManager = wsManager;
        this.renderer = renderer;

        this.messages = [];
        this.currentChatId = null;
        this.metadata = {};

        this.listeners = {
            'chat_loaded': [],
            'chat_cleared': [],
            'message_added': [],
            'state_changed': []
        };

        this.setupWebSocketHandlers();
        console.log('ChatStateManager initialized');
    }

    /**
     * Load current chat from backend (on page reload)
     */
    async loadCurrentChat() {
        try {
            console.log('Loading current chat from backend...');
            const response = await fetch('/api/chat/current', {
                headers: {
                    'X-Session-ID': window.sessionId || ''
                }
            });

            const data = await response.json();

            if (data.status === 'success' && data.chat) {
                console.log(`Loaded current chat with ${data.chat.messages ? data.chat.messages.length : 0} messages`);

                this.messages = data.chat.messages || [];
                this.currentChatId = data.chat.id || null;
                this.metadata = data.chat.metadata || {};

                // Render all messages
                if (this.messages.length > 0) {
                    this.renderAll();
                    console.log('✅ Chat messages rendered on page load');
                } else {
                    console.log('No messages to render (new chat)');
                }

                this.emit('chat_loaded', data.chat);
                this.emit('state_changed');
            } else {
                console.log('No active chat or empty response');
            }

        } catch (error) {
            console.error('Failed to load current chat:', error);
            // Don't show error to user - might be a new chat
        }
    }

    /**
     * Load an archived chat by ID
     * Backend will replace g.chat and broadcast chat_replaced event
     */
    async loadChat(chatId) {
        try {
            const response = await fetch(`/api/chat/load/${chatId}`, {
                method: 'POST',
                headers: {
                    'X-Session-ID': window.sessionId || ''
                }
            });

            const data = await response.json();

            if (data.status === 'success') {
                console.log('Chat load request successful, waiting for sync event...');
            } else {
                throw new Error(data.error || 'Failed to load chat');
            }

        } catch (error) {
            console.error('Failed to load chat:', error);
            this.showError(`Failed to load chat: ${error.message}`);
            throw error;
        }
    }

    /**
     * Start a new chat (archives current conversation)
     * Backend will archive and broadcast chat_cleared event
     */
    async startNewChat() {
        try {
            const response = await fetch('/api/chat/new', {
                method: 'POST',
                headers: {
                    'X-Session-ID': window.sessionId || ''
                }
            });

            const data = await response.json();

            if (data.status === 'success') {
                console.log('New chat request successful, waiting for sync event...');
            } else {
                throw new Error(data.error || 'Failed to start new chat');
            }

        } catch (error) {
            console.error('Failed to start new chat:', error);
            this.showError(`Failed to start new chat: ${error.message}`);
            throw error;
        }
    }

    /**
     * Add a message to state (called by WebSocket handler)
     */
    addMessage(message) {
        this.messages.push(message);
        this.emit('message_added', message);
        this.emit('state_changed');
    }

    /**
     * Replace entire chat state (called by WebSocket handler on chat_replaced event)
     */
    replaceChat(chatData) {
        console.log('Replacing chat state with loaded chat:', chatData);

        this.messages = chatData.messages || [];
        this.currentChatId = chatData.id || null;
        this.metadata = chatData.metadata || {};

        this.emit('chat_loaded', chatData);
        this.emit('state_changed');

        this.renderAll();

        this.showSuccess('Chat loaded successfully');
    }

    /**
     * Clear chat state (called by WebSocket handler on chat_cleared event)
     */
    clearChat() {
        console.log('Clearing chat state');

        this.messages = [];
        this.currentChatId = null;
        this.metadata = {};

        this.emit('chat_cleared');
        this.emit('state_changed');

        this.renderer.clearContent();

        this.showSuccess('New chat started. Previous conversation archived.');
    }

    /**
     * Re-render all messages in UI
     */
    renderAll() {
        console.log(`Rendering ${this.messages.length} messages`);

        this.renderer.clearContent();

        // On first render, use bootstrap_message_count from metadata
        // On subsequent renders, keep the same startIdx we used before
        if (this._bootstrapStartIdx === undefined) {
            this._bootstrapStartIdx = this.metadata.bootstrap_message_count || 0;
        }

        const startIdx = this._bootstrapStartIdx;

        if (startIdx > 0) {
            console.log(`Skipping first ${startIdx} bootstrap messages`);
        }

        for (let i = startIdx; i < this.messages.length; i++) {
            try {
                // Pass the actual array index (i) so tool_info and edit functionality work correctly
                this.renderer.renderMessage(this.messages[i], i);
            } catch (error) {
                console.error('Failed to render message:', error, this.messages[i]);
            }
        }
    }

    /**
     * Setup WebSocket event handlers for state synchronization
     */
    setupWebSocketHandlers() {
        if (!this.wsManager) {
            console.warn('WebSocket manager not available for ChatStateManager');
            return;
        }

        if (typeof this.wsManager.on !== 'function') {
            console.warn('WebSocket manager does not have on() method');
            return;
        }

        // Track if we've loaded the chat to avoid double-loading
        let chatLoaded = false;

        // Load chat immediately via HTTP, don't wait for WebSocket
        this.loadCurrentChat().then(() => {
            chatLoaded = true;
            console.log('✅ Initial chat load complete');
        }).catch(err => {
            console.warn('Failed to load chat via HTTP, will retry on session_init:', err);
        });

        this.wsManager.on('session_init', async (data) => {
            console.log('Session initialized');
            // Only reload if initial load failed or if this is a fresh session
            if (!chatLoaded) {
                await this.loadCurrentChat();
            } else {
                console.log('Chat already loaded, skipping duplicate load');
            }
        });

        this.wsManager.on('chat_replaced', (data) => {
            console.log('chat_replaced event received:', data);
            this.replaceChat(data.chat);
        });

        this.wsManager.on('chat_cleared', (data) => {
            console.log('chat_cleared event received:', data);
            this.clearChat();
        });

        this.wsManager.on('message_added', (data) => {
            console.log('message_added event received:', data);

            const incomingMsg = data.message;
            const lastMsg = this.messages[this.messages.length - 1];

            // DEBUG: Log tool_info to check if model is present
            if (incomingMsg.metadata && incomingMsg.metadata.tool_info) {
                console.log('🔍 tool_info received:', incomingMsg.metadata.tool_info);
                incomingMsg.metadata.tool_info.forEach((tool, idx) => {
                    console.log(`  Tool ${idx}: ${tool.name}, model: ${tool.model || 'MISSING'}`);
                });
            }

            // Simple logic: if last message is assistant and incoming is assistant, replace it
            // (content/metadata has grown via add_message concatenation)
            if (lastMsg && lastMsg.role === 'assistant' && incomingMsg.role === 'assistant') {
                console.log('Updating existing assistant message (content/metadata grew)');
                const messageIndex = this.messages.length - 1;
                this.messages[messageIndex] = incomingMsg;
                this.emit('state_changed');

                // Remove old DOM element and re-render
                const contentWrapper = document.getElementById('content-wrapper');
                if (contentWrapper) {
                    const oldMessageBlock = contentWrapper.querySelector(`[data-message-index="${messageIndex}"]`);
                    if (oldMessageBlock) {
                        oldMessageBlock.remove();
                    }
                }

                // Re-render the updated message
                this.renderer.renderMessage(incomingMsg, messageIndex);
            } else {
                // New message (user message or new assistant response from agent loop)
                console.log('Adding new message');
                this.addMessage(incomingMsg);
                const messageIndex = this.messages.length - 1;
                this.renderer.renderMessage(incomingMsg, messageIndex);
            }
        });

        this.wsManager.on('chat_truncated', (data) => {
            console.log('chat_truncated event received:', data);
            this.messages = this.messages.slice(0, data.index + 1);
            this.renderAll();
            this.emit('state_changed');
        });

        this.wsManager.on('message_deleted', (data) => {
            console.log('message_deleted event received:', data);
            if (data.index >= 0 && data.index < this.messages.length) {
                this.messages.splice(data.index, 1);
                this.renderAll();
                this.emit('state_changed');
            }
        });

        console.log('WebSocket handlers registered for chat sync');
    }

    /**
     * Event emitter pattern for reactive UI updates
     */
    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    emit(event, data) {
        if (this.listeners[event]) {
            for (const callback of this.listeners[event]) {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in ${event} listener:`, error);
                }
            }
        }
    }

    /**
     * Show success message to user
     */
    showSuccess(message) {
        if (typeof showToast === 'function') {
            showToast(message, 'success');
        } else {
            console.log('✅', message);
        }
    }

    /**
     * Show error message to user
     */
    showError(message) {
        if (typeof showToast === 'function') {
            showToast(message, 'error');
        } else {
            console.error('❌', message);
        }
    }
}

window.ChatStateManager = ChatStateManager;
