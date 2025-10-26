/**
 * @typedef {Object} Message
 * @property {string} role - Message role ('user', 'assistant', 'system')
 * @property {string} content - Message content (may include XML-tagged tool calls)
 * @property {MessageMetadata} [metadata] - Optional message metadata
 */

/**
 * @typedef {Object} MessageMetadata
 * @property {string} [timestamp] - ISO 8601 timestamp
 * @property {string} [model] - LLM model used for generation
 * @property {ToolInfo[]} [tool_info] - Information about tools used
 * @property {number} [message_id] - Unique message identifier
 */

/**
 * @typedef {Object} ToolInfo
 * @property {string} name - Tool name
 * @property {string} [model] - Model used for tool invocation
 * @property {number} [execution_time] - Execution time in seconds
 * @property {string} [status] - Execution status ('success', 'error')
 */

/**
 * @typedef {Object} ChatData
 * @property {string} [id] - Unique chat identifier
 * @property {Message[]} messages - Array of chat messages
 * @property {ChatMetadata} [metadata] - Chat-level metadata
 */

/**
 * @typedef {Object} ChatMetadata
 * @property {string} [created_at] - ISO 8601 creation timestamp
 * @property {string} [updated_at] - ISO 8601 last update timestamp
 * @property {number} [bootstrap_message_count] - Number of initial bootstrap messages to skip in rendering
 */

/**
 * @typedef {Object} APIResponse
 * @property {string} status - Response status ('success' or 'error')
 * @property {*} [data] - Response data (structure varies by endpoint)
 * @property {string} [error] - Error message if status is 'error'
 * @property {string} [code] - Error code if status is 'error'
 */

/**
 * @typedef {Object} EventListeners
 * @property {Function[]} chat_loaded - Listeners for chat load events
 * @property {Function[]} chat_cleared - Listeners for chat clear events
 * @property {Function[]} message_added - Listeners for message add events
 * @property {Function[]} state_changed - Listeners for generic state change events
 */

/**
 * ChatStateManager - Frontend state management for chat
 *
 * Provides single source of truth for chat state on the client side.
 * Syncs with backend via API calls and WebSocket events.
 * Enables chat loading without page reload.
 *
 * @class
 * @fires ChatStateManager#chat_loaded
 * @fires ChatStateManager#chat_cleared
 * @fires ChatStateManager#message_added
 * @fires ChatStateManager#state_changed
 */
class ChatStateManager {
    /**
     * Creates a new ChatStateManager instance
     *
     * @param {Object} wsManager - WebSocket manager instance with on() method for event subscription
     * @param {Object} renderer - Content renderer instance with renderMessage() and clearContent() methods
     */
    constructor(wsManager, renderer) {
        /**
         * WebSocket manager for real-time communication
         * @type {Object}
         * @private
         */
        this.wsManager = wsManager;

        /**
         * Content renderer for displaying messages
         * @type {Object}
         * @private
         */
        this.renderer = renderer;

        /**
         * Array of all messages in the current chat
         * @type {Message[]}
         */
        this.messages = [];

        /**
         * Current chat identifier (null for new/unsaved chats)
         * @type {string|null}
         */
        this.currentChatId = null;

        /**
         * Chat-level metadata
         * @type {ChatMetadata}
         */
        this.metadata = {};

        /**
         * Event listener registry
         * @type {EventListeners}
         * @private
         */
        this.listeners = {
            'chat_loaded': [],
            'chat_cleared': [],
            'message_added': [],
            'state_changed': []
        };

        /**
         * Rendering queue to prevent race conditions during rapid updates
         * @type {Promise<void>}
         * @private
         */
        this.renderingQueue = Promise.resolve();

        /**
         * Pending render operation (currently unused, reserved for future use)
         * @type {Function|null}
         * @private
         */
        this.pendingRender = null;

        /**
         * Bootstrap message start index (cached to prevent re-calculation)
         * @type {number|undefined}
         * @private
         */
        this._bootstrapStartIdx = undefined;

        this.setupWebSocketHandlers();
        console.log('ChatStateManager initialized');
    }

    /**
     * Load current chat from backend (on page reload)
     * Fetches chat state from /api/chat/current and renders all messages
     *
     * @async
     * @returns {Promise<void>}
     * @fires ChatStateManager#chat_loaded
     * @fires ChatStateManager#state_changed
     */
    async loadCurrentChat() {
        try {
            console.log('Loading current chat from backend...');
            const url = window.backendConnector ? await window.backendConnector.buildURL('/api/chat/current') : '/api/chat/current';
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout for chat load

            try {
                const response = await fetch(url, {
                    headers: {
                        'X-Session-ID': window.sessionId || ''
                    },
                    signal: controller.signal
                });
                clearTimeout(timeoutId);

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
                clearTimeout(timeoutId);
                throw error;
            }
        } catch (error) {
            console.error('Failed to load current chat:', error);
            // Don't show error to user - might be a new chat
        }
    }

    /**
     * Load an archived chat by ID
     * Backend will replace g.chat and broadcast chat_replaced event
     *
     * @async
     * @param {string} chatId - Unique identifier of the chat to load
     * @returns {Promise<void>}
     * @throws {Error} If chat load request fails or backend returns error status
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
     *
     * @async
     * @returns {Promise<void>}
     * @throws {Error} If new chat request fails or backend returns error status
     */
    async startNewChat() {
        try {
            const url = window.backendConnector ? await window.backendConnector.buildURL('/api/chat/new') : '/api/chat/new';
            const response = await fetch(url, {
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
     *
     * @param {Message} message - Message object to add to chat state
     * @fires ChatStateManager#message_added
     * @fires ChatStateManager#state_changed
     */
    addMessage(message) {
        this.messages.push(message);
        this.emit('message_added', message);
        this.emit('state_changed');
    }

    /**
     * Replace entire chat state (called by WebSocket handler on chat_replaced event)
     * Used when loading an archived chat from the sidebar
     *
     * @param {ChatData} chatData - Complete chat data including messages, id, and metadata
     * @fires ChatStateManager#chat_loaded
     * @fires ChatStateManager#state_changed
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
     * Triggered when user starts a new chat
     *
     * @fires ChatStateManager#chat_cleared
     * @fires ChatStateManager#state_changed
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
     * Clears the DOM and re-renders all messages from current state.
     * Respects bootstrap_message_count to skip initial system messages.
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
     * Queue a render operation to prevent race conditions
     * Ensures renders happen sequentially, preventing concurrent DOM modifications
     *
     * @param {Function} renderFn - Async function that performs the render
     * @returns {Promise<void>} Promise that resolves when render completes
     */
    queueRender(renderFn) {
        // Chain the new render after the current queue completes
        this.renderingQueue = this.renderingQueue
            .then(() => renderFn())
            .catch(err => {
                console.error('Render error:', err);
            });
        return this.renderingQueue;
    }

    /**
     * Setup WebSocket event handlers for state synchronization
     * Registers handlers for session_init, chat_replaced, chat_cleared,
     * message_added, chat_truncated, and message_deleted events
     *
     * @private
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

        this.wsManager.on('message_added', async (data) => {
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

            // Queue the render to prevent race conditions from rapid sequential updates
            await this.queueRender(async () => {
                // Simple logic: if last message is assistant and incoming is assistant, replace it
                // (content/metadata has grown via add_message concatenation)
                if (lastMsg && lastMsg.role === 'assistant' && incomingMsg.role === 'assistant') {
                    console.log('Updating existing assistant message (content/metadata grew)');
                    const messageIndex = this.messages.length - 1;
                    this.messages[messageIndex] = incomingMsg;
                    this.emit('state_changed');

                    // Check if we have an optimistic message to remove
                    const contentWrapper = document.getElementById('content-wrapper');
                    const optimisticMsgId = window.app && window.app.currentOptimisticMessageId;

                    if (contentWrapper && optimisticMsgId) {
                        const optimisticElement = document.getElementById(optimisticMsgId);
                        if (optimisticElement) {
                            console.log('💬 Removing optimistic message, real assistant response arrived:', optimisticMsgId);
                            optimisticElement.remove();
                            window.app.currentOptimisticMessageId = null;
                        }
                    }

                    // Remove old DOM element and re-render
                    if (contentWrapper) {
                        const oldMessageBlock = contentWrapper.querySelector(`[data-message-index="${messageIndex}"]`);
                        if (oldMessageBlock) {
                            oldMessageBlock.remove();
                        }
                    }

                    // Re-render the updated message
                    await this.renderer.renderMessage(incomingMsg, messageIndex);

                    // Update interrupt button state based on whether response is complete
                    this.updateInterruptButtonState(incomingMsg);
                } else {
                    // New message (user message or new assistant response from agent loop)
                    console.log('Adding new message');

                    // Check if we have an optimistic message to remove (only for NEW assistant messages)
                    if (incomingMsg.role === 'assistant') {
                        const contentWrapper = document.getElementById('content-wrapper');
                        const optimisticMsgId = window.app && window.app.currentOptimisticMessageId;

                        if (contentWrapper && optimisticMsgId) {
                            const optimisticElement = document.getElementById(optimisticMsgId);
                            if (optimisticElement) {
                                console.log('💬 Removing optimistic message, real assistant response arrived:', optimisticMsgId);
                                optimisticElement.remove();
                                window.app.currentOptimisticMessageId = null;
                            }
                        }
                    }

                    this.addMessage(incomingMsg);
                    const messageIndex = this.messages.length - 1;
                    await this.renderer.renderMessage(incomingMsg, messageIndex);

                    // Show interrupt button for new assistant messages
                    if (incomingMsg.role === 'assistant') {
                        this.showInterruptButton();
                    }
                }
            });
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
     * Update interrupt button state based on message content
     * Shows interrupt button while response is generating, hides when complete
     *
     * @param {Message} message - Message to check for completion
     * @private
     */
    updateInterruptButtonState(message) {
        // Check if message contains visualresponse (response is complete)
        const hasVisualResponse = message.content && (
            message.content.includes('<visualresponse>') ||
            message.content.includes('<response>')
        );

        if (hasVisualResponse) {
            // Response is complete, disable button visually (but keep it functional)
            this.hideInterruptButton();
        } else {
            // Still generating, keep button active
            this.showInterruptButton();
        }
    }

    /**
     * Show interrupt button (visually active)
     * Delegates to window.app.showInterruptButton() if available
     *
     * @private
     */
    showInterruptButton() {
        if (window.app && typeof window.app.showInterruptButton === 'function') {
            window.app.showInterruptButton();
        }
    }

    /**
     * Hide interrupt button (visually disabled, but still functional)
     * Delegates to window.app.hideInterruptButton() if available
     *
     * @private
     */
    hideInterruptButton() {
        if (window.app && typeof window.app.hideInterruptButton === 'function') {
            window.app.hideInterruptButton();
        }
    }

    /**
     * Register an event listener
     * Implements event emitter pattern for reactive UI updates
     *
     * @param {string} event - Event name ('chat_loaded', 'chat_cleared', 'message_added', 'state_changed')
     * @param {Function} callback - Callback function to invoke when event is emitted
     */
    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    /**
     * Emit an event to all registered listeners
     * Catches and logs errors from individual listeners to prevent cascade failures
     *
     * @param {string} event - Event name to emit
     * @param {*} [data] - Data to pass to event listeners
     * @private
     */
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
     * Uses showToast() if available, otherwise logs to console
     *
     * @param {string} message - Success message to display
     * @private
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
     * Uses showToast() if available, otherwise logs to console
     *
     * @param {string} message - Error message to display
     * @private
     */
    showError(message) {
        if (typeof showToast === 'function') {
            showToast(message, 'error');
        } else {
            console.error('❌', message);
        }
    }
}

/**
 * Chat loaded event
 * Fired when a chat is loaded from backend (initial load or archive load)
 *
 * @event ChatStateManager#chat_loaded
 * @type {ChatData}
 */

/**
 * Chat cleared event
 * Fired when the chat is cleared (new chat started)
 *
 * @event ChatStateManager#chat_cleared
 * @type {undefined}
 */

/**
 * Message added event
 * Fired when a new message is added to the chat
 *
 * @event ChatStateManager#message_added
 * @type {Message}
 */

/**
 * State changed event
 * Fired whenever any state change occurs (catch-all event)
 *
 * @event ChatStateManager#state_changed
 * @type {undefined}
 */

window.ChatStateManager = ChatStateManager;
