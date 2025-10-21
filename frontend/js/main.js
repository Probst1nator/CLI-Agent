/**
 * Visual Response Interface Main Application Controller
 *
 * The central orchestrator for the CLI-Agent visual response interface.
 * This class coordinates all major components and handles initialization,
 * configuration, and lifecycle management.
 *
 * Key responsibilities:
 * - Initialize and coordinate ContentRenderer and WebSocketManager
 * - Setup theme switching and user preferences
 * - Configure markdown rendering with Mermaid diagram support
 * - Handle library loading and dependency management
 * - Manage chat input and form submissions
 * - Provide syntax highlighting for code blocks
 *
 * Architecture:
 * VisualResponseApp
 * ├── ContentRenderer (handles visualization rendering)
 * ├── WebSocketManager (handles agent communication)
 * ├── Theme Manager (handles light/dark mode)
 * └── Chat Interface (handles user input)
 *
 * @class VisualResponseApp
 * @author CLI-Agent Development Team
 * @version 1.0.0
 */

// ===== TYPE DEFINITIONS =====

/**
 * System configuration object matching API response structure
 * @typedef {Object} SystemConfiguration
 * @property {string[]} selected_llms - Array of selected LLM model IDs (sent from API)
 * @property {string[]} [fallback_models] - Array of fallback model IDs (computed from selected_llms)
 * @property {string|null} router_llm - Router LLM model ID or null
 * @property {string|null} planner_llm - Planner LLM model ID or null
 * @property {Object.<string, any>} [additionalConfig] - Additional dynamic configuration
 */

/**
 * LLM model information structure
 * @typedef {Object} LLMModel
 * @property {string} id - Unique model identifier
 * @property {string} name - Human-readable model name
 * @property {string} provider - Model provider (e.g., 'google', 'ollama', 'groq')
 * @property {string[]} [capabilities] - Model capabilities array
 * @property {number} [context_window] - Context window size
 * @property {boolean} [isNoneOption] - Whether this represents a "None" option
 */

/**
 * Message history entry for localStorage persistence
 * @typedef {Object} MessageHistoryEntry
 * @property {'user'|'agent'} type - Type of message
 * @property {number} timestamp - Unix timestamp
 * @property {UserMessageData|AgentMessageData} data - Message data
 */

/**
 * User message data structure
 * @typedef {Object} UserMessageData
 * @property {string} message - User's message content
 * @property {number} timestamp - Message timestamp
 */

/**
 * Agent message data structure
 * @typedef {Object} AgentMessageData
 * @property {Object} messageData - The actual message content/data
 * @property {string} messageData.type - Message type (e.g., 'html', 'markdown', 'text')
 * @property {string} messageData.content - Message content
 * @property {ToolInfo[]} [messageData.tool_info] - Tool execution information
 * @property {string} [originalId] - Original message ID
 * @property {number} timestamp - Message timestamp
 */

/**
 * Tool execution information
 * @typedef {Object} ToolInfo
 * @property {string} name - Tool name
 * @property {'success'|'error'} status - Execution status
 * @property {number} duration_ms - Execution duration in milliseconds
 * @property {string} [error] - Error message if status is 'error'
 * @property {string} [args] - Tool arguments
 * @property {string} [context_returned] - Tool output/context
 */

/**
 * Permission request data structure
 * @typedef {Object} PermissionData
 * @property {string} id - Permission request ID
 * @property {string} tool_name - Name of the tool requesting permission
 * @property {string} description - Description of what the tool wants to do
 * @property {Object} [details] - Additional permission details
 */

/**
 * LLM selector options configuration
 * @typedef {Object} LLMSelectorOptions
 * @property {string} placeholder - Placeholder text
 * @property {string} id - Unique selector ID
 * @property {boolean} [includeNoneOption] - Whether to include "None" option
 * @property {function(LLMModel): void} onSelectionChange - Selection change callback
 * @property {function(SearchableLLMSelectbox, LLMModel[], SystemConfiguration): void} onModelsLoaded - Models loaded callback
 */

/**
 * Console log entry structure
 * @typedef {Object} ConsoleLogEntry
 * @property {string} timestamp - Formatted timestamp
 * @property {string} line - Log line content
 * @property {'info'|'warning'|'error'|'debug'} level - Log level
 */

/**
 * Upload response structure
 * @typedef {Object} UploadResponse
 * @property {'success'|'error'} status - Upload status
 * @property {string} message - Status message
 * @property {string[]} [files] - Array of uploaded file paths
 */

/**
 * File upload status types
 * @typedef {'uploading'|'success'|'error'} UploadStatusType
 */

/**
 * Theme type
 * @typedef {'light'|'dark'} ThemeType
 */
class VisualResponseApp {
    /**
     * Initialize the Visual Response Application
     * Sets up all components and establishes the application state
     */
    constructor() {
        /** @type {ContentRenderer|null} */
        this.renderer = null;

        /** @type {WebSocketManager|null} */
        this.wsManager = null;

        /** @type {HTMLInputElement|null} */
        this.chatInput = null;

        /** @type {HTMLFormElement|null} */
        this.chatForm = null;

        /** @type {MessageHistoryEntry[]} */
        this.messageHistory = [];

        /** @type {HTMLInputElement|null} */
        this.fileInput = null;

        /** @type {HTMLButtonElement|null} */
        this.fileUploadBtn = null;

        /** @type {HTMLButtonElement|null} */
        this.interruptBtn = null;

        /** @type {Object.<string, SearchableLLMSelectbox>} */
        this.llmSelectors = {};

        /** @type {Object.<string, boolean>} */
        this.llmSelectorsLoaded = { router: false, planner: false, primary: false, fallback1: false, fallback2: false };

        /** @type {SystemConfiguration|null} */
        this.systemConfiguration = null;

        /** @type {Object.<string, LLMModel|null>} */
        this.llmSelections = {};

        /** @type {LLMModel|null} */
        this.plannerLLMSelection = null;

        /** @type {boolean} */
        this.isInitializing = true;

        /** @type {boolean} */
        this.initializedFromSystem = false;

        /** @type {boolean} */
        this.isRestoring = false;

        /** @type {boolean} */
        this.agentGenerating = false;

        /** @type {string|null} */
        this.activeProfileSlot = null;

        /** @type {Object.<string, Object>} */
        this.profileSlots = {};

        /** @type {number|null} */
        this.autoSaveTimeout = null;

        /** @type {Logger} */
        this.logger = window.visualLogger;

        this.logger.info('VisualResponseApp starting initialization');
        this.logger.startTiming('app_initialization');

        this.init();
    }

    init() {
        console.group('🚀 Visual Response App Initialization');
        this.checkLibraries();
        this.initializeComponents();
        this.setupEventListeners();
        this.setupStateSync();
        this.setDarkTheme();
        // loadMessageHistory() is now triggered by the WebSocket config message from server
        // This ensures we respect the continue/clear_history flags from the CLI
        console.log('✅ Visual Response App initialized successfully');
        console.groupEnd();
    }

    setupStateSync() {
        // Listen for state changes from backend
        if (this.wsManager) {
            this.wsManager.on('state_changed', (data) => {
                console.log('📡 State changed:', data.state);
                this.updateButtonStates(data.state);
            });

            // Listen for message_added events to trigger TTS
            this.wsManager.on('message_added', async (data) => {
                // Only trigger TTS for assistant messages with visualresponse/response tags
                if (this.speakModeEnabled && data.message && data.message.role === 'assistant' && data.message.content) {
                    // Extract text ONLY from visualresponse or response tags
                    const content = data.message.content;
                    const visualResponseMatch = /<visualresponse>\s*(?:<action>.*?<\/action>\s*)?<type>.*?<\/type>\s*<content>([\s\S]*?)<\/content>\s*<\/visualresponse>/i.exec(content);
                    const responseMatch = /<response>(.*?)<\/response>/is.exec(content);

                    let text = '';
                    if (visualResponseMatch) {
                        text = visualResponseMatch[1].trim();
                        // Strip HTML and markdown
                        text = text.replace(/<[^>]+>/g, ' ').replace(/[#*_`]/g, '');
                    } else if (responseMatch) {
                        text = responseMatch[1].trim();
                    }
                    // No fallback - only speak visualresponse/response content

                    text = text.replace(/\s+/g, ' ').trim();

                    if (text && text.trim()) {
                        // Check if user has interacted yet (browser autoplay policy)
                        if (!this.audioManager.hasUserInteraction) {
                            console.log('⏸️ TTS queued - waiting for user interaction (browser autoplay policy)');
                            this.pendingTTS = text;
                            this.showAutoplayPrompt();
                        } else {
                            console.log('🔊 Final response received, speak mode active - triggering TTS');
                            try {
                                await this.audioManager.playTTS(text, 'auto');
                            } catch (error) {
                                console.error('TTS playback failed:', error);
                            }
                        }
                    }
                }
            });

            // Legacy: Listen for message_end events (not currently sent by backend)
            this.wsManager.on('message_end', async (data) => {
                if (this.speakModeEnabled && data.content) {
                    console.log('🔊 Message ended, speak mode active - triggering TTS');

                    // Extract text from visualresponse or response tags first
                    const content = data.content;
                    const visualResponseMatch = /<visualresponse>\s*(?:<action>.*?<\/action>\s*)?<type>.*?<\/type>\s*<content>([\s\S]*?)<\/content>\s*<\/visualresponse>/i.exec(content);
                    const responseMatch = /<response>(.*?)<\/response>/is.exec(content);

                    let text = '';
                    if (visualResponseMatch) {
                        text = visualResponseMatch[1].trim();
                        // Strip HTML and markdown
                        text = text.replace(/<[^>]+>/g, ' ').replace(/[#*_`]/g, '');
                    } else if (responseMatch) {
                        text = responseMatch[1].trim();
                    } else {
                        // Fallback to extractTextContent for non-visualresponse messages
                        text = this.extractTextContent(content);
                    }

                    text = text.replace(/\s+/g, ' ').trim();

                    if (text && text.trim()) {
                        try {
                            await this.audioManager.playTTS(text, 'auto');
                        } catch (error) {
                            console.error('TTS playback failed:', error);
                        }
                    }
                }
            });
        }

        // Fetch initial state when connected
        this.fetchAgentState();
    }

    async fetchAgentState() {
        try {
            const url = window.backendConnector ? await window.backendConnector.buildURL('/api/agent/state') : '/api/agent/state';
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

            try {
                const response = await fetch(url, { signal: controller.signal });
                clearTimeout(timeoutId);
                const data = await response.json();
                if (data.status === 'success' && data.state) {
                    console.log('📥 Initial state loaded:', data.state);
                    this.updateButtonStates(data.state);
                }
            } catch (error) {
                clearTimeout(timeoutId);
                throw error;
            }
        } catch (error) {
            console.warn('Failed to fetch agent state:', error);
        }
    }

    updateButtonStates(state) {
        // Update voice button
        if (this.voiceBtn) {
            if (state.voice_mode && !state.tts_only_mode) {
                this.voiceBtn.classList.add('active');
                this.voiceBtn.disabled = false;
            } else {
                this.voiceBtn.classList.remove('active');
            }
        }

        // Update speak button and sync frontend state
        // Only update if user hasn't manually overridden speak mode
        if (this.speakBtn && !this.speakModeManualOverride) {
            if (state.tts_only_mode) {
                this.speakBtn.classList.add('active');
                this.speakBtn.disabled = false;
                this.voiceBtn.classList.remove('active');
                this.speakModeEnabled = true;
            } else if (state.voice_mode) {
                // Voice mode includes TTS, so enable TTS playback and show as active
                this.speakBtn.classList.add('active');
                this.speakBtn.disabled = false;
                this.speakModeEnabled = true;
            } else {
                this.speakBtn.classList.remove('active');
                this.speakModeEnabled = false;
            }
        }

        // Note: auto_execute, online_mode, local_mode don't have visible buttons in the UI
        // but you can add them here if needed
    }

    extractTextContent(content) {
        /**
         * Extract plain text from agent response, stripping XML tags and other markup.
         * Used for TTS to get clean readable text.
         */
        if (!content || typeof content !== 'string') {
            return '';
        }

        let text = content;

        // Remove visualresponse XML blocks
        text = text.replace(/<visualresponse>[\s\S]*?<\/visualresponse>/gi, '');

        // Remove other common XML-like tags
        text = text.replace(/<[^>]+>/g, '');

        // Remove code blocks
        text = text.replace(/```[\s\S]*?```/g, '');

        // Remove inline code
        text = text.replace(/`[^`]+`/g, '');

        // Clean up extra whitespace
        text = text.replace(/\s+/g, ' ').trim();

        return text;
    }

    async speakLatestResponse() {
        /**
         * Extract and speak the latest assistant message's visualresponse content.
         * This is called when the volume button is toggled on.
         */
        if (!this.chatState || !this.chatState.messages) {
            console.warn('No chat state available');
            return;
        }

        // Find the last assistant message
        const messages = this.chatState.messages;
        let lastAssistantMessage = null;
        for (let i = messages.length - 1; i >= 0; i--) {
            if (messages[i].role === 'assistant') {
                lastAssistantMessage = messages[i];
                break;
            }
        }

        if (!lastAssistantMessage || !lastAssistantMessage.content) {
            console.warn('No assistant message found to speak');
            return;
        }

        // Extract text from visualresponse content
        const content = lastAssistantMessage.content;

        // Try to extract visualresponse content first
        const visualResponseMatch = /<visualresponse>\s*(?:<action>.*?<\/action>\s*)?<type>.*?<\/type>\s*<content>([\s\S]*?)<\/content>\s*<\/visualresponse>/i.exec(content);
        const responseMatch = /<response>(.*?)<\/response>/is.exec(content);

        let textToSpeak = '';

        if (visualResponseMatch) {
            // Extract from visualresponse
            textToSpeak = visualResponseMatch[1].trim();
            // If it's HTML or markdown, strip tags
            textToSpeak = textToSpeak.replace(/<[^>]+>/g, ' ').replace(/[#*_`]/g, '');
        } else if (responseMatch) {
            // Extract from response tag
            textToSpeak = responseMatch[1].trim();
        } else {
            // Fallback: use extractTextContent to clean the entire message
            textToSpeak = this.extractTextContent(content);
        }

        // Clean up whitespace
        textToSpeak = textToSpeak.replace(/\s+/g, ' ').trim();

        if (textToSpeak && this.audioManager) {
            console.log('🔊 Speaking latest response:', textToSpeak.substring(0, 100) + '...');
            try {
                await this.audioManager.playTTS(textToSpeak, 'auto');
            } catch (error) {
                console.error('Failed to play TTS:', error);
            }
        } else {
            console.warn('No text to speak or audio manager not available');
        }
    }

    handleMessageEdit(messageBlock, messageIndex) {
        const contentDiv = messageBlock.querySelector('.message-content');
        if (!contentDiv) return;

        // Get raw content from chat state instead of rendered text
        let originalContent = contentDiv.textContent;
        if (this.chatState && this.chatState.messages && this.chatState.messages[messageIndex]) {
            // Use the raw message content from chat state (includes XML for assistant messages)
            originalContent = this.chatState.messages[messageIndex].content || originalContent;
        }

        const textarea = document.createElement('textarea');
        textarea.className = 'message-edit-textarea';
        textarea.value = originalContent;

        // Auto-resize function
        const autoResize = () => {
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        };

        // Auto-resize on input
        textarea.addEventListener('input', autoResize);

        const saveBtn = document.createElement('button');
        saveBtn.className = 'message-edit-save';
        saveBtn.textContent = '✓ Save';

        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'message-edit-cancel';
        cancelBtn.textContent = '✗ Cancel';

        const btnContainer = document.createElement('div');
        btnContainer.className = 'message-edit-buttons';
        btnContainer.appendChild(saveBtn);
        btnContainer.appendChild(cancelBtn);

        contentDiv.replaceWith(textarea);
        messageBlock.appendChild(btnContainer);

        // Set initial height after adding to DOM
        setTimeout(() => {
            autoResize();
            textarea.focus();
        }, 0);

        const saveEdit = async () => {
            const newContent = textarea.value.trim();
            if (!newContent || newContent === originalContent) {
                cancelEdit();
                return;
            }

            try {
                const url = window.backendConnector ? await window.backendConnector.buildURL('/api/chat/message/edit') : '/api/chat/message/edit';
                const response = await fetch(url, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message_index: parseInt(messageBlock.dataset.messageIndex),
                        new_content: newContent
                    })
                });

                const data = await response.json();
                if (data.status === 'success') {
                    // Update chat state with new content
                    if (this.chatState && this.chatState.messages && this.chatState.messages[messageIndex]) {
                        this.chatState.messages[messageIndex].content = newContent;
                    }

                    // Re-render the entire message properly
                    btnContainer.remove();
                    if (this.chatState) {
                        await this.chatState.renderAll();
                    } else {
                        // Fallback: just replace with text content
                        const newContentDiv = document.createElement('div');
                        newContentDiv.className = 'message-content';
                        newContentDiv.textContent = newContent;
                        textarea.replaceWith(newContentDiv);
                    }

                    messageBlock.classList.add('message-edited');
                    setTimeout(() => messageBlock.classList.remove('message-edited'), 2000);
                } else {
                    alert('Failed to save edit: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                console.error('Edit save error:', error);
                alert('Failed to save edit');
            }
        };

        const cancelEdit = async () => {
            // Re-render the message properly instead of just showing text
            btnContainer.remove();
            if (this.chatState) {
                await this.chatState.renderAll();
            } else {
                // Fallback: just replace with text content
                const newContentDiv = document.createElement('div');
                newContentDiv.className = 'message-content';
                newContentDiv.textContent = originalContent;
                textarea.replaceWith(newContentDiv);
            }
        };

        saveBtn.onclick = saveEdit;
        cancelBtn.onclick = cancelEdit;

        textarea.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                cancelEdit();
            } else if (e.key === 'Enter' && e.ctrlKey) {
                saveEdit();
            }
        });
    }

    async handleMessageDelete(messageIndex) {
        const confirmed = confirm(
            `Delete this message?\n\n` +
            `This will permanently remove this message from the chat history.\n\n` +
            `Are you sure?`
        );

        if (!confirmed) {
            return;
        }

        try {
            const response = await fetch(`/api/chat/message/delete/${messageIndex}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-ID': window.sessionId || ''
                }
            });

            const data = await response.json();

            if (data.status === 'success') {
                console.log(`Message deleted at index ${messageIndex}`);

                // Remove the message block from DOM
                const contentWrapper = document.getElementById('content-wrapper');
                if (contentWrapper) {
                    const messageBlocks = contentWrapper.querySelectorAll('.message-block');
                    if (messageBlocks[messageIndex]) {
                        messageBlocks[messageIndex].remove();

                        // Update data-message-index for all subsequent messages
                        messageBlocks.forEach((block, index) => {
                            if (index > messageIndex) {
                                block.dataset.messageIndex = index - 1;
                            }
                        });
                    }
                }

                const notification = document.createElement('div');
                notification.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #dc3545; color: white; padding: 12px 20px; border-radius: 4px; z-index: 10000; font-size: 14px;';
                notification.textContent = `✓ Message deleted`;
                document.body.appendChild(notification);

                setTimeout(() => {
                    notification.remove();
                }, 2000);
            } else {
                alert('Failed to delete message: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Failed to delete message:', error);
            alert('Failed to delete message. Check console for details.');
        }
    }

    async handleMessageReload(messageIndex) {
        const confirmed = confirm(
            `Continue from this message?\n\n` +
            `This will remove all messages after this point and allow you to continue the conversation from here.\n\n` +
            `Are you sure?`
        );

        if (!confirmed) {
            return;
        }

        try {
            const response = await fetch(`/api/chat/truncate/${messageIndex}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-ID': window.sessionId || ''
                }
            });

            const data = await response.json();

            if (data.status === 'success') {
                console.log(`Chat truncated to ${data.new_length} messages (removed ${data.removed_count})`);

                const contentWrapper = document.getElementById('content-wrapper');
                if (contentWrapper) {
                    const messageBlocks = contentWrapper.querySelectorAll('.message-block');
                    messageBlocks.forEach((block, index) => {
                        if (index > messageIndex) {
                            block.remove();
                        }
                    });
                }

                this.chatInput.focus();

                const notification = document.createElement('div');
                notification.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #28a745; color: white; padding: 12px 20px; border-radius: 4px; z-index: 10000; font-size: 14px;';
                notification.textContent = `✓ Chat truncated. ${data.removed_count} message(s) removed. Continue your conversation...`;
                document.body.appendChild(notification);

                setTimeout(() => {
                    notification.remove();
                }, 3000);
            } else {
                alert('Failed to truncate chat: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Failed to truncate chat:', error);
            alert('Failed to truncate chat. Check console for details.');
        }
    }

    displayInstanceInfo() {
        // Show which instance/port this tab is connected to
        const port = window.location.port;
        let instanceInfo = '';
        
        if (port === '7861') {
            instanceInfo = '🟢 Instance 1 (Port 7861)';
        } else if (port === '7862') {
            instanceInfo = '🟡 Instance 2 (Port 7862)';  
        } else if (port === '7863') {
            instanceInfo = '🔵 Instance 3 (Port 7863)';
        } else if (port === '7860') {
            instanceInfo = '⚪ Session Manager (Port 7860)';
        } else {
            instanceInfo = `⚫ Unknown Instance (Port ${port})`;
        }
        
        // Add instance indicator to header
        const header = document.querySelector('.header h1, h1') || document.querySelector('body');
        if (header) {
            const instanceDiv = document.createElement('div');
            instanceDiv.id = 'instance-indicator';
            instanceDiv.style.cssText = `
                position: fixed; 
                top: 10px; 
                right: 10px; 
                background: rgba(0, 0, 0, 0.8); 
                color: white; 
                padding: 8px 12px; 
                border-radius: 5px; 
                font-size: 12px; 
                font-family: monospace; 
                z-index: 10000;
                border: 1px solid #444;
            `;
            instanceDiv.textContent = instanceInfo;
            document.body.appendChild(instanceDiv);
        }
        
        console.log('🏷️ Connected to:', instanceInfo);
    }

    checkLibraries() {
        console.groupCollapsed('📚 Library Dependencies');
        console.log('jQuery:', typeof $ !== 'undefined' ? '✅ loaded' : '❌ MISSING');
        console.log('Marked:', typeof marked !== 'undefined' ? '✅ loaded' : '❌ MISSING');
        console.log('Mermaid:', typeof mermaid !== 'undefined' ? '✅ loaded' : '❌ MISSING');
        console.log('Plotly:', typeof Plotly !== 'undefined' ? '✅ loaded' : '❌ MISSING');
        console.log('3Dmol:', typeof $3Dmol !== 'undefined' ? '✅ loaded' : '❌ MISSING');
        console.log('svgPanZoom:', typeof svgPanZoom !== 'undefined' ? '✅ loaded' : '❌ MISSING');
        console.log('React:', typeof React !== 'undefined' ? '✅ loaded' : '❌ MISSING');
        console.log('ReactDOM:', typeof ReactDOM !== 'undefined' ? '✅ loaded' : '❌ MISSING');
        console.log('Vue:', typeof Vue !== 'undefined' ? '✅ loaded' : '❌ MISSING');
        console.log('Babel:', typeof Babel !== 'undefined' ? '✅ loaded' : '❌ MISSING');
        console.groupEnd();
    }

    initializeComponents() {
        // Initialize Mermaid with proper configuration
        if (typeof mermaid !== 'undefined') {
            mermaid.initialize({
                startOnLoad: false,
                theme: 'default',
                securityLevel: 'loose',
                htmlLabels: true
            });
            window.visualLogger.debug('Mermaid initialized successfully');
        } else {
            console.error('Mermaid library not loaded!');
        }

        // Configure Marked.js renderer for enhanced code block handling
        this.setupMarkedRenderer();

        // Initialize renderer
        this.renderer = new ContentRenderer();

        // Initialize WebSocket manager
        const statusDiv = document.getElementById('status');
        const contentWrapper = document.getElementById('content-wrapper');
        this.wsManager = new WebSocketManager(this.renderer, statusDiv, contentWrapper);

        // Set app reference for loading indicator control
        this.wsManager.setApp(this);

        // Expose WebSocket manager globally for permission responses
        window.wsManager = this.wsManager;

        // Initialize ChatStateManager for new API integration
        if (typeof ChatStateManager !== 'undefined') {
            this.chatState = new ChatStateManager(this.wsManager, this.renderer);
            window.chatState = this.chatState;
            this.logger.info('ChatStateManager initialized');
        } else {
            this.logger.warn('ChatStateManager not available');
        }

        // Initialize AudioManager for voice features
        if (typeof AudioManager !== 'undefined') {
            this.audioManager = new AudioManager();
            window.audioManager = this.audioManager;
            this.logger.info('AudioManager initialized');
        } else {
            this.logger.warn('AudioManager not available');
        }

        // Get chat elements
        this.chatForm = document.getElementById('chat-form');
        this.chatInput = document.getElementById('chat-input');
        
        // File upload elements
        this.fileInput = document.getElementById('file-input');
        this.fileUploadBtn = document.getElementById('file-upload-btn');
        this.screenshotBtn = document.getElementById('screenshot-btn');
        this.screenCaptureBtn = document.getElementById('screen-capture-btn');
        this.voiceBtn = document.getElementById('voice-btn');
        this.speakBtn = document.getElementById('speak-btn');

        // Initialize screen capture manager
        this.screenCaptureManager = new ScreenCaptureManager();
        this.screenCaptureEnabled = false;

        // Frontend-controlled speak mode state
        this.speakModeEnabled = false;
        this.speakModeManualOverride = false; // Track if user manually toggled speak mode
        this.pendingTTS = null; // Queue TTS waiting for user interaction

        // Initialize Dynamic LLM Manager for multi-model fallback support
        this.initializeDynamicLLMManager();
    }

    setupMarkedRenderer() {
        if (typeof marked === 'undefined') return;

        const renderer = new marked.Renderer();
        const originalCodeRenderer = renderer.code;

        renderer.code = (code, language, escaped) => {
            // Handle case where code might be an object with a 'text' property
            let codeStr;
            if (typeof code === 'object' && code !== null) {
                codeStr = code.text || code.raw || String(code);
            } else {
                codeStr = String(code || '');
            }

            window.visualLogger.debug('Marked renderer processing code block', { language, escaped, codeLength: codeStr.length });

            // Handle both 'mermaid' and undefined/null language cases by checking code content
            const isMermaidCode = language === 'mermaid' ||
                                 ((!language || language === 'undefined') &&
                                  (codeStr.trim().startsWith('graph ') ||
                                   codeStr.trim().startsWith('flowchart ') ||
                                   codeStr.trim().startsWith('sequenceDiagram') ||
                                   codeStr.trim().startsWith('classDiagram') ||
                                   codeStr.trim().startsWith('erDiagram')));

            if (isMermaidCode) {
                window.visualLogger.debug('Mermaid code block detected');
                const uniqueId = 'mermaid-embedded-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
                const placeholder = '<div class="mermaid-placeholder" data-mermaid-id="' + uniqueId + '" data-mermaid-definition="' + encodeURIComponent(codeStr) + '" style="text-align:center;"></div>';
                return placeholder;
            }

            // Apply syntax highlighting for regular code blocks
            const highlightedCode = this.highlightCode(codeStr, language);
            return '<pre><code class="language-' + (language || 'text') + '">' + highlightedCode + '</code></pre>';
        };

        marked.use({ renderer });
        console.log('Marked renderer configured');
    }

    /**
     * Apply basic syntax highlighting to code
     * @param {string} code - Code to highlight
     * @param {string} language - Programming language
     * @returns {string} Highlighted code HTML
     */
    highlightCode(code, language) {
        if (!language || language === 'text') return code;

        // Basic patterns for common languages
        const patterns = {
            javascript: [
                { pattern: /\b(function|const|let|var|if|else|for|while|return|class|import|export|async|await)\b/g, className: 'hljs-keyword' },
                { pattern: /(["'`])((?:\\.|(?!\1)[^\\])*)\1/g, className: 'hljs-string' },
                { pattern: /\/\/.*$/gm, className: 'hljs-comment' },
                { pattern: /\/\*[\s\S]*?\*\//g, className: 'hljs-comment' },
                { pattern: /\b(\d+\.?\d*)\b/g, className: 'hljs-number' }
            ],
            python: [
                { pattern: /\b(def|class|if|else|elif|for|while|import|from|return|try|except|with|as)\b/g, className: 'hljs-keyword' },
                { pattern: /(["'`])((?:\\.|(?!\1)[^\\])*)\1/g, className: 'hljs-string' },
                { pattern: /#.*$/gm, className: 'hljs-comment' },
                { pattern: /\b(\d+\.?\d*)\b/g, className: 'hljs-number' }
            ]
        };

        if (patterns[language]) {
            let highlightedCode = code;
            patterns[language].forEach(pattern => {
                highlightedCode = highlightedCode.replace(pattern.pattern, (match) => {
                    return '<span class="' + pattern.className + '">' + match + '</span>';
                });
            });
            return highlightedCode;
        }

        return code;
    }

    setupEventListeners() {
        if (!this.chatForm || !this.chatInput) {
            console.error('Chat form elements not found');
            return;
        }

        this.chatForm.addEventListener('submit', (event) => {
            event.preventDefault();
            this.handleChatSubmit();
        });

        // File upload event listeners
        this.setupFileUpload();

        // Screenshot button event listener
        this.setupScreenshot();

        // Screen capture button event listener
        this.setupScreenCaptureButton();

        // Voice button event listener
        this.setupVoiceButton();

        // Speak button event listener
        this.setupSpeakButton();

        // Global JS toggle event listener
        this.setupGlobalJSToggle();


        // Setup interrupt button
        const interruptBtn = document.getElementById('interrupt-btn');
        if (interruptBtn) {
            this.setupInterruptButton(interruptBtn);
        }

        // Iframe focus management
        this.setupIframeFocusManagement();

        // Agent message height management
        // this.setupAgentMessageHeightManagement();

        // Profile slot event listeners
        this.setupProfileSlots();
    }

    async handleChatSubmit() {
        const message = this.chatInput.value.trim();

        if (message && this.wsManager.isConnected()) {
            const images = [];

            // Disable send button to prevent double-send
            const sendButton = document.querySelector('#chat-form button[type="submit"]');
            if (sendButton) {
                sendButton.disabled = true;
                sendButton.classList.add('processing');
            }

            try {
                // Capture screen if continuous capture is enabled
                if (this.screenCaptureEnabled) {
                    try {
                        console.log('📸 Capturing screenshot from continuous stream...');
                        const screenshot = await this.screenCaptureManager.captureFromStream();
                        images.push(screenshot);
                        console.log('✅ Screenshot captured');
                    } catch (error) {
                        console.error('❌ Screenshot capture failed:', error);
                    }
                }

                // Only show interrupt button for non-command messages (commands don't generate agent responses)
                if (!message.startsWith('/')) {
                    this.showInterruptButton();
                }

                // Send the message with images and tool configuration
                // Note: Don't display user message here - the message_added WebSocket event will handle rendering
                const messageWithConfig = {
                    message: message,
                    images: images,
                    toolConfig: window.ToolManager ? window.ToolManager.getToolConfiguration() : null
                };

                const sendSuccess = this.wsManager.sendMessage(JSON.stringify(messageWithConfig));

                if (!sendSuccess) {
                    throw new Error('Failed to send message via WebSocket');
                }

                // Show loading indicator AFTER message is sent (will appear after user message)
                if (!message.startsWith('/')) {
                    setTimeout(() => {
                        this.showLoadingIndicator();
                    }, 50);
                }

                // Clear input only after successful send
                this.chatInput.value = '';
                this.chatInput.focus();

                // Reset screenshot state if a screenshot was ready (screenshot has been sent)
                this.resetScreenshotState();

            } catch (error) {
                console.error('❌ Failed to send message:', error);

                // Show error state in loading indicator
                this.showErrorIndicator('Failed to send message. Please try again.');

                // Hide interrupt button since message failed to send
                this.hideInterruptButton();

                // Restore input placeholder with error feedback
                const originalPlaceholder = this.chatInput.placeholder;
                this.chatInput.placeholder = '❌ Message failed to send';
                this.chatInput.style.borderColor = '#dc3545';

                setTimeout(() => {
                    this.chatInput.placeholder = originalPlaceholder;
                    this.chatInput.style.borderColor = '';
                }, 3000);
            } finally {
                // Re-enable send button
                if (sendButton) {
                    sendButton.disabled = false;
                    sendButton.classList.remove('processing');
                }
            }

        } else if (!this.wsManager.isConnected()) {
            // Visual feedback for connection error
            const originalPlaceholder = this.chatInput.placeholder;
            this.chatInput.placeholder = '❌ Not connected to agent';
            this.chatInput.style.borderColor = '#dc3545';

            setTimeout(() => {
                this.chatInput.placeholder = originalPlaceholder;
                this.chatInput.style.borderColor = '';
            }, 2000);
        }
    }

    /**
     * Display a user message in the chat
     * @param {string} message - User's message text
     */
    async displayUserMessage(message) {
        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper) return;

        const currentTimestamp = Date.now();

        // Create message block
        const messageBlock = document.createElement('div');
        messageBlock.className = 'message-block user-message';

        // Fetch current chat state to get accurate message count (including system messages)
        let messageIndex;
        try {
            const url = window.backendConnector ? await window.backendConnector.buildURL('/api/chat/current') : '/api/chat/current';
            const response = await fetch(url);
            const data = await response.json();
            if (data.status === 'success' && data.chat && data.chat.messages) {
                // The new message will be at the current length index (after it's added by backend)
                messageIndex = data.chat.messages.length;
            } else {
                // Fallback
                messageIndex = contentWrapper.querySelectorAll('.message-block').length + 1;
            }
        } catch (error) {
            console.warn('Failed to fetch chat state for message index:', error);
            // Fallback: assume 1 system message + visible messages
            messageIndex = contentWrapper.querySelectorAll('.message-block').length + 1;
        }
        messageBlock.dataset.messageIndex = messageIndex;

        // Create timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'agent-msg-head';
        timestamp.textContent = new Date(currentTimestamp).toLocaleTimeString();

        // Create content
        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = message;

        // Create edit button
        const editBtn = document.createElement('button');
        editBtn.className = 'message-edit-btn';
        editBtn.innerHTML = '✏️';
        editBtn.title = 'Edit message';
        editBtn.onclick = () => this.handleMessageEdit(messageBlock, messageIndex);

        // Assemble message
        messageBlock.appendChild(timestamp);
        messageBlock.appendChild(content);
        messageBlock.appendChild(editBtn);

        // Add to content wrapper
        contentWrapper.appendChild(messageBlock);

        // Save user message to history
        this.saveMessageToHistory('user', {
            message: message,
            timestamp: currentTimestamp
        });

        // Force immediate scroll after user message
        this.scrollToBottom();

        // Note: Loading indicator is now shown in handleChatSubmit() BEFORE sending
        // to provide instant feedback. No need to show it again here.

        // Scroll again after a brief delay to ensure all content is rendered
        setTimeout(() => this.scrollToBottom(), 10);
    }

    /**
     * Unified autoscroll function to ensure consistent chat scrolling behavior
     * Uses a combination of methods for maximum compatibility and smooth scrolling
     */
    scrollToBottom() {
        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper) {
            console.warn('Content wrapper not found for scrolling');
            return;
        }

        // Single efficient scroll operation - much faster than multiple attempts
        const scrollToEnd = () => {
            // Use scrollIntoView for the most reliable scrolling
            const lastChild = contentWrapper.lastElementChild;
            if (lastChild) {
                try {
                    lastChild.scrollIntoView({ behavior: 'instant', block: 'end', inline: 'nearest' });
                } catch (e) {
                    // Fallback to direct scroll assignment
                    contentWrapper.scrollTop = contentWrapper.scrollHeight;
                }
            } else {
                // Direct scroll if no children
                contentWrapper.scrollTop = contentWrapper.scrollHeight;
            }
        };

        // Immediate scroll
        scrollToEnd();

        // Single delayed scroll for dynamic content (much faster than multiple timeouts)
        requestAnimationFrame(() => {
            scrollToEnd();
        });
    }

    showSendIndicator() {
        const indicator = document.getElementById('send-indicator');
        if (indicator) {
            indicator.classList.add('show');
            setTimeout(() => {
                indicator.classList.remove('show');
            }, 1500);
        }
    }

    setupFileUpload() {
        if (!this.fileInput || !this.fileUploadBtn || !this.chatInput) {
            console.error('File upload elements not found');
            return;
        }

        // Upload button click
        this.fileUploadBtn.addEventListener('click', () => {
            this.fileInput.click();
        });

        // File input change
        this.fileInput.addEventListener('change', (event) => {
            this.handleFiles(event.target.files);
        });

        // Drag and drop events on chat input field
        this.chatInput.addEventListener('dragover', (event) => {
            event.preventDefault();
            this.chatInput.classList.add('drag-over');
        });

        this.chatInput.addEventListener('dragleave', (event) => {
            event.preventDefault();
            // Only remove drag-over if we're actually leaving the input element
            if (!this.chatInput.contains(event.relatedTarget)) {
                this.chatInput.classList.remove('drag-over');
            }
        });

        this.chatInput.addEventListener('drop', (event) => {
            event.preventDefault();
            this.chatInput.classList.remove('drag-over');
            this.handleFiles(event.dataTransfer.files);
        });

        // Prevent default drag behaviors on the document
        document.addEventListener('dragover', (event) => {
            event.preventDefault();
        });

        document.addEventListener('drop', (event) => {
            event.preventDefault();
        });
    }

    setupScreenshot() {
        if (!this.screenshotBtn) {
            console.error('Screenshot button not found');
            return;
        }

        // Screenshot button click
        this.screenshotBtn.addEventListener('click', () => {
            console.log('📸 Screenshot button clicked');

            // Send screenshot command to agent
            if (this.wsManager && this.wsManager.isConnected()) {
                this.wsManager.sendMessage('/screenshot');

                // Set capturing state
                this.setScreenshotCapturing();
            } else {
                console.warn('WebSocket not connected, screenshot command not sent');

                // Visual feedback for connection error
                const originalIcon = this.screenshotBtn.innerHTML;
                this.screenshotBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2"/></svg>';
                this.screenshotBtn.style.borderColor = '#dc3545';

                setTimeout(() => {
                    this.screenshotBtn.innerHTML = originalIcon;
                    this.screenshotBtn.style.borderColor = '';
                }, 1500);
            }
        });
    }

    /**
     * Set screenshot button to capturing state (loading spinner)
     */
    setScreenshotCapturing() {
        if (!this.screenshotBtn) return;

        this.screenshotBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="2"/><path d="M12 1v6m0 6v6m-6-6h6m6 0h-6" stroke="currentColor" stroke-width="2"/></svg>';
        this.screenshotBtn.style.opacity = '0.7';
        this.screenshotBtn.disabled = true;
        this.screenshotBtn.classList.remove('screenshot-ready');
    }

    /**
     * Set screenshot button to ready state (green glow)
     */
    setScreenshotReady() {
        if (!this.screenshotBtn) return;

        // Restore original icon
        this.screenshotBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="3" width="18" height="18" rx="2" ry="2" stroke="currentColor" stroke-width="2"/><circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/><path d="M21 15L16 10L5 21" stroke="currentColor" stroke-width="2"/></svg>';
        this.screenshotBtn.style.opacity = '';
        this.screenshotBtn.disabled = false;
        this.screenshotBtn.classList.add('screenshot-ready');

        console.log('📸 Screenshot ready - button styled with green indicator');
    }

    /**
     * Reset screenshot button to default state
     */
    resetScreenshotState() {
        if (!this.screenshotBtn) return;

        // Restore original icon
        this.screenshotBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="3" width="18" height="18" rx="2" ry="2" stroke="currentColor" stroke-width="2"/><circle cx="8.5" cy="8.5" r="1.5" fill="currentColor"/><path d="M21 15L16 10L5 21" stroke="currentColor" stroke-width="2"/></svg>';
        this.screenshotBtn.style.opacity = '';
        this.screenshotBtn.disabled = false;
        this.screenshotBtn.classList.remove('screenshot-ready');

        console.log('📸 Screenshot state reset to default');
    }

    setupScreenCaptureButton() {
        if (!this.screenCaptureBtn) {
            console.error('Screen capture button not found');
            return;
        }

        // Check if screen capture is supported
        if (!this.screenCaptureManager.isSupported()) {
            this.screenCaptureBtn.disabled = true;
            this.screenCaptureBtn.title = 'Screen capture not supported in this browser';
            this.screenCaptureBtn.style.opacity = '0.5';
            console.warn('Screen capture API not supported');
            return;
        }

        // Screen capture button click - toggle continuous capture mode
        this.screenCaptureBtn.addEventListener('click', async () => {
            console.log('🎥 Screen capture button clicked');

            if (this.screenCaptureEnabled) {
                // Disable continuous capture mode
                this.screenCaptureEnabled = false;
                this.screenCaptureManager.stopCapture();
                this.screenCaptureBtn.classList.remove('enabled');
                this.screenCaptureBtn.title = 'Enable screen capture - captures on message send';
                console.log('🔴 Screen capture disabled');
            } else {
                // Enable continuous capture mode
                try {
                    // Start continuous stream
                    await this.screenCaptureManager.startContinuousCapture({
                        audio: false,
                        preferCurrentTab: false
                    });

                    // Permission granted, enable auto-capture
                    this.screenCaptureEnabled = true;
                    this.screenCaptureBtn.classList.add('enabled');
                    this.screenCaptureBtn.title = 'Screen capture enabled - click to disable';
                    console.log('🟢 Continuous screen capture enabled - will capture on message send');

                } catch (error) {
                    console.error('❌ Screen capture failed:', error);
                    alert(`Screen capture failed: ${error.message}`);
                    this.screenCaptureEnabled = false;
                    this.screenCaptureBtn.classList.remove('enabled');
                }
            }
        });
    }

    setupVoiceButton() {
        if (!this.voiceBtn) {
            console.error('Voice button not found');
            return;
        }

        // Voice button click
        this.voiceBtn.addEventListener('click', async () => {
            console.log('🎤 Voice button clicked');

            // Clear manual override - let backend state control TTS
            this.speakModeManualOverride = false;

            // Send voice command to agent
            if (this.wsManager && this.wsManager.isConnected()) {
                this.wsManager.sendMessage('/voice');

                // Visual feedback - toggle active state
                const wasActive = this.voiceBtn.classList.contains('active');
                this.voiceBtn.classList.toggle('active');
                const isActive = this.voiceBtn.classList.contains('active');

                // If voice is activated, enable speak mode
                if (isActive) {
                    this.speakModeEnabled = true;
                    this.speakBtn.classList.add('active');
                    this.speakBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M11 5L6 9H2V15H6L11 19V5Z" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M15.54 8.46C16.4774 9.39764 17.0039 10.6692 17.0039 11.995C17.0039 13.3208 16.4774 14.5924 15.54 15.53" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M18.07 5.93C19.9447 7.80528 20.9979 10.3453 20.9979 13C20.9979 15.6547 19.9447 18.1947 18.07 20.07" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
                    this.speakBtn.title = 'Speak mode enabled - click to disable';
                    console.log('🔊 Speak mode auto-enabled with voice mode');
                }

                // Handle microphone recording
                if (this.audioManager) {
                    // Check if VAD mode is enabled
                    const vadToggle = document.getElementById('vad-toggle');
                    const vadEnabled = vadToggle && vadToggle.checked;

                    if (isActive) {
                        // Start recording (VAD or manual)
                        try {
                            if (vadEnabled) {
                                console.log('🎤 Starting VAD recording (auto-stop on silence)');
                                this.voiceBtn.classList.add('vad-listening');

                                const transcription = await this.audioManager.startRecordingWithVAD();

                                this.voiceBtn.classList.remove('vad-listening', 'vad-recording');
                                console.log('🎤 VAD recording complete, transcription:', transcription);

                                if (transcription && this.wsManager) {
                                    this.wsManager.sendMessage(transcription);
                                }

                                this.voiceBtn.classList.remove('active');
                                this.voiceBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 1C13.66 1 15 2.34 15 4V12C15 13.66 13.66 15 12 15C10.34 15 9 13.66 9 12V4C9 2.34 10.34 1 12 1Z" stroke="currentColor" stroke-width="2"/><path d="M19 10V12C19 16.42 15.42 20 11 20H13C17.42 20 21 16.42 21 12V10" stroke="currentColor" stroke-width="2"/><path d="M12 20V24" stroke="currentColor" stroke-width="2"/><path d="M8 24H16" stroke="currentColor" stroke-width="2"/></svg>';
                            } else {
                                console.log('🎤 Starting manual recording');
                                await this.audioManager.startRecording();
                            }
                        } catch (error) {
                            console.error('Failed to start recording:', error);
                            // Revert button state on error
                            this.voiceBtn.classList.remove('active', 'vad-listening', 'vad-recording');
                            return;
                        }
                    } else {
                        // Stop manual recording and send audio (VAD auto-stops)
                        if (!vadEnabled) {
                            try {
                                const transcription = await this.audioManager.stopRecording();
                                console.log('🎤 Recording stopped, transcription:', transcription);
                                if (transcription && this.wsManager) {
                                    this.wsManager.sendMessage(transcription);
                                }
                            } catch (error) {
                                console.error('Failed to stop recording:', error);
                            }
                        }
                    }
                }

                // Change icon based on state
                if (isActive) {
                    // Show microphone with "active" styling
                    this.voiceBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 1C13.66 1 15 2.34 15 4V12C15 13.66 13.66 15 12 15C10.34 15 9 13.66 9 12V4C9 2.34 10.34 1 12 1Z" fill="currentColor"/><path d="M19 10V12C19 16.42 15.42 20 11 20H13C17.42 20 21 16.42 21 12V10" stroke="currentColor" stroke-width="2"/><path d="M12 20V24" stroke="currentColor" stroke-width="2"/><path d="M8 24H16" stroke="currentColor" stroke-width="2"/></svg>';
                    this.voiceBtn.title = 'Voice mode enabled - click to disable';
                } else {
                    // Show regular microphone icon
                    this.voiceBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 1C13.66 1 15 2.34 15 4V12C15 13.66 13.66 15 12 15C10.34 15 9 13.66 9 12V4C9 2.34 10.34 1 12 1Z" stroke="currentColor" stroke-width="2"/><path d="M19 10V12C19 16.42 15.42 20 11 20H13C17.42 20 21 16.42 21 12V10" stroke="currentColor" stroke-width="2"/><path d="M12 20V24" stroke="currentColor" stroke-width="2"/><path d="M8 24H16" stroke="currentColor" stroke-width="2"/></svg>';
                    this.voiceBtn.title = 'Toggle voice mode (mic input + TTS) - /voice';
                }

                // Brief feedback animation
                this.voiceBtn.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    this.voiceBtn.style.transform = '';
                }, 150);

            } else {
                console.warn('WebSocket not connected, voice command not sent');

                // Visual feedback for connection error
                const originalIcon = this.voiceBtn.innerHTML;
                this.voiceBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2"/></svg>';
                this.voiceBtn.style.borderColor = '#dc3545';

                setTimeout(() => {
                    this.voiceBtn.innerHTML = originalIcon;
                    this.voiceBtn.style.borderColor = '';
                }, 1500);
            }
        });

        this.setupVolumeSlider();
    }

    setupVolumeSlider() {
        const volumeSlider = document.getElementById('tts-volume-slider');
        const volumeLabel = document.querySelector('.volume-label');

        if (!volumeSlider || !volumeLabel) {
            console.warn('Volume slider elements not found');
            return;
        }

        volumeSlider.addEventListener('input', (e) => {
            const volume = parseInt(e.target.value);
            volumeLabel.textContent = `${volume}%`;

            const percentage = volume / 100;
            e.target.style.background = `linear-gradient(to top, var(--primary-color) 0%, var(--primary-color) ${volume}%, var(--border-color) ${volume}%)`;

            if (this.audioManager) {
                this.audioManager.setVolume(volume);
            }
        });

        volumeSlider.value = 100;
        volumeLabel.textContent = '100%';
        volumeSlider.style.background = `linear-gradient(to top, var(--primary-color) 0%, var(--primary-color) 100%, var(--border-color) 100%)`;

        console.log('Volume slider initialized');
    }

    setupSpeakButton() {
        if (!this.speakBtn) {
            console.error('Speak button not found');
            return;
        }

        // Speak button click
        this.speakBtn.addEventListener('click', async () => {
            console.log('🔊 Speak button clicked');

            // Toggle frontend state (no backend command needed)
            this.speakModeEnabled = !this.speakModeEnabled;
            this.speakModeManualOverride = true; // Mark as manually overridden

            // Visual feedback - toggle active state
            this.speakBtn.classList.toggle('active', this.speakModeEnabled);

            // If speak is activated, deactivate voice mode
            if (this.speakModeEnabled) {
                this.voiceBtn.classList.remove('active');
                console.log('🔊 Speak mode enabled (TTS auto-play on)');

                // Speak the latest assistant visualresponse content
                await this.speakLatestResponse();
            } else {
                // If speak is deactivated, stop all playing audio
                if (this.audioManager && typeof this.audioManager.stopAudio === 'function') {
                    this.audioManager.stopAudio();
                    console.log('🔇 Speak mode disabled (stopped audio playback)');
                }
            }

            // Change icon based on state
            if (this.speakModeEnabled) {
                // Show speaker with "active" styling (filled)
                this.speakBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M11 5L6 9H2V15H6L11 19V5Z" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M15.54 8.46C16.4774 9.39764 17.0039 10.6692 17.0039 11.995C17.0039 13.3208 16.4774 14.5924 15.54 15.53" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M18.07 5.93C19.9447 7.80528 20.9979 10.3453 20.9979 13C20.9979 15.6547 19.9447 18.1947 18.07 20.07" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
                this.speakBtn.title = 'Speak mode enabled - click to disable';
            } else {
                // Show regular speaker icon
                this.speakBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M11 5L6 9H2V15H6L11 19V5Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M15.54 8.46C16.4774 9.39764 17.0039 10.6692 17.0039 11.995C17.0039 13.3208 16.4774 14.5924 15.54 15.53" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M18.07 5.93C19.9447 7.80528 20.9979 10.3453 20.9979 13C20.9979 15.6547 19.9447 18.1947 18.07 20.07" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
                this.speakBtn.title = 'Enable speak mode (TTS auto-play)';
            }

            // Brief feedback animation
            this.speakBtn.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.speakBtn.style.transform = '';
            }, 150);
        });
    }

    /**
     * Handle uploaded files
     * @param {FileList} files - Files to upload
     * @returns {Promise<void>}
     */
    async handleFiles(files) {
        if (!files || files.length === 0) {
            return;
        }

        this.showUploadStatus('Uploading files...', 'uploading');

        try {
            const formData = new FormData();
            
            // Add all files to the form data
            for (let i = 0; i < files.length; i++) {
                formData.append('file', files[i]);
            }

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.status === 'success') {
                this.showUploadStatus(result.message, 'success');
                
                // Add file paths to input field
                if (result.files && result.files.length > 0) {
                    this.addFilesToInput(result.files);
                }
                
                // Clear the file input
                this.fileInput.value = '';
                
                // Hide status after 3 seconds
                setTimeout(() => {
                    this.hideUploadStatus();
                }, 3000);
                
            } else {
                this.showUploadStatus(result.message || 'Upload failed', 'error');
                setTimeout(() => {
                    this.hideUploadStatus();
                }, 5000);
            }

        } catch (error) {
            console.error('Upload error:', error);
            this.showUploadStatus('Upload failed: ' + error.message, 'error');
            setTimeout(() => {
                this.hideUploadStatus();
            }, 5000);
        }
    }

    /**
     * Add file paths to the chat input field
     * @param {string[]} filePaths - Array of file paths to add
     */
    addFilesToInput(filePaths) {
        const currentValue = this.chatInput.value;
        const pathsText = filePaths.join(' ');
        
        if (currentValue.trim()) {
            this.chatInput.value = currentValue + ' ' + pathsText;
        } else {
            this.chatInput.value = pathsText;
        }
        
        // Focus on the input field
        this.chatInput.focus();
    }

    /**
     * Show upload status message
     * @param {string} message - Status message to display
     * @param {UploadStatusType} type - Status type
     */
    showUploadStatus(message, type) {
        const statusDiv = document.getElementById('upload-status');
        if (statusDiv) {
            statusDiv.className = `file-upload-status ${type}`;
            statusDiv.textContent = message;
            statusDiv.style.display = 'block';
        }
    }

    hideUploadStatus() {
        const statusDiv = document.getElementById('upload-status');
        if (statusDiv) {
            statusDiv.style.display = 'none';
        }
    }

    async handleScreenCaptureBase64(base64Image) {
        if (!this.wsManager || !this.wsManager.isConnected()) {
            throw new Error('WebSocket not connected');
        }

        const imageUploadMessage = {
            type: 'image_upload',
            image: base64Image,
            timestamp: Date.now(),
            source: 'screen_capture'
        };

        const success = this.wsManager.sendMessage(JSON.stringify(imageUploadMessage));

        if (!success) {
            throw new Error('Failed to send image via WebSocket');
        }

        console.log('📤 Screen capture base64 sent via WebSocket');
    }

    setupGlobalJSToggle() {
        const globalJsToggle = document.getElementById('global-js-toggle');
        if (!globalJsToggle) {
            console.warn('Global JS toggle button not found');
            return;
        }

        // Initialize button state based on stored preference
        const isGlobalJsEnabled = window.JavaScriptUtils ? window.JavaScriptUtils.getGlobalJSState() : true;
        this.updateGlobalJSToggleButton(globalJsToggle, isGlobalJsEnabled);

        // Set up click event listener
        globalJsToggle.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();

            if (!window.JavaScriptUtils) {
                console.error('JavaScriptUtils not available');
                return;
            }

            // Toggle the global state
            const newState = window.JavaScriptUtils.toggleGlobalJSState();

            // Update button appearance
            this.updateGlobalJSToggleButton(globalJsToggle, newState);

            // Apply to all current messages with staggered timing
            this.applyGlobalJSStateToMessages(newState);

            window.visualLogger.debug(`Global JS ${newState ? 'enabled' : 'disabled'}`);
        });
    }

    updateGlobalJSToggleButton(button, enabled) {
        if (enabled) {
            button.className = 'js-enabled';
            button.querySelector('.js-text').textContent = 'JS Enabled';
            button.title = 'Disable JavaScript for all messages';
        } else {
            button.className = 'js-disabled';
            button.querySelector('.js-text').textContent = 'JS Disabled';
            button.title = 'Enable JavaScript for all messages';
        }
    }

    applyGlobalJSStateToMessages(enabled) {
        // Find all message containers with JS content
        const messageContainers = document.querySelectorAll('[id*="msg-"]');
        const jsMessages = [];

        messageContainers.forEach(container => {
            const messageId = container.id;
            if (messageId && messageId.startsWith('msg-')) {
                // Check if this message has JS content (iframe or JS toggle button)
                const iframe = container.querySelector('iframe');
                const jsToggle = container.querySelector('.msg-js-toggle');
                if (iframe || jsToggle) {
                    jsMessages.push({
                        container,
                        messageId,
                        jsToggle,
                        iframe
                    });
                }
            }
        });

        window.visualLogger.debug(`Applying global JS ${enabled ? 'enable' : 'disable'} to ${jsMessages.length} messages`);

        // Apply changes with 200ms intervals
        jsMessages.forEach((msgData, index) => {
            setTimeout(() => {
                this.applyJSStateToMessage(msgData, enabled);
            }, index * 200);
        });
    }

    applyJSStateToMessage(msgData, enabled) {
        const { container, messageId, jsToggle, iframe } = msgData;

        if (!window.JavaScriptUtils) {
            console.error('JavaScriptUtils not available');
            return;
        }

        // Update the message's individual JS state
        window.JavaScriptUtils.setMessageJSState(messageId, enabled);

        // Update individual toggle button appearance if present
        if (jsToggle) {
            jsToggle.className = `msg-js-toggle ${enabled ? 'enabled' : 'disabled'}`;
            jsToggle.textContent = enabled ? 'JS ON' : 'JS OFF';
            jsToggle.title = `Click to ${enabled ? 'disable' : 'enable'} JavaScript for this message`;
        }

        // Trigger re-rendering if we have access to the content renderer and original data
        if (iframe && this.renderer && window.ContentRenderer) {
            try {
                // Try to get original HTML content from data attributes or reconstruct
                const originalContent = this.getOriginalContentForMessage(messageId, iframe);
                if (originalContent) {
                    const data = { type: 'html', content: originalContent };
                    this.renderer.reRenderMessage(container.querySelector('.message-content') || container, data, messageId, enabled);
                    window.visualLogger.debug(`Re-rendered message ${messageId} with JS ${enabled ? 'enabled' : 'disabled'}`);
                } else {
                    window.visualLogger.warn(`Could not retrieve original content for message ${messageId}`);
                }
            } catch (error) {
                console.warn(`Failed to re-render message ${messageId}:`, error);
            }
        }

        window.visualLogger.debug(`Applied JS ${enabled ? 'enable' : 'disable'} to message ${messageId}`);
    }

    getOriginalContentForMessage(messageId, iframe) {
        // Try to get content from iframe srcdoc
        if (iframe.srcdoc) {
            return iframe.srcdoc;
        }

        // Try to get content from iframe contentDocument
        try {
            if (iframe.contentDocument) {
                return iframe.contentDocument.documentElement.outerHTML;
            }
        } catch (e) {
            // Cross-origin restrictions
        }

        // Try to get from data attributes if we stored it there
        const messageContainer = document.getElementById(messageId);
        if (messageContainer && messageContainer.dataset.originalContent) {
            return messageContainer.dataset.originalContent;
        }

        return null;
    }

    setupInterruptButton(interruptBtn) {
        // Track agent state to show/hide button appropriately
        this.agentGenerating = false;

        // Button starts in inactive (grayed out) state
        interruptBtn.classList.remove('active', 'processing');

        interruptBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();

            // Always allow interrupt - no state checks
            // This ensures the system can always return to waiting for user input

            if (!this.wsManager || !this.wsManager.isConnected()) {
                console.warn('Cannot send interrupt - WebSocket not connected');
                this.showInterruptError('WebSocket not connected');
                return;
            }

            // Send interrupt signal
            const success = this.wsManager.sendInterruptSignal();

            if (success) {
                // Visual feedback
                const originalIcon = interruptBtn.querySelector('.interrupt-icon').textContent;

                interruptBtn.classList.add('processing');
                interruptBtn.querySelector('.interrupt-icon').textContent = '⏳';

                // Reset button state after a delay
                setTimeout(() => {
                    interruptBtn.classList.remove('processing', 'active');
                    interruptBtn.querySelector('.interrupt-icon').textContent = originalIcon;
                }, 2000);

                console.log('Interrupt signal sent to agent - forcing return to user input state');
            } else {
                this.showInterruptError('Failed to send interrupt signal');
            }
        });

        // Store reference for external access
        this.interruptBtn = interruptBtn;
    }

    showInterruptButton() {
        if (this.interruptBtn) {
            this.interruptBtn.classList.add('active');
            this.interruptBtn.classList.remove('processing');
            this.agentGenerating = true;
        }
    }

    hideInterruptButton() {
        if (this.interruptBtn) {
            this.interruptBtn.classList.remove('active', 'processing');
            this.agentGenerating = false;
        }
    }

    showInterruptError(message) {
        // Show a brief error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'interrupt-error-message';
        errorDiv.textContent = `⚠️ ${message}`;
        errorDiv.style.cssText = `
            position: fixed;
            top: 60px;
            right: 20px;
            background-color: #dc3545;
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.9rem;
            z-index: 10001;
            animation: fadeInOut 3s ease-in-out;
        `;

        document.body.appendChild(errorDiv);

        // Remove after animation
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 3000);
    }

    showResetConfirmationDialog() {
        return new Promise((resolve) => {
            // Create modal overlay
            const overlay = document.createElement('div');
            overlay.className = 'reset-dialog-overlay';

            // Create dialog
            const dialog = document.createElement('div');
            dialog.className = 'reset-dialog';

            // Create dialog content
            const content = document.createElement('div');
            content.className = 'reset-dialog-content';

            // Title
            const title = document.createElement('h3');
            title.textContent = '🗑️ Clear Chat History';
            title.className = 'reset-dialog-title';

            // Message
            const message = document.createElement('p');
            message.innerHTML = 'This will clear the conversation history and start fresh.<br><br><strong>This action cannot be undone.</strong>';
            message.className = 'reset-dialog-message';

            // Progress area with stopwatch
            const progressArea = document.createElement('div');
            progressArea.className = 'reset-dialog-progress';
            progressArea.style.display = 'none';

            const progressText = document.createElement('div');
            progressText.textContent = 'Resetting agent...';
            progressText.className = 'reset-progress-text';

            const progressStopwatch = document.createElement('div');
            progressStopwatch.textContent = '0.0s';
            progressStopwatch.className = 'reset-progress-stopwatch';

            progressArea.appendChild(progressText);
            progressArea.appendChild(progressStopwatch);

            // Buttons
            const buttonArea = document.createElement('div');
            buttonArea.className = 'reset-dialog-buttons';

            const cancelBtn = document.createElement('button');
            cancelBtn.textContent = 'Cancel';
            cancelBtn.className = 'reset-dialog-btn reset-dialog-cancel';

            const confirmBtn = document.createElement('button');
            confirmBtn.textContent = 'Reset Agent';
            confirmBtn.className = 'reset-dialog-btn reset-dialog-confirm';

            buttonArea.appendChild(cancelBtn);
            buttonArea.appendChild(confirmBtn);

            // Assemble dialog
            content.appendChild(title);
            content.appendChild(message);
            content.appendChild(progressArea);
            content.appendChild(buttonArea);
            dialog.appendChild(content);
            overlay.appendChild(dialog);

            // Event handlers
            const cleanup = () => {
                overlay.remove();
            };

            cancelBtn.addEventListener('click', () => {
                cleanup();
                resolve(false);
            });

            confirmBtn.addEventListener('click', () => {
                // Hide buttons and message, show progress
                buttonArea.style.display = 'none';
                message.style.display = 'none';
                progressArea.style.display = 'flex';

                // Start progress stopwatch
                const startTime = Date.now();
                const progressTimer = setInterval(() => {
                    const elapsed = (Date.now() - startTime) / 1000;
                    progressStopwatch.textContent = elapsed.toFixed(1) + 's';
                }, 100);

                // Auto-close after reset completes (2 seconds)
                setTimeout(() => {
                    clearInterval(progressTimer);
                    cleanup();
                }, 2000);

                resolve(true);
            });

            // Close on overlay click
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    cleanup();
                    resolve(false);
                }
            });

            // Add to DOM
            document.body.appendChild(overlay);

            // Focus the confirm button
            setTimeout(() => confirmBtn.focus(), 100);
        });
    }

    setDarkTheme() {
        // Always use dark theme
        document.documentElement.setAttribute('data-theme', 'dark');
    }
    

    /**
     * Show loading indicator while waiting for agent response
     */
    showLoadingIndicator() {
        // Remove any existing loading indicator first
        this.hideLoadingIndicator();

        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper) return;

        // Create loading indicator
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'loading-indicator show';
        loadingIndicator.id = 'agent-loading-indicator';

        const loadingContent = document.createElement('div');
        loadingContent.className = 'loading-content';

        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';

        const loadingText = document.createElement('div');
        loadingText.className = 'loading-text loading-dots';

        // Random loading messages for variety and humor
        const loadingMessages = [
            // Original classics
            'Wrestling with algorithms...',
            'Compiling insights...',
            'Channeling patterns...',
            'Consulting the matrix...',
            'Decoding some universe...',
            'Summoning creativity...',
            'Pondering hallucinations...',
            'Untangling complexity...',
            'Calibrating neurons...',
            // 'Shining...',
            
            // Biblically inspired additions
            'Parting the data sea...',
            'Wrangling with no-mans logic...',
            'Gathering manna from the cloud...',
            'Consulting the ancient scrolls...',
            'Turning over a new byte...',

            // Modern / Tech-Humor additions
            'Negotiating with other AIs...',      // The feeling that AI has a mind of its own
            'Appeasing the algorithms...',      // A nod to the mysterious power of social media/search algorithms
            'Waking up the servers...',         // A classic, simple and relatable
            'Reticulating splines...',          // A famous, nonsensical loading message from SimCity
            'Synergizing the paradigms...',     // Poking fun at corporate buzzwords
            'Compiling patterns...',     // A fun take on a standard developer task
            'Avoiding the singularity...',       // A humorous, high-stakes take on working with advanced AI

            // Philosphical
            'Pondering the cosmic giggle...',           // Watts: The playful, non-serious nature of existence
            'Ingesting a heroic dose of data...',          // McKenna: A nod to his ideas on profound experiences
            'Searching for synchronicities...',// Jung: Looking for meaningful, acausal connections
            'Observing before judging...',       // Watts: A Taoist/Zen approach to letting the answer emerge
            'Increasing local novelty...',              // McKenna: A reference to his Novelty Theory
            'Querying the collective unconscious...',   // Jung: Accessing the shared, universal repository of archetypes
            'Juggling the prickly and the gooey...'     // Watts: His metaphor for the interplay of order and chaos
        ];
        
        const randomMessage = loadingMessages[Math.floor(Math.random() * loadingMessages.length)];
        loadingText.textContent = randomMessage;

        // Create stopwatch element
        const stopwatch = document.createElement('div');
        stopwatch.className = 'loading-stopwatch';
        stopwatch.textContent = '0.0s';

        // Start stopwatch timer
        const startTime = Date.now();
        const stopwatchTimer = setInterval(() => {
            const elapsed = (Date.now() - startTime) / 1000;
            stopwatch.textContent = elapsed.toFixed(1) + 's';
        }, 100);

        // Store timer ID on the loading indicator for cleanup
        loadingIndicator.stopwatchTimer = stopwatchTimer;

        loadingContent.appendChild(spinner);
        loadingContent.appendChild(loadingText);
        loadingContent.appendChild(stopwatch);
        loadingIndicator.appendChild(loadingContent);

        // Add to content wrapper
        contentWrapper.appendChild(loadingIndicator);

        // Scroll to show the loading indicator
        this.scrollToBottom();
    }

    /**
     * Hide loading indicator when agent responds
     */
    hideLoadingIndicator() {
        const loadingIndicator = document.getElementById('agent-loading-indicator');
        if (loadingIndicator) {
            // Clear the stopwatch timer if it exists
            if (loadingIndicator.stopwatchTimer) {
                clearInterval(loadingIndicator.stopwatchTimer);
            }
            loadingIndicator.remove();
        }
    }

    /**
     * Show error indicator when message send fails
     * @param {string} errorMessage - Error message to display
     */
    showErrorIndicator(errorMessage) {
        // Remove any existing loading indicator first
        this.hideLoadingIndicator();

        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper) return;

        // Create error indicator
        const errorIndicator = document.createElement('div');
        errorIndicator.className = 'loading-indicator error show';
        errorIndicator.id = 'agent-loading-indicator';

        const errorContent = document.createElement('div');
        errorContent.className = 'loading-content';

        const errorIcon = document.createElement('div');
        errorIcon.className = 'loading-spinner';
        errorIcon.textContent = '❌';

        const errorText = document.createElement('div');
        errorText.className = 'loading-text';
        errorText.textContent = errorMessage;

        errorContent.appendChild(errorIcon);
        errorContent.appendChild(errorText);
        errorIndicator.appendChild(errorContent);

        // Add to content wrapper
        contentWrapper.appendChild(errorIndicator);

        // Auto-hide error after 4 seconds
        setTimeout(() => {
            if (errorIndicator.parentNode) {
                errorIndicator.remove();
            }
        }, 4000);

        // Scroll to show the error indicator
        this.scrollToBottom();
    }

    /**
     * Save message to history and localStorage
     * @param {'user'|'agent'} type - 'user' or 'agent'
     * @param {UserMessageData|AgentMessageData} data - Message data
     */
    saveMessageToHistory(type, data) {
        // Enhanced validation and debugging
        if (!type || !data) {
            console.error('❌ Invalid message data for history save:', { type, data });
            return;
        }

        // Prevent saving transient UI elements that should not be restored on page reload
        if (type === 'agent' && data.messageData) {
            // FIXED: Removed 'user_message' from transient types - this was incorrectly filtering user messages
            const transientTypes = ['permission_request', 'config'];
            if (transientTypes.includes(data.messageData.type)) {
                window.visualLogger.debug(`Skipping storage of transient message type: ${data.messageData.type}`);
                return;
            }
        }

        // Validate user message structure
        if (type === 'user') {
            if (!data.message || typeof data.message !== 'string') {
                console.error('❌ Invalid user message structure - missing or invalid message field:', data);
                return;
            }
            window.visualLogger.debug(`Saving user message to history: "${data.message.substring(0, 50)}${data.message.length > 50 ? '...' : ''}"`);
        }

        // Validate agent message structure
        if (type === 'agent') {
            if (!data.messageData) {
                console.error('❌ Invalid agent message structure - missing messageData field:', data);
                return;
            }
            window.visualLogger.debug(`Saving agent message to history: type=${data.messageData.type}, id=${data.originalId || 'unknown'}`);
        }

        const messageEntry = {
            type: type,
            timestamp: Date.now(),
            data: data
        };

        this.messageHistory.push(messageEntry);

        // Keep only last 50 messages to prevent localStorage bloat
        if (this.messageHistory.length > 50) {
            const removedCount = this.messageHistory.length - 50;
            this.messageHistory = this.messageHistory.slice(-50);
            window.visualLogger.debug(`Trimmed ${removedCount} old messages from history to maintain 50 message limit`);
        }

        try {
            localStorage.setItem('agent-message-history', JSON.stringify(this.messageHistory));
            window.visualLogger.debug(`Message history saved: ${this.messageHistory.length} total messages`);
        } catch (e) {
            console.error('❌ Failed to save message history to localStorage:', e);

            // Try to recover by clearing and saving current message only
            try {
                this.messageHistory = [messageEntry];
                localStorage.setItem('agent-message-history', JSON.stringify(this.messageHistory));
                console.log('🔧 Recovered by clearing history and saving current message only');
            } catch (recoveryError) {
                console.error('❌ Recovery attempt also failed:', recoveryError);
            }
        }
    }

    /**
     * Load message history from localStorage and restore messages
     */
    loadMessageHistory() {
        try {
            const stored = localStorage.getItem('agent-message-history');
            window.visualLogger.debug('Checking localStorage for message history', { hasData: !!stored });

            if (stored) {
                this.messageHistory = JSON.parse(stored);

                // Clean up any legacy permission requests that may have been incorrectly stored
                const originalLength = this.messageHistory.length;
                this.messageHistory = this.messageHistory.filter(entry => {
                    // CRITICAL FIX: Never filter out actual user messages (type: 'user')
                    if (entry.type === 'user') {
                        window.visualLogger.debug('Preserving user message during history cleanup:', entry.data?.message?.substring(0, 50) + '...');
                        return true; // Always keep user messages
                    }

                    // Only filter agent messages with transient types
                    if (entry.type === 'agent' && entry.data && entry.data.messageData) {
                        // Removed 'user_message' from transient types - this was incorrectly filtering user messages
                        const transientTypes = ['permission_request', 'config'];
                        if (transientTypes.includes(entry.data.messageData.type)) {
                            window.visualLogger.debug(`Removing legacy transient message: ${entry.data.messageData.type}`);
                            return false;
                        }
                    }
                    return true;
                });

                // Save cleaned history back to localStorage if we removed anything
                if (this.messageHistory.length < originalLength) {
                    window.visualLogger.debug(`Cleaned ${originalLength - this.messageHistory.length} transient messages from history`);
                    try {
                        localStorage.setItem('agent-message-history', JSON.stringify(this.messageHistory));
                    } catch (e) {
                        console.warn('Failed to save cleaned message history:', e);
                    }
                }

                window.visualLogger.info(`Continue mode: Loading ${this.messageHistory.length} messages from previous session`);

                // Count user vs agent messages with detailed breakdown
                const userMessages = this.messageHistory.filter(m => m.type === 'user');
                const agentMessages = this.messageHistory.filter(m => m.type === 'agent');
                window.visualLogger.debug(`Message breakdown: ${userMessages.length} user, ${agentMessages.length} agent`);

                // Debug: Log first few characters of each user message to verify content
                if (userMessages.length > 0) {
                    window.visualLogger.debug('User messages found for restoration:');
                    userMessages.forEach((msg, i) => {
                        const preview = msg.data?.message?.substring(0, 50) || 'No message content';
                        const timestamp = new Date(msg.data?.timestamp || msg.timestamp).toLocaleTimeString();
                        window.visualLogger.debug(`  ${i + 1}. [${timestamp}] "${preview}${msg.data?.message?.length > 50 ? '...' : ''}"`);
                    });
                } else {
                    window.visualLogger.warn('No user messages found in history - this may indicate the bug');
                }

                // Add flag to indicate we're in restoration mode
                this.isRestoring = true;
                this.restoreMessages().then(() => {
                    this.isRestoring = false;
                });

                // Show a temporary restoration indicator
                this.showRestorationFeedback(this.messageHistory.length);
            } else {
                window.visualLogger.info('Continue mode: No previous session history found in localStorage');
            }
        } catch (e) {
            window.visualLogger.error('Failed to load message history from localStorage:', {}, e);
            this.messageHistory = [];
        }
    }

    /**
     * Show temporary feedback when restoring messages
     */
    showRestorationFeedback(messageCount) {
        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper) return;

        // Create restoration indicator
        const indicator = document.createElement('div');
        indicator.className = 'restoration-indicator';
        indicator.innerHTML = `
            <div class="restoration-content">
                <span class="restoration-icon">🔄</span>
                <span class="restoration-text">Restoring ${messageCount} messages from previous session...</span>
            </div>
        `;

        contentWrapper.appendChild(indicator);

        // Remove indicator after restoration
        setTimeout(() => {
            if (indicator.parentNode) {
                indicator.remove();
                console.log(`✅ Successfully restored ${messageCount} messages`);
            }
        }, 2000);
    }

    /**
     * Restore messages from history to the UI
     */
    async restoreMessages() {
        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper || this.messageHistory.length === 0) return;

        // Clear any existing content first to ensure clean restoration
        contentWrapper.innerHTML = '';

        // Sort messages by timestamp to ensure chronological order
        const sortedHistory = [...this.messageHistory].sort((a, b) => a.timestamp - b.timestamp);

        console.log(`🔄 Starting restoration of ${sortedHistory.length} messages...`);

        for (const [index, entry] of sortedHistory.entries()) {
            try {
                console.log(`📝 Restoring message ${index + 1}/${sortedHistory.length}: type=${entry.type}`);

                if (entry.type === 'user') {
                    console.log(`👤 Processing user message: "${entry.data?.message?.substring(0, 30)}..."`);
                    this.restoreUserMessage(entry.data);
                } else if (entry.type === 'agent') {
                    console.log(`🤖 Processing agent message: id=${entry.data?.originalId || 'unknown'}`);
                    await this.restoreAgentMessage(entry.data);
                } else {
                    console.warn(`⚠️ Unknown message type during restoration: ${entry.type}`);
                }
            } catch (e) {
                console.error(`❌ Failed to restore message ${index + 1}:`, e, entry);
            }
        }

        // Scroll to bottom after restoration
        if (this.wsManager) {
            this.wsManager.scrollToBottom();
        }

        console.log(`Restored ${sortedHistory.length} messages in chronological order`);
    }

    /**
     * Restore a user message to the UI
     * @param {UserMessageData} data - User message data
     */
    restoreUserMessage(data) {
        console.log('👤 Restoring user message:', data);

        // Enhanced validation and debugging
        if (!data) {
            console.error('❌ User message restoration failed: No data provided');
            return;
        }

        if (!data.message) {
            console.error('❌ User message restoration failed: No message content in data:', data);
            return;
        }

        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper) {
            console.error('❌ Content wrapper not found for user message restoration');
            return;
        }

        try {
            // Create message block
            const messageBlock = document.createElement('div');
            messageBlock.className = 'message-block user-message';

            // Create timestamp with fallback handling
            const timestamp = document.createElement('div');
            timestamp.className = 'agent-msg-head';

            // Handle both data.timestamp and legacy timestamp formats
            const timestampValue = data.timestamp || Date.now();
            const timeString = new Date(timestampValue).toLocaleTimeString();
            timestamp.textContent = timeString;

            // Create content
            const content = document.createElement('div');
            content.className = 'message-content';
            content.textContent = data.message;

            // Assemble message
            messageBlock.appendChild(timestamp);
            messageBlock.appendChild(content);
            contentWrapper.appendChild(messageBlock);

            console.log(`✅ User message restored successfully: "${data.message.substring(0, 50)}${data.message.length > 50 ? '...' : ''}"`);

        } catch (error) {
            console.error('❌ Error during user message restoration:', error, 'Data:', data);
        }
    }

    /**
     * Restore an agent message to the UI
     * @param {AgentMessageData} data - Agent message data
     */
    async restoreAgentMessage(data) {
        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper || !this.renderer) return;

        const messageId = data.originalId || 'restored-' + Date.now();

        // Create message block
        const messageBlock = document.createElement('div');
        messageBlock.className = 'message-block agent-message';
        messageBlock.id = messageId;

        // Create timestamp header with tool info if available
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'agent-msg-head';
        const timestampText = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : 'Restored';

        // Base timestamp content
        let timestampContent = `<span class="timestamp-text">${timestampText} - ${data.originalId ? 'ID: ' + data.originalId : 'Restored'}</span>`;

        // Add tool execution info if available in restored data
        const toolInfo = data.messageData && data.messageData.tool_info;
        if (toolInfo && Array.isArray(toolInfo) && toolInfo.length > 0) {
            const toolCount = toolInfo.length;
            const toolNames = toolInfo.map(tool => tool.name).join(', ');

            timestampContent += `<span class="tool-info-summary" title="Click to expand tool details">
                🔧 ${toolCount} tool${toolCount > 1 ? 's' : ''}: ${toolNames}
            </span>`;
        }

        timestampDiv.innerHTML = timestampContent;
        messageBlock.appendChild(timestampDiv);

        // Add detailed tool info section for restored messages (initially hidden)
        if (toolInfo && Array.isArray(toolInfo) && toolInfo.length > 0) {
            const toolDetails = document.createElement('div');
            toolDetails.className = 'tool-details hidden';

            let detailsHTML = '<div class="tool-details-header">Tool Execution Details (Restored):</div>';
            toolInfo.forEach(tool => {
                const statusIcon = tool.status === 'success' ? '✅' : '❌';
                const errorText = tool.error ? ` (${tool.error})` : '';
                detailsHTML += `
                    <div class="tool-detail-item">
                        ${statusIcon} <strong>${tool.name}</strong> - ${tool.duration_ms}ms${errorText}
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

        // Create message content container
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageBlock.appendChild(messageContent);

        contentWrapper.appendChild(messageBlock);

        // Render the content using the existing renderer
        await this.renderer.renderContent(messageContent, data.messageData, messageId, timestampDiv);
    }

    /**
     * Clear message history
     */
    clearMessageHistory() {
        this.messageHistory = [];
        try {
            localStorage.removeItem('agent-message-history');
        } catch (e) {
            console.warn('Failed to clear message history from localStorage:', e);
        }
        console.log('Message history cleared');
    }

    /**
     * Setup iframe focus management for interactive content
     * Allows users to click into iframes to capture keyboard/mouse events
     */
    setupIframeFocusManagement() {
        let focusedIframe = null;
        let iframeClickOverlay = null;

        // Create click-to-focus overlay for iframes
        const createIframeOverlay = (iframe) => {
            const overlay = document.createElement('div');
            overlay.className = 'iframe-focus-overlay';
            overlay.style.cssText = `
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: transparent;
                cursor: pointer;
                z-index: 10;
                border: 2px dashed transparent;
                transition: border-color 0.2s ease;
            `;

            overlay.addEventListener('mouseenter', () => {
                overlay.style.borderColor = 'rgba(100, 149, 237, 0.5)';
            });

            overlay.addEventListener('mouseleave', () => {
                overlay.style.borderColor = 'transparent';
            });

            overlay.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.focusIframe(iframe, overlay);
            });

            return overlay;
        };

        // Focus an iframe and enable interaction
        this.focusIframe = (iframe, overlay) => {
            // Remove focus from any previously focused iframe
            if (focusedIframe && focusedIframe !== iframe) {
                this.unfocusIframe(focusedIframe);
            }

            focusedIframe = iframe;

            // Hide the overlay to allow direct interaction
            if (overlay) {
                overlay.style.display = 'none';
            }

            // Add focused state styling
            iframe.style.border = '2px solid rgba(100, 149, 237, 0.8)';
            iframe.style.boxShadow = '0 0 10px rgba(100, 149, 237, 0.3)';

            // Add focus indicator
            const indicator = document.createElement('div');
            indicator.className = 'iframe-focus-indicator';
            indicator.innerHTML = '🎯 Interactive Mode - Click outside to exit';
            indicator.style.cssText = `
                position: absolute;
                top: -25px;
                left: 0;
                background: rgba(100, 149, 237, 0.9);
                color: white;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 12px;
                z-index: 15;
                pointer-events: none;
            `;

            const container = iframe.parentElement;
            if (container.style.position !== 'absolute' && container.style.position !== 'relative') {
                container.style.position = 'relative';
            }
            container.appendChild(indicator);

            // Focus the iframe content window
            try {
                iframe.contentWindow.focus();
            } catch (e) {
                console.warn('Could not focus iframe content window:', e);
            }

            console.log('Iframe focused - interactive mode enabled');
        };

        // Unfocus an iframe and restore overlay
        this.unfocusIframe = (iframe) => {
            if (!iframe) return;

            // Remove focused state styling
            iframe.style.border = '';
            iframe.style.boxShadow = '';

            // Remove focus indicator
            const indicator = iframe.parentElement.querySelector('.iframe-focus-indicator');
            if (indicator) {
                indicator.remove();
            }

            // Restore overlay for the iframe
            const container = iframe.parentElement;
            let overlay = container.querySelector('.iframe-focus-overlay');
            if (overlay) {
                overlay.style.display = 'block';
            }

            focusedIframe = null;
            console.log('Iframe unfocused - click-to-focus mode restored');
        };

        // Set up mutation observer to handle dynamically added iframes
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        // Check if the node itself is an iframe
                        if (node.tagName === 'IFRAME') {
                            this.setupIframeInteraction(node, createIframeOverlay);
                        }

                        // Check for iframes within the added node
                        const iframes = node.querySelectorAll ? node.querySelectorAll('iframe') : [];
                        iframes.forEach(iframe => {
                            this.setupIframeInteraction(iframe, createIframeOverlay);
                        });
                    }
                });
            });
        });

        // Start observing
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Setup existing iframes
        document.querySelectorAll('iframe').forEach(iframe => {
            this.setupIframeInteraction(iframe, createIframeOverlay);
        });

        // Global click handler to unfocus iframes when clicking outside
        document.addEventListener('click', (e) => {
            if (focusedIframe && !focusedIframe.contains(e.target) && !e.target.closest('iframe')) {
                this.unfocusIframe(focusedIframe);
            }
        });

        // Prevent arrow keys from scrolling the main page when iframe is focused
        document.addEventListener('keydown', (e) => {
            if (focusedIframe && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
                // Only prevent if the active element is not an input field in the main page
                if (!e.target.matches('input, textarea, [contenteditable="true"]')) {
                    e.preventDefault();
                    e.stopPropagation();
                }
            }
        });

        console.log('Iframe focus management initialized');
    }

    /**
     * Setup interaction overlay for a specific iframe
     */
    setupIframeInteraction(iframe, createOverlayFn) {
        // Skip if already set up
        if (iframe.dataset.interactionSetup === 'true') {
            return;
        }

        const container = iframe.parentElement;
        if (!container) return;

        // Ensure container has relative positioning for overlay placement
        if (container.style.position !== 'absolute' && container.style.position !== 'relative') {
            container.style.position = 'relative';
        }

        // Create and add overlay
        const overlay = createOverlayFn(iframe);
        container.appendChild(overlay);

        // Mark as set up
        iframe.dataset.interactionSetup = 'true';

        console.log('Interactive overlay added to iframe');
    }

    /**
     * Setup agent message height management
     * Makes agent messages scale to 90% of content-wrapper height with hover scrolling
     */
    setupAgentMessageHeightManagement() {
        const updateAgentMessageHeights = () => {
            const contentWrapper = document.getElementById('content-wrapper');
            if (!contentWrapper) return;

            const contentWrapperHeight = contentWrapper.clientHeight;
            const maxAgentMessageHeight = Math.floor(contentWrapperHeight * 0.9);

            // Apply height to all agent messages
            const agentMessages = document.querySelectorAll('.message-block.agent-message');
            agentMessages.forEach(message => {
                message.style.maxHeight = maxAgentMessageHeight + 'px';
            });

            console.log(`Updated agent message max-height to ${maxAgentMessageHeight}px (90% of ${contentWrapperHeight}px)`);
        };

        // Initial setup
        updateAgentMessageHeights();

        // Update on window resize
        window.addEventListener('resize', updateAgentMessageHeights);

        // Update whenever new messages are added (using MutationObserver)
        const contentWrapper = document.getElementById('content-wrapper');
        if (contentWrapper) {
            const observer = new MutationObserver((mutations) => {
                let hasNewAgentMessage = false;
                mutations.forEach((mutation) => {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE &&
                            (node.classList?.contains('agent-message') ||
                             node.querySelector?.('.message-block.agent-message'))) {
                            hasNewAgentMessage = true;
                        }
                    });
                });

                if (hasNewAgentMessage) {
                    // Small delay to ensure DOM is fully updated
                    setTimeout(updateAgentMessageHeights, 10);
                }
            });

            observer.observe(contentWrapper, {
                childList: true,
                subtree: true
            });
        }

        console.log('Agent message height management initialized');
    }

    /**
     * Initialize the LLM selectors in the header
     */
    /**
     * Initialize Dynamic LLM Manager for multi-model fallback support
     */
    initializeDynamicLLMManager() {
        // Check if DynamicLLMManager is available
        if (typeof DynamicLLMManager === 'undefined') {
            console.warn('DynamicLLMManager not available, LLM selectors will not be initialized');
            return;
        }

        // Instantiate the Dynamic LLM Manager
        this.dynamicLLMManager = new DynamicLLMManager();
        window.visualLogger.info('Dynamic LLM Manager initialized successfully');
    }

    /**
     * Initialize selectors from system configuration
     */
    initializeFromSystemConfiguration() {
        // DEBUG: Log what we received from the API
        console.log('🔍 WEB UI PERSISTENCE DEBUG:');
        console.log('  → systemConfiguration received:', this.systemConfiguration);

        if (this.systemConfiguration && this.systemConfiguration.selected_llms) {
            const fallbackModels = this.systemConfiguration.selected_llms;
            console.log(`Initializing UI with ${fallbackModels.length} fallback models:`, fallbackModels);

            // Initialize router LLM from saved configuration, defaults to None
            const savedRouterLLM = this.systemConfiguration.router_llm;
            if (this.llmSelectors.router) {
                if (savedRouterLLM) {
                    this.llmSelectors.router.setSelectedModel(savedRouterLLM);
                    console.log(`Set router LLM selector to saved selection: ${savedRouterLLM}`);
                } else {
                    this.llmSelectors.router.setSelectedModel(null);
                    console.log(`Set router LLM selector to None (default)`);
                }
            }

            // Initialize planner LLM from saved configuration, or default to first fallback model
            const savedPlannerLLM = this.systemConfiguration.planner_llm;
            if (savedPlannerLLM && this.llmSelectors.planner) {
                this.llmSelectors.planner.setSelectedModel(savedPlannerLLM);
                console.log(`Set planner LLM selector to saved selection: ${savedPlannerLLM}`);
            } else if (fallbackModels.length >= 1 && this.llmSelectors.planner) {
                this.llmSelectors.planner.setSelectedModel(fallbackModels[0]);
                console.log(`Set planner LLM selector to default (first fallback): ${fallbackModels[0]}`);
            }

            // Initialize override LLM from saved configuration, defaults to None
            const savedOverrideLLM = this.systemConfiguration.override_llm;
            if (this.llmSelectors.override) {
                if (savedOverrideLLM) {
                    this.llmSelectors.override.setSelectedModel(savedOverrideLLM);
                    console.log(`Set override LLM selector to saved selection: ${savedOverrideLLM}`);
                } else {
                    this.llmSelectors.override.setSelectedModel(null);
                    console.log(`Set override LLM selector to None (default)`);
                }
            }

            // Display first 3 fallback models in the UI
            if (fallbackModels.length >= 1 && this.llmSelectors.primary) {
                this.llmSelectors.primary.setSelectedModel(fallbackModels[0]);
                console.log(`Set primary selector to: ${fallbackModels[0]}`);
            }

            if (fallbackModels.length >= 2 && this.llmSelectors.fallback1) {
                this.llmSelectors.fallback1.setSelectedModel(fallbackModels[1]);
                console.log(`Set fallback1 selector to: ${fallbackModels[1]}`);
            }

            if (fallbackModels.length >= 3 && this.llmSelectors.fallback2) {
                this.llmSelectors.fallback2.setSelectedModel(fallbackModels[2]);
                console.log(`Set fallback2 selector to: ${fallbackModels[2]}`);
            }

            // Note if there's a 4th model that will be removed on next change
            if (fallbackModels.length >= 4) {
                console.log(`Note: 4th fallback model (${fallbackModels[3]}) will be removed on next UI change`);
            }

            // Set a flag to indicate we've initialized from system config
            this.initializedFromSystem = true;
        } else {
            console.log('No system configuration available, falling back to localStorage...');
            this.loadLLMSelections();
        }

        // Initialization is complete - allow selection change events
        this.isInitializing = false;
    }

    /**
     * Handle LLM selection changes
     * @param {string} position - 'primary', 'fallback1', or 'fallback2'
     * @param {LLMModel|null} model - Selected model object
     */
    handleLLMSelectionChange(position, model) {

        // Guard against updates during initialization
        if (this.isInitializing) {
            return;
        }

        // Store current selections
        if (!this.llmSelections) {
            this.llmSelections = {};
        }
        this.llmSelections[position] = model;

        // Ensure we have the current state of all selectboxes before sending command
        if (!this.llmSelections.primary && this.llmSelectors.primary) {
            this.llmSelections.primary = this.llmSelectors.primary.getSelectedModel();
        }
        if (!this.llmSelections.fallback1 && this.llmSelectors.fallback1) {
            this.llmSelections.fallback1 = this.llmSelectors.fallback1.getSelectedModel();
        }
        if (!this.llmSelections.fallback2 && this.llmSelectors.fallback2) {
            this.llmSelections.fallback2 = this.llmSelectors.fallback2.getSelectedModel();
        }

        // Send fallbacks command as chat message
        // Handle None option properly: if model exists and has isNoneOption=true, use 'none'
        // If model exists and has a real id, use that id
        // If model doesn't exist, use 'none' as fallback
        const primary = this.llmSelections.primary ?
            (this.llmSelections.primary.isNoneOption ? 'none' : this.llmSelections.primary.id) : 'none';
        const fallback1 = this.llmSelections.fallback1 ?
            (this.llmSelections.fallback1.isNoneOption ? 'none' : this.llmSelections.fallback1.id) : 'none';
        const fallback2 = this.llmSelections.fallback2 ?
            (this.llmSelections.fallback2.isNoneOption ? 'none' : this.llmSelections.fallback2.id) : 'none';
        const command = `/fallbacks ${primary} ${fallback1} ${fallback2}`;
        console.log('📤 Sending fallbacks change command:', command);

        if (this.wsManager && this.wsManager.isConnected()) {
            this.wsManager.sendMessage(command);
        } else {
            console.warn('WebSocket not connected, fallbacks command not sent');
        }

        // Save to localStorage for persistence
        try {
            const allSelections = {
                primary: this.llmSelections.primary?.id || null,
                fallback1: this.llmSelections.fallback1?.id || null,
                fallback2: this.llmSelections.fallback2?.id || null,
                timestamp: Date.now()
            };
            localStorage.setItem('llm-selections', JSON.stringify(allSelections));
            console.log('LLM selections saved to localStorage');
        } catch (e) {
            console.warn('Failed to save LLM selections to localStorage:', e);
        }

        this.scheduleAutoSave();
    }

    /**
     * Handle router LLM selection changes
     * @param {LLMModel|null} model - Selected router LLM model
     */
    handleRouterLLMSelectionChange(model) {

        // Guard against updates during initialization
        if (this.isInitializing) {
            return;
        }

        // Send router LLM command as chat message
        const modelKey = model?.id || 'none';
        const command = `/router_llm ${modelKey}`;
        console.log('📤 Sending router LLM change command:', command);

        if (this.wsManager && this.wsManager.isConnected()) {
            this.wsManager.sendMessage(command);
        } else {
            console.warn('WebSocket not connected, router LLM command not sent');
        }

        // Update local storage
        try {
            const routerSelection = {
                router_llm: model?.id || null,
                timestamp: Date.now()
            };

            localStorage.setItem('router-llm-selection', JSON.stringify(routerSelection));
            console.log('Router LLM selection saved to localStorage');
        } catch (e) {
            console.warn('Failed to save router LLM selection to localStorage:', e);
        }

        this.scheduleAutoSave();
    }

    /**
     * Handle planner LLM selection changes
     * @param {LLMModel|null} model - Selected planner LLM model
     */
    handlePlannerLLMSelectionChange(model) {

        // Guard against updates during initialization
        if (this.isInitializing) {
            return;
        }

        // Store current planner selection
        if (!this.plannerLLMSelection) {
            this.plannerLLMSelection = null;
        }
        this.plannerLLMSelection = model;

        // Send planner LLM command as chat message
        const modelKey = model?.id || 'none';
        const command = `/planner_llm ${modelKey}`;
        console.log('📤 Sending planner LLM change command:', command);

        if (this.wsManager && this.wsManager.isConnected()) {
            this.wsManager.sendMessage(command);
        } else {
            console.warn('WebSocket not connected, planner LLM command not sent');
        }

        // Save to localStorage for persistence
        try {
            const plannerSelection = {
                planner_llm: model?.id || null,
                timestamp: Date.now()
            };
            localStorage.setItem('planner-llm-selection', JSON.stringify(plannerSelection));
            console.log('Planner LLM selection saved to localStorage');
        } catch (e) {
            console.warn('Failed to save planner LLM selection to localStorage:', e);
        }

        this.scheduleAutoSave();
    }

    /**
     * Handle override LLM selection changes
     * @param {LLMModel|null} model - Selected override LLM model
     */
    handleOverrideLLMSelectionChange(model) {

        // Guard against updates during initialization
        if (this.isInitializing) {
            return;
        }

        // Store current override selection
        if (!this.overrideLLMSelection) {
            this.overrideLLMSelection = null;
        }
        this.overrideLLMSelection = model;

        // Send override LLM command as chat message
        const modelKey = model?.id || 'none';
        const command = `/override_llm ${modelKey}`;
        console.log('📤 Sending override LLM change command:', command);

        if (this.wsManager && this.wsManager.isConnected()) {
            this.wsManager.sendMessage(command);
        } else {
            console.warn('WebSocket not connected, override LLM command not sent');
        }

        // Save to localStorage for persistence
        try {
            const overrideSelection = {
                override_llm: model?.id || null,
                timestamp: Date.now()
            };
            localStorage.setItem('override-llm-selection', JSON.stringify(overrideSelection));
            console.log('Override LLM selection saved to localStorage');
        } catch (e) {
            console.warn('Failed to save override LLM selection to localStorage:', e);
        }

        this.scheduleAutoSave();
    }

    /**
     * Load and restore LLM selections from localStorage
     */
    loadLLMSelections() {
        try {
            const stored = localStorage.getItem('llm-selections');
            if (stored) {
                const selections = JSON.parse(stored);
                console.log('Attempting to restore LLM selections:', selections);

                let restoredCount = 0;

                // Restore selections to selectboxes
                if (selections.primary && this.llmSelectors.primary) {
                    this.llmSelectors.primary.setSelectedModel(selections.primary);
                    console.log(`Restoring primary selection: ${selections.primary}`);
                    restoredCount++;
                }
                if (selections.fallback1 && this.llmSelectors.fallback1) {
                    this.llmSelectors.fallback1.setSelectedModel(selections.fallback1);
                    console.log(`Restoring fallback1 selection: ${selections.fallback1}`);
                    restoredCount++;
                }
                if (selections.fallback2 && this.llmSelectors.fallback2) {
                    this.llmSelectors.fallback2.setSelectedModel(selections.fallback2);
                    console.log(`Restoring fallback2 selection: ${selections.fallback2}`);
                    restoredCount++;
                }

                console.log(`LLM selections restoration completed: ${restoredCount} selections restored`);
            } else {
                console.log('No stored LLM selections found in localStorage');
            }
        } catch (e) {
            console.warn('Failed to load LLM selections from localStorage:', e);
        }
    }

    setupProfileSlots() {
        const profileSlots = document.querySelectorAll('.profile-slot');

        profileSlots.forEach(slot => {
            slot.addEventListener('click', () => {
                const slotNumber = slot.dataset.slot;
                this.handleProfileSlotClick(slotNumber);
            });
        });

        this.loadProfileSlots();
    }

    async loadProfileSlots() {
        try {
            const url = window.backendConnector ? await window.backendConnector.buildURL('/api/profiles') : '/api/profiles';
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

            try {
                const response = await fetch(url, { signal: controller.signal });
                clearTimeout(timeoutId);
                const data = await response.json();

                if (data.status === 'success') {
                    this.profileSlots = {};

                    data.profiles.forEach(profile => {
                        this.profileSlots[profile.id] = profile;
                    });

                    this.updateProfileSlotUI();

                    if (data.active_profile) {
                        this.activeProfileSlot = data.active_profile;
                        this.markSlotAsActive(data.active_profile);
                    }
                }
            } catch (error) {
                clearTimeout(timeoutId);
                throw error;
            }
        } catch (error) {
            console.error('Failed to load profile slots:', error);
        }
    }

    updateProfileSlotUI() {
        const profileSlots = document.querySelectorAll('.profile-slot');

        profileSlots.forEach(slot => {
            const slotNumber = slot.dataset.slot;

            if (this.profileSlots[slotNumber]) {
                slot.classList.add('has-profile');
            } else {
                slot.classList.remove('has-profile');
            }
        });
    }

    markSlotAsActive(slotNumber) {
        const profileSlots = document.querySelectorAll('.profile-slot');

        profileSlots.forEach(slot => {
            if (slot.dataset.slot === slotNumber) {
                slot.classList.add('active');
            } else {
                slot.classList.remove('active');
            }
        });
    }

    async handleProfileSlotClick(slotNumber) {
        const profile = this.profileSlots[slotNumber];

        if (profile) {
            await this.applyProfile(slotNumber);
        } else {
            await this.saveCurrentConfigToSlot(slotNumber);
        }
    }

    async applyProfile(slotNumber) {
        if (this.isInitializing) return;

        if (this.wsManager && this.wsManager.isConnected()) {
            this.wsManager.sendMessage(`/profile_load ${slotNumber}`);

            this.activeProfileSlot = slotNumber;
            this.markSlotAsActive(slotNumber);

            setTimeout(async () => {
                await this.loadProfileSlots();
                await this.refreshSelectorsFromBackend();
            }, 200);
        }
    }

    async refreshSelectorsFromBackend() {
        try {
            const url = window.backendConnector ? await window.backendConnector.buildURL('/api/llms') : '/api/llms';
            const response = await fetch(url);
            const data = await response.json();

            if (data.status === 'success' && data.current_configuration) {
                const config = data.current_configuration;

                this.isInitializing = true;

                if (this.llmSelectors.override) {
                    this.llmSelectors.override.setSelectedModel(config.override_llm || null);
                }

                if (this.llmSelectors.router) {
                    this.llmSelectors.router.setSelectedModel(config.router_llm || null);
                }

                if (this.llmSelectors.planner) {
                    this.llmSelectors.planner.setSelectedModel(config.planner_llm || null);
                }

                if (config.selected_llms && config.selected_llms.length > 0) {
                    if (this.llmSelectors.primary) {
                        this.llmSelectors.primary.setSelectedModel(config.selected_llms[0] || null);
                    }
                    if (this.llmSelectors.fallback1) {
                        this.llmSelectors.fallback1.setSelectedModel(config.selected_llms[1] || null);
                    }
                    if (this.llmSelectors.fallback2) {
                        this.llmSelectors.fallback2.setSelectedModel(config.selected_llms[2] || null);
                    }
                }

                this.isInitializing = false;

                console.log('✅ Refreshed selectors from backend configuration');
            }
        } catch (error) {
            console.error('Failed to refresh selectors from backend:', error);
            this.isInitializing = false;
        }
    }

    async saveCurrentConfigToSlot(slotNumber) {
        if (this.wsManager && this.wsManager.isConnected()) {
            this.wsManager.sendMessage(`/profile_save ${slotNumber}`);

            setTimeout(async () => {
                await this.loadProfileSlots();
                this.activeProfileSlot = slotNumber;
                this.markSlotAsActive(slotNumber);
            }, 100);
        }
    }

    scheduleAutoSave() {
        if (!this.activeProfileSlot) return;

        if (this.autoSaveTimeout) {
            clearTimeout(this.autoSaveTimeout);
        }

        this.autoSaveTimeout = setTimeout(() => {
            this.saveCurrentConfigToSlot(this.activeProfileSlot);
        }, 2000);
    }

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

}

/**
 * Initialize the application when DOM is ready and all dependencies are loaded
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
    window.VisualResponseApp = app;
    window.agentInterface = app;  // For message edit/delete buttons
}

/**
 * Global permission response function
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

/**
 * Global interrupt signal function
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

/**
 * Global tool detail toggle function
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

/**
 * Console Log Manager
 * Handles the console sidebar functionality for displaying raw CLI-Agent output
 */
class ConsoleLogManager {
    constructor() {
        /** @type {boolean} */
        this.consoleVisible = false;

        /** @type {ConsoleLogEntry[]} */
        this.consoleLog = [];

        /** @type {number} */
        this.maxLogLines = 5000;

        // Performance optimization properties
        /** @type {boolean} */
        this.consoleUpdatePending = false;
        /** @type {number} */
        this.lastRenderTime = 0;
        /** @type {number} */
        this.renderThrottleMs = 16; // ~60fps max

        /** @type {number} */
        this.consoleWidth = 350; // Default width

        /** @type {boolean} */
        this.isResizing = false;

        this.init();
    }

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

    setupToggleButton() {
        const toggleBtn = document.getElementById('console-toggle-btn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                this.toggleConsole();
            });
        }
    }

    setupClearButton() {
        const clearBtn = document.getElementById('console-clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearConsole();
            });
        }
    }

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

    normalizeLevelFromBackend(level) {
        const normalized = (level || 'INFO').toLowerCase().trim();
        if (normalized.includes('error')) return 'error';
        if (normalized.includes('warn')) return 'warning';
        if (normalized.includes('debug')) return 'debug';
        return 'info';
    }

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

    disconnectSSE() {
        if (this.sseConnection) {
            this.sseConnection.close();
            this.sseConnection = null;
        }
    }

    startAutoRefresh(intervalMs = 5000) {
        this.stopAutoRefresh();
        if (this.logSource === 'file') {
            this.autoRefreshInterval = setInterval(() => {
                this.fetchLogFile();
            }, intervalMs);
        }
    }

    stopAutoRefresh() {
        this.disconnectSSE();

        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
        }
    }

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

    /**
     * Add a log line to the console
     * @param {string} line - Log line content
     * @param {'info'|'warning'|'error'|'debug'} level - Log level
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

    /**
     * Add multiple log lines efficiently (for file-based streaming)
     * @param {Array} lines - Array of {line, level} objects
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

    /**
     * Optimized console rendering using DocumentFragment for better performance
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

    // Legacy method for backwards compatibility
    renderConsoleLog() {
        this.renderConsoleLogOptimized();
    }

    /**
     * Get CSS class for log level
     * @param {string} level - Log level
     * @returns {string} CSS class name
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

    /**
     * Remove ANSI escape codes from text
     * @param {string} text - Text with ANSI codes
     * @returns {string} Cleaned text
     */
    cleanAnsiCodes(text) {
        // Remove ANSI escape sequences for colors, cursor control, etc.
        return text.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '')
                  .replace(/\x1b\].*?\x07/g, '')
                  .replace(/\x1b\[.*?[@-~]/g, '')
                  .replace(/\u001b\[[0-9;]*[a-zA-Z]/g, '');
    }

    /**
     * Escape HTML entities in text
     * @param {string} text - Text to escape
     * @returns {string} HTML-escaped text
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Schedule a console update using requestAnimationFrame for optimal performance
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

    /**
     * Perform the actual console update with optimized DOM manipulation
     */
    performConsoleUpdate() {
        this.renderConsoleLogOptimized();
        this.scrollToBottomOptimized();
    }

    /**
     * Optimized scroll to bottom - single efficient operation
     */
    scrollToBottomOptimized() {
        const consoleContent = document.getElementById('console-content');
        if (consoleContent) {
            // Single efficient scroll operation
            consoleContent.scrollTop = consoleContent.scrollHeight;
        }
    }

    // Legacy method for backwards compatibility
    scrollToBottom() {
        this.scrollToBottomOptimized();
    }

    clearConsole() {
        this.consoleLog = [];
        this.renderConsoleLog();
    }

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
}

// Global console log manager instance
let consoleLogManager = null;

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