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
class VisualResponseApp {
    /**
     * Initialize the Visual Response Application
     * Sets up all components and establishes the application state
     */
    constructor() {
        this.renderer = null;
        this.wsManager = null;
        this.chatInput = null;
        this.chatForm = null;
        this.messageHistory = [];

        // Initialization protection flag
        this.isInitializing = true;

        // Get logger reference
        this.logger = window.visualLogger;

        this.logger.info('VisualResponseApp starting initialization');
        this.logger.startTiming('app_initialization');

        this.init();
    }

    init() {
        console.log('Initializing Visual Response App...');
        this.checkLibraries();
        this.initializeComponents();
        this.setupEventListeners();
        this.setupThemeSwitcher();
        this.loadMessageHistory();
        console.log('Visual Response App initialized successfully');
    }

    checkLibraries() {
        console.log('Library status check:');
        console.log('- jQuery:', typeof $ !== 'undefined' ? 'loaded' : 'MISSING');
        console.log('- Marked:', typeof marked !== 'undefined' ? 'loaded' : 'MISSING');
        console.log('- Mermaid:', typeof mermaid !== 'undefined' ? 'loaded' : 'MISSING');
        console.log('- Plotly:', typeof Plotly !== 'undefined' ? 'loaded' : 'MISSING');
        console.log('- 3Dmol:', typeof $3Dmol !== 'undefined' ? 'loaded' : 'MISSING');
        console.log('- svgPanZoom:', typeof svgPanZoom !== 'undefined' ? 'loaded' : 'MISSING');
        console.log('- React:', typeof React !== 'undefined' ? 'loaded' : 'MISSING');
        console.log('- ReactDOM:', typeof ReactDOM !== 'undefined' ? 'loaded' : 'MISSING');
        console.log('- Vue:', typeof Vue !== 'undefined' ? 'loaded' : 'MISSING');
        console.log('- Babel:', typeof Babel !== 'undefined' ? 'loaded' : 'MISSING');
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
            console.log('Mermaid initialized successfully');
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

        // Get chat elements
        this.chatForm = document.getElementById('chat-form');
        this.chatInput = document.getElementById('chat-input');
        
        // File upload elements
        this.fileInput = document.getElementById('file-input');
        this.fileUploadBtn = document.getElementById('file-upload-btn');

        // Initialize LLM selectors
        this.initializeLLMSelectors();
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

            console.log('Marked renderer called with code: "' + codeStr.substring(0, 50) + '...", language: "' + language + '", escaped: ' + escaped);
            console.log('Full code object:', code);

            // Handle both 'mermaid' and undefined/null language cases by checking code content
            const isMermaidCode = language === 'mermaid' ||
                                 ((!language || language === 'undefined') &&
                                  (codeStr.trim().startsWith('graph ') ||
                                   codeStr.trim().startsWith('flowchart ') ||
                                   codeStr.trim().startsWith('sequenceDiagram') ||
                                   codeStr.trim().startsWith('classDiagram') ||
                                   codeStr.trim().startsWith('erDiagram')));

            if (isMermaidCode) {
                console.log('Mermaid code block detected!');
                const uniqueId = 'mermaid-embedded-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
                const placeholder = '<div class="mermaid-placeholder" data-mermaid-id="' + uniqueId + '" data-mermaid-definition="' + encodeURIComponent(codeStr) + '" style="text-align:center;"></div>';
                console.log('Generated placeholder:', placeholder);
                return placeholder;
            }

            // Apply syntax highlighting for regular code blocks
            const highlightedCode = this.highlightCode(codeStr, language);
            return '<pre><code class="language-' + (language || 'text') + '">' + highlightedCode + '</code></pre>';
        };

        marked.use({ renderer });
        console.log('Marked renderer configured');
    }

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

        // Global JS toggle event listener
        this.setupGlobalJSToggle();


        // Reset history button event listener
        this.setupResetHistoryButton();

        // Iframe focus management
        this.setupIframeFocusManagement();

        // Agent message height management
        // this.setupAgentMessageHeightManagement();
    }

    handleChatSubmit() {
        const message = this.chatInput.value.trim();

        if (message && this.wsManager.isConnected()) {
            // Display user message immediately
            this.displayUserMessage(message);

            // Show interrupt button when sending user message (agent will start generating)
            this.showInterruptButton();

            // Send the message
            this.wsManager.sendMessage(message);
            this.chatInput.value = '';
            this.chatInput.focus();

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

    displayUserMessage(message) {
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

        // Save user message to history
        this.saveMessageToHistory('user', {
            message: message,
            timestamp: currentTimestamp
        });

        // Force immediate scroll after user message
        this.scrollToBottom();

        // Show loading indicator after user message
        this.showLoadingIndicator();

        // Scroll again after loading indicator is added
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

        // Force immediate scroll with multiple methods
        const scrollToEnd = () => {
            const scrollHeight = contentWrapper.scrollHeight;
            const clientHeight = contentWrapper.clientHeight;
            const maxScrollTop = Math.max(0, scrollHeight - clientHeight);
            
            console.log('Scrolling attempt:', { 
                scrollHeight, 
                clientHeight, 
                currentScrollTop: contentWrapper.scrollTop,
                maxScroll: maxScrollTop,
                targetScroll: scrollHeight + 100 // Extra padding to ensure bottom
            });
            
            // Method 1: Direct scrollTop assignment with extra padding
            contentWrapper.scrollTop = scrollHeight + 100;
            
            // Method 2: scrollTo with extra padding
            contentWrapper.scrollTo(0, scrollHeight + 100);
            
            // Method 3: Force scroll to absolute maximum
            contentWrapper.scrollTop = contentWrapper.scrollHeight;
            
            // Method 4: Alternative scrollIntoView approach
            const lastChild = contentWrapper.lastElementChild;
            if (lastChild) {
                try {
                    lastChild.scrollIntoView({ behavior: 'instant', block: 'end', inline: 'nearest' });
                } catch (e) {
                    lastChild.scrollIntoView(false); // Fallback
                }
            }
            
            // Method 5: One more forced scroll after DOM settles
            setTimeout(() => {
                contentWrapper.scrollTop = contentWrapper.scrollHeight;
            }, 1);
        };

        // Immediate scroll
        scrollToEnd();
        
        // Delayed scrolls to handle dynamic content
        requestAnimationFrame(() => {
            scrollToEnd();
            setTimeout(scrollToEnd, 10);
            setTimeout(scrollToEnd, 50);
            setTimeout(scrollToEnd, 100);
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

            console.log(`Global JS ${newState ? 'enabled' : 'disabled'}`);
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

        console.log(`Applying global JS ${enabled ? 'enable' : 'disable'} to ${jsMessages.length} messages`);

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
                    console.log(`Re-rendered message ${messageId} with JS ${enabled ? 'enabled' : 'disabled'}`);
                } else {
                    console.log(`Could not retrieve original content for message ${messageId}`);
                }
            } catch (error) {
                console.warn(`Failed to re-render message ${messageId}:`, error);
            }
        }

        console.log(`Applied JS ${enabled ? 'enable' : 'disable'} to message ${messageId}`);
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

    setupResetHistoryButton() {
        const resetHistoryBtn = document.getElementById('reset-history-btn');
        if (!resetHistoryBtn) {
            console.warn('Reset history button not found');
            return;
        }

        // Setup interrupt button
        const interruptBtn = document.getElementById('interrupt-btn');
        if (interruptBtn) {
            this.setupInterruptButton(interruptBtn);
        } else {
            console.warn('Interrupt button not found');
        }

        resetHistoryBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();

            // Show custom reset confirmation dialog with stopwatch
            this.showResetConfirmationDialog().then((confirmed) => {
            if (confirmed) {
                // Send reset command to backend
                if (this.wsManager && this.wsManager.isConnected()) {
                    this.wsManager.sendMessage('/reset');

                    // Visual feedback
                    const originalText = resetHistoryBtn.querySelector('.reset-text').textContent;
                    resetHistoryBtn.querySelector('.reset-text').textContent = 'Resetting...';
                    resetHistoryBtn.disabled = true;

                    // Clear UI immediately for instant feedback
                    const contentWrapper = document.getElementById('content-wrapper');
                    if (contentWrapper) {
                        contentWrapper.innerHTML = '';
                    }

                    // Clear local storage
                    this.clearMessageHistory();

                    // Reset button state after delay
                    setTimeout(() => {
                        resetHistoryBtn.querySelector('.reset-text').textContent = originalText;
                        resetHistoryBtn.disabled = false;
                    }, 2000);

                    console.log('Agent reset requested by user');
                } else {
                    // Fallback to UI-only clear if WebSocket not available
                    console.warn('WebSocket not connected, falling back to UI-only clear');

                    const contentWrapper = document.getElementById('content-wrapper');
                    if (contentWrapper) {
                        contentWrapper.innerHTML = '';
                    }
                    this.clearMessageHistory();

                    // Visual feedback for fallback
                    const originalText = resetHistoryBtn.querySelector('.reset-text').textContent;
                    resetHistoryBtn.querySelector('.reset-text').textContent = 'Reset (UI Only)';

                    setTimeout(() => {
                        resetHistoryBtn.querySelector('.reset-text').textContent = originalText;
                    }, 2000);
                }
            }
            }); // Close the promise callback
        });
    }


    setupInterruptButton(interruptBtn) {
        // Track agent state to show/hide button appropriately
        this.agentGenerating = false;

        // Button starts in inactive (grayed out) state
        interruptBtn.classList.remove('active', 'processing');

        interruptBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();

            // Only process clicks when button is active
            if (!interruptBtn.classList.contains('active')) {
                return;
            }

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

                interruptBtn.classList.remove('active');
                interruptBtn.classList.add('processing');
                interruptBtn.querySelector('.interrupt-icon').textContent = '⏳';
                // Keep button clickable - users should be able to send multiple interrupts

                // Reset button state after a delay (agent should stop soon)
                setTimeout(() => {
                    this.hideInterruptButton();

                    // Reset button state
                    interruptBtn.classList.remove('processing');
                    interruptBtn.querySelector('.interrupt-icon').textContent = originalIcon;
                }, 3000);

                console.log('Interrupt signal sent to agent');
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

    setupThemeSwitcher() {
        const lightThemeBtn = document.getElementById('light-theme-btn');
        const darkThemeBtn = document.getElementById('dark-theme-btn');

        const setTheme = (theme) => {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('agent-theme', theme);
        };

        if (lightThemeBtn) {
            lightThemeBtn.addEventListener('click', () => setTheme('light'));
        }

        if (darkThemeBtn) {
            darkThemeBtn.addEventListener('click', () => setTheme('dark'));
        }

        // Load saved theme or default to dark
        setTheme(localStorage.getItem('agent-theme') || 'dark');
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
     * Save message to history and localStorage
     * @param {string} type - 'user' or 'agent'
     * @param {object} data - Message data
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
                console.log(`Skipping storage of transient message type: ${data.messageData.type}`);
                return;
            }
        }

        // Validate user message structure
        if (type === 'user') {
            if (!data.message || typeof data.message !== 'string') {
                console.error('❌ Invalid user message structure - missing or invalid message field:', data);
                return;
            }
            console.log(`💾 Saving user message to history: "${data.message.substring(0, 50)}${data.message.length > 50 ? '...' : ''}"`);
        }

        // Validate agent message structure
        if (type === 'agent') {
            if (!data.messageData) {
                console.error('❌ Invalid agent message structure - missing messageData field:', data);
                return;
            }
            console.log(`💾 Saving agent message to history: type=${data.messageData.type}, id=${data.originalId || 'unknown'}`);
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
            console.log(`🧹 Trimmed ${removedCount} old messages from history to maintain 50 message limit`);
        }

        try {
            localStorage.setItem('agent-message-history', JSON.stringify(this.messageHistory));
            console.log(`📊 Message history saved: ${this.messageHistory.length} total messages`);
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
            console.log('🔍 Checking localStorage for message history...');
            console.log('Raw stored data:', stored);

            if (stored) {
                this.messageHistory = JSON.parse(stored);

                // Clean up any legacy permission requests that may have been incorrectly stored
                const originalLength = this.messageHistory.length;
                this.messageHistory = this.messageHistory.filter(entry => {
                    // CRITICAL FIX: Never filter out actual user messages (type: 'user')
                    if (entry.type === 'user') {
                        console.log('✅ Preserving user message during history cleanup:', entry.data?.message?.substring(0, 50) + '...');
                        return true; // Always keep user messages
                    }

                    // Only filter agent messages with transient types
                    if (entry.type === 'agent' && entry.data && entry.data.messageData) {
                        // Removed 'user_message' from transient types - this was incorrectly filtering user messages
                        const transientTypes = ['permission_request', 'config'];
                        if (transientTypes.includes(entry.data.messageData.type)) {
                            console.log(`Removing legacy transient message: ${entry.data.messageData.type}`);
                            return false;
                        }
                    }
                    return true;
                });

                // Save cleaned history back to localStorage if we removed anything
                if (this.messageHistory.length < originalLength) {
                    console.log(`Cleaned ${originalLength - this.messageHistory.length} transient messages from history`);
                    try {
                        localStorage.setItem('agent-message-history', JSON.stringify(this.messageHistory));
                    } catch (e) {
                        console.warn('Failed to save cleaned message history:', e);
                    }
                }

                console.log('📋 Parsed message history:', this.messageHistory);
                console.log(`🔄 Continue mode: Loading ${this.messageHistory.length} messages from previous session`);

                // Count user vs agent messages with detailed breakdown
                const userMessages = this.messageHistory.filter(m => m.type === 'user');
                const agentMessages = this.messageHistory.filter(m => m.type === 'agent');
                console.log(`📊 Message breakdown: ${userMessages.length} user, ${agentMessages.length} agent`);

                // Debug: Log first few characters of each user message to verify content
                if (userMessages.length > 0) {
                    console.log('👤 User messages found for restoration:');
                    userMessages.forEach((msg, i) => {
                        const preview = msg.data?.message?.substring(0, 50) || 'No message content';
                        const timestamp = new Date(msg.data?.timestamp || msg.timestamp).toLocaleTimeString();
                        console.log(`  ${i + 1}. [${timestamp}] "${preview}${msg.data?.message?.length > 50 ? '...' : ''}"`);
                    });
                } else {
                    console.log('⚠️ No user messages found in history - this may indicate the bug');
                }

                // Add flag to indicate we're in restoration mode
                this.isRestoring = true;
                this.restoreMessages().then(() => {
                    this.isRestoring = false;
                });

                // Show a temporary restoration indicator
                this.showRestorationFeedback(this.messageHistory.length);
            } else {
                console.log('🔄 Continue mode: No previous session history found in localStorage');
            }
        } catch (e) {
            console.warn('❌ Failed to load message history from localStorage:', e);
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
     * @param {object} data - User message data
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
     * @param {object} data - Agent message data
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
    initializeLLMSelectors() {
        // Check if SearchableLLMSelectbox is available
        if (typeof SearchableLLMSelectbox === 'undefined') {
            console.warn('SearchableLLMSelectbox not available, LLM selectors will not be initialized');
            return;
        }

        // Initialize the LLM selectors (including router and planner)
        this.llmSelectors = {};
        this.llmSelectorsLoaded = { router: false, planner: false, primary: false, fallback1: false, fallback2: false };

        // Store system configuration for reference
        this.systemConfiguration = null;

        // Create a callback to track when all selectors have loaded their models
        const onModelsLoaded = (selector, models, systemConfig) => {
            // Store the system configuration from the first selector that loads
            if (systemConfig && !this.systemConfiguration) {
                this.systemConfiguration = systemConfig;
                console.log('System configuration received:', systemConfig);
            }

            // Find which selector this is
            const selectorName = Object.keys(this.llmSelectors).find(key => this.llmSelectors[key] === selector);
            if (selectorName) {
                this.llmSelectorsLoaded[selectorName] = true;
                console.log(`Models loaded for ${selectorName} selector: ${models.length} models`);

                // Check if all selectors have loaded their models
                const allLoaded = Object.values(this.llmSelectorsLoaded).every(loaded => loaded);
                if (allLoaded) {
                    console.log('All LLM selectors have loaded their models...');
                    this.initializeFromSystemConfiguration();
                }
            }
        };

        // Router LLM selector
        const routerContainer = document.getElementById('router-llm-selector');
        if (routerContainer) {
            this.llmSelectors.router = new SearchableLLMSelectbox(routerContainer, {
                placeholder: 'Router LLM...',
                id: 'llm-router',
                includeNoneOption: true,  // Special flag for router to include None option
                onSelectionChange: (model) => this.handleRouterLLMSelectionChange(model),
                onModelsLoaded: onModelsLoaded
            });
        }

        // Planner LLM selector
        const plannerContainer = document.getElementById('planner-llm-selector');
        if (plannerContainer) {
            this.llmSelectors.planner = new SearchableLLMSelectbox(plannerContainer, {
                placeholder: 'Planner LLM...',
                id: 'llm-planner',
                includeNoneOption: true,
                onSelectionChange: (model) => this.handlePlannerLLMSelectionChange(model),
                onModelsLoaded: onModelsLoaded
            });
        }

        // Primary LLM selector
        const primaryContainer = document.getElementById('llm-selector-primary');
        if (primaryContainer) {
            this.llmSelectors.primary = new SearchableLLMSelectbox(primaryContainer, {
                placeholder: 'Primary LLM...',
                id: 'llm-primary',
                includeNoneOption: true,
                onSelectionChange: (model) => this.handleLLMSelectionChange('primary', model),
                onModelsLoaded: onModelsLoaded
            });
        }

        // First fallback LLM selector
        const fallback1Container = document.getElementById('llm-selector-fallback1');
        if (fallback1Container) {
            this.llmSelectors.fallback1 = new SearchableLLMSelectbox(fallback1Container, {
                placeholder: 'Fallback 1...',
                id: 'llm-fallback1',
                includeNoneOption: true,
                onSelectionChange: (model) => this.handleLLMSelectionChange('fallback1', model),
                onModelsLoaded: onModelsLoaded
            });
        }

        // Second fallback LLM selector
        const fallback2Container = document.getElementById('llm-selector-fallback2');
        if (fallback2Container) {
            this.llmSelectors.fallback2 = new SearchableLLMSelectbox(fallback2Container, {
                placeholder: 'Fallback 2...',
                id: 'llm-fallback2',
                includeNoneOption: true,
                onSelectionChange: (model) => this.handleLLMSelectionChange('fallback2', model),
                onModelsLoaded: onModelsLoaded
            });
        }

        console.log('LLM selectors initialized successfully');
    }

    /**
     * Initialize selectors from system configuration
     */
    initializeFromSystemConfiguration() {
        // DEBUG: Log what we received from the API
        console.log('🔍 WEB UI PERSISTENCE DEBUG:');
        console.log('  → systemConfiguration received:', this.systemConfiguration);

        if (this.systemConfiguration && this.systemConfiguration.fallback_models) {
            const fallbackModels = this.systemConfiguration.fallback_models;
            console.log(`Initializing UI with ${fallbackModels.length} fallback models:`, fallbackModels);

            // Initialize router LLM from saved configuration, defaults to None
            const savedRouterLLM = this.systemConfiguration.router_llm;
            if (this.llmSelectors.router) {
                if (savedRouterLLM) {
                    console.log(`🔍 BUG HUNT: Setting router LLM to saved selection: ${savedRouterLLM} (isInitializing=${this.isInitializing})`);
                    this.llmSelectors.router.setSelectedModel(savedRouterLLM);
                    console.log(`Set router LLM selector to saved selection: ${savedRouterLLM}`);
                } else {
                    console.log(`🔍 BUG HUNT: Setting router LLM to None (default) (isInitializing=${this.isInitializing})`);
                    this.llmSelectors.router.setSelectedModel(null);
                    console.log(`Set router LLM selector to None (default)`);
                }
            }

            // Initialize planner LLM from saved configuration, or default to first fallback model
            const savedPlannerLLM = this.systemConfiguration.planner_llm;
            if (savedPlannerLLM && this.llmSelectors.planner) {
                console.log(`🔍 BUG HUNT: Setting planner LLM to saved selection: ${savedPlannerLLM} (isInitializing=${this.isInitializing})`);
                this.llmSelectors.planner.setSelectedModel(savedPlannerLLM);
                console.log(`Set planner LLM selector to saved selection: ${savedPlannerLLM}`);
            } else if (fallbackModels.length >= 1 && this.llmSelectors.planner) {
                console.log(`🔍 BUG HUNT: Setting planner LLM to default: ${fallbackModels[0]} (isInitializing=${this.isInitializing})`);
                this.llmSelectors.planner.setSelectedModel(fallbackModels[0]);
                console.log(`Set planner LLM selector to default (first fallback): ${fallbackModels[0]}`);
            }

            // Display first 3 fallback models in the UI
            if (fallbackModels.length >= 1 && this.llmSelectors.primary) {
                console.log(`🔍 BUG HUNT: Setting primary LLM to: ${fallbackModels[0]} (isInitializing=${this.isInitializing})`);
                this.llmSelectors.primary.setSelectedModel(fallbackModels[0]);
                console.log(`Set primary selector to: ${fallbackModels[0]}`);
            }

            if (fallbackModels.length >= 2 && this.llmSelectors.fallback1) {
                console.log(`🔍 BUG HUNT: Setting fallback1 LLM to: ${fallbackModels[1]} (isInitializing=${this.isInitializing})`);
                this.llmSelectors.fallback1.setSelectedModel(fallbackModels[1]);
                console.log(`Set fallback1 selector to: ${fallbackModels[1]}`);
            }

            if (fallbackModels.length >= 3 && this.llmSelectors.fallback2) {
                console.log(`🔍 BUG HUNT: Setting fallback2 LLM to: ${fallbackModels[2]} (isInitializing=${this.isInitializing})`);
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
        console.log('🔍 BUG HUNT: About to disable initialization flag...');
        this.isInitializing = false;
        console.log('🔍 BUG HUNT: VisualResponseApp initialization complete - selection events now enabled');
    }

    /**
     * Handle LLM selection changes
     * @param {string} position - 'primary', 'fallback1', or 'fallback2'
     * @param {Object} model - Selected model object
     */
    handleLLMSelectionChange(position, model) {
        console.log(`🔍 BUG HUNT: LLM selection changed - ${position}:`, model);
        console.log(`🔍 BUG HUNT: isInitializing = ${this.isInitializing}`);

        // Guard against updates during initialization
        if (this.isInitializing) {
            console.log(`🚫 BUG HUNT: Blocking LLM selection change during initialization - ${position}`);
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
    }

    /**
     * Handle router LLM selection changes
     */
    handleRouterLLMSelectionChange(model) {
        console.log('🔍 BUG HUNT: Router LLM selection changed:', model);
        console.log(`🔍 BUG HUNT: isInitializing = ${this.isInitializing}`);

        // Guard against updates during initialization
        if (this.isInitializing) {
            console.log(`🚫 BUG HUNT: Blocking router LLM selection change during initialization`);
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
    }

    /**
     * Handle planner LLM selection changes
     */
    handlePlannerLLMSelectionChange(model) {
        console.log('🔍 BUG HUNT: Planner LLM selection changed:', model);
        console.log(`🔍 BUG HUNT: isInitializing = ${this.isInitializing}`);

        // Guard against updates during initialization
        if (this.isInitializing) {
            console.log(`🚫 BUG HUNT: Blocking planner LLM selection change during initialization`);
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

}

// Initialize the application when DOM is ready and all dependencies are loaded
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
    new VisualResponseApp();
}

// Global permission response function
window.sendPermissionResponse = function(permissionId, result, action = {}) {
    console.log('sendPermissionResponse called', { permissionId, result, action });

    if (window.wsManager && window.wsManager.sendPermissionResponse) {
        return window.wsManager.sendPermissionResponse(permissionId, result, action);
    } else {
        console.error('WebSocket manager not available for permission response');
        return false;
    }
};

// Global interrupt signal function
window.sendInterruptSignal = function() {
    console.log('sendInterruptSignal called');

    if (window.wsManager && window.wsManager.sendInterruptSignal) {
        return window.wsManager.sendInterruptSignal();
    } else {
        console.error('WebSocket manager not available for interrupt signal');
        return false;
    }
};

// Global tool detail toggle function
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
        this.consoleVisible = false;
        this.consoleLog = [];
        this.maxLogLines = 1000;
        this.consoleWidth = 350; // Default width
        this.isResizing = false;
        this.init();
    }

    init() {
        this.setupToggleButton();
        this.setupClearButton();
        this.setupResizeHandle();
        this.loadConsoleState();
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

    toggleConsole() {
        this.consoleVisible = !this.consoleVisible;
        this.updateConsoleVisibility();
        this.saveConsoleState();
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

    addLogLine(line, level = 'info') {
        const timestamp = new Date().toLocaleTimeString();

        // Store the raw unfiltered line (just trimmed)
        const rawLine = line.trim();

        const logEntry = {
            timestamp,
            line: rawLine,
            level
        };

        this.consoleLog.push(logEntry);

        // Keep only the most recent lines
        if (this.consoleLog.length > this.maxLogLines) {
            this.consoleLog = this.consoleLog.slice(-this.maxLogLines);
        }

        this.renderConsoleLog();
        this.scrollToBottom();
    }

    renderConsoleLog() {
        const consoleContent = document.getElementById('console-content');
        if (!consoleContent) return;

        // Show last 100 lines for performance
        const displayLines = this.consoleLog.slice(-100);

        if (displayLines.length === 0) {
            consoleContent.innerHTML = `
                <div class="console-line">
                    <span class="console-timestamp">[${new Date().toLocaleTimeString()}]</span>
                    <span class="console-text">Console log will appear here...</span>
                </div>
            `;
            return;
        }

        consoleContent.innerHTML = displayLines.map(entry => {
            const levelClass = this.getLevelClass(entry.level);
            return `
                <div class="console-line">
                    <span class="console-timestamp">[${entry.timestamp}]</span>
                    <span class="console-text ${levelClass}">${this.escapeHtml(entry.line)}</span>
                </div>
            `;
        }).join('');
    }

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

    cleanAnsiCodes(text) {
        // Remove ANSI escape sequences for colors, cursor control, etc.
        return text.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '')
                  .replace(/\x1b\].*?\x07/g, '')
                  .replace(/\x1b\[.*?[@-~]/g, '')
                  .replace(/\u001b\[[0-9;]*[a-zA-Z]/g, '');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom() {
        const consoleContent = document.getElementById('console-content');
        if (consoleContent) {
            consoleContent.scrollTop = consoleContent.scrollHeight;
        }
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

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, checking dependencies...');

    // Initialize console log manager
    consoleLogManager = new ConsoleLogManager();

    // Make it globally accessible for WebSocket messages
    window.consoleLogManager = consoleLogManager;

    initializeApp();
});