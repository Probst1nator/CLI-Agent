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

        // Clear history button event listener
        this.setupClearHistoryButton();

        // Iframe focus management
        this.setupIframeFocusManagement();
    }

    handleChatSubmit() {
        const message = this.chatInput.value.trim();

        if (message && this.wsManager.isConnected()) {
            // Display user message immediately
            this.displayUserMessage(message);

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

    setupClearHistoryButton() {
        const clearHistoryBtn = document.getElementById('clear-history-btn');
        if (!clearHistoryBtn) {
            console.warn('Clear history button not found');
            return;
        }

        clearHistoryBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();

            // Show custom reset confirmation dialog with stopwatch
            this.showResetConfirmationDialog().then((confirmed) => {
            if (confirmed) {
                // Send agent reset command to backend
                if (this.wsManager && this.wsManager.isConnected()) {
                    this.wsManager.sendMessage('__AGENT_RESET__');

                    // Visual feedback
                    const originalText = clearHistoryBtn.querySelector('.clear-text').textContent;
                    clearHistoryBtn.querySelector('.clear-text').textContent = 'Resetting...';
                    clearHistoryBtn.disabled = true;

                    // Clear UI immediately for instant feedback
                    const contentWrapper = document.getElementById('content-wrapper');
                    if (contentWrapper) {
                        contentWrapper.innerHTML = '';
                    }

                    // Clear local storage
                    this.clearMessageHistory();

                    // Reset button state after delay
                    setTimeout(() => {
                        clearHistoryBtn.querySelector('.clear-text').textContent = originalText;
                        clearHistoryBtn.disabled = false;
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
                    const originalText = clearHistoryBtn.querySelector('.clear-text').textContent;
                    clearHistoryBtn.querySelector('.clear-text').textContent = 'Cleared (UI Only)';

                    setTimeout(() => {
                        clearHistoryBtn.querySelector('.clear-text').textContent = originalText;
                    }, 2000);
                }
            }
            }); // Close the promise callback
        });
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
            title.textContent = '🔄 Agent Reset Confirmation';
            title.className = 'reset-dialog-title';

            // Message
            const message = document.createElement('p');
            message.innerHTML = 'This will completely reset the agent and clear all:<br>• Conversation history<br>• Tool states and context<br>• Images and metadata<br><br><strong>This action cannot be undone.</strong>';
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
            'Appeasing the algorithm...',      // A nod to the mysterious power of social media/search algorithms
            'Waking up the servers...',         // A classic, simple and relatable
            'Reticulating splines...',          // A famous, nonsensical loading message from SimCity
            'Synergizing the paradigms...',     // Poking fun at corporate buzzwords
            'Compiling patterns...',     // A fun take on a standard developer task
            'Avoiding the singularity...',       // A humorous, high-stakes take on working with advanced AI

            // Philosphical
            'Pondering the cosmic giggle...',           // Watts: The playful, non-serious nature of existence
            'Ingesting a heroic dose of data...',          // McKenna: A nod to his ideas on profound experiences
            'Searching for synchronicities...',// Jung: Looking for meaningful, acausal connections
            'Surrendering to the data stream...',       // Watts: A Taoist/Zen approach to letting the answer emerge
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
        const messageEntry = {
            type: type,
            timestamp: Date.now(),
            data: data
        };

        this.messageHistory.push(messageEntry);

        // Keep only last 50 messages to prevent localStorage bloat
        if (this.messageHistory.length > 50) {
            this.messageHistory = this.messageHistory.slice(-50);
        }

        try {
            localStorage.setItem('agent-message-history', JSON.stringify(this.messageHistory));
        } catch (e) {
            console.warn('Failed to save message history to localStorage:', e);
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
                console.log('📋 Parsed message history:', this.messageHistory);
                console.log(`🔄 Continue mode: Loading ${this.messageHistory.length} messages from previous session`);

                // Count user vs agent messages
                const userCount = this.messageHistory.filter(m => m.type === 'user').length;
                const agentCount = this.messageHistory.filter(m => m.type === 'agent').length;
                console.log(`📊 Message breakdown: ${userCount} user, ${agentCount} agent`);

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

        for (const entry of sortedHistory) {
            try {
                if (entry.type === 'user') {
                    this.restoreUserMessage(entry.data);
                } else if (entry.type === 'agent') {
                    await this.restoreAgentMessage(entry.data);
                }
            } catch (e) {
                console.warn('Failed to restore message:', e, entry);
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
        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper) {
            console.warn('❌ Content wrapper not found for user message restoration');
            return;
        }

        // Create message block
        const messageBlock = document.createElement('div');
        messageBlock.className = 'message-block user-message';

        // Create timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'agent-msg-head';
        timestamp.textContent = new Date(data.timestamp).toLocaleTimeString();

        // Create content
        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = data.message;

        // Assemble message
        messageBlock.appendChild(timestamp);
        messageBlock.appendChild(content);
        contentWrapper.appendChild(messageBlock);

        console.log('✅ User message restored successfully');
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

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, checking dependencies...');
    initializeApp();
});