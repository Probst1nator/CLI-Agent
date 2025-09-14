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

        // Create message block
        const messageBlock = document.createElement('div');
        messageBlock.className = 'message-block user-message';

        // Create timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = new Date().toLocaleTimeString();

        // Create content
        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = message;

        // Assemble message
        messageBlock.appendChild(timestamp);
        messageBlock.appendChild(content);

        // Add to content wrapper
        contentWrapper.appendChild(messageBlock);
        
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
            
            console.log('Scrolling attempt:', { 
                scrollHeight, 
                clientHeight, 
                currentScrollTop: contentWrapper.scrollTop,
                maxScroll: scrollHeight - clientHeight 
            });
            
            // Method 1: Direct scrollTop assignment (most reliable)
            contentWrapper.scrollTop = scrollHeight;
            
            // Method 2: scrollTo with immediate behavior
            contentWrapper.scrollTo(0, scrollHeight);
            
            // Method 3: Smooth scroll for better UX (if supported)
            try {
                contentWrapper.scrollTo({
                    top: scrollHeight,
                    left: 0,
                    behavior: 'smooth'
                });
            } catch (e) {
                console.log('Smooth scroll not supported');
            }
            
            // Method 4: Alternative scrollIntoView approach
            const lastChild = contentWrapper.lastElementChild;
            if (lastChild) {
                try {
                    lastChild.scrollIntoView({ behavior: 'smooth', block: 'end' });
                } catch (e) {
                    lastChild.scrollIntoView(false); // Fallback
                }
            }
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
            'Shining...',
            
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
            'Compiling...',     // A fun take on a standard developer task
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

        loadingContent.appendChild(spinner);
        loadingContent.appendChild(loadingText);
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
            loadingIndicator.remove();
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

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, checking dependencies...');
    initializeApp();
});