/**
 * JavaScript Detection and Message State Management Utilities
 * 
 * Provides utilities for detecting JavaScript in HTML content and managing
 * per-message JavaScript execution state.
 */

class JavaScriptUtils {
    /**
     * Detects if HTML content contains JavaScript
     * @param {string} htmlContent - The HTML content to analyze
     * @returns {boolean} True if JavaScript is detected
     */
    static detectJavaScript(htmlContent) {
        if (!htmlContent || typeof htmlContent !== 'string') {
            return false;
        }
        
        // Convert to lowercase for case-insensitive matching
        const content = htmlContent.toLowerCase();
        
        // Check for script tags
        if (content.includes('<script')) {
            return true;
        }
        
        // Check for inline event handlers
        const eventHandlers = [
            'onclick=', 'onload=', 'onmouseover=', 'onmouseout=', 'onchange=',
            'onsubmit=', 'onkeypress=', 'onkeydown=', 'onkeyup=', 'onfocus=',
            'onblur=', 'onresize=', 'onscroll=', 'ondblclick=', 'oncontextmenu=',
            'oninput=', 'oninvalid=', 'onreset=', 'onsearch=', 'onselect=',
            'ondrag=', 'ondrop=', 'ondragover=', 'ondragstart=', 'ondragend='
        ];
        
        for (const handler of eventHandlers) {
            if (content.includes(handler)) {
                return true;
            }
        }
        
        // Check for javascript: URLs
        if (content.includes('javascript:')) {
            return true;
        }
        
        // Check for common JavaScript patterns in attributes
        if (content.match(/href\s*=\s*["']javascript:/i)) {
            return true;
        }
        
        return false;
    }
    
    /**
     * Gets the JavaScript execution state for a specific message
     * @param {string} messageId - The unique message identifier
     * @returns {boolean} True if JavaScript is enabled for this message
     */
    static getMessageJSState(messageId) {
        if (!messageId) return true; // Default to enabled
        
        try {
            const state = localStorage.getItem(`msg-js-${messageId}`);
            return state !== 'false'; // Default to true unless explicitly disabled
        } catch (e) {
            console.warn('Failed to read message JS state:', e);
            return true;
        }
    }
    
    /**
     * Sets the JavaScript execution state for a specific message
     * @param {string} messageId - The unique message identifier
     * @param {boolean} enabled - Whether JavaScript should be enabled
     */
    static setMessageJSState(messageId, enabled) {
        if (!messageId) return;
        
        try {
            localStorage.setItem(`msg-js-${messageId}`, enabled.toString());
        } catch (e) {
            console.warn('Failed to save message JS state:', e);
        }
    }
    
    /**
     * Toggles the JavaScript execution state for a specific message
     * @param {string} messageId - The unique message identifier
     * @returns {boolean} The new state after toggling
     */
    static toggleMessageJSState(messageId) {
        const currentState = this.getMessageJSState(messageId);
        const newState = !currentState;
        this.setMessageJSState(messageId, newState);
        return newState;
    }
    
    /**
     * Cleans up old message states to prevent localStorage bloat
     * Removes states older than 30 days
     */
    static cleanupOldStates() {
        try {
            const keys = Object.keys(localStorage);
            const msgJsKeys = keys.filter(key => key.startsWith('msg-js-'));
            
            // If we have more than 100 message states, clean up the oldest ones
            if (msgJsKeys.length > 100) {
                // Sort by timestamp (assuming message IDs contain timestamps)
                msgJsKeys.sort();
                
                // Remove oldest entries, keeping only the most recent 50
                const toRemove = msgJsKeys.slice(0, msgJsKeys.length - 50);
                toRemove.forEach(key => {
                    localStorage.removeItem(key);
                });
                
                console.log(`Cleaned up ${toRemove.length} old message JS states`);
            }
        } catch (e) {
            console.warn('Failed to cleanup old message states:', e);
        }
    }
    
    /**
     * Creates a toggle button element for a specific message
     * @param {string} messageId - The unique message identifier
     * @param {Function} onToggle - Callback function when toggle is clicked
     * @returns {HTMLElement} The toggle button element
     */
    static createToggleButton(messageId, onToggle) {
        const isEnabled = this.getMessageJSState(messageId);
        
        const button = document.createElement('button');
        button.className = `msg-js-toggle ${isEnabled ? 'enabled' : 'disabled'}`;
        button.textContent = isEnabled ? 'JS ON' : 'JS OFF';
        button.title = `Click to ${isEnabled ? 'disable' : 'enable'} JavaScript for this message`;
        button.setAttribute('data-message-id', messageId);
        
        button.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            const newState = this.toggleMessageJSState(messageId);
            
            // Update button appearance
            button.className = `msg-js-toggle ${newState ? 'enabled' : 'disabled'}`;
            button.textContent = newState ? 'JS ON' : 'JS OFF';
            button.title = `Click to ${newState ? 'disable' : 'enable'} JavaScript for this message`;
            
            // Call the callback
            if (onToggle) {
                onToggle(messageId, newState);
            }
        });
        
        return button;
    }
    
    /**
     * Creates a reset button element for a specific message
     * Performs the cycle: whatever_state → js_off → js_on
     * @param {string} messageId - The unique message identifier
     * @param {Function} onReset - Callback function when reset is clicked
     * @returns {HTMLElement} The reset button element
     */
    static createResetButton(messageId, onReset) {
        const button = document.createElement('button');
        button.className = 'msg-reset-button';
        button.textContent = '⟳';
        button.title = 'Reset: JS OFF → JS ON';
        button.setAttribute('data-message-id', messageId);
        
        button.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            // Find the corresponding toggle button to update its appearance
            const toggleButton = document.querySelector(`.msg-js-toggle[data-message-id="${messageId}"]`);
            
            // Cycle: whatever_state → js_off → js_on
            this.setMessageJSState(messageId, false); // First set to OFF
            
            // Update toggle button appearance for OFF state
            if (toggleButton) {
                toggleButton.className = 'msg-js-toggle disabled';
                toggleButton.textContent = 'JS OFF';
                toggleButton.title = 'Click to enable JavaScript for this message';
            }
            
            // Call callback for JS OFF state
            if (onReset && typeof onReset === 'function') {
                onReset(messageId, false);
            }
            
            // Then immediately set to ON after a brief delay for visual feedback
            setTimeout(() => {
                this.setMessageJSState(messageId, true);
                
                // Update toggle button appearance for ON state
                if (toggleButton) {
                    toggleButton.className = 'msg-js-toggle enabled';
                    toggleButton.textContent = 'JS ON';
                    toggleButton.title = 'Click to disable JavaScript for this message';
                }
                
                // Call callback for JS ON state
                if (onReset && typeof onReset === 'function') {
                    onReset(messageId, true);
                }
            }, 100);
        });
        
        return button;
    }
    
    /**
     * Creates a copy button element for a specific message
     * Copies the iframe HTML content to clipboard
     * @param {string} messageId - The unique message identifier
     * @param {string} originalContent - The original HTML content
     * @returns {HTMLElement} The copy button element
     */
    static createCopyButton(messageId, originalContent) {
        const button = document.createElement('button');
        button.className = 'msg-copy-button';
        button.textContent = 'COPY';
        button.title = 'Copy HTML content to clipboard';
        button.setAttribute('data-message-id', messageId);
        
        button.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            try {
                // Try to copy the original HTML content
                let contentToCopy = originalContent;
                
                // If we can access the iframe, get its content
                const messageBlock = document.getElementById(messageId);
                if (messageBlock) {
                    const iframe = messageBlock.querySelector('iframe');
                    if (iframe && iframe.contentDocument) {
                        try {
                            // Get the full HTML from the iframe
                            contentToCopy = iframe.contentDocument.documentElement.outerHTML;
                        } catch (accessError) {
                            // If we can't access iframe content due to security, use original
                            console.log('Using original content due to iframe access restrictions');
                        }
                    }
                }
                
                // Copy to clipboard
                await navigator.clipboard.writeText(contentToCopy);
                
                // Show success feedback
                const originalText = button.textContent;
                const originalClass = button.className;
                
                button.textContent = '✓';
                button.className = 'msg-copy-button success';
                
                // Reset after 1 second
                setTimeout(() => {
                    button.textContent = originalText;
                    button.className = originalClass;
                }, 1000);
                
            } catch (err) {
                console.error('Failed to copy to clipboard:', err);
                
                // Show error feedback
                const originalText = button.textContent;
                button.textContent = '✗';
                
                setTimeout(() => {
                    button.textContent = originalText;
                }, 1000);
            }
        });
        
        return button;
    }
    
    /**
     * Creates a save button element for a specific message
     * Shows a popup to specify directory for saving HTML/CSS/JS files
     * @param {string} messageId - The unique message identifier
     * @param {string} originalContent - The original HTML content
     * @returns {HTMLElement} The save button element
     */
    static createSaveButton(messageId, originalContent) {
        const button = document.createElement('button');
        button.className = 'msg-save-button';
        button.textContent = 'SAVE';
        button.title = 'Save HTML/CSS/JS files to directory';
        button.setAttribute('data-message-id', messageId);
        
        button.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            this.showSavePopup(messageId, originalContent, button);
        });
        
        return button;
    }

    /**
     * Creates an expand button element for a specific message
     * Toggles between normal and fullscreen view of the message
     * @param {string} messageId - The unique message identifier
     * @returns {HTMLElement} The expand button element
     */
    static createExpandButton(messageId) {
        const button = document.createElement('button');
        button.className = 'msg-expand-button';
        button.textContent = '⛶';
        button.title = 'Expand to full screen';
        button.setAttribute('data-message-id', messageId);

        button.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();

            this.toggleMessageExpansion(messageId, button);
        });

        return button;
    }

    /**
     * Toggles the expansion state of a message
     * @param {string} messageId - The unique message identifier
     * @param {HTMLElement} expandButton - The expand button element
     */
    static toggleMessageExpansion(messageId, expandButton) {
        const messageBlock = document.getElementById(messageId);
        if (!messageBlock) return;

        const isExpanded = messageBlock.classList.contains('expanded-message');

        if (isExpanded) {
            // Collapse the message
            this.collapseMessage(messageBlock, expandButton);
        } else {
            // Before expanding, ensure no other messages are expanded
            this.collapseAllMessages();

            // Expand the message
            this.expandMessage(messageBlock, expandButton);
        }
    }

    /**
     * Expands a message to full screen
     * @param {HTMLElement} messageBlock - The message block element
     * @param {HTMLElement} expandButton - The expand button element
     */
    static expandMessage(messageBlock, expandButton) {
        // Add expanded class
        messageBlock.classList.add('expanded-message');

        // Update button
        expandButton.textContent = '🗕';
        expandButton.title = 'Exit full screen';
        expandButton.classList.add('expanded');

        // Create overlay background
        const overlay = document.createElement('div');
        overlay.className = 'message-overlay';
        overlay.setAttribute('data-message-id', messageBlock.id);

        // Add overlay to body
        document.body.appendChild(overlay);
        document.body.classList.add('message-expanded');

        // Handle escape key and overlay click
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                this.collapseMessage(messageBlock, expandButton);
            }
        };

        const handleOverlayClick = (e) => {
            if (e.target === overlay) {
                this.collapseMessage(messageBlock, expandButton);
            }
        };

        overlay.addEventListener('click', handleOverlayClick);
        document.addEventListener('keydown', handleEscape);

        // Store event handlers for cleanup
        overlay._escapeHandler = handleEscape;
        overlay._overlayClickHandler = handleOverlayClick;

        // Adjust iframe height if present
        const iframe = messageBlock.querySelector('iframe');
        if (iframe) {
            iframe.style.height = 'calc(100vh - 120px)';
        }

        console.log(`Expanded message ${messageBlock.id} to full screen`);
    }

    /**
     * Collapses a message from full screen
     * @param {HTMLElement} messageBlock - The message block element
     * @param {HTMLElement} expandButton - The expand button element
     */
    static collapseMessage(messageBlock, expandButton) {
        // Remove expanded class
        messageBlock.classList.remove('expanded-message');

        // Update button
        expandButton.textContent = '⛶';
        expandButton.title = 'Expand to full screen';
        expandButton.classList.remove('expanded');

        // Remove overlay
        const overlay = document.querySelector(`.message-overlay[data-message-id="${messageBlock.id}"]`);
        if (overlay) {
            // Remove event listeners
            if (overlay._escapeHandler) {
                document.removeEventListener('keydown', overlay._escapeHandler);
            }
            if (overlay._overlayClickHandler) {
                overlay.removeEventListener('click', overlay._overlayClickHandler);
            }

            overlay.remove();
        }

        document.body.classList.remove('message-expanded');

        // Reset iframe height if present
        const iframe = messageBlock.querySelector('iframe');
        if (iframe) {
            iframe.style.height = '';
        }

        console.log(`Collapsed message ${messageBlock.id} from full screen`);
    }

    /**
     * Collapses all currently expanded messages
     * This ensures only one message can be expanded at a time
     */
    static collapseAllMessages() {
        const expandedMessages = document.querySelectorAll('.expanded-message');
        expandedMessages.forEach(messageBlock => {
            const expandButton = messageBlock.querySelector('.msg-expand-button');
            if (expandButton) {
                this.collapseMessage(messageBlock, expandButton);
            }
        });

        // Also clean up any orphaned overlays and body classes
        const overlays = document.querySelectorAll('.message-overlay');
        overlays.forEach(overlay => overlay.remove());
        document.body.classList.remove('message-expanded');
    }

    /**
     * Shows the save popup modal
     * @param {string} messageId - The unique message identifier
     * @param {string} originalContent - The original HTML content
     * @param {HTMLElement} saveButton - The save button for feedback
     */
    static showSavePopup(messageId, originalContent, saveButton) {
        // Remove existing popup if any
        const existing = document.getElementById('save-popup-overlay');
        if (existing) {
            existing.remove();
        }
        
        // Create popup overlay
        const overlay = document.createElement('div');
        overlay.id = 'save-popup-overlay';
        overlay.className = 'save-popup-overlay';
        
        // Create popup content
        overlay.innerHTML = `
            <div class="save-popup">
                <h3>Save Files</h3>
                <div class="save-popup-form">
                    <label for="save-directory">Directory path:</label>
                    <input 
                        type="text" 
                        id="save-directory" 
                        class="save-popup-input"
                        placeholder="/path/to/directory (e.g., ./downloads, /home/user/projects)"
                        value="./downloads"
                    />
                    <label for="save-filename">Base filename:</label>
                    <input 
                        type="text" 
                        id="save-filename" 
                        class="save-popup-input"
                        placeholder="filename (without extension)"
                        value="content"
                    />
                    <div class="save-popup-buttons">
                        <button class="save-popup-button secondary" id="save-cancel">Cancel</button>
                        <button class="save-popup-button primary" id="save-confirm">Save Files</button>
                    </div>
                </div>
            </div>
        `;
        
        // Add to document
        document.body.appendChild(overlay);
        
        // Show with animation
        requestAnimationFrame(() => {
            overlay.classList.add('show');
        });
        
        // Focus the directory input
        const directoryInput = overlay.querySelector('#save-directory');
        directoryInput.focus();
        directoryInput.select();
        
        // Handle cancel
        overlay.querySelector('#save-cancel').addEventListener('click', () => {
            this.hideSavePopup(overlay);
        });
        
        // Handle overlay click (close)
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                this.hideSavePopup(overlay);
            }
        });
        
        // Handle escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                this.hideSavePopup(overlay);
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);
        
        // Handle confirm/save
        overlay.querySelector('#save-confirm').addEventListener('click', () => {
            const directory = overlay.querySelector('#save-directory').value.trim();
            const filename = overlay.querySelector('#save-filename').value.trim();
            
            if (!directory || !filename) {
                alert('Please specify both directory and filename');
                return;
            }
            
            this.saveFiles(messageId, originalContent, directory, filename, saveButton);
            this.hideSavePopup(overlay);
        });
        
        // Handle Enter key in inputs
        overlay.querySelectorAll('input').forEach(input => {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    overlay.querySelector('#save-confirm').click();
                }
            });
        });
    }
    
    /**
     * Hides the save popup modal
     * @param {HTMLElement} overlay - The popup overlay element
     */
    static hideSavePopup(overlay) {
        overlay.classList.remove('show');
        setTimeout(() => {
            if (overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
        }, 300);
    }
    
    /**
     * Saves the HTML content as separate files
     * @param {string} messageId - The unique message identifier
     * @param {string} originalContent - The original HTML content
     * @param {string} directory - The target directory
     * @param {string} filename - The base filename
     * @param {HTMLElement} saveButton - The save button for feedback
     */
    static async saveFiles(messageId, originalContent, directory, filename, saveButton) {
        try {
            // Get content to save (try iframe first, fallback to original)
            let contentToSave = originalContent;
            const messageBlock = document.getElementById(messageId);
            if (messageBlock) {
                const iframe = messageBlock.querySelector('iframe');
                if (iframe && iframe.contentDocument) {
                    try {
                        contentToSave = iframe.contentDocument.documentElement.outerHTML;
                    } catch (accessError) {
                        console.log('Using original content due to iframe access restrictions');
                    }
                }
            }
            
            // Parse HTML to extract CSS and JS
            const parser = new DOMParser();
            const doc = parser.parseFromString(contentToSave, 'text/html');
            
            // Extract CSS
            const styleElements = doc.querySelectorAll('style');
            let cssContent = '';
            styleElements.forEach(style => {
                cssContent += style.textContent + '\n\n';
            });
            
            // Extract JS
            const scriptElements = doc.querySelectorAll('script');
            let jsContent = '';
            scriptElements.forEach(script => {
                if (script.textContent.trim()) {
                    jsContent += script.textContent + '\n\n';
                }
            });
            
            // Create clean HTML (remove style and script tags, link to external files)
            const cleanDoc = doc.cloneNode(true);
            cleanDoc.querySelectorAll('style').forEach(el => el.remove());
            cleanDoc.querySelectorAll('script').forEach(el => el.remove());
            
            // Add links to external CSS and JS files
            if (cssContent.trim()) {
                const cssLink = cleanDoc.createElement('link');
                cssLink.rel = 'stylesheet';
                cssLink.href = `${filename}.css`;
                cleanDoc.head.appendChild(cssLink);
            }
            
            if (jsContent.trim()) {
                const jsScript = cleanDoc.createElement('script');
                jsScript.src = `${filename}.js`;
                cleanDoc.body.appendChild(jsScript);
            }
            
            const cleanHtml = cleanDoc.documentElement.outerHTML;
            
            // Prepare file data
            const files = [
                { name: `${filename}.html`, content: cleanHtml, type: 'text/html' },
                { name: `${filename}.css`, content: cssContent || '/* No CSS content found */', type: 'text/css' },
                { name: `${filename}.js`, content: jsContent || '// No JavaScript content found', type: 'text/javascript' }
            ];
            
            // Since we can't actually save files from browser, we'll download them
            this.downloadFiles(files, directory, saveButton);
            
        } catch (err) {
            console.error('Failed to save files:', err);
            
            // Show error feedback
            const originalText = saveButton.textContent;
            saveButton.textContent = '✗';
            
            setTimeout(() => {
                saveButton.textContent = originalText;
            }, 2000);
            
            alert('Failed to save files: ' + err.message);
        }
    }
    
    /**
     * Downloads files as separate downloads
     * @param {Array} files - Array of file objects {name, content, type}
     * @param {string} directory - The target directory (for display purposes)
     * @param {HTMLElement} saveButton - The save button for feedback
     */
    static downloadFiles(files, directory, saveButton) {
        try {
            files.forEach(file => {
                const blob = new Blob([file.content], { type: file.type });
                const url = URL.createObjectURL(blob);

                const a = document.createElement('a');
                a.href = url;
                a.download = file.name;
                a.style.display = 'none';

                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);

                URL.revokeObjectURL(url);
            });

            // Show success feedback
            const originalText = saveButton.textContent;
            const originalClass = saveButton.className;

            saveButton.textContent = '✓';
            saveButton.className = 'msg-save-button success';

            setTimeout(() => {
                saveButton.textContent = originalText;
                saveButton.className = originalClass;
            }, 2000);

        } catch (err) {
            console.error('Failed to download files:', err);

            // Show error feedback
            const originalText = saveButton.textContent;
            saveButton.textContent = originalText;
        }
    }

    /**
     * Gets the global JavaScript execution state
     * @returns {boolean} True if JavaScript is enabled globally
     */
    static getGlobalJSState() {
        try {
            const state = localStorage.getItem('global-js-enabled');
            return state !== 'false'; // Default to true unless explicitly disabled
        } catch (e) {
            console.warn('Failed to read global JS state:', e);
            return true;
        }
    }

    /**
     * Sets the global JavaScript execution state
     * @param {boolean} enabled - Whether JavaScript should be enabled globally
     */
    static setGlobalJSState(enabled) {
        try {
            localStorage.setItem('global-js-enabled', enabled.toString());
        } catch (e) {
            console.warn('Failed to save global JS state:', e);
        }
    }

    /**
     * Toggles the global JavaScript execution state
     * @returns {boolean} The new state after toggling
     */
    static toggleGlobalJSState() {
        const currentState = this.getGlobalJSState();
        const newState = !currentState;
        this.setGlobalJSState(newState);
        return newState;
    }

    /**
     * Applies a JavaScript state to a specific message with visual feedback
     * Used by the global toggle system with staggered timing
     * @param {string} messageId - The unique message identifier
     * @param {boolean} enabled - Whether JavaScript should be enabled
     * @param {Function} onUpdate - Optional callback when update is complete
     */
    static applyJSStateToMessage(messageId, enabled, onUpdate = null) {
        if (!messageId) return;

        // Update the stored state
        this.setMessageJSState(messageId, enabled);

        // Find the message container and update UI elements
        const container = document.getElementById(messageId);
        if (container) {
            // Update toggle button appearance if present
            const toggleButton = container.querySelector('.msg-js-toggle');
            if (toggleButton) {
                toggleButton.className = `msg-js-toggle ${enabled ? 'enabled' : 'disabled'}`;
                toggleButton.textContent = enabled ? 'JS ON' : 'JS OFF';
                toggleButton.title = `Click to ${enabled ? 'disable' : 'enable'} JavaScript for this message`;
            }

            // Call the callback if provided
            if (onUpdate && typeof onUpdate === 'function') {
                onUpdate(messageId, enabled, container);
            }
        }
    }

    /**
     * Gets the effective JavaScript state for a message considering global state
     * @param {string} messageId - The unique message identifier
     * @returns {boolean} True if JavaScript should be enabled for this message
     */
    static getEffectiveMessageJSState(messageId) {
        const globalEnabled = this.getGlobalJSState();
        if (!globalEnabled) {
            return false; // Global override
        }
        return this.getMessageJSState(messageId);
    }

    /**
     * Counts the number of messages with JavaScript content on the current page
     * @returns {number} Number of messages with JS content
     */
    static countJSMessages() {
        const messageContainers = document.querySelectorAll('[id*="msg-"]');
        let count = 0;

        messageContainers.forEach(container => {
            const messageId = container.id;
            if (messageId && messageId.startsWith('msg-')) {
                const iframe = container.querySelector('iframe');
                const jsToggle = container.querySelector('.msg-js-toggle');
                if (iframe || jsToggle) {
                    count++;
                }
            }
        });

        return count;
    }
}

// Initialize cleanup on page load
document.addEventListener('DOMContentLoaded', () => {
    JavaScriptUtils.cleanupOldStates();

    // Listen for storage changes to sync across tabs
    window.addEventListener('storage', (e) => {
        if (e.key === 'global-js-enabled') {
            const globalJsToggle = document.getElementById('global-js-toggle');
            if (globalJsToggle && window.VisualResponseApp) {
                const newState = e.newValue !== 'false';
                window.VisualResponseApp.updateGlobalJSToggleButton(globalJsToggle, newState);
                console.log('Global JS state synchronized across tabs:', newState);
            }
        }
    });
});

// Export for use by other modules
window.JavaScriptUtils = JavaScriptUtils;