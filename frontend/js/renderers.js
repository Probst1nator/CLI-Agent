// tools/visual_assets/js/renderers.js

/**
 * Renders HTML content within a sandboxed iframe to isolate it from the main document.
 * @param {HTMLElement} targetElement - The container element to render into.
 * @param {string} htmlContent - The complete HTML string to render.
 */
function renderHTML(targetElement, htmlContent, messageId = null) {
    // Decode HTML entities if the content appears to be HTML-encoded
    if (htmlContent.includes('&lt;') || htmlContent.includes('&gt;') || htmlContent.includes('&amp;')) {
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = htmlContent;
        htmlContent = tempDiv.textContent || tempDiv.innerText || '';
        window.visualLogger.info('Decoded HTML entities from content', {
            originalLength: htmlContent.length,
            decodedLength: htmlContent.length
        });
    }

    // Check if JavaScript execution is enabled for this specific message (considering global state)
    let jsEnabled = true; // Default to enabled
    if (messageId && window.JavaScriptUtils) {
        jsEnabled = window.JavaScriptUtils.getEffectiveMessageJSState(messageId);
    }

    // Check if the HTML content contains JavaScript
    const hasJS = window.JavaScriptUtils ? window.JavaScriptUtils.detectJavaScript(htmlContent) : false;
    
    // If JS is disabled and content contains JS, show content in height-limited container
    if (!jsEnabled && hasJS) {
        // Create height-limited container
        const container = document.createElement('div');
        container.className = 'js-disabled-content';

        // Add disabled indicator at the top
        const indicator = document.createElement('div');
        indicator.className = 'js-disabled-indicator';

        // Create text content
        const indicatorText = document.createElement('span');
        indicatorText.textContent = 'JavaScript Disabled - Content Limited (Scroll to see more)';

        // Create expand button
        const expandBtn = document.createElement('button');
        expandBtn.className = 'js-disabled-expand-btn';
        expandBtn.innerHTML = '🔍 Expand';
        expandBtn.title = 'Expand to show full content height';

        // Add elements to indicator
        indicator.appendChild(indicatorText);
        indicator.appendChild(expandBtn);

        // Create content wrapper
        const contentWrapper = document.createElement('div');
        contentWrapper.style.padding = '8px';

        // Create iframe with JS disabled (sandbox without allow-scripts)
        const iframe = document.createElement('iframe');
        iframe.style.width = '100%';
        iframe.style.border = 'none';
        iframe.style.minHeight = '200px';

        // Remove/neutralize JavaScript from content for safer display
        let safeContent = htmlContent;
        // Remove script tags
        safeContent = safeContent.replace(/<script[\s\S]*?<\/script>/gi, '<!-- Script removed for safety -->');
        // Remove event handlers
        safeContent = safeContent.replace(/\son\w+\s*=\s*["'][^"']*["']/gi, '');
        // Remove javascript: URLs
        safeContent = safeContent.replace(/javascript:[^"'\s>]*/gi, '#');

        iframe.srcdoc = `
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <base href="${window.location.href}">
                <style>
                    body {
                        margin: 0;
                        font-family: inherit;
                        background: transparent;
                        color: #e0e0e0;
                    }
                    * {
                        max-width: 100%;
                        color: inherit;
                    }
                </style>
                <script>
                    // Override console.error to suppress source map warnings
                    const originalError = console.error;
                    console.error = function(...args) {
                        const message = args[0];
                        if (typeof message === 'string' && (
                            message.includes('Source map error') ||
                            message.includes('DevTools failed to load') ||
                            message.includes('sourceMappingURL') ||
                            message.includes('Failed to load source map')
                        )) {
                            return; // Suppress source map errors
                        }
                        return originalError.apply(console, args);
                    };
                </script>
            </head>
            <body>${safeContent}</body>
            </html>
        `;

        // Assemble the components
        contentWrapper.appendChild(iframe);
        container.appendChild(indicator);
        container.appendChild(contentWrapper);

        targetElement.innerHTML = '';
        targetElement.appendChild(container);

        // Store original height for expand/collapse functionality
        let isExpanded = false;
        let originalHeight = 0;

        // Adjust iframe height after loading
        iframe.onload = function() {
            try {
                originalHeight = iframe.contentDocument.body.scrollHeight;
                iframe.style.height = originalHeight + 'px';
            } catch (e) {
                // Fallback height if cross-origin issues
                originalHeight = 350;
                iframe.style.height = originalHeight + 'px';
            }
        };

        // Add expand/collapse functionality
        expandBtn.addEventListener('click', function() {
            isExpanded = !isExpanded;

            if (isExpanded) {
                // Expand to full height
                container.classList.add('expanded');
                iframe.style.height = originalHeight + 'px';
                expandBtn.innerHTML = '🔼 Collapse';
                expandBtn.title = 'Collapse to limited height';
                expandBtn.classList.add('expanded');
                indicatorText.textContent = 'JavaScript Disabled - Content Fully Expanded';
            } else {
                // Collapse to limited height
                container.classList.remove('expanded');
                iframe.style.height = originalHeight + 'px';
                expandBtn.innerHTML = '🔍 Expand';
                expandBtn.title = 'Expand to show full content height';
                expandBtn.classList.remove('expanded');
                indicatorText.textContent = 'JavaScript Disabled - Content Limited (Scroll to see more)';
            }
        });

        return;
    }

    const iframe = document.createElement('iframe');

    // Use srcdoc instead of writing to contentDocument - this is more secure
    // and doesn't require allow-same-origin
    iframe.sandbox = jsEnabled ? 
        'allow-scripts allow-forms allow-popups allow-popups-to-escape-sandbox allow-downloads' :
        'allow-forms allow-popups allow-popups-to-escape-sandbox allow-downloads';
    iframe.style.width = '100%';
    iframe.style.height = 'auto'; // Initial height
    iframe.style.border = 'none';
    iframe.style.overflow = 'hidden'; // Prevent scrollbars during resize
    iframe.setAttribute('loading', 'lazy');

    // Height adjustment state
    let lastHeight = 400;
    let heightAdjustmentCount = 0;
    let heightDebounceTimeout = null;
    const MAX_HEIGHT_ADJUSTMENTS = 8; // Allow more adjustments for better accuracy
    const HEIGHT_DEBOUNCE_MS = 150; // Reduce debounce time for faster response

    // Declare messageListener variable that will be defined later
    let messageListener;

    // Add height-reporting script to the HTML content
    const heightReportingScript = `
        <script>
            let reportCount = 0;
            const MAX_REPORTS = 10;

            function reportHeight() {
                try {
                    if (reportCount >= MAX_REPORTS) {
                        console.log('Maximum height reports reached, stopping');
                        return;
                    }

                    // Get the most accurate height measurement
                    const heights = [
                        document.documentElement.scrollHeight,
                        document.documentElement.offsetHeight,
                        document.body.scrollHeight,
                        document.body.offsetHeight
                    ];

                    // Use the largest height but add some intelligence
                    let bestHeight = Math.max(...heights);

                    // If document has minimal content, use a more conservative height
                    const contentElements = document.querySelectorAll('p, div, img, h1, h2, h3, h4, h5, h6, ul, ol, table');
                    if (contentElements.length <= 2 && bestHeight > 400) {
                        // For simple content, be more conservative
                        bestHeight = Math.min(bestHeight, 300);
                    }

                    // Remove any excessive height caused by margins/padding
                    const bodyStyle = window.getComputedStyle(document.body);
                    const marginBottom = parseInt(bodyStyle.marginBottom) || 0;
                    const paddingBottom = parseInt(bodyStyle.paddingBottom) || 0;

                    // Reduce height if there's excessive bottom spacing
                    if (marginBottom + paddingBottom > 50) {
                        bestHeight = Math.max(bestHeight - (marginBottom + paddingBottom - 20), 150);
                    }

                    reportCount++;
                    parent.postMessage({type: 'resize', height: bestHeight}, '*');
                    console.log('Height reported:', bestHeight, 'px (report #' + reportCount + ')');
                } catch(e) {
                    console.log('Could not report height:', e);
                }
            }

            // Report height when DOM is ready
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', reportHeight);
            } else {
                reportHeight();
            }

            // Report height when images load
            window.addEventListener('load', function() {
                setTimeout(reportHeight, 50);
                setTimeout(reportHeight, 200); // Second check after images should be loaded
            });

            // Watch for dynamic content changes with throttling
            if (window.ResizeObserver) {
                let resizeTimeout;
                new ResizeObserver(() => {
                    clearTimeout(resizeTimeout);
                    resizeTimeout = setTimeout(reportHeight, 100);
                }).observe(document.body);
            }
        </script>
    `;

    // Add CSP meta tag and base tag to iframe content to allow external scripts and fix source maps
    const baseTag = `<base href="${window.location.href}">`;
    const cspMetaTag = '<meta http-equiv="Content-Security-Policy" content="default-src \'self\'; script-src \'self\' \'unsafe-inline\' \'unsafe-eval\' https://unpkg.com https://cdn.plot.ly https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; style-src \'self\' \'unsafe-inline\' https://cdn.jsdelivr.net; font-src \'self\' data: https://cdn.jsdelivr.net; img-src \'self\' data: blob: https://* http://localhost:* http://127.0.0.1:*; frame-src \'self\' data: blob:; connect-src \'self\' ws: wss:; object-src \'none\'; base-uri \'self\'; form-action \'self\'">';

    // Add CSS to fix text color inheritance and disable source maps
    const colorFixCSS = `
        <style>
            body {
                color: #e0e0e0 !important;
            }
            * {
                color: inherit;
            }
            h1, h2, h3, h4, h5, h6, p, span, div, td, th {
                color: inherit !important;
            }
        </style>
    `;

    // Add script to disable source map loading and handle console errors
    const sourceMapDisableScript = `
        <script>
            // Override console.error to suppress source map warnings
            const originalError = console.error;
            console.error = function(...args) {
                const message = args[0];
                if (typeof message === 'string' && (
                    message.includes('Source map error') ||
                    message.includes('DevTools failed to load') ||
                    message.includes('sourceMappingURL') ||
                    message.includes('Failed to load source map')
                )) {
                    return; // Suppress source map errors
                }
                return originalError.apply(console, args);
            };

            // Disable source map loading by overriding fetch for .map files
            const originalFetch = window.fetch;
            if (originalFetch) {
                window.fetch = function(url, options) {
                    if (typeof url === 'string' && url.endsWith('.map')) {
                        return Promise.reject(new Error('Source maps disabled in iframe'));
                    }
                    return originalFetch.apply(this, arguments);
                };
            }

            // Override XMLHttpRequest for .map files
            const originalOpen = XMLHttpRequest.prototype.open;
            XMLHttpRequest.prototype.open = function(method, url, ...args) {
                if (typeof url === 'string' && url.endsWith('.map')) {
                    // Prevent source map requests
                    return;
                }
                return originalOpen.apply(this, [method, url, ...args]);
            };
        </script>
    `;

    // Inject CSP, base tag, styling, and source map disable script into the HTML content
    let modifiedHtmlContent = htmlContent;

    // First, inject meta tags, styling, and source map disable script in head
    const headContent = baseTag + cspMetaTag + colorFixCSS + sourceMapDisableScript;
    if (htmlContent.toLowerCase().includes('</head>')) {
        modifiedHtmlContent = modifiedHtmlContent.replace('</head>', headContent + '</head>');
    } else if (htmlContent.toLowerCase().includes('<head>')) {
        modifiedHtmlContent = modifiedHtmlContent.replace('<head>', '<head>' + headContent);
    } else {
        // If no head tag, add it
        modifiedHtmlContent = modifiedHtmlContent.replace('<html>', '<html><head>' + headContent + '</head>');
    }

    // Then inject height reporting script
    if (modifiedHtmlContent.toLowerCase().includes('</body>')) {
        modifiedHtmlContent = modifiedHtmlContent.replace('</body>', heightReportingScript + '</body>');
    } else if (modifiedHtmlContent.toLowerCase().includes('</html>')) {
        modifiedHtmlContent = modifiedHtmlContent.replace('</html>', heightReportingScript + '</html>');
    } else {
        modifiedHtmlContent = modifiedHtmlContent + heightReportingScript;
    }

    // Update iframe with modified content
    iframe.srcdoc = modifiedHtmlContent;

    // Add message listener
    window.addEventListener('message', messageListener);

    // Function to resize iframe with conservative content estimation fallback
    function resizeIframe() {
        try {
            // More conservative height estimation based on content
            const contentLines = htmlContent.split('\n').length;
            const hasImages = htmlContent.includes('<img');
            const hasLargeContent = htmlContent.length > 1000;
            const contentLength = htmlContent.replace(/<[^>]*>/g, '').trim().length; // Text content only

            // Base height on actual content, not just lines
            let estimatedHeight = Math.max(150, Math.min(500, Math.max(
                contentLines * 12, // Reduced from 20px per line
                contentLength * 0.3 // Rough character-based estimation
            )));

            if (hasImages) {
                estimatedHeight += 200; // Reduced from 300px
            }

            if (hasLargeContent) {
                estimatedHeight += 100; // Reduced from 200px
            }

            // No max height limit - let content expand fully

            iframe.style.height = estimatedHeight + 'px';
            lastHeight = estimatedHeight; // Update lastHeight for container sizing
            window.visualLogger.debug(`Iframe height estimated at ${estimatedHeight}px (conservative fallback)`);

        } catch (e) {
            window.visualLogger.warn('Could not resize iframe automatically:', e.message);
            iframe.style.height = originalHeight + 'px'; // More conservative default
            lastHeight = originalHeight; // Update lastHeight for container sizing
        }
    }

    // Cleanup function to remove message listener and clear timeouts
    iframe.cleanup = () => {
        window.removeEventListener('message', messageListener);
        if (heightDebounceTimeout) {
            clearTimeout(heightDebounceTimeout);
        }
    };

    // Set up event handlers
    iframe.onload = () => {
        window.visualLogger.info('iFrame content rendered successfully using srcdoc.');

        // Resize after a short delay to allow content to render
        setTimeout(() => {
            resizeIframe();
        }, 100);

        // Also resize after images might have loaded
        setTimeout(() => {
            resizeIframe();
        }, 1000);
    };

    iframe.onerror = (e) => {
        window.visualLogger.error("An error occurred loading the iFrame content.", { error: e });
        targetElement.textContent = "An error occurred while loading the iFrame.";
    };

    // Clear the target element and create resizable container
    targetElement.innerHTML = '';

    // Create container for resizable HTML messages
    const resizableContainer = document.createElement('div');
    resizableContainer.className = 'html-message-container';

    // Create resize handle
    const resizeHandle = document.createElement('div');
    resizeHandle.className = 'html-resize-handle';
    resizeHandle.title = 'Drag to resize message height';

    // Add drag functionality to resize handle
    let isResizing = false;
    let startY = 0;
    let startHeight = 0;
    let resizeIndicator = null;

    resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startY = e.clientY;
        startHeight = resizableContainer.offsetHeight;

        // Create and show resize indicator
        resizeIndicator = document.createElement('div');
        resizeIndicator.className = 'resize-indicator';
        resizeIndicator.textContent = startHeight + 'px';
        document.body.appendChild(resizeIndicator);
        setTimeout(() => resizeIndicator.classList.add('show'), 10);

        // Prevent text selection and add global mouse events
        document.body.style.userSelect = 'none';
        document.body.style.cursor = 'ns-resize';

        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        const deltaY = e.clientY - startY;
        const newHeight = Math.max(150, Math.min(window.innerHeight * 0.8, startHeight + deltaY));

        // Update container height
        resizableContainer.style.height = newHeight + 'px';

        // Update iframe height to fill container minus handle space (12px)
        const iframeHeight = newHeight - 12;
        iframe.style.height = iframeHeight + 'px';

        // Update lastHeight for consistency with automatic sizing
        lastHeight = iframeHeight;

        if (resizeIndicator) {
            resizeIndicator.textContent = Math.round(newHeight) + 'px';
        }

        e.preventDefault();
    });

    document.addEventListener('mouseup', () => {
        if (!isResizing) return;

        isResizing = false;
        document.body.style.userSelect = '';
        document.body.style.cursor = '';

        // Hide and remove resize indicator
        if (resizeIndicator) {
            resizeIndicator.classList.remove('show');
            setTimeout(() => {
                if (resizeIndicator && resizeIndicator.parentNode) {
                    resizeIndicator.parentNode.removeChild(resizeIndicator);
                }
                resizeIndicator = null;
            }, 200);
        }
    });

    // Set initial height based on content (container = iframe + handle)
    resizableContainer.style.height = (lastHeight + 12) + 'px';

    // Append iframe and resize handle to container
    resizableContainer.appendChild(iframe);
    resizableContainer.appendChild(resizeHandle);

    // Append container to target element
    targetElement.appendChild(resizableContainer);

    // Now create the message listener with access to resizableContainer
    messageListener = (event) => {
        if (event.source === iframe.contentWindow && event.data.type === 'resize') {
            const rawHeight = event.data.height;
            const newHeight = Math.max(150, rawHeight + 10); // No max height limit - expand to full content

            // Only adjust if height difference is meaningful
            const heightDifference = Math.abs(newHeight - lastHeight);
            if (heightDifference < 8 && heightAdjustmentCount > 1) {
                window.visualLogger.debug(`Skipping height adjustment: difference too small (${heightDifference}px)`);
                return;
            }

            // Allow more adjustments but with exponential backoff for later ones
            if (heightAdjustmentCount >= MAX_HEIGHT_ADJUSTMENTS) {
                window.visualLogger.debug(`Maximum height adjustments reached (${MAX_HEIGHT_ADJUSTMENTS}), ignoring further changes`);
                return;
            }

            // Clear existing timeout and set new one with dynamic delay
            if (heightDebounceTimeout) {
                clearTimeout(heightDebounceTimeout);
            }

            // Use shorter delay for early adjustments, longer for later ones
            const dynamicDelay = Math.min(HEIGHT_DEBOUNCE_MS * (1 + heightAdjustmentCount * 0.3), 500);

            heightDebounceTimeout = setTimeout(() => {
                // Update iframe height
                iframe.style.height = newHeight + 'px';
                lastHeight = newHeight;
                heightAdjustmentCount++;

                // Also update container height to accommodate iframe + handle (12px)
                resizableContainer.style.height = (newHeight + 12) + 'px';

                window.visualLogger.debug(`Iframe height adjusted to ${newHeight}px (raw: ${rawHeight}px, adjustment #${heightAdjustmentCount})`);
            }, dynamicDelay);
        }
    };

    // Add message listener to window
    window.addEventListener('message', messageListener);

    // Function to resize iframe with conservative content estimation fallback
    function resizeIframe() {
        try {
            // More conservative height estimation based on content
            const contentLines = htmlContent.split('\n').length;
            const hasImages = htmlContent.includes('<img');
            const hasLargeContent = htmlContent.length > 1000;
            const contentLength = htmlContent.replace(/<[^>]*>/g, '').trim().length; // Text content only

            // Base height on actual content, not just lines
            let estimatedHeight = Math.max(150, Math.min(500, Math.max(
                contentLines * 12, // Reduced from 20px per line
                contentLength * 0.3 // Rough character-based estimation
            )));

            if (hasImages) {
                estimatedHeight += 200; // Reduced from 300px
            }

            if (hasLargeContent) {
                estimatedHeight += 100; // Reduced from 200px
            }

            // No max height limit - let content expand fully

            iframe.style.height = estimatedHeight + 'px';
            lastHeight = estimatedHeight; // Update lastHeight for container sizing

            // Update container height to match
            resizableContainer.style.height = (estimatedHeight + 12) + 'px';

            window.visualLogger.debug(`Iframe height estimated at ${estimatedHeight}px (conservative fallback)`);

        } catch (e) {
            window.visualLogger.warn('Could not resize iframe automatically:', e.message);
            const fallbackHeight = 300; // More conservative default
            iframe.style.height = fallbackHeight + 'px';
            lastHeight = fallbackHeight; // Update lastHeight for container sizing
            resizableContainer.style.height = (fallbackHeight + 12) + 'px';
        }
    }

    // Cleanup function to remove message listener and clear timeouts
    iframe.cleanup = () => {
        window.removeEventListener('message', messageListener);
        if (heightDebounceTimeout) {
            clearTimeout(heightDebounceTimeout);
        }
    };

    // Set up event handlers
    iframe.onload = () => {
        window.visualLogger.info('iFrame content rendered successfully using srcdoc.');

        // Resize after a short delay to allow content to render
        setTimeout(() => {
            resizeIframe();
        }, 100);

        // Also resize after images might have loaded
        setTimeout(() => {
            resizeIframe();
        }, 1000);
    };

    iframe.onerror = (e) => {
        window.visualLogger.error("An error occurred loading the iFrame content.", { error: e });
        targetElement.textContent = "An error occurred while loading the iFrame.";
    };
}

/**
 * Renders Markdown content, including support for Mermaid diagrams and LaTeX.
 * @param {HTMLElement} targetElement - The container element to render into.
 * @param {string} markdownContent - The Markdown string.
 */
function renderMarkdown(targetElement, markdownContent) {
    if (typeof marked === 'undefined') {
        window.visualLogger.warn("Marked.js not loaded. Cannot render Markdown.");
        targetElement.textContent = markdownContent;
        return;
    }
    targetElement.innerHTML = marked.parse(markdownContent);
    renderMermaidDiagrams(targetElement);
    renderLatex(targetElement);
}

/**
 * Finds and renders any Mermaid diagram placeholders in the target element.
 * @param {HTMLElement} targetElement - The element to scan for mermaid blocks.
 */
async function renderMermaidDiagrams(targetElement) {
    if (typeof mermaid === 'undefined') {
        window.visualLogger.warn("Mermaid.js not loaded. Skipping diagram rendering.");
        return;
    }
    try {
        // Use a more specific selector if your markdown renderer creates them
        const placeholders = targetElement.querySelectorAll('.mermaid');
        if (placeholders.length > 0) {
            // mermaid.run() is the modern, preferred API
            await mermaid.run({ nodes: placeholders });
            window.visualLogger.debug(`Rendered ${placeholders.length} mermaid diagrams.`);
        }
    } catch (error) {
        window.visualLogger.error("Failed to render Mermaid diagram.", { error: error.message });
    }
}

/**
 * Renders LaTeX math expressions using KaTeX.
 * @param {HTMLElement} targetElement - The element to scan for LaTeX expressions.
 */
function renderLatex(targetElement) {
    if (typeof renderMathInElement === 'undefined') {
        window.visualLogger.warn("KaTeX auto-render not loaded. Skipping LaTeX rendering.");
        return;
    }
    try {
        renderMathInElement(targetElement, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\[', right: '\\]', display: true},
                {left: '\\(', right: '\\)', display: false}
            ],
            throwOnError: false,
            strict: false
        });
        window.visualLogger.debug('LaTeX expressions rendered successfully.');
    } catch (error) {
        window.visualLogger.error("Failed to render LaTeX expressions.", { error: error.message });
    }
}


// --- Main Content Rendering Logic ---
const renderers = {
    'html': renderHTML,
    'markdown': renderMarkdown,
    'image_upload_ack': (container, content) => {
        // Silent acknowledgment - no visual output needed
        console.log('Image upload acknowledged:', content);
    }
};

/**
 * This class is a simplified placeholder to be instantiated by main.js
 * It exposes the main rendering function.
 */
class ContentRenderer {
    async renderContent(container, data, messageId = null, timestampDiv = null) {
        const { type, content } = data;
        const renderer = renderers[type];

        if (renderer) {
            window.visualLogger.debug(`Rendering content with '${type}' renderer.`, data);
            try {
                // Add controls to HTML message headers
                if (type === 'html' && messageId && timestampDiv && window.JavaScriptUtils) {
                    const hasJS = window.JavaScriptUtils.detectJavaScript(content);

                    // Always show JS controls for HTML messages (JS detection is just for informational purposes)
                    timestampDiv.classList.add('has-js-toggle');

                    const resetButton = window.JavaScriptUtils.createResetButton(
                        messageId,
                        (msgId, enabled) => this.reRenderMessage(container, data, msgId, enabled)
                    );

                    const toggleButton = window.JavaScriptUtils.createToggleButton(
                        messageId,
                        (msgId, enabled) => this.reRenderMessage(container, data, msgId, enabled)
                    );

                    const copyButton = window.JavaScriptUtils.createCopyButton(
                        messageId,
                        content
                    );

                    const saveButton = window.JavaScriptUtils.createSaveButton(
                        messageId,
                        content
                    );

                    timestampDiv.appendChild(resetButton);
                    timestampDiv.appendChild(toggleButton);
                    timestampDiv.appendChild(copyButton);
                    timestampDiv.appendChild(saveButton);

                    const jsStatus = hasJS ? 'detected' : 'not detected';
                    window.visualLogger.debug(`Added JS controls to header for message ${messageId} (JS ${jsStatus})`);

                    window.visualLogger.debug(`Added expand button and controls to header for message ${messageId}`);
                }
                
                // Pass messageId to renderer for JS state checking
                if (renderer === renderHTML && messageId) {
                    renderer(container, content, messageId);
                } else {
                    renderer(container, content);
                }
            } catch (e) {
                window.visualLogger.error(`${type} rendering failed:`, { error: e });
                container.innerHTML = `<pre class="render-error">Error rendering ${type}: ${e.message}</pre>`;
            }
        } else {
            window.visualLogger.warn(`No renderer found for type: ${type}`);
            container.innerHTML = `<pre>Unsupported content type: ${type}</pre>`;
        }
    }
    
    // Re-render a specific message when JS toggle is clicked
    reRenderMessage(container, data, messageId, jsEnabled) {
        window.visualLogger.debug(`Re-rendering message ${messageId} with JS ${jsEnabled ? 'enabled' : 'disabled'}`);

        // Check if message was expanded before re-rendering
        const messageBlock = document.getElementById(messageId);

        // Clear container
        container.innerHTML = '';

        // Re-render with current JS state
        if (data.type === 'html') {
            // Use the updated renderHTML function which handles JS disabled state
            renderHTML(container, data.content, messageId);
        } else {
            // For non-HTML content, just re-render normally
            const renderer = renderers[data.type];
            if (renderer) {
                renderer(container, data.content);
            }
        }
    }


    /**
     * Render a message to the chat interface
     * This is a convenience wrapper for ChatStateManager compatibility
     * @param {Object} message - Message object with type, content, etc.
     * @param {number} messageIndex - Index of message in chat history (for edit functionality)
     */
    async renderMessage(message, messageIndex) {
        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper) {
            console.error('Content wrapper not found');
            return;
        }

        // Determine if this is a user message
        const isUserMessage = message.role === 'user';

        // Create message block structure similar to WebSocketManager
        const messageBlock = document.createElement('div');
        messageBlock.className = isUserMessage ? 'message-block user-message' : 'message-block agent-message';

        // Assign message index for edit functionality
        if (messageIndex !== undefined) {
            messageBlock.dataset.messageIndex = messageIndex;
        }

        const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        messageBlock.id = messageId;

        // Create timestamp header with flex layout (using same style for both user and assistant messages)
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'agent-msg-head';

        // Left: timestamp
        const timestampText = document.createElement('span');
        timestampText.className = 'timestamp-text';
        timestampText.textContent = new Date().toLocaleTimeString();
        timestampDiv.appendChild(timestampText);

        messageBlock.appendChild(timestampDiv);

        // Add loading-tools-header for loaded messages (same structure as real-time)
        // Always parse XML to find tool calls, then enhance with metadata
        let toolInfo = [];
        let visualResponseTool = null;
        if (!isUserMessage && message.content) {
            const xmlToolCalls = this.parseToolCallsFromXML(message.content);
            const metadataToolInfo = message.metadata && message.metadata.tool_info ? message.metadata.tool_info : null;
            toolInfo = this.mergeToolInfoWithMetadata(xmlToolCalls, metadataToolInfo);

            // Extract visualresponse/response tool for model display before filtering
            visualResponseTool = toolInfo.find(tool => ['visualresponse', 'response'].includes(tool.name));

            // Filter out user-facing tags (visualresponse, response) - they're rendered as content, not tool boxes
            toolInfo = toolInfo.filter(tool => !['visualresponse', 'response'].includes(tool.name));
        }

        // Center: model info (from visualresponse tool_info or response_model metadata)
        const modelSpan = document.createElement('span');
        modelSpan.className = 'message-model-info';
        modelSpan.style.color = '#888';
        modelSpan.style.fontSize = '0.9em';
        modelSpan.style.flex = '1';
        modelSpan.style.textAlign = 'center';

        // Try visualResponseTool.model first, then fallback to metadata.response_model
        const modelToDisplay = (visualResponseTool && visualResponseTool.model)
            || (message.metadata && message.metadata.response_model);

        if (modelToDisplay) {
            modelSpan.textContent = `[${modelToDisplay}]`;
        }
        timestampDiv.appendChild(modelSpan);

        // Right: button container
        const buttonContainer = document.createElement('span');
        buttonContainer.className = 'message-buttons';
        buttonContainer.style.display = 'flex';
        buttonContainer.style.gap = '4px';
        buttonContainer.style.alignItems = 'center';
        timestampDiv.appendChild(buttonContainer);

        if (!isUserMessage && toolInfo && Array.isArray(toolInfo) && toolInfo.length > 0) {
            const toolsHeader = document.createElement('div');
            toolsHeader.className = 'loading-tools-header';

            toolInfo.forEach((tool, index) => {
                const toolItem = document.createElement('div');

                let itemClass = tool.status === 'success' ? 'loading-tool-item completed' : 'loading-tool-item failed';

                // Apply special styling for think and todos tags
                if (tool.name === 'todos') {
                    itemClass += ' special-todos';
                } else if (tool.name === 'think') {
                    itemClass += ' special-think';
                }

                toolItem.className = itemClass;
                toolItem.id = `loading-tool-${messageId}-${index}`;

                const statusIcon = tool.status === 'success' ? '✅' : '❌';
                const modelDisplay = tool.model ? `<span class="loading-tool-model">[${this.escapeHtml(tool.model)}]</span>` : '';
                const durationDisplay = tool.duration_ms ? `${tool.duration_ms}ms` : '0ms';

                toolItem.innerHTML = `
                    <div class="loading-tool-main">
                        <span class="loading-tool-name">${this.escapeHtml(tool.name)}</span>
                        ${modelDisplay}
                        <span class="loading-tool-status completed">
                            <span class="loading-tool-icon">${statusIcon}</span>
                            <span class="loading-tool-duration">${durationDisplay}</span>
                        </span>
                    </div>
                    <div class="loading-tool-details collapsed">
                        <div class="loading-tool-args">${this.escapeHtml(tool.args || 'No arguments')}</div>
                        <div class="loading-tool-output">${this.escapeHtml(tool.context_returned || 'No output')}</div>
                    </div>
                `;

                // Add click handler for expandable details
                const mainDiv = toolItem.querySelector('.loading-tool-main');
                mainDiv.addEventListener('click', () => {
                    const details = toolItem.querySelector('.loading-tool-details');
                    const isCollapsed = details.classList.contains('collapsed');

                    if (isCollapsed) {
                        details.classList.remove('collapsed');
                        details.classList.add('expanded');
                    } else {
                        details.classList.remove('expanded');
                        details.classList.add('collapsed');
                    }
                });

                toolsHeader.appendChild(toolItem);
            });

            messageBlock.appendChild(toolsHeader);
        }

        // Create content container
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        // For user messages, use simple text content
        if (isUserMessage) {
            messageContent.textContent = message.content || '';
        }

        messageBlock.appendChild(messageContent);

        // Add buttons to the button container in the header
        if (messageIndex !== undefined && window.agentInterface) {
            // Add edit button for all messages (user and assistant)
            const editBtn = document.createElement('button');
            editBtn.className = 'message-edit-btn';
            editBtn.innerHTML = '✏️';
            editBtn.title = 'Edit message';
            editBtn.onclick = () => window.agentInterface.handleMessageEdit(messageBlock, messageIndex);
            buttonContainer.appendChild(editBtn);

            // Add delete button for all messages
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'message-delete-btn';
            deleteBtn.innerHTML = '🗑️';
            deleteBtn.title = 'Delete message';
            deleteBtn.onclick = () => window.agentInterface.handleMessageDelete(messageIndex);
            buttonContainer.appendChild(deleteBtn);

            // Add reload button for all messages (to continue from this point)
            const reloadBtn = document.createElement('button');
            reloadBtn.className = 'message-reload-btn';
            reloadBtn.innerHTML = '🔄';
            reloadBtn.title = 'Continue from here';
            reloadBtn.onclick = () => window.agentInterface.handleMessageReload(messageIndex);
            buttonContainer.appendChild(reloadBtn);
        }

        // Add raw/rendered view toggle button for assistant messages
        if (!isUserMessage && messageIndex !== undefined) {
            const toggleViewBtn = document.createElement('button');
            toggleViewBtn.className = 'message-view-toggle-btn';
            toggleViewBtn.innerHTML = '📄';
            toggleViewBtn.title = 'Toggle raw/rendered view';
            toggleViewBtn.dataset.showingRaw = 'false';
            buttonContainer.appendChild(toggleViewBtn);
        }

        // Append to content wrapper
        contentWrapper.appendChild(messageBlock);

        // Create raw content container (hidden by default)
        const rawContent = document.createElement('div');
        rawContent.className = 'message-raw-content';
        rawContent.style.display = 'none';
        rawContent.style.whiteSpace = 'pre-wrap';
        rawContent.style.fontFamily = 'monospace';
        rawContent.style.fontSize = '12px';
        rawContent.style.padding = '10px';
        rawContent.style.background = 'rgba(0,0,0,0.05)';
        rawContent.style.borderRadius = '4px';
        rawContent.style.marginTop = '8px';
        rawContent.textContent = message.content || '';
        messageBlock.appendChild(rawContent);

        // Render the actual content (for agent messages)
        if (!isUserMessage) {
            // If message has a type field (visual response), use renderContent
            // Otherwise, check if we should hide raw XML when tool_info is present
            if (message.type) {
                await this.renderContent(messageContent, message, messageId, timestampDiv);
            } else if (toolInfo && toolInfo.length > 0) {
                // Message has tool executions - strip tool XML tags but preserve user-facing content
                let cleanedContent = message.content || '';

                // Remove tool XML tags (but NOT visualresponse/response which contain user content)
                const userFacingTags = ['visualresponse', 'response'];
                toolInfo.forEach(tool => {
                    // Skip user-facing tags - they should be rendered, not hidden
                    if (userFacingTags.includes(tool.name)) {
                        return;
                    }

                    // Remove both empty and non-empty versions of the tag
                    const emptyPattern = new RegExp(`<${tool.name}\\s*/>`, 'gi');
                    const fullPattern = new RegExp(`<${tool.name}[^>]*>[\\s\\S]*?</${tool.name}>`, 'gi');
                    cleanedContent = cleanedContent.replace(emptyPattern, '');
                    cleanedContent = cleanedContent.replace(fullPattern, '');
                });

                // Also remove <context> blocks as they're shown in tool details
                cleanedContent = cleanedContent.replace(/<context>\s*[\s\S]*?\s*<\/context>/gi, '');

                // Check if there's visualresponse or response content to render
                // Pattern handles optional <action> tag and any whitespace/newlines
                const visualResponseMatch = /<visualresponse>\s*(?:<action>.*?<\/action>\s*)?<type>(.*?)<\/type>\s*<content>([\s\S]*?)<\/content>\s*<\/visualresponse>/i.exec(message.content);
                const responseMatch = /<response>(.*?)<\/response>/is.exec(message.content);

                // Skip rendering visualresponse content if it's in tool_info (payload will handle display)
                const hasVisualResponseInTools = message.tool_info && Array.isArray(message.tool_info) &&
                    message.tool_info.some(tool => ['visualresponse', 'response'].includes(tool.tool_name?.toLowerCase()));

                if (visualResponseMatch && !hasVisualResponseInTools) {
                    // Render visualresponse content (HTML or markdown)
                    const vizType = visualResponseMatch[1].trim();
                    const vizContent = visualResponseMatch[2].trim();
                    await this.renderContent(messageContent, {type: vizType, content: vizContent}, messageId, timestampDiv);
                } else if (responseMatch && !hasVisualResponseInTools) {
                    // Render plain response text
                    messageContent.textContent = responseMatch[1].trim();
                } else if (cleanedContent.trim()) {
                    // Show remaining text content (if any) after removing tool tags
                    messageContent.textContent = cleanedContent.trim();
                } else {
                    // No content left after removing tool tags
                    messageContent.style.display = 'none';
                }
            } else {
                // Plain text assistant message - render exactly as received
                messageContent.textContent = message.content || '';
            }

            // Add toggle functionality
            const toggleViewBtn = messageBlock.querySelector('.message-view-toggle-btn');
            const toolsHeader = messageBlock.querySelector('.loading-tools-header');

            if (toggleViewBtn) {
                toggleViewBtn.addEventListener('click', () => {
                    const isShowingRaw = toggleViewBtn.dataset.showingRaw === 'true';

                    if (isShowingRaw) {
                        // Switch to rendered view
                        messageContent.style.display = '';
                        rawContent.style.display = 'none';
                        if (toolsHeader) toolsHeader.style.display = '';
                        toggleViewBtn.dataset.showingRaw = 'false';
                        toggleViewBtn.innerHTML = '📄';
                        toggleViewBtn.title = 'Show raw XML';
                    } else {
                        // Switch to raw view
                        messageContent.style.display = 'none';
                        rawContent.style.display = 'block';
                        if (toolsHeader) toolsHeader.style.display = 'none';
                        toggleViewBtn.dataset.showingRaw = 'true';
                        toggleViewBtn.innerHTML = '🎨';
                        toggleViewBtn.title = 'Show rendered view';
                    }
                });
            }
        }

        // Scroll to bottom
        contentWrapper.scrollTop = contentWrapper.scrollHeight;
    }

    clearContent() {
        const contentWrapper = document.getElementById('content-wrapper');
        if (contentWrapper) {
            contentWrapper.innerHTML = '';
            window.visualLogger.debug('Chat content cleared');
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Parse XML tool calls from message content
     * Returns array of tool info objects with name, args, and context
     * Preserves original document order
     */
    parseToolCallsFromXML(content) {
        const toolCalls = [];

        // Common tool tags to look for
        const toolTags = ['bash', 'readfile', 'writefile', 'editfile', 'grepsearch', 'searchweb',
                          'viewpermissions', 'think', 'todos', 'visualresponse', 'insertline',
                          'deletelines', 'readdirectory', 'searchtext', 'taskmanager', 'homeassistant',
                          'paralleltools'];

        // Find all tool tags with their positions
        const matches = [];

        toolTags.forEach(tag => {
            const pattern = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, 'gi');
            let match;

            while ((match = pattern.exec(content)) !== null) {
                matches.push({
                    position: match.index,
                    name: tag,
                    args: match[1].trim().substring(0, 500),
                    fullMatch: match[0]
                });
            }
        });

        // Sort by position to maintain document order
        matches.sort((a, b) => a.position - b.position);

        // Convert to tool call objects
        matches.forEach(match => {
            toolCalls.push({
                name: match.name,
                args: match.args,
                status: 'success',
                duration_ms: null,
                model: null,
                context_returned: '',
                context_length: 0
            });
        });

        // Parse <context> blocks and match with non-visualresponse tools in order
        const contextPattern = /<context>\s*([\s\S]*?)\s*<\/context>/gi;
        let contextMatch;
        let nonVisualToolIndex = 0;

        while ((contextMatch = contextPattern.exec(content)) !== null) {
            const contextContent = contextMatch[1].trim();

            // Find the next non-visualresponse tool to attach this context to
            while (nonVisualToolIndex < toolCalls.length) {
                if (toolCalls[nonVisualToolIndex].name !== 'visualresponse' &&
                    toolCalls[nonVisualToolIndex].name !== 'think' &&
                    toolCalls[nonVisualToolIndex].name !== 'todos') {
                    toolCalls[nonVisualToolIndex].context_returned = contextContent.substring(0, 500);
                    toolCalls[nonVisualToolIndex].context_length = contextContent.length;
                    nonVisualToolIndex++;
                    break;
                }
                nonVisualToolIndex++;
            }
        }

        return toolCalls;
    }

    /**
     * Merge parsed XML tool calls with metadata
     * Metadata provides timing and model info, XML provides structure
     */
    mergeToolInfoWithMetadata(xmlToolCalls, metadata) {
        if (!metadata || !Array.isArray(metadata)) {
            return xmlToolCalls;
        }

        // Match by index and merge information
        return xmlToolCalls.map((xmlTool, index) => {
            const metadataTool = metadata[index];

            if (metadataTool && metadataTool.name === xmlTool.name) {
                // Metadata takes precedence for timing/model info
                return {
                    ...xmlTool,
                    duration_ms: metadataTool.duration_ms || xmlTool.duration_ms,
                    model: metadataTool.model || xmlTool.model,
                    status: metadataTool.status || xmlTool.status,
                    error: metadataTool.error || xmlTool.error
                };
            }

            return xmlTool;
        });
    }
}

// Export for use in main script
window.ContentRenderer = ContentRenderer;