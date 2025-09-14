// tools/visual_assets/js/renderers.js

/**
 * Renders HTML content within a sandboxed iframe to isolate it from the main document.
 * @param {HTMLElement} targetElement - The container element to render into.
 * @param {string} htmlContent - The complete HTML string to render.
 */
function renderHTML(targetElement, htmlContent) {
    const iframe = document.createElement('iframe');

    // Use srcdoc instead of writing to contentDocument - this is more secure
    // and doesn't require allow-same-origin
    iframe.sandbox = 'allow-scripts allow-forms allow-popups allow-popups-to-escape-sandbox allow-downloads';
    iframe.style.width = '100%';
    iframe.style.height = '400px'; // Initial height
    iframe.style.border = 'none';
    iframe.style.overflow = 'hidden'; // Prevent scrollbars during resize
    iframe.setAttribute('loading', 'lazy');

    // Height adjustment state
    let lastHeight = 400;
    let heightAdjustmentCount = 0;
    let heightDebounceTimeout = null;
    const MAX_HEIGHT_ADJUSTMENTS = 8; // Allow more adjustments for better accuracy
    const HEIGHT_DEBOUNCE_MS = 150; // Reduce debounce time for faster response

    // Listen for height messages from iframe content with improved logic
    const messageListener = (event) => {
        if (event.source === iframe.contentWindow && event.data.type === 'resize') {
            const rawHeight = event.data.height;
            const newHeight = Math.max(150, Math.min(1200, rawHeight + 10)); // Reduce padding from 20px to 10px

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
                iframe.style.height = newHeight + 'px';
                lastHeight = newHeight;
                heightAdjustmentCount++;
                window.visualLogger.debug(`Iframe height adjusted to ${newHeight}px (raw: ${rawHeight}px, adjustment #${heightAdjustmentCount})`);
            }, dynamicDelay);
        }
    };

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

    // Inject the height reporting script into the HTML content
    let modifiedHtmlContent = htmlContent;
    if (htmlContent.toLowerCase().includes('</body>')) {
        modifiedHtmlContent = htmlContent.replace('</body>', heightReportingScript + '</body>');
    } else if (htmlContent.toLowerCase().includes('</html>')) {
        modifiedHtmlContent = htmlContent.replace('</html>', heightReportingScript + '</html>');
    } else {
        modifiedHtmlContent = htmlContent + heightReportingScript;
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

            // Don't exceed reasonable bounds
            estimatedHeight = Math.min(estimatedHeight, 600);

            iframe.style.height = estimatedHeight + 'px';
            window.visualLogger.debug(`Iframe height estimated at ${estimatedHeight}px (conservative fallback)`);

        } catch (e) {
            window.visualLogger.warn('Could not resize iframe automatically:', e.message);
            iframe.style.height = '300px'; // More conservative default
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
        setTimeout(resizeIframe, 100);

        // Also resize after images might have loaded
        setTimeout(resizeIframe, 1000);
    };

    iframe.onerror = (e) => {
        window.visualLogger.error("An error occurred loading the iFrame content.", { error: e });
        targetElement.textContent = "An error occurred while loading the iFrame.";
    };

    // Clear the target element and append the new iframe.
    targetElement.innerHTML = '';
    targetElement.appendChild(iframe);
}

/**
 * Renders Markdown content, including support for Mermaid diagrams.
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


// --- Main Content Rendering Logic ---
const renderers = {
    'html': renderHTML,
    'markdown': renderMarkdown,
};

/**
 * This class is a simplified placeholder to be instantiated by main.js
 * It exposes the main rendering function.
 */
class ContentRenderer {
    async renderContent(container, data) {
        const { type, content } = data;
        const renderer = renderers[type];

        if (renderer) {
            window.visualLogger.debug(`Rendering content with '${type}' renderer.`, data);
            try {
                renderer(container, content);
            } catch (e) {
                window.visualLogger.error(`${type} rendering failed:`, { error: e });
                container.innerHTML = `<pre class="render-error">Error rendering ${type}: ${e.message}</pre>`;
            }
        } else {
            window.visualLogger.warn(`No renderer found for type: ${type}`);
            container.innerHTML = `<pre>Unsupported content type: ${type}</pre>`;
        }
    }
}

// Export for use in main script
window.ContentRenderer = ContentRenderer;