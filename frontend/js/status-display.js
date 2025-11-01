/**
 * StatusDisplay - Real-time backend operation status display
 *
 * Displays LLM generation metrics in a single line above tool-toggles-container.
 * Integrates tool execution status into loading-tool-items within messages.
 */

class StatusDisplay {
    constructor(wsManager) {
        this.wsManager = wsManager;
        this.currentToolItem = null;
        this.currentToolId = null;
        this.activeGenerations = new Map(); // Track multiple concurrent generations: generation_id -> metrics data

        // Listen to real-time status events from backend
        this.wsManager.on('token_stream', (data) => this.handleTokenStream(data));  // Legacy support
        this.wsManager.on('generation_metrics', (data) => this.handleGenerationMetrics(data));
        this.wsManager.on('generation_complete', (data) => this.handleGenerationComplete(data));
        this.wsManager.on('tool_update', (data) => this.handleToolUpdate(data));

        console.log('✅ StatusDisplay initialized (generation metrics in dedicated line)');
    }

    /**
     * Update the generation status line display with all active generations
     */
    updateGenerationStatusLine() {
        const statusLine = document.getElementById('generation-status-line');
        if (!statusLine) return;

        const contentEl = statusLine.querySelector('.generation-status-content');
        if (!contentEl) return;

        // If no active generations, hide the line
        if (this.activeGenerations.size === 0) {
            statusLine.classList.add('hidden');
            return;
        }

        // Show the line and build content
        statusLine.classList.remove('hidden');

        // Build display for each active generation
        const generationDisplays = [];
        this.activeGenerations.forEach((data, generationId) => {
            const modelName = data.model_key ? data.model_key.split('/').pop() : 'LLM';
            const elapsed = (data.elapsed_ms / 1000).toFixed(1);
            const tokensPerSecond = data.tokens_per_second.toFixed(1);

            generationDisplays.push(
                `<span class="generation-item">` +
                `<strong>${this.escapeHtml(modelName)}</strong>: ` +
                `${data.tokens_generated} tokens, ${tokensPerSecond} t/s, ${elapsed}s` +
                `</span>`
            );
        });

        contentEl.innerHTML = `⚡ Generating response → ${generationDisplays.join(' <span class="generation-separator">|</span> ')}`;
    }

    /**
     * Get or create the tools container header within the current message being generated
     * Always looks up the current message dynamically to avoid stale references
     */
    getOrCreateToolsContainer() {
        // Find the last message block (the one currently being generated)
        const contentWrapper = document.getElementById('content-wrapper');
        if (!contentWrapper) return null;

        const messageBlocks = contentWrapper.querySelectorAll('.message-block');
        const lastMessageBlock = messageBlocks[messageBlocks.length - 1];

        if (!lastMessageBlock) {
            console.warn('No message block found for tools');
            return null;
        }

        // Look for existing tools container in this message block
        let toolsContainer = lastMessageBlock.querySelector('.loading-tools-header');

        if (!toolsContainer) {
            // Create container within the message block
            toolsContainer = document.createElement('div');
            toolsContainer.className = 'loading-tools-header';

            // Insert before the message content (if it exists) or append
            const messageContent = lastMessageBlock.querySelector('.message-content');
            if (messageContent) {
                lastMessageBlock.insertBefore(toolsContainer, messageContent);
            } else {
                lastMessageBlock.appendChild(toolsContainer);
            }
        }

        // Always return fresh container, don't cache to avoid stale references
        return toolsContainer;
    }

    /**
     * Create or update a loading-tool-item for the current tool
     */
    getOrCreateToolItem(toolName) {
        const container = this.getOrCreateToolsContainer();

        // Look for existing tool item for this tool
        let toolItem = container.querySelector(`[data-tool-name="${toolName}"]`);

        if (!toolItem) {
            // Create new tool item
            toolItem = document.createElement('div');
            toolItem.className = 'loading-tool-item running';
            toolItem.setAttribute('data-tool-name', toolName);

            toolItem.innerHTML = `
                <div class="loading-tool-main">
                    <span class="loading-tool-name">${this.escapeHtml(toolName)}</span>
                    <span class="loading-tool-status running">
                        <span class="loading-tool-spinner">⏳</span>
                    </span>
                </div>
                <div class="loading-tool-details expanded">
                    <div class="loading-tool-status-data"></div>
                </div>
            `;

            container.appendChild(toolItem);
            this.currentToolItem = toolItem;
            this.currentToolId = toolName;
        }

        return toolItem;
    }

    /**
     * Escape HTML for safe display
     */
    escapeHtml(unsafe) {
        if (!unsafe) return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    /**
     * Handle generation metrics events (real-time LLM generation with generation_id)
     * Updates the dedicated generation status line above tool-toggles-container
     */
    handleGenerationMetrics(data) {
        if (!data || !data.generation_id) return;

        try {
            // Store/update generation data
            this.activeGenerations.set(data.generation_id, {
                generation_id: data.generation_id,
                model_key: data.model_key,
                tokens_generated: data.tokens_generated,
                elapsed_ms: data.elapsed_ms,
                tokens_per_second: data.tokens_per_second
            });

            // Update the status line display
            this.updateGenerationStatusLine();
        } catch (e) {
            console.error('Error handling generation_metrics:', e);
        }
    }

    /**
     * Handle generation complete events (LLM generation finished)
     * Removes the generation from tracking and updates the status line
     */
    handleGenerationComplete(data) {
        if (!data || !data.generation_id) return;

        try {
            // Remove from active generations
            this.activeGenerations.delete(data.generation_id);

            // Update the status line (will hide if no active generations remain)
            this.updateGenerationStatusLine();
        } catch (e) {
            console.error('Error handling generation_complete:', e);
        }
    }

    /**
     * Handle token streaming events (real-time LLM generation)
     * LEGACY: Kept for backward compatibility with old token_stream events
     * Uses a fake generation_id for tracking
     */
    handleTokenStream(data) {
        if (!data) return;

        try {
            // Use a legacy generation ID for backward compatibility
            const legacyGenerationId = 'legacy_generation';

            const tokensPerSecond = data.tokens_generated > 0 && data.elapsed_ms > 0
                ? (data.tokens_generated / (data.elapsed_ms / 1000))
                : 0;

            // Store/update generation data
            this.activeGenerations.set(legacyGenerationId, {
                generation_id: legacyGenerationId,
                model_key: 'LLM',
                tokens_generated: data.tokens_generated,
                elapsed_ms: data.elapsed_ms,
                tokens_per_second: tokensPerSecond
            });

            // Update the status line display
            this.updateGenerationStatusLine();
        } catch (e) {
            console.error('Error handling token_stream:', e);
        }
    }

    /**
     * Handle tool update events (tool execution status)
     */
    handleToolUpdate(data) {
        if (!data) return;

        try {
            const toolName = data.tool_name || 'Unknown Tool';

            if (data.status === 'running') {
                // Create or update tool item when tool starts
                const toolItem = this.getOrCreateToolItem(toolName);
                const statusDataEl = toolItem.querySelector('.loading-tool-status-data');

                if (statusDataEl && data.description) {
                    statusDataEl.innerHTML = `
                        <div style="font-size: 0.85rem; padding: 8px; background: rgba(255, 193, 7, 0.08); border-radius: 4px;">
                            <div style="margin-bottom: 4px;"><strong>🔧 Running</strong></div>
                            <div style="opacity: 0.8; font-size: 0.8rem;">${this.escapeHtml(this.truncateDescription(data.description))}</div>
                        </div>
                    `;
                }
            } else if (data.status === 'success') {
                // Update tool item to completed state
                const toolItem = this.getOrCreateToolItem(toolName);
                toolItem.className = 'loading-tool-item completed';

                const statusEl = toolItem.querySelector('.loading-tool-status');
                const statusDataEl = toolItem.querySelector('.loading-tool-status-data');
                const duration = (data.duration_ms / 1000).toFixed(1);

                if (statusEl) {
                    statusEl.className = 'loading-tool-status success';
                    statusEl.innerHTML = `
                        <span class="loading-tool-icon">✅</span>
                        <span class="loading-tool-duration">${duration}s</span>
                    `;
                }

                if (statusDataEl) {
                    statusDataEl.innerHTML = `
                        <div style="font-size: 0.85rem; padding: 8px; background: rgba(76, 175, 80, 0.08); border-radius: 4px; color: #27ae60;">
                            <strong>✓ Completed in ${duration}s</strong>
                        </div>
                    `;
                }
            } else if (data.status === 'error') {
                // Update tool item to failed state
                const toolItem = this.getOrCreateToolItem(toolName);
                toolItem.className = 'loading-tool-item failed';

                const statusEl = toolItem.querySelector('.loading-tool-status');
                const statusDataEl = toolItem.querySelector('.loading-tool-status-data');
                const duration = (data.duration_ms / 1000).toFixed(1);

                if (statusEl) {
                    statusEl.className = 'loading-tool-status error';
                    statusEl.innerHTML = `
                        <span class="loading-tool-icon">❌</span>
                        <span class="loading-tool-duration">${duration}s</span>
                    `;
                }

                if (statusDataEl) {
                    statusDataEl.innerHTML = `
                        <div style="font-size: 0.85rem; padding: 8px; background: rgba(244, 67, 54, 0.08); border-radius: 4px; color: #e74c3c;">
                            <div style="margin-bottom: 4px;"><strong>✗ Failed after ${duration}s</strong></div>
                            <div style="opacity: 0.9; font-size: 0.8rem;">${this.escapeHtml(this.truncateDescription(data.error || 'Unknown error'))}</div>
                        </div>
                    `;
                }
            }
        } catch (e) {
            console.error('Error handling tool_update:', e);
        }
    }

    /**
     * Truncate long descriptions for display
     */
    truncateDescription(text, maxLength = 80) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    /**
     * Clear all tool items (e.g., on new chat)
     */
    clearAllTools() {
        // Clear all tools containers in all messages
        const allToolsContainers = document.querySelectorAll('.loading-tools-header');
        allToolsContainers.forEach(container => {
            container.innerHTML = '';
            container.remove();
        });

        this.currentToolItem = null;
        this.currentToolId = null;
        this.activeGenerations.clear(); // Clear generation tracking
    }
}
