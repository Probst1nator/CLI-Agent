/**
 * Tool Registry - Singleton for tool metadata management
 * Fetches and caches available tools from the server
 * Provides single source of truth for tool names and metadata
 * @class
 */
class ToolRegistry {
    /**
     * Initialize tool registry (singleton pattern)
     * @returns {ToolRegistry} Singleton instance
     */
    constructor() {
        if (ToolRegistry._instance) {
            return ToolRegistry._instance;
        }

        ToolRegistry._instance = this;

        /** @type {ToolDefinition[]} Available tools */
        this.tools = [];
        /** @type {string[]} Tool names array */
        this.toolNames = [];
        /** @type {Map<string, ToolDefinition>} Tool lookup map */
        this.toolMap = new Map();
        /** @type {boolean} Whether tools have been loaded */
        this.isReady = false;
        /** @type {boolean} Whether currently loading */
        this.isLoading = false;
        /** @type {Promise<ToolDefinition[]>|null} Pending load promise */
        this.loadPromise = null;
        /** @type {boolean} Whether initial fetch failed */
        this._failedInitially = false;
        /** @type {ToolDefinition[]} Fallback tools (empty - uses dynamic discovery) */
        this.fallbackTools = [];
    }

    /**
     * Initialize tool registry from server
     * @param {boolean} [force=false] - Force reload even if already ready
     * @returns {Promise<ToolDefinition[]>} Loaded tools
     */
    async initialize(force = false) {
        if (this.isReady && !force) {
            return this.tools;
        }

        if (this.isLoading && !force) {
            return this.loadPromise;
        }

        this.isLoading = true;
        this.loadPromise = this.fetchTools();

        try {
            await this.loadPromise;
        } finally {
            this.isLoading = false;
        }

        return this.tools;
    }

    /**
     * Fetch tools from backend API
     * @returns {Promise<void>}
     * @private
     */
    async fetchTools() {
        try {
            window.logDebug('[ToolRegistry] Fetching tools from: /api/tools');
            const response = window.backendConnector
                ? await window.backendConnector.fetch('/api/tools')
                : await fetch('/api/tools');

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`[ToolRegistry] HTTP ${response.status}: ${response.statusText}`);
                console.error('[ToolRegistry] Response body:', errorText.substring(0, 500));
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.tools && Array.isArray(data.tools)) {
                this.tools = data.tools.map(tool => ({
                    name: tool.name || tool.delim,
                    displayName: tool.displayName || tool.info?.name || tool.name,
                    description: tool.description || tool.info?.description || '',
                    delim: tool.delim
                }));

                this.toolNames = this.tools.map(t => t.name);

                this.tools.forEach(tool => {
                    this.toolMap.set(tool.name, tool);
                });

                this.isReady = true;
                const wasFailedInitially = this._failedInitially;
                this._failedInitially = false;

                window.visualLogger?.info(`ToolRegistry: Loaded ${this.tools.length} tools from server`);
                window.logDebug('[ToolRegistry] Successfully loaded', this.tools.length, 'tools');

                if (wasFailedInitially && window.ToolManager) {
                    window.logDebug('[ToolRegistry] Retried after initial failure - refreshing ToolManager');
                    window.ToolManager.refresh();
                }
            } else {
                throw new Error('Invalid tools data format received from server');
            }
        } catch (error) {
            console.error('[ToolRegistry] Fetch failed:', error);
            window.visualLogger?.error('ToolRegistry: Error fetching tools from server:', error);
            this.useFallbackTools();
        }
    }

    /**
     * Use fallback tools when API fetch fails (empty - relies on dynamic discovery)
     * @private
     */
    useFallbackTools() {
        this.tools = [];
        this.toolNames = [];
        this.toolMap.clear();
        this.isReady = true;
        this._failedInitially = true;
        window.visualLogger?.warn('ToolRegistry: API fetch failed, will use dynamic tool discovery from XML content');
    }

    /**
     * Get array of tool names
     * @returns {string[]} Tool names
     */
    getToolNames() {
        if (!this.isReady) {
            window.visualLogger?.warn('ToolRegistry: getToolNames called before ready, returning empty array (will use dynamic discovery)');
            return [];
        }
        return this.toolNames;
    }

    /**
     * Get array of tool definitions
     * @returns {ToolDefinition[]} Tool definitions
     */
    getTools() {
        if (!this.isReady) {
            window.visualLogger?.warn('ToolRegistry: getTools called before ready, returning empty array (will use dynamic discovery)');
            return [];
        }
        return this.tools;
    }

    /**
     * Get tool definition by name
     * @param {string} name - Tool name
     * @returns {ToolDefinition|undefined} Tool definition
     */
    getTool(name) {
        return this.toolMap.get(name);
    }

    /**
     * Check if any tools are loaded
     * @returns {boolean} True if tools exist
     */
    hasTools() {
        return this.tools.length > 0;
    }

    /**
     * Wait for tools to be loaded
     * @returns {Promise<void>}
     */
    async waitForReady() {
        if (this.isReady) {
            return;
        }

        if (this.isLoading) {
            await this.loadPromise;
            return;
        }

        await this.initialize();
    }

    /**
     * Retry loading tools if initial fetch failed
     * @returns {Promise<void>}
     */
    async retryIfFailed() {
        if (this._failedInitially) {
            window.logDebug('[ToolRegistry] Retrying after initial failure...');
            await this.initialize(true);
        }
    }
}

// Create singleton instance
window.ToolRegistry = new ToolRegistry();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = ToolRegistry;
}
