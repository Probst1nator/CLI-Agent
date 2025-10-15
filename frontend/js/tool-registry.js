/**
 * Tool Registry
 *
 * Singleton that fetches and caches available tools from the server.
 * Provides a single source of truth for tool names and metadata.
 */

class ToolRegistry {
    constructor() {
        if (ToolRegistry._instance) {
            return ToolRegistry._instance;
        }

        ToolRegistry._instance = this;

        this.tools = [];
        this.toolNames = [];
        this.toolMap = new Map();
        this.isReady = false;
        this.isLoading = false;
        this.loadPromise = null;

        // No more hardcoded fallback list - we'll use empty array and let
        // the system discover tools dynamically from XML content
        this.fallbackTools = [];
    }

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

    async fetchTools() {
        try {
            const url = window.backendConnector ? await window.backendConnector.buildURL('/api/tools') : '/api/tools';
            console.log('[ToolRegistry] Fetching tools from:', url);
            const response = await fetch(url);
            if (!response.ok) {
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
                console.log('[ToolRegistry] Successfully loaded', this.tools.length, 'tools');
                
                // Notify that tools are ready (especially important if we failed initially)
                if (wasFailedInitially && window.ToolManager) {
                    console.log('[ToolRegistry] Retried after initial failure - refreshing ToolManager');
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

    useFallbackTools() {
        // No hardcoded fallback list anymore - start with empty array
        // Tools will be discovered dynamically from XML content via generic pattern matching
        this.tools = [];
        this.toolNames = [];
        this.toolMap.clear();

        this.isReady = true;
        this._failedInitially = true; // Mark that we failed, will retry when backend ready
        window.visualLogger?.warn('ToolRegistry: API fetch failed, will use dynamic tool discovery from XML content');
    }

    getToolNames() {
        if (!this.isReady) {
            window.visualLogger?.warn('ToolRegistry: getToolNames called before ready, returning empty array (will use dynamic discovery)');
            return [];
        }
        return this.toolNames;
    }

    getTools() {
        if (!this.isReady) {
            window.visualLogger?.warn('ToolRegistry: getTools called before ready, returning empty array (will use dynamic discovery)');
            return [];
        }
        return this.tools;
    }

    getTool(name) {
        return this.toolMap.get(name);
    }

    hasTools() {
        return this.tools.length > 0;
    }

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
    
    async retryIfFailed() {
        // If we failed initially, retry now that backend might be ready
        if (this._failedInitially) {
            console.log('[ToolRegistry] Retrying after initial failure...');
            await this.initialize(true); // Force re-init
        }
    }
}

window.ToolRegistry = new ToolRegistry();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = ToolRegistry;
}
