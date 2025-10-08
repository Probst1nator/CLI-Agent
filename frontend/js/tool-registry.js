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

        this.fallbackTools = [
            { name: 'searchweb', displayName: 'Web Search' },
            { name: 'bash', displayName: 'Terminal' },
            { name: 'readfile', displayName: 'Read File' },
            { name: 'writefile', displayName: 'Write File' },
            { name: 'editfile', displayName: 'Edit File' },
            { name: 'grepsearch', displayName: 'Text Search' },
            { name: 'visualresponse', displayName: 'Visual Response' },
            { name: 'taskmanager', displayName: 'Task Manager' },
            { name: 'readdirectory', displayName: 'Read Directory' },
            { name: 'insertline', displayName: 'Insert Line' },
            { name: 'deletelines', displayName: 'Delete Lines' },
            { name: 'paralleltools', displayName: 'Parallel Tools' },
            { name: 'searchresource', displayName: 'Search Resource' },
            { name: 'viewpermissions', displayName: 'View Permissions' },
            { name: 'homeassistant', displayName: 'Home Assistant' },
            { name: 'think', displayName: 'Think' },
            { name: 'todos', displayName: 'Todos' }
        ];
    }

    async initialize() {
        if (this.isReady) {
            return this.tools;
        }

        if (this.isLoading) {
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
            const response = await fetch('/api/tools');
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
                window.visualLogger?.info(`ToolRegistry: Loaded ${this.tools.length} tools from server`);
            } else {
                throw new Error('Invalid tools data format received from server');
            }
        } catch (error) {
            window.visualLogger?.error('ToolRegistry: Error fetching tools from server:', error);
            this.useFallbackTools();
        }
    }

    useFallbackTools() {
        this.tools = this.fallbackTools;
        this.toolNames = this.tools.map(t => t.name);

        this.tools.forEach(tool => {
            this.toolMap.set(tool.name, tool);
        });

        this.isReady = true;
        window.visualLogger?.warn(`ToolRegistry: Using fallback tool list with ${this.tools.length} tools`);
    }

    getToolNames() {
        if (!this.isReady) {
            window.visualLogger?.warn('ToolRegistry: getToolNames called before ready, returning fallback names');
            return this.fallbackTools.map(t => t.name);
        }
        return this.toolNames;
    }

    getTools() {
        if (!this.isReady) {
            window.visualLogger?.warn('ToolRegistry: getTools called before ready, returning fallback tools');
            return this.fallbackTools;
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
}

window.ToolRegistry = new ToolRegistry();

if (typeof module !== 'undefined' && module.exports) {
    module.exports = ToolRegistry;
}
