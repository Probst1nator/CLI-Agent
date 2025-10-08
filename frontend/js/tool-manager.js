/**
 * Tool Manager
 *
 * Manages the enable/disable state of agent tools through a UI with checkboxes.
 * Allows users to control which tools the agent can use.
 */

class ToolManager {
    constructor() {
        this.tools = [];
        this.disabledTools = new Set();
        this.containerElement = null;

        // Load saved state from localStorage
        this.loadToolStates();

        // Initialize after DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initialize());
        } else {
            this.initialize();
        }
    }

    /**
     * Initialize the tool manager
     */
    initialize() {
        this.containerElement = document.getElementById('tool-toggles');

        if (!this.containerElement) {
            console.warn('Tool toggles container not found');
            return;
        }

        // Tools will be dynamically fetched from the server
        this.tools = [];

        // Fetch tools from server and then create toggles
        this.fetchAvailableTools().then(() => {
            this.createToolToggles();
            window.visualLogger?.info('ToolManager initialized with', this.tools.length, 'tools from server');
        }).catch(error => {
            console.error('Failed to fetch tools, falling back to default list:', error);
            this.fallbackToDefaultTools();
            this.createToolToggles();
            window.visualLogger?.warn('ToolManager initialized with', this.tools.length, 'fallback tools');
        });
    }

    /**
     * Fetch available tools from the server
     */
    async fetchAvailableTools() {
        try {
            const response = await fetch('/api/tools');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.tools && Array.isArray(data.tools)) {
                this.tools = data.tools.map(tool => ({
                    name: tool.name || tool.delim,
                    displayName: tool.displayName || tool.info?.name
                }));

                window.visualLogger?.info(`Fetched ${this.tools.length} tools from server:`, this.tools.map(t => t.name));
            } else {
                throw new Error('Invalid tools data format received from server');
            }
        } catch (error) {
            window.visualLogger?.error('Error fetching tools from server:', error);
            throw error;
        }
    }

    /**
     * Fallback to a default set of common tools if server fetch fails
     */
    fallbackToDefaultTools() {
        this.tools = [
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
            { name: 'searchtext', displayName: 'Search Text' },
            { name: 'viewpermissions', displayName: 'View Permissions' }
        ];

        window.visualLogger?.warn('Using fallback tool list with', this.tools.length, 'tools');
    }

    /**
     * Create the tool toggle checkboxes
     */
    createToolToggles() {
        if (!this.containerElement) return;

        // Clear existing content
        this.containerElement.innerHTML = '';

        // Add "Tools:" header as first item
        const headerItem = document.createElement('div');
        headerItem.className = 'tool-toggle-item';
        headerItem.textContent = 'Tools:';
        this.containerElement.appendChild(headerItem);

        this.tools.forEach(tool => {
            const isEnabled = !this.disabledTools.has(tool.name);

            // Create container for this tool
            const toolItem = document.createElement('div');
            toolItem.className = `tool-toggle-item ${isEnabled ? 'enabled' : 'disabled'}`;

            // Create checkbox
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `tool-${tool.name}`;
            checkbox.checked = isEnabled;
            checkbox.addEventListener('change', (e) => {
                this.toggleTool(tool.name, e.target.checked);
            });

            // Create label
            const label = document.createElement('label');
            label.htmlFor = `tool-${tool.name}`;
            label.textContent = tool.displayName;

            // Assemble
            toolItem.appendChild(checkbox);
            toolItem.appendChild(label);
            this.containerElement.appendChild(toolItem);
        });
    }

    /**
     * Toggle a tool's enabled/disabled state
     */
    toggleTool(toolName, enabled) {
        const toolItem = document.querySelector(`.tool-toggle-item:has(#tool-${toolName})`);

        if (enabled) {
            this.disabledTools.delete(toolName);
            toolItem?.classList.remove('disabled');
            toolItem?.classList.add('enabled');
        } else {
            this.disabledTools.add(toolName);
            toolItem?.classList.remove('enabled');
            toolItem?.classList.add('disabled');
        }

        this.saveToolStates();
        this.notifyToolStateChange(toolName, enabled);

        window.visualLogger?.info(`Tool ${toolName} ${enabled ? 'enabled' : 'disabled'}`);
    }

    /**
     * Check if a tool is enabled
     */
    isToolEnabled(toolName) {
        return !this.disabledTools.has(toolName);
    }

    /**
     * Get list of enabled tools
     */
    getEnabledTools() {
        return this.tools
            .filter(tool => this.isToolEnabled(tool.name))
            .map(tool => tool.name);
    }

    /**
     * Get list of disabled tools
     */
    getDisabledTools() {
        return Array.from(this.disabledTools);
    }

    /**
     * Enable all tools
     */
    enableAllTools() {
        this.disabledTools.clear();
        this.tools.forEach(tool => {
            const checkbox = document.getElementById(`tool-${tool.name}`);
            if (checkbox) checkbox.checked = true;
        });
        this.saveToolStates();
        this.createToolToggles(); // Refresh UI
        window.visualLogger?.info('All tools enabled');
    }

    /**
     * Disable all tools
     */
    disableAllTools() {
        this.tools.forEach(tool => {
            this.disabledTools.add(tool.name);
            const checkbox = document.getElementById(`tool-${tool.name}`);
            if (checkbox) checkbox.checked = false;
        });
        this.saveToolStates();
        this.createToolToggles(); // Refresh UI
        window.visualLogger?.info('All tools disabled');
    }

    /**
     * Reset to default (all enabled)
     */
    resetToDefaults() {
        this.enableAllTools();
    }

    /**
     * Save tool states to localStorage
     */
    saveToolStates() {
        try {
            const disabledToolsArray = Array.from(this.disabledTools);
            localStorage.setItem('agent-disabled-tools', JSON.stringify(disabledToolsArray));
        } catch (e) {
            console.warn('Failed to save tool states:', e);
        }
    }

    /**
     * Load tool states from localStorage
     */
    loadToolStates() {
        try {
            const saved = localStorage.getItem('agent-disabled-tools');
            if (saved) {
                const disabledToolsArray = JSON.parse(saved);
                this.disabledTools = new Set(disabledToolsArray);
            }
        } catch (e) {
            console.warn('Failed to load tool states:', e);
            this.disabledTools = new Set();
        }
    }

    /**
     * Notify other components about tool state changes
     */
    notifyToolStateChange(toolName, enabled) {
        // Dispatch custom event for other components to listen to
        window.dispatchEvent(new CustomEvent('toolStateChanged', {
            detail: { toolName, enabled, disabledTools: this.getDisabledTools() }
        }));
    }

    /**
     * Get tool information for sending to server
     */
    getToolConfiguration() {
        return {
            enabledTools: this.getEnabledTools(),
            disabledTools: this.getDisabledTools(),
            totalTools: this.tools.length
        };
    }
}

// Create global instance
window.ToolManager = new ToolManager();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ToolManager;
}