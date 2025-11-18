/**
 * Tool Manager - UI-based tool enable/disable management
 * Manages the enable/disable state of agent tools through checkboxes
 * Persists state to localStorage for consistency across sessions
 * @class
 */
class ToolManager {
    /**
     * Initialize tool manager with state persistence
     * Loads saved tool states from localStorage and initializes DOM when ready
     */
    constructor() {
        /** @type {ToolDefinition[]} Array of available tools */
        this.tools = [];
        /** @type {Set<string>} Set of disabled tool names */
        this.disabledTools = new Set();
        /** @type {HTMLElement|null} Container for tool toggle checkboxes */
        this.containerElement = null;

        this.loadToolStates();

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initialize());
        } else {
            this.initialize();
        }
    }

    /**
     * Initialize the tool manager from ToolRegistry
     * @returns {Promise<void>}
     * @async
     */
    async initialize() {
        this.containerElement = document.getElementById('tool-toggles');

        if (!this.containerElement) {
            console.warn('Tool toggles container not found');
            return;
        }

        if (window.ToolRegistry) {
            await window.ToolRegistry.waitForReady();
            this.tools = window.ToolRegistry.getTools();
            window.visualLogger?.info('ToolManager initialized with', this.tools.length, 'tools from ToolRegistry');
        } else {
            window.visualLogger?.warn('ToolRegistry not available, using fallback');
            this.fallbackToDefaultTools();
        }

        this.createToolToggles();

        // Sync initial state from localStorage with backend
        this.syncWithBackend();
    }

    /**
     * Use empty tool list when ToolRegistry unavailable
     * Tools will be discovered dynamically as they are used
     * @private
     */
    fallbackToDefaultTools() {
        this.tools = [];
        window.visualLogger?.warn('ToolRegistry unavailable - tool manager will remain empty (tools discovered dynamically)');
    }

    /**
     * Create checkbox toggle UI for all tools
     * Renders tool-toggle-item elements with checkboxes and labels
     * @private
     */
    createToolToggles() {
        window.logDebug('[ToolManager] createToolToggles called with', this.tools.length, 'tools');
        window.logDebug('[ToolManager] containerElement:', this.containerElement);

        if (!this.containerElement) {
            console.error('[ToolManager] No container element found!');
            return;
        }

        this.containerElement.innerHTML = '';

        const headerItem = document.createElement('div');
        headerItem.className = 'tool-toggle-item';
        headerItem.textContent = 'Tools:';
        this.containerElement.appendChild(headerItem);

        window.logDebug('[ToolManager] About to create toggles for tools:', this.tools.map(t => t.name));

        this.tools.forEach(tool => {
            const isEnabled = !this.disabledTools.has(tool.name);

            const toolItem = document.createElement('div');
            toolItem.className = `tool-toggle-item ${isEnabled ? 'enabled' : 'disabled'}`;

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `tool-${tool.name}`;
            checkbox.checked = isEnabled;
            checkbox.addEventListener('change', (e) => {
                this.toggleTool(tool.name, e.target.checked);
            });

            const label = document.createElement('label');
            label.htmlFor = `tool-${tool.name}`;
            label.textContent = tool.displayName;

            toolItem.appendChild(checkbox);
            toolItem.appendChild(label);
            this.containerElement.appendChild(toolItem);
        });
    }

    /**
     * Toggle tool enabled/disabled state and update UI
     * @param {string} toolName - Name of tool to toggle
     * @param {boolean} enabled - Whether tool should be enabled
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
        this.syncWithBackend();  // Sync with backend
        this.notifyToolStateChange(toolName, enabled);

        window.visualLogger?.info(`Tool ${toolName} ${enabled ? 'enabled' : 'disabled'}`);
    }

    /**
     * Check if a tool is enabled
     * @param {string} toolName - Name of tool to check
     * @returns {boolean} True if tool is enabled
     */
    isToolEnabled(toolName) {
        return !this.disabledTools.has(toolName);
    }

    /**
     * Get array of enabled tool names
     * @returns {string[]} List of enabled tool names
     */
    getEnabledTools() {
        return this.tools
            .filter(tool => this.isToolEnabled(tool.name))
            .map(tool => tool.name);
    }

    /**
     * Get array of disabled tool names
     * @returns {string[]} List of disabled tool names
     */
    getDisabledTools() {
        return Array.from(this.disabledTools);
    }

    /**
     * Enable all tools and refresh UI
     */
    enableAllTools() {
        this.disabledTools.clear();
        this.tools.forEach(tool => {
            const checkbox = document.getElementById(`tool-${tool.name}`);
            if (checkbox) checkbox.checked = true;
        });
        this.saveToolStates();
        this.syncWithBackend();  // Sync with backend
        this.createToolToggles();
        window.visualLogger?.info('All tools enabled');
    }

    /**
     * Disable all tools and refresh UI
     */
    disableAllTools() {
        this.tools.forEach(tool => {
            this.disabledTools.add(tool.name);
            const checkbox = document.getElementById(`tool-${tool.name}`);
            if (checkbox) checkbox.checked = false;
        });
        this.saveToolStates();
        this.syncWithBackend();  // Sync with backend
        this.createToolToggles();
        window.visualLogger?.info('All tools disabled');
    }

    /**
     * Reset all tools to default (all enabled)
     */
    resetToDefaults() {
        this.enableAllTools();
    }

    /**
     * Refresh tool list from ToolRegistry and update UI
     * @returns {Promise<void>}
     * @async
     */
    async refresh() {
        window.logDebug('[ToolManager] refresh() called');
        if (window.ToolRegistry) {
            await window.ToolRegistry.waitForReady();
            this.tools = window.ToolRegistry.getTools();
            window.logDebug('[ToolManager] Got tools from registry:', this.tools);
            this.createToolToggles();
            window.visualLogger?.info('ToolManager refreshed with', this.tools.length, 'tools');
        } else {
            console.error('[ToolManager] ToolRegistry not available in refresh()');
        }
    }

    /**
     * Persist disabled tool names to localStorage
     * @private
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
     * Sync disabled tools list with backend
     * @private
     * @returns {Promise<void>}
     */
    async syncWithBackend() {
        try {
            const disabledToolsArray = Array.from(this.disabledTools);

            const response = window.backendConnector
                ? await window.backendConnector.fetch('api/tools/disabled', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Session-ID': window.sessionId || ''
                    },
                    body: JSON.stringify({
                        disabled_tools: disabledToolsArray
                    })
                })
                : await fetch('api/tools/disabled', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Session-ID': window.sessionId || ''
                    },
                    body: JSON.stringify({
                        disabled_tools: disabledToolsArray
                    })
                });

            const data = await response.json();

            if (data.status !== 'success') {
                console.error('Failed to sync disabled tools with backend:', data.error);
                window.visualLogger?.warn('Failed to sync tool state with backend:', data.error);
            } else {
                window.visualLogger?.debug(`Synced ${disabledToolsArray.length} disabled tools with backend`);
            }
        } catch (e) {
            console.error('Failed to sync with backend:', e);
            window.visualLogger?.warn('Failed to sync tool state with backend');
        }
    }

    /**
     * Load disabled tool names from localStorage
     * @private
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
     * Emit custom event for tool state changes
     * Other components can listen with window.addEventListener('toolStateChanged', callback)
     * @param {string} toolName - Tool that changed
     * @param {boolean} enabled - Whether tool is now enabled
     * @private
     */
    notifyToolStateChange(toolName, enabled) {
        window.dispatchEvent(new CustomEvent('toolStateChanged', {
            detail: { toolName, enabled, disabledTools: this.getDisabledTools() }
        }));
    }

    /**
     * Get current tool configuration for sending to server
     * @returns {{enabledTools: string[], disabledTools: string[], totalTools: number}} Configuration object
     */
    getToolConfiguration() {
        return {
            enabledTools: this.getEnabledTools(),
            disabledTools: this.getDisabledTools(),
            totalTools: this.tools.length
        };
    }
}

// Create global singleton instance
window.ToolManager = new ToolManager();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ToolManager;
}
