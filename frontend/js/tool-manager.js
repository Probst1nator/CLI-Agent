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
    }

    /**
     * Fallback to empty tool list if ToolRegistry is unavailable
     * Tools will be discovered dynamically as they are used
     */
    fallbackToDefaultTools() {
        this.tools = [];
        window.visualLogger?.warn('ToolRegistry unavailable - tool manager will remain empty (tools discovered dynamically)');
    }

    /**
     * Create the tool toggle checkboxes
     */
    createToolToggles() {
        console.log('[ToolManager] createToolToggles called with', this.tools.length, 'tools');
        console.log('[ToolManager] containerElement:', this.containerElement);
        
        if (!this.containerElement) {
            console.error('[ToolManager] No container element found!');
            return;
        }

        // Clear existing content
        this.containerElement.innerHTML = '';

        // Add "Tools:" header as first item
        const headerItem = document.createElement('div');
        headerItem.className = 'tool-toggle-item';
        headerItem.textContent = 'Tools:';
        this.containerElement.appendChild(headerItem);

        console.log('[ToolManager] About to create toggles for tools:', this.tools.map(t => t.name));
        
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
     * Refresh tool list from ToolRegistry
     */
    async refresh() {
        console.log('[ToolManager] refresh() called');
        if (window.ToolRegistry) {
            await window.ToolRegistry.waitForReady();
            this.tools = window.ToolRegistry.getTools();
            console.log('[ToolManager] Got tools from registry:', this.tools);
            this.createToolToggles();
            window.visualLogger?.info('ToolManager refreshed with', this.tools.length, 'tools');
        } else {
            console.error('[ToolManager] ToolRegistry not available in refresh()');
        }
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