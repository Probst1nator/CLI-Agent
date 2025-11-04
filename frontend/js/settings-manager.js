/**
 * Settings Manager - Permission settings UI
 *
 * Manages the permissions settings panel that replaces the chat area temporarily.
 * Settings are stored per-profile in LLM profiles.
 */

class SettingsManager {
    constructor() {
        this.settingsPanel = null;
        this.settingsBtn = null;
        this.closeBtn = null;
        this.saveBtn = null;
        this.resetBtn = null;
        this.chatArea = null;
        this.mainContentArea = null;

        // Cached state to restore chat when closing
        this.cachedDisplay = {
            chatHTML: '',
            chatDisplayStyle: '',
            mainContentDisplayStyle: ''
        };

        // Available tools cache
        this.availableTools = [];
    }

    /**
     * Initialize the settings manager
     */
    async init() {
        // Get DOM elements
        this.settingsPanel = document.getElementById('settings-panel');
        this.settingsBtn = document.getElementById('settings-btn');
        this.closeBtn = document.getElementById('settings-close-btn');
        this.saveBtn = document.getElementById('settings-save-btn');
        this.resetBtn = document.getElementById('settings-reset-btn');
        this.chatArea = document.getElementById('content-wrapper');
        this.mainContentArea = document.querySelector('.main-content-area');

        if (!this.settingsPanel || !this.settingsBtn || !this.chatArea) {
            console.error('[SettingsManager] Required DOM elements not found');
            return;
        }

        // Bind events
        this.settingsBtn.addEventListener('click', () => this.toggle());
        this.closeBtn.addEventListener('click', () => this.close());
        this.saveBtn.addEventListener('click', () => this.save());
        this.resetBtn.addEventListener('click', () => this.reset());

        // Update rule counts on textarea input
        const allowTextarea = document.getElementById('allow-patterns');
        const askTextarea = document.getElementById('ask-patterns');
        const denyTextarea = document.getElementById('deny-patterns');

        allowTextarea.addEventListener('input', () => this.updateRuleCount('allow'));
        askTextarea.addEventListener('input', () => this.updateRuleCount('ask'));
        denyTextarea.addEventListener('input', () => this.updateRuleCount('deny'));

        console.log('[SettingsManager] Initialized');
    }

    /**
     * Toggle settings panel open/closed
     */
    toggle() {
        const isHidden = this.settingsPanel.classList.contains('hidden');
        if (isHidden) {
            this.open();
        } else {
            this.close();
        }
    }

    /**
     * Open the settings panel (replaces chat area)
     */
    async open() {
        console.log('[SettingsManager] Opening settings panel');

        // Cache current display state
        this.cachedDisplay.chatHTML = this.chatArea.innerHTML;
        this.cachedDisplay.chatDisplayStyle = this.chatArea.style.display;
        this.cachedDisplay.mainContentDisplayStyle = this.mainContentArea.style.display;

        // Hide main content area (chat + sidebars)
        this.mainContentArea.style.display = 'none';

        // Show settings panel
        this.settingsPanel.classList.remove('hidden');

        // Load current profile's permissions and tools
        await Promise.all([
            this.loadPermissions(),
            this.loadTools()
        ]);

        console.log('[SettingsManager] Settings panel opened');
    }

    /**
     * Close the settings panel (restore chat area)
     */
    close() {
        console.log('[SettingsManager] Closing settings panel');

        // Hide settings panel
        this.settingsPanel.classList.add('hidden');

        // Restore main content area
        this.mainContentArea.style.display = this.cachedDisplay.mainContentDisplayStyle || '';

        // Restore chat HTML from cache (no re-render needed)
        this.chatArea.innerHTML = this.cachedDisplay.chatHTML;
        this.chatArea.style.display = this.cachedDisplay.chatDisplayStyle || '';

        console.log('[SettingsManager] Settings panel closed, chat state restored');
    }

    /**
     * Load current profile's permissions from backend
     */
    async loadPermissions() {
        try {
            const url = window.backendConnector
                ? await window.backendConnector.buildURL('/api/permissions')
                : '/api/permissions';

            const response = await fetch(url);
            const data = await response.json();

            if (data.status !== 'success') {
                throw new Error(data.error || 'Failed to load permissions');
            }

            const permissions = data.permissions || { allow: [], deny: [], ask: [] };

            // Populate textareas
            document.getElementById('allow-patterns').value = permissions.allow.join('\n');
            document.getElementById('ask-patterns').value = permissions.ask.join('\n');
            document.getElementById('deny-patterns').value = permissions.deny.join('\n');

            // Update counts
            this.updateRuleCount('allow');
            this.updateRuleCount('ask');
            this.updateRuleCount('deny');

            console.log(`[SettingsManager] Loaded permissions for profile: ${data.profile_id || 'none'}`);
        } catch (error) {
            console.error('[SettingsManager] Failed to load permissions:', error);
            this.showError(`Failed to load permissions: ${error.message}`);
        }
    }

    /**
     * Load available tools from backend
     */
    async loadTools() {
        try {
            const url = window.backendConnector
                ? await window.backendConnector.buildURL('/api/tools')
                : '/api/tools';

            const response = await fetch(url);
            const data = await response.json();

            if (data.status !== 'success') {
                throw new Error(data.error || 'Failed to load tools');
            }

            // Filter out system tools that should not be configurable
            const systemTools = ['visualresponse', 'think', 'response', 'todos'];
            this.availableTools = (data.tools || []).filter(tool =>
                !systemTools.includes(tool.name.toLowerCase())
            );

            // Render tool grid
            this.renderToolGrid();

            console.log(`[SettingsManager] Loaded ${this.availableTools.length} tools`);
        } catch (error) {
            console.error('[SettingsManager] Failed to load tools:', error);
            this.showError(`Failed to load tools: ${error.message}`);
        }
    }

    /**
     * Render the tool permission grid
     */
    renderToolGrid() {
        const grid = document.getElementById('tools-permission-grid');

        if (this.availableTools.length === 0) {
            grid.innerHTML = '<div class="tools-grid-empty">No tools available</div>';
            return;
        }

        // Get current permissions
        const allowPatterns = document.getElementById('allow-patterns').value.split('\n').filter(Boolean);
        const askPatterns = document.getElementById('ask-patterns').value.split('\n').filter(Boolean);
        const denyPatterns = document.getElementById('deny-patterns').value.split('\n').filter(Boolean);

        // Build tool items
        const toolItems = this.availableTools.map(tool => {
            const toolName = tool.name.toLowerCase();

            // Check if tool has explicit permission
            const isAllowed = allowPatterns.some(p => p.toLowerCase() === toolName);
            const isAsked = askPatterns.some(p => p.toLowerCase() === toolName);
            const isDenied = denyPatterns.some(p => p.toLowerCase() === toolName);

            // Determine current state (deny > ask > allow)
            // If not in any list, default to 'ask' (which is the system default behavior)
            let currentState = 'ask';
            if (isDenied) currentState = 'deny';
            else if (isAsked) currentState = 'ask';
            else if (isAllowed) currentState = 'allow';

            return `
                <div class="tool-permission-item" data-tool="${toolName}">
                    <div class="tool-name">${tool.displayName || tool.name}</div>
                    <div class="tool-description">${tool.description.substring(0, 80)}${tool.description.length > 80 ? '...' : ''}</div>
                    <div class="tool-permission-controls">
                        <label class="permission-radio">
                            <input type="radio" name="perm-${toolName}" value="allow" ${currentState === 'allow' ? 'checked' : ''}>
                            <span class="permission-label allow">Allow</span>
                        </label>
                        <label class="permission-radio">
                            <input type="radio" name="perm-${toolName}" value="ask" ${currentState === 'ask' ? 'checked' : ''}>
                            <span class="permission-label ask">Ask</span>
                        </label>
                        <label class="permission-radio">
                            <input type="radio" name="perm-${toolName}" value="deny" ${currentState === 'deny' ? 'checked' : ''}>
                            <span class="permission-label deny">Deny</span>
                        </label>
                    </div>
                </div>
            `;
        }).join('');

        grid.innerHTML = toolItems;

        // Add event listeners for radio buttons
        const radioButtons = grid.querySelectorAll('input[type="radio"]');
        radioButtons.forEach(radio => {
            radio.addEventListener('change', (e) => this.handleToolPermissionChange(e));
        });
    }

    /**
     * Handle tool permission radio button change
     */
    handleToolPermissionChange(event) {
        const radio = event.target;
        const toolName = radio.name.replace('perm-', '');
        const permission = radio.value;

        // Get current patterns
        const allowPatterns = document.getElementById('allow-patterns').value.split('\n').filter(Boolean);
        const askPatterns = document.getElementById('ask-patterns').value.split('\n').filter(Boolean);
        const denyPatterns = document.getElementById('deny-patterns').value.split('\n').filter(Boolean);

        // Remove tool from all lists
        const removeFromList = (list) => list.filter(p => p.toLowerCase() !== toolName);
        let newAllow = removeFromList(allowPatterns);
        let newAsk = removeFromList(askPatterns);
        let newDeny = removeFromList(denyPatterns);

        // Add to appropriate list
        if (permission === 'allow') {
            newAllow.push(toolName);
        } else if (permission === 'ask') {
            newAsk.push(toolName);
        } else if (permission === 'deny') {
            newDeny.push(toolName);
        }

        // Update textareas
        document.getElementById('allow-patterns').value = newAllow.join('\n');
        document.getElementById('ask-patterns').value = newAsk.join('\n');
        document.getElementById('deny-patterns').value = newDeny.join('\n');

        // Update counts
        this.updateRuleCount('allow');
        this.updateRuleCount('ask');
        this.updateRuleCount('deny');
    }

    /**
     * Update rule count display
     */
    updateRuleCount(type) {
        const textarea = document.getElementById(`${type}-patterns`);
        const countSpan = document.getElementById(`${type}-count`);

        if (!textarea || !countSpan) return;

        const lines = textarea.value.split('\n').filter(line => line.trim().length > 0);
        countSpan.textContent = lines.length;
    }

    /**
     * Save permissions to backend
     */
    async save() {
        try {
            console.log('[SettingsManager] Saving permissions...');

            // Parse textareas into arrays
            const allow = document.getElementById('allow-patterns').value
                .split('\n')
                .map(line => line.trim())
                .filter(Boolean);
            const ask = document.getElementById('ask-patterns').value
                .split('\n')
                .map(line => line.trim())
                .filter(Boolean);
            const deny = document.getElementById('deny-patterns').value
                .split('\n')
                .map(line => line.trim())
                .filter(Boolean);

            const url = window.backendConnector
                ? await window.backendConnector.buildURL('/api/permissions')
                : '/api/permissions';

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-ID': window.sessionId || ''
                },
                body: JSON.stringify({ allow, ask, deny })
            });

            const data = await response.json();

            if (data.status !== 'success') {
                throw new Error(data.error || 'Failed to save permissions');
            }

            console.log('[SettingsManager] Permissions saved successfully');
            this.showSuccess('Permissions saved successfully!');

            // Close settings panel after save
            setTimeout(() => this.close(), 1000);
        } catch (error) {
            console.error('[SettingsManager] Failed to save permissions:', error);
            this.showError(`Failed to save permissions: ${error.message}`);
        }
    }

    /**
     * Reset permissions to default template
     */
    async reset() {
        if (!confirm('Reset permissions to default template? This will overwrite current settings.')) {
            return;
        }

        try {
            console.log('[SettingsManager] Resetting permissions to default template...');

            const url = window.backendConnector
                ? await window.backendConnector.buildURL('/api/permissions/reset')
                : '/api/permissions/reset';

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'X-Session-ID': window.sessionId || ''
                }
            });

            const data = await response.json();

            if (data.status !== 'success') {
                throw new Error(data.error || 'Failed to reset permissions');
            }

            console.log('[SettingsManager] Permissions reset to default template');

            // Reload permissions
            await this.loadPermissions();
            this.renderToolGrid();

            this.showSuccess('Permissions reset to default template!');
        } catch (error) {
            console.error('[SettingsManager] Failed to reset permissions:', error);
            this.showError(`Failed to reset permissions: ${error.message}`);
        }
    }

    /**
     * Show success message
     */
    showSuccess(message) {
        // Simple alert for now - could be replaced with a toast notification
        console.log('[SettingsManager] Success:', message);
        // Show temporary message in header
        const originalText = this.saveBtn.textContent;
        this.saveBtn.textContent = '✓ Saved!';
        this.saveBtn.classList.add('success-flash');
        setTimeout(() => {
            this.saveBtn.textContent = originalText;
            this.saveBtn.classList.remove('success-flash');
        }, 2000);
    }

    /**
     * Show error message
     */
    showError(message) {
        console.error('[SettingsManager] Error:', message);
        alert(`Error: ${message}`);
    }
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', async () => {
        window.settingsManager = new SettingsManager();
        await window.settingsManager.init();
    });
} else {
    // DOM already loaded
    window.settingsManager = new SettingsManager();
    window.settingsManager.init();
}
