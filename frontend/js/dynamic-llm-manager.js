/**
 * DynamicLLMManager - Manages dynamic LLM model selection with fallback chains
 *
 * Features:
 * - Add/remove LLM selectors dynamically for each role (override, router, planner, fallbacks)
 * - Random model selection when adding new selectors
 * - Remove selector when "None" is selected
 * - Persist configuration to backend
 * - Sync with backend on load
 */

class DynamicLLMManager {
    constructor() {
        this.availableModels = [];
        this.roleContainers = {
            override: document.getElementById('override-llm-selectors-container'),
            router: document.getElementById('router-llm-selectors-container'),
            planner: document.getElementById('planner-llm-selectors-container'),
            fallbacks: document.getElementById('fallbacks-llm-selectors-container')
        };

        // Track selector instances for each role
        this.selectors = {
            override: [],
            router: [],
            planner: [],
            fallbacks: []
        };

        // Counter for unique IDs
        this.selectorIdCounter = 0;

        this.logger = window.visualLogger;
        this.logger.info('[DynamicLLMManager] Initializing...');

        this.init();
    }

    /**
     * Initialize the manager
     */
    init() {
        this.setupAddButtons();
        this.loadConfiguration();
    }

    /**
     * Setup event listeners for "+" buttons
     */
    setupAddButtons() {
        const addButtons = document.querySelectorAll('.add-llm-selector-btn');
        addButtons.forEach(button => {
            const role = button.getAttribute('data-role');
            button.addEventListener('click', () => this.addSelector(role));
        });

        this.logger.info('[DynamicLLMManager] Add buttons configured');
    }

    /**
     * Load configuration from backend
     */
    async loadConfiguration() {
        try {
            const url = window.backendConnector
                ? await window.backendConnector.buildURL('/api/llms')
                : '/api/llms';

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);

            try {
                const response = await fetch(url, {
                    signal: controller.signal,
                    headers: { 'X-Session-ID': window.sessionId || '' }
                });
                clearTimeout(timeoutId);

                const data = await response.json();

                if (data.status === 'success') {
                    this.availableModels = data.models || [];
                    const config = data.current_configuration || {};

                    // Debug: Log what we received
                    console.log('[DynamicLLMManager] Available models:', this.availableModels.length);
                    console.log('[DynamicLLMManager] Config received:', config);
                    console.log('[DynamicLLMManager] override_llms:', config.override_llms);
                    console.log('[DynamicLLMManager] router_llms:', config.router_llms);
                    console.log('[DynamicLLMManager] planner_llms:', config.planner_llms);
                    console.log('[DynamicLLMManager] selected_llms:', config.selected_llms);

                    // Load multi-model arrays (new format)
                    this.renderSelectors('override', config.override_llms || []);
                    this.renderSelectors('router', config.router_llms || []);
                    this.renderSelectors('planner', config.planner_llms || []);
                    this.renderSelectors('fallbacks', config.selected_llms || []);

                    this.logger.info('[DynamicLLMManager] Configuration loaded successfully');
                } else {
                    throw new Error(data.error || 'Failed to load configuration');
                }
            } catch (error) {
                clearTimeout(timeoutId);
                throw error;
            }
        } catch (error) {
            this.logger.error('[DynamicLLMManager] Failed to load configuration:', error);
            // Initialize with empty selectors if loading fails
            this.renderSelectors('fallbacks', [this.getRandomModel()]);
        }
    }

    /**
     * Render selectors for a specific role
     * @param {string} role - Role name (override, router, planner, fallbacks)
     * @param {Array<string>} models - Array of model IDs to render
     */
    renderSelectors(role, models) {
        console.log(`[DynamicLLMManager] renderSelectors called for ${role} with models:`, models);

        const container = this.roleContainers[role];
        if (!container) {
            this.logger.warn(`[DynamicLLMManager] Container not found for role: ${role}`);
            console.error(`[DynamicLLMManager] Container #${role}-llm-selectors-container not found in DOM`);
            return;
        }

        console.log(`[DynamicLLMManager] Container found for ${role}:`, container);

        // Clear existing selectors
        container.innerHTML = '';
        this.selectors[role] = [];

        // Add at least one selector for each role, or all configured models
        const modelsToRender = models.length > 0 ? models : [this.getRandomModel()];
        console.log(`[DynamicLLMManager] Models to render for ${role}:`, modelsToRender);

        modelsToRender.forEach((modelId, index) => {
            console.log(`[DynamicLLMManager] Creating selector ${index} for ${role} with model: ${modelId}`);
            this.createSelector(role, modelId, index);
        });

        this.logger.info(`[DynamicLLMManager] Rendered ${modelsToRender.length} selectors for ${role}`);
    }

    /**
     * Create a new selector for a role
     * @param {string} role - Role name
     * @param {string} selectedModel - Model ID to select (or null for random)
     * @param {number} index - Index position for label
     */
    createSelector(role, selectedModel = null, index = null) {
        const container = this.roleContainers[role];
        if (!container) {
            this.logger.warn(`[DynamicLLMManager] Container not found for role: ${role}`);
            return;
        }

        // Generate unique ID for this selector
        const selectorId = `llm-selector-${role}-${this.selectorIdCounter++}`;

        // Create wrapper div
        const wrapper = document.createElement('div');
        wrapper.className = 'dynamic-llm-selector-item';
        wrapper.setAttribute('data-role', role);
        wrapper.setAttribute('data-selector-id', selectorId);

        // Add label for fallbacks role
        if (role === 'fallbacks') {
            const label = document.createElement('span');
            label.className = 'llm-fallback-label';
            const position = index !== null ? index + 1 : this.selectors[role].length + 1;
            label.textContent = this.getOrdinalLabel(position);
            wrapper.appendChild(label);
        }

        // Create selector container div
        const selectorDiv = document.createElement('div');
        selectorDiv.id = selectorId;
        wrapper.appendChild(selectorDiv);

        // Append to container
        container.appendChild(wrapper);

        // Use selected model or random if not provided
        const modelToSelect = selectedModel || this.getRandomModel();

        // Initialize SearchableLLMSelectbox
        console.log(`[DynamicLLMManager] Checking for SearchableLLMSelectbox:`, typeof window.SearchableLLMSelectbox);
        console.log(`[DynamicLLMManager] Available models count:`, this.availableModels.length);
        console.log(`[DynamicLLMManager] Model to select:`, modelToSelect);
        console.log(`[DynamicLLMManager] selectorDiv element:`, selectorDiv);

        if (window.SearchableLLMSelectbox) {
            console.log(`[DynamicLLMManager] Creating SearchableLLMSelectbox for ${selectorId}`);

            // Create selectbox with correct constructor signature
            const selectbox = new window.SearchableLLMSelectbox(selectorDiv, {
                placeholder: `Select ${role} LLM...`,
                id: selectorId,
                initialModels: this.availableModels,
                includeNoneOption: true,
                onSelectionChange: (newModel) => this.onSelectionChange(role, selectorId, newModel)
            });

            // Set the initial selection
            if (modelToSelect) {
                selectbox.selectModel(modelToSelect);
            }

            this.selectors[role].push({
                id: selectorId,
                selectbox: selectbox,
                wrapper: wrapper
            });

            console.log(`[DynamicLLMManager] Successfully created selector ${selectorId} for ${role}`);
            this.logger.debug(`[DynamicLLMManager] Created selector ${selectorId} for ${role}`);
        } else {
            console.error('[DynamicLLMManager] SearchableLLMSelectbox class not found on window object');
            this.logger.error('[DynamicLLMManager] SearchableLLMSelectbox class not found');
        }
    }

    /**
     * Add a new selector to a role
     * @param {string} role - Role name
     */
    addSelector(role) {
        this.createSelector(role, this.getRandomModel(), this.selectors[role].length);
        this.saveConfiguration(role);
        this.logger.info(`[DynamicLLMManager] Added new selector to ${role}`);
    }

    /**
     * Remove a selector from a role
     * @param {string} role - Role name
     * @param {string} selectorId - Selector ID to remove
     */
    removeSelector(role, selectorId) {
        const selectorIndex = this.selectors[role].findIndex(s => s.id === selectorId);
        if (selectorIndex === -1) {
            this.logger.warn(`[DynamicLLMManager] Selector not found: ${selectorId}`);
            return;
        }

        // Prevent removing the last fallback model (must have at least one default)
        if (role === 'fallbacks' && this.selectors[role].length === 1) {
            console.warn('[DynamicLLMManager] Cannot remove last fallback model - at least one default model required');
            alert('You must have at least one default model. Please select a different model instead of removing this one.');
            return;
        }

        const selector = this.selectors[role][selectorIndex];

        // Remove from DOM
        if (selector.wrapper && selector.wrapper.parentNode) {
            selector.wrapper.parentNode.removeChild(selector.wrapper);
        }

        // Remove from tracking array
        this.selectors[role].splice(selectorIndex, 1);

        // Update labels for fallbacks
        if (role === 'fallbacks') {
            this.updateFallbackLabels();
        }

        this.saveConfiguration(role);
        this.logger.info(`[DynamicLLMManager] Removed selector ${selectorId} from ${role}`);
    }

    /**
     * Handle selection change event
     * @param {string} role - Role name
     * @param {string} selectorId - Selector ID
     * @param {Object|null} newModel - New model object or null for "None"
     */
    onSelectionChange(role, selectorId, newModel) {
        console.log(`[DynamicLLMManager] Selection changed for ${role}:`, newModel);
        this.logger.debug(`[DynamicLLMManager] Selection changed: ${role} - ${selectorId}`, newModel);

        // If "None" selected, remove the selector
        // Check multiple conditions: model is null, model.id is null, or isNoneOption flag
        if (newModel === null || (newModel && (newModel.id === null || newModel.isNoneOption === true))) {
            console.log(`[DynamicLLMManager] Removing selector ${selectorId} for ${role}`);
            this.removeSelector(role, selectorId);
        } else {
            console.log(`[DynamicLLMManager] Saving configuration for ${role}`);
            this.saveConfiguration(role);
        }
    }

    /**
     * Save configuration to backend
     * @param {string} role - Role name to save
     */
    async saveConfiguration(role) {
        try {
            // Collect all model selections for this role
            const models = this.selectors[role]
                .map(s => {
                    const model = s.selectbox.getSelectedModel();
                    return model ? model.id : null;
                })
                .filter(m => m && m !== 'none');

            const url = window.backendConnector
                ? await window.backendConnector.buildURL('/api/llms/config')
                : '/api/llms/config';

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-ID': window.sessionId || ''
                },
                body: JSON.stringify({
                    role: role,
                    models: models
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.logger.info(`[DynamicLLMManager] Configuration saved for ${role}:`, models);
            } else {
                throw new Error(data.error || 'Failed to save configuration');
            }
        } catch (error) {
            this.logger.error(`[DynamicLLMManager] Failed to save configuration for ${role}:`, error);
        }
    }

    /**
     * Get a random model from available models
     * @returns {string} Random model ID
     */
    getRandomModel() {
        if (this.availableModels.length === 0) {
            return 'gemini-flash-latest'; // Fallback default
        }
        const randomIndex = Math.floor(Math.random() * this.availableModels.length);
        return this.availableModels[randomIndex].id;
    }

    /**
     * Get ordinal label for fallback position
     * @param {number} position - Position number (1-based)
     * @returns {string} Ordinal label (1st, 2nd, 3rd, etc.)
     */
    getOrdinalLabel(position) {
        const suffixes = ['th', 'st', 'nd', 'rd'];
        const value = position % 100;
        const suffix = suffixes[(value - 20) % 10] || suffixes[value] || suffixes[0];
        return position + suffix;
    }

    /**
     * Update fallback labels after adding/removing selectors
     */
    updateFallbackLabels() {
        this.selectors.fallbacks.forEach((selector, index) => {
            const label = selector.wrapper.querySelector('.llm-fallback-label');
            if (label) {
                label.textContent = this.getOrdinalLabel(index + 1);
            }
        });
    }
}

// Export to window for global access
window.DynamicLLMManager = DynamicLLMManager;

// Log module load
if (window.visualLogger) {
    window.visualLogger.info('[dynamic-llm-manager.js] Module loaded');
}
