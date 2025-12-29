/**
 * DynamicLLMManager - Manages dynamic LLM model selection with fallback chains
 *
 * Features:
 * - Add/remove LLM selectors dynamically for each role (override, router, planner, fallbacks)
 * - Empty selection by default (user must explicitly choose models)
 * - Remove selector when "None" is selected
 * - Persist configuration to backend
 * - Sync with backend on load
 *
 * @class DynamicLLMManager
 * @author CLI-Agent Development Team
 * @version 1.0.0
 */

/**
 * @typedef {Object} LLMConfig
 * @property {Array<string>} override_llms - Override LLM model IDs
 * @property {Array<string>} autocomplete_llms - Autocomplete LLM model IDs
 * @property {Array<string>} router_llms - Router LLM model IDs
 * @property {Array<string>} planner_llms - Planner LLM model IDs
 * @property {Array<string>} selected_llms - Fallback LLM model IDs (in priority order)
 */

/**
 * @typedef {Object} SelectorInstance
 * @property {string} id - Unique selector ID
 * @property {SearchableLLMSelectbox} selectbox - Selectbox instance
 * @property {HTMLElement} wrapper - Wrapper element containing selector
 */

/**
 * @typedef {Object} RoleContainers
 * @property {HTMLElement|null} override - Container for override LLM selectors
 * @property {HTMLElement|null} autocomplete - Container for autocomplete LLM selectors
 * @property {HTMLElement|null} router - Container for router LLM selectors
 * @property {HTMLElement|null} planner - Container for planner LLM selectors
 * @property {HTMLElement|null} fallbacks - Container for fallback LLM selectors
 */

/**
 * @typedef {Object} RoleSelectors
 * @property {Array<SelectorInstance>} override - Override selector instances
 * @property {Array<SelectorInstance>} autocomplete - Autocomplete selector instances
 * @property {Array<SelectorInstance>} router - Router selector instances
 * @property {Array<SelectorInstance>} planner - Planner selector instances
 * @property {Array<SelectorInstance>} fallbacks - Fallback selector instances
 */

class DynamicLLMManager {
    /**
     * Initialize the dynamic LLM manager
     */
    constructor() {
        /**
         * Available LLM models from server
         * @type {Array<Object>}
         */
        this.availableModels = [];

        /**
         * Container elements for each role
         * @type {RoleContainers}
         */
        this.roleContainers = {
            override: document.getElementById('override-llm-selectors-container'),
            autocomplete: document.getElementById('autocomplete-llm-selectors-container'),
            router: document.getElementById('router-llm-selectors-container'),
            planner: document.getElementById('planner-llm-selectors-container'),
            fallbacks: document.getElementById('fallbacks-llm-selectors-container')
        };

        /**
         * Track selector instances for each role
         * @type {RoleSelectors}
         */
        this.selectors = {
            override: [],
            autocomplete: [],
            router: [],
            planner: [],
            fallbacks: []
        };

        /**
         * Counter for unique IDs
         * @type {number}
         */
        this.selectorIdCounter = 0;

        /**
         * Flag to prevent saving during initialization
         * @type {boolean}
         */
        this.isInitializing = true;

        /**
         * Pending save requests by role (to debounce rapid changes)
         * @type {Object<string, number>}
         */
        this.pendingSaveTimeouts = {};

        /**
         * Logger instance
         * @type {Object}
         */
        this.logger = window.visualLogger;

        this.logger.info('[DynamicLLMManager] Initializing...');

        this.init();
    }

    /**
     * Initialize the manager
     * Sets up add buttons and loads configuration from backend
     * @returns {void}
     */
    init() {
        this.setupAddButtons();
        this.loadConfiguration();
    }

    /**
     * Setup event listeners for "+" buttons
     * @returns {void}
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
     * Fetches available models and current LLM configuration, then renders selectors
     * @async
     * @returns {Promise<void>}
     */
    async loadConfiguration() {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);

            try {
                const response = await fetch('api/llms', {
                    signal: controller.signal,
                    headers: { 'X-Session-ID': window.sessionId || '' }
                });
                clearTimeout(timeoutId);

                const data = await response.json();

                if (data.status === 'success') {
                    this.availableModels = data.models || [];
                    const config = data.current_configuration || {};

                    // Debug: Log what we received
                    window.logDebug('[DynamicLLMManager] Available models:', this.availableModels.length);
                    window.logDebug('[DynamicLLMManager] Config received:', config);
                    window.logDebug('[DynamicLLMManager] override_llms:', config.override_llms);
                    window.logDebug('[DynamicLLMManager] autocomplete_llms:', config.autocomplete_llms);
                    window.logDebug('[DynamicLLMManager] router_llms:', config.router_llms);
                    window.logDebug('[DynamicLLMManager] planner_llms:', config.planner_llms);
                    window.logDebug('[DynamicLLMManager] selected_llms:', config.selected_llms);

                    // Load multi-model arrays (new format)
                    this.renderSelectors('override', config.override_llms || []);
                    this.renderSelectors('autocomplete', config.autocomplete_llms || []);
                    this.renderSelectors('router', config.router_llms || []);
                    this.renderSelectors('planner', config.planner_llms || []);
                    this.renderSelectors('fallbacks', config.selected_llms || []);

                    // Initialization complete - allow saves from now on
                    this.isInitializing = false;

                    // Notify listeners about initial state after loading configuration
                    this.notifySendButtonUpdateListeners();

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
            // Initialize with no selectors if loading fails - user must select models
            this.renderSelectors('fallbacks', []);
            this.renderSelectors('router', []);
            this.renderSelectors('planner', []);
            this.renderSelectors('autocomplete', []);
            this.renderSelectors('override', []);
            // Initialization complete even if failed - allow saves from now on
            this.isInitializing = false;

            // Notify listeners about initial state even on error
            this.notifySendButtonUpdateListeners();
        }
    }

    /**
     * Render selectors for a specific role
     * @param {string} role - Role name (override, router, planner, fallbacks)
     * @param {Array<string>} models - Array of model IDs to render
     * @returns {void}
     */
    renderSelectors(role, models) {
        window.logDebug(`[DynamicLLMManager] renderSelectors called for ${role} with models:`, models);

        const container = this.roleContainers[role];
        if (!container) {
            this.logger.warn(`[DynamicLLMManager] Container not found for role: ${role}`);
            console.error(`[DynamicLLMManager] Container #${role}-llm-selectors-container not found in DOM`);
            return;
        }

        window.logDebug(`[DynamicLLMManager] Container found for ${role}:`, container);

        // Clear existing selectors
        container.innerHTML = '';
        this.selectors[role] = [];

        // Determine which models to render:
        // Only render configured models. Empty if config is empty.
        // User must explicitly select models - no randomization or auto-selection.
        let modelsToRender = [];

        if (models.length > 0) {
            // Configuration has models for this role - use them
            modelsToRender = models;
        }
        // If config is empty, render nothing - user must select models

        window.logDebug(`[DynamicLLMManager] Models to render for ${role}:`, modelsToRender);

        modelsToRender.forEach((modelId, index) => {
            window.logDebug(`[DynamicLLMManager] Creating selector ${index} for ${role} with model: ${modelId}`);
            this.createSelector(role, modelId, index);
        });

        this.logger.info(`[DynamicLLMManager] Rendered ${modelsToRender.length} selectors for ${role}`);
    }

    /**
     * Create a new selector for a role
     * @param {string} role - Role name (override, router, planner, fallbacks)
     * @param {string|null} [selectedModel=null] - Model ID to select (or null for random)
     * @param {number|null} [index=null] - Index position for label
     * @returns {void}
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

        // Use selected model if provided, otherwise null (empty selection)
        const modelToSelect = selectedModel || null;

        // Initialize SearchableLLMSelectbox
        window.logDebug(`[DynamicLLMManager] Checking for SearchableLLMSelectbox:`, typeof window.SearchableLLMSelectbox);
        window.logDebug(`[DynamicLLMManager] Available models count:`, this.availableModels.length);
        window.logDebug(`[DynamicLLMManager] Model to select:`, modelToSelect);
        window.logDebug(`[DynamicLLMManager] selectorDiv element:`, selectorDiv);

        if (window.SearchableLLMSelectbox) {
            window.logDebug(`[DynamicLLMManager] Creating SearchableLLMSelectbox for ${selectorId}`);

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

            window.logDebug(`[DynamicLLMManager] Successfully created selector ${selectorId} for ${role}`);
            this.logger.debug(`[DynamicLLMManager] Created selector ${selectorId} for ${role}`);
        } else {
            console.error('[DynamicLLMManager] SearchableLLMSelectbox class not found on window object');
            this.logger.error('[DynamicLLMManager] SearchableLLMSelectbox class not found');
        }
    }

    /**
     * Add a new selector to a role (with empty selection - user must choose model)
     * @param {string} role - Role name (override, autocomplete, router, planner, fallbacks)
     * @returns {void}
     */
    addSelector(role) {
        // Don't pass a selected model - user must choose
        this.createSelector(role, null, this.selectors[role].length);
        this.saveConfiguration(role);
        this.logger.info(`[DynamicLLMManager] Added new selector to ${role}`);
    }

    /**
     * Remove a selector from a role
     * @param {string} role - Role name
     * @param {string} selectorId - Selector ID to remove
     * @returns {void}
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

        // Notify listeners about the removal (for send button state update)
        this.notifySendButtonUpdateListeners();
    }

    /**
     * Handle selection change event
     * @param {string} role - Role name
     * @param {string} selectorId - Selector ID
     * @param {Object|null} newModel - New model object or null for "None"
     * @returns {void}
     */
    onSelectionChange(role, selectorId, newModel) {
        window.logDebug(`[DynamicLLMManager] Selection changed for ${role}:`, newModel);
        this.logger.debug(`[DynamicLLMManager] Selection changed: ${role} - ${selectorId}`, newModel);

        // If "None" selected, remove the selector
        // Check multiple conditions: model is null, model.id is null, or isNoneOption flag
        if (newModel === null || (newModel && (newModel.id === null || newModel.isNoneOption === true))) {
            window.logDebug(`[DynamicLLMManager] Removing selector ${selectorId} for ${role}`);
            this.removeSelector(role, selectorId);
        } else {
            window.logDebug(`[DynamicLLMManager] Saving configuration for ${role}`);
            this.saveConfiguration(role);
        }

        // Notify listeners about the selection change (for send button state update)
        this.notifySendButtonUpdateListeners();
    }

    /**
     * Save configuration to backend
     * @async
     * @param {string} role - Role name to save
     * @returns {Promise<void>}
     */
    async saveConfiguration(role) {
        // Skip saving during initialization to prevent overwriting loaded config
        if (this.isInitializing) {
            window.logDebug(`[DynamicLLMManager] Skipping save during initialization for ${role}`);
            return;
        }

        // Skip saving during profile loading to prevent overwriting other profiles
        if (window.app && window.app.isLoadingProfile) {
            window.logDebug(`[DynamicLLMManager] Skipping save during profile load for ${role}`);
            return;
        }

        // Debounce: Clear any pending save for this role and schedule a new one
        // This prevents spamming the backend when user makes rapid changes
        if (this.pendingSaveTimeouts[role]) {
            clearTimeout(this.pendingSaveTimeouts[role]);
        }

        this.pendingSaveTimeouts[role] = setTimeout(async () => {
            try {
                // Collect all model selections for this role
                const models = this.selectors[role]
                    .map(s => {
                        const model = s.selectbox.getSelectedModel();
                        return model ? model.id : null;
                    })
                    .filter(m => m && m !== 'none');

            const response = await fetch('api/llms/config', {
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

                // Update active profile in frontend state
                this.updateActiveProfileState(role, models);

                // Trigger auto-save (with debouncing) to persist changes and check for empty profiles
                if (window.app && window.app.scheduleAutoSave) {
                    window.logDebug(`[DynamicLLMManager] Calling scheduleAutoSave() after ${role} update`);
                    window.app.scheduleAutoSave();
                } else {
                    console.warn('[DynamicLLMManager] window.app.scheduleAutoSave not available');
                }
            } else {
                throw new Error(data.error || 'Failed to save configuration');
            }
            } catch (error) {
                this.logger.error(`[DynamicLLMManager] Failed to save configuration for ${role}:`, error);
            } finally {
                delete this.pendingSaveTimeouts[role];
            }
        }, 300); // 300ms debounce - wait for rapid changes to settle
    }

    /**
     * Update currently loaded profile state in the frontend after config change
     * @param {string} role - Role that was updated (router, planner, override, fallbacks)
     * @param {Array<string>} models - New model(s) for this role
     */
    updateActiveProfileState(role, models) {
        if (!window.app) {
            console.warn('[DynamicLLMManager] window.app not available in updateActiveProfileState');
            return;
        }

        // Update currently loaded profile (not always profile 1)
        const profileId = window.app.currentLoadedProfile;
        console.log(`[updateActiveProfileState] Updating profile ${profileId} for role ${role}:`, models);

        const profile = window.app.profileSlots[profileId];

        if (!profile) {
            // Create profile if it doesn't exist yet
            console.log(`[updateActiveProfileState] Profile ${profileId} doesn't exist, creating it`);
            window.app.profileSlots[profileId] = {
                id: profileId,
                override: [],
                router: [],
                planner: [],
                fallbacks: []
            };
        }

        // Update the profile with new role configuration (store as arrays)
        if (role === 'router') {
            window.app.profileSlots[profileId].router = models;
        } else if (role === 'planner') {
            window.app.profileSlots[profileId].planner = models;
        } else if (role === 'override') {
            window.app.profileSlots[profileId].override = models;
        } else if (role === 'fallbacks') {
            window.app.profileSlots[profileId].fallbacks = models;
        }

        console.log(`[updateActiveProfileState] Profile ${profileId} after update:`, window.app.profileSlots[profileId]);

        // Update the profile slot UI to refresh tooltips
        window.app.updateProfileSlotUI();

        this.logger.info(`[DynamicLLMManager] Updated profile '${profileId}' with ${role}:`, models);
    }

    /**
     * Get current configuration for all roles
     * @returns {Object} Current config with override, autocomplete, router, planner, fallbacks arrays
     */
    getCurrentConfig() {
        const getModelsForRole = (role) => {
            return this.selectors[role]
                .map(s => {
                    const model = s.selectbox.getSelectedModel();
                    return model ? model.id : null;
                })
                .filter(m => m && m !== 'none');
        };

        return {
            override: getModelsForRole('override'),
            autocomplete: getModelsForRole('autocomplete'),
            router: getModelsForRole('router'),
            planner: getModelsForRole('planner'),
            fallbacks: getModelsForRole('fallbacks')
        };
    }

    /**
     * Check if any LLM is selected across all roles
     * Returns true if at least one model is selected in any role
     * @returns {boolean} True if any LLM is selected
     */
    hasAnyLLMSelected() {
        const config = this.getCurrentConfig();
        return config.override.length > 0 ||
               config.autocomplete.length > 0 ||
               config.router.length > 0 ||
               config.planner.length > 0 ||
               config.fallbacks.length > 0;
    }

    /**
     * Register a callback to be called when LLM selection changes
     * This is used to update the send button state
     * @param {function():void} callback - Callback function to call on selection change
     * @returns {void}
     */
    addSendButtonUpdateListener(callback) {
        if (!this.sendButtonUpdateCallbacks) {
            this.sendButtonUpdateCallbacks = [];
        }
        this.sendButtonUpdateCallbacks.push(callback);
    }

    /**
     * Notify registered listeners about selection change
     * @private
     * @returns {void}
     */
    notifySendButtonUpdateListeners() {
        if (this.sendButtonUpdateCallbacks) {
            this.sendButtonUpdateCallbacks.forEach(callback => {
                try {
                    callback();
                } catch (error) {
                    console.error('[DynamicLLMManager] Error in send button update callback:', error);
                }
            });
        }
    }

    /**
     * Get ordinal label for fallback position
     * @param {number} position - Position number (1-based)
     * @returns {string} Ordinal label (1st, 2nd, 3rd, 4th, etc.)
     */
    getOrdinalLabel(position) {
        const suffixes = ['th', 'st', 'nd', 'rd'];
        const value = position % 100;
        const suffix = suffixes[(value - 20) % 10] || suffixes[value] || suffixes[0];
        return position + suffix;
    }

    /**
     * Update fallback labels after adding/removing selectors
     * Ensures labels stay sequential (1st, 2nd, 3rd, etc.)
     * @returns {void}
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
