/**
 * Searchable LLM Selectbox Component
 *
 * A reusable component for selecting LLM models with search functionality,
 * provider grouping, and keyboard navigation support.
 *
 * Features:
 * - Real-time search/filtering
 * - Provider grouping (Google, Groq, Ollama, etc.)
 * - Keyboard navigation (arrow keys, Enter, Escape)
 * - Click-outside-to-close behavior
 * - Custom styling to match CLI-Agent Visual Response UI theme
 *
 * @class SearchableLLMSelectbox
 * @author CLI-Agent Development Team
 * @version 1.0.0
 */

/**
 * @typedef {Object} LLMSelectboxOptions
 * @property {string} [placeholder='Select LLM...'] - Placeholder text for the selectbox
 * @property {string} [id] - Unique ID for this selectbox instance
 * @property {function(LLMModel|null):void} [onSelectionChange] - Callback when selection changes
 * @property {function(SearchableLLMSelectbox, Array<LLMModel>, Object):void} [onModelsLoaded] - Callback when models are loaded
 * @property {Array<LLMModel>} [initialModels=[]] - Initial model list (optional)
 * @property {boolean} [includeNoneOption=false] - Whether to include a "None" option
 */

/**
 * @typedef {Object} LLMModel
 * @property {string|null} id - Unique model identifier (null for None option)
 * @property {string} name - Display name of the model
 * @property {string} provider - Provider name (Google, Groq, Ollama, etc.)
 * @property {number|string} context_window - Context window size or 'N/A'
 * @property {Object|null} pricing - Pricing information
 * @property {Array<string>} capabilities - Model capabilities (vision, function-calling, etc.)
 * @property {boolean} [isNoneOption] - True if this is the "None" option
 * @property {string} [status] - Model status ('available' or 'failed')
 * @property {string} [failure_reason] - Failure reason if status is 'failed'
 */

/**
 * @typedef {Object} SystemConfiguration
 * @property {string|null} override_llm - Current override LLM model ID
 * @property {string|null} router_llm - Current router LLM model ID
 * @property {string|null} planner_llm - Current planner LLM model ID
 * @property {Array<string>} selected_llms - Current fallback LLM model IDs
 */

class SearchableLLMSelectbox {
    /**
     * Initialize the searchable LLM selectbox
     * @param {HTMLElement} container - The container element to attach the selectbox to
     * @param {LLMSelectboxOptions} [options={}] - Configuration options
     */
    constructor(container, options = {}) {
        /**
         * Container element for the selectbox
         * @type {HTMLElement}
         */
        this.container = container;

        /**
         * Configuration options
         * @type {LLMSelectboxOptions}
         */
        this.options = {
            placeholder: options.placeholder || 'Select LLM...',
            id: options.id || 'llm-selector-' + Date.now(),
            onSelectionChange: options.onSelectionChange || (() => {}),
            onModelsLoaded: options.onModelsLoaded || (() => {}),
            initialModels: options.initialModels || [],
            includeNoneOption: options.includeNoneOption || false
        };

        /**
         * All available models
         * @type {Array<LLMModel>}
         */
        this.models = [];

        /**
         * Filtered models based on search query
         * @type {Array<LLMModel>}
         */
        this.filteredModels = [];

        /**
         * Currently selected model
         * @type {LLMModel|null}
         */
        this.selectedModel = null;

        /**
         * Whether dropdown is currently open
         * @type {boolean}
         */
        this.isOpen = false;

        /**
         * Current search query
         * @type {string}
         */
        this.searchQuery = '';

        /**
         * Index of currently focused option (for keyboard navigation)
         * @type {number}
         */
        this.focusedIndex = -1;

        /**
         * System configuration from server
         * @type {SystemConfiguration|null}
         */
        this.systemConfiguration = null;

        /**
         * Deferred selection (model ID to select once models are loaded)
         * @type {string|null}
         */
        this.deferredSelection = null;

        /**
         * Main selectbox element
         * @type {HTMLElement|null}
         */
        this.selectbox = null;

        /**
         * Trigger button element
         * @type {HTMLElement|null}
         */
        this.trigger = null;

        /**
         * Label element displaying selected model
         * @type {HTMLElement|null}
         */
        this.label = null;

        /**
         * Arrow indicator element
         * @type {HTMLElement|null}
         */
        this.arrow = null;

        /**
         * Dropdown container element
         * @type {HTMLElement|null}
         */
        this.dropdown = null;

        /**
         * Search input element
         * @type {HTMLInputElement|null}
         */
        this.searchInput = null;

        /**
         * Options container element
         * @type {HTMLElement|null}
         */
        this.optionsContainer = null;

        /**
         * Failure display element (shown below selectbox when failed model selected)
         * @type {HTMLElement|null}
         */
        this.failureDisplay = null;

        /**
         * Global click handler for click-outside-to-close
         * @type {function(Event):void|null}
         */
        this.globalClickHandler = null;

        /**
         * Window resize handler
         * @type {function(Event):void|null}
         */
        this.resizeHandler = null;

        this.init();
    }

    /**
     * Initialize the selectbox DOM structure and event listeners
     * @returns {void}
     */
    init() {
        this.createDOM();
        this.setupEventListeners();

        // Set initial width based on placeholder
        this.updateSelectboxWidth();

        // Load models from server if not provided initially
        if (this.options.initialModels.length === 0) {
            this.loadModelsFromServer();
        } else {
            this.setModels(this.options.initialModels);
            // Notify that models are loaded (for initial models)
            this.options.onModelsLoaded(this, this.options.initialModels, {});
        }
    }

    /**
     * Create the DOM structure for the selectbox
     * @returns {void}
     */
    createDOM() {
        this.container.innerHTML = `
            <div class="llm-selectbox" id="${this.options.id}">
                <div class="llm-selectbox-trigger" tabindex="0">
                    <span class="llm-selectbox-label">${this.options.placeholder}</span>
                    <span class="llm-selectbox-arrow">▼</span>
                </div>
                <div class="llm-selectbox-dropdown" style="display: none;">
                    <div class="llm-selectbox-search">
                        <input type="text" placeholder="Search models..." class="llm-search-input">
                    </div>
                    <div class="llm-selectbox-options">
                        <div class="llm-selectbox-loading">Loading models...</div>
                    </div>
                </div>
                <div class="llm-selectbox-failure-display llm-failure-reason" style="display: none;"></div>
            </div>
        `;

        // Get DOM references
        this.selectbox = this.container.querySelector('.llm-selectbox');
        this.trigger = this.container.querySelector('.llm-selectbox-trigger');
        this.label = this.container.querySelector('.llm-selectbox-label');
        this.arrow = this.container.querySelector('.llm-selectbox-arrow');
        this.dropdown = this.container.querySelector('.llm-selectbox-dropdown');
        this.searchInput = this.container.querySelector('.llm-search-input');
        this.optionsContainer = this.container.querySelector('.llm-selectbox-options');
        this.failureDisplay = this.container.querySelector('.llm-selectbox-failure-display');
    }

    /**
     * Setup event listeners for interaction
     * Handles keyboard navigation (arrow keys, Enter, Escape), click events,
     * search input, and global click-outside-to-close behavior
     * @returns {void}
     */
    setupEventListeners() {
        // Trigger click
        this.trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleDropdown();
        });

        // Trigger keyboard navigation
        this.trigger.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'Enter':
                case ' ':
                    e.preventDefault();
                    this.toggleDropdown();
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    if (!this.isOpen) {
                        this.openDropdown();
                    } else {
                        this.focusNextOption();
                    }
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    if (this.isOpen) {
                        this.focusPreviousOption();
                    }
                    break;
                case 'Escape':
                    if (this.isOpen) {
                        this.closeDropdown();
                    }
                    break;
            }
        });

        // Search input
        this.searchInput.addEventListener('input', (e) => {
            this.searchQuery = e.target.value.toLowerCase();
            this.filterAndRenderOptions();
        });

        this.searchInput.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    this.focusNextOption();
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    this.focusPreviousOption();
                    break;
                case 'Enter':
                    e.preventDefault();
                    this.selectFocusedOption();
                    break;
                case 'Escape':
                    this.closeDropdown();
                    break;
            }
        });

        // Global click to close - store the handler so it can be removed later
        this.globalClickHandler = (e) => {
            // Don't close dropdown if clicking on form inputs or other interactive elements
            const isFormElement = e.target.matches('input, textarea, button, select, [contenteditable]');
            const isInsideSelectbox = this.selectbox.contains(e.target);
            const isChatArea = e.target.closest('.chat-input-container, #chat-input, #chat-form');

            // Only close dropdown if it's open AND we're not clicking on protected elements
            if (this.isOpen && !isInsideSelectbox && !isFormElement && !isChatArea) {
                this.closeDropdown();
            }
        };
        document.addEventListener('click', this.globalClickHandler, { passive: true });

        // Add window resize listener to update width on screen size changes
        this.resizeHandler = () => {
            this.updateSelectboxWidth();
        };
        window.addEventListener('resize', this.resizeHandler);
    }

    /**
     * Load models from the server
     * @async
     * @returns {Promise<void>}
     */
    async loadModelsFromServer() {
        try {
            this.showLoading(true);
            const response = await fetch('api/llms');
            const data = await response.json();

            if (data.status === 'success') {
                this.setModels(data.models);
                window.visualLogger.debug(`Loaded ${data.models.length} LLM models from server`);
                // Store current system configuration for reference
                this.systemConfiguration = data.current_configuration || {};
                window.visualLogger.debug('Current system configuration:', this.systemConfiguration);
                // Notify that models are loaded
                this.options.onModelsLoaded(this, data.models, this.systemConfiguration);
            } else {
                throw new Error(data.message || 'Failed to load models');
            }
        } catch (error) {
            console.error('Failed to load LLM models:', error);
            this.showError('Failed to load models. Please refresh the page.');
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Set the available models
     * @param {Array<LLMModel>} models - Array of model objects
     * @returns {void}
     */
    setModels(models) {
        this.models = [...models];

        // Add None option if specified
        if (this.options.includeNoneOption) {
            const noneOption = {
                id: null,
                name: 'None',
                provider: 'System',
                context_window: 'N/A',
                pricing: null,
                capabilities: [],
                isNoneOption: true
            };
            this.models.unshift(noneOption);  // Add at the beginning
        }

        this.filteredModels = [...this.models];
        this.filterAndRenderOptions();

        // Handle deferred selection if there was one
        if (this.deferredSelection) {
            const modelId = this.deferredSelection;
            this.deferredSelection = null; // Clear the deferred selection

            const model = this.models.find(m => m.id === modelId);
            if (model) {
                this.selectedModel = model;
                this.label.textContent = model.name;
                this.label.title = `${model.provider}: ${model.name}`;

                // Show/hide failure reason for deferred selection
                if (model.status === 'failed' && model.failure_reason) {
                    this.failureDisplay.textContent = model.failure_reason;
                    this.failureDisplay.style.display = 'block';
                } else {
                    this.failureDisplay.style.display = 'none';
                }

                this.updateSelectboxWidth();
                window.visualLogger.debug(`Successfully restored deferred selection: ${modelId}`);
                // Don't call onSelectionChange for deferred selections to avoid duplicate events
            } else {
                console.warn(`Deferred model ${modelId} still not found after models loaded`);
            }
        }
    }

    /**
     * Filter models based on search query and render options
     * @returns {void}
     */
    filterAndRenderOptions() {
        // Filter models by search query
        this.filteredModels = this.models.filter(model => {
            const searchText = this.searchQuery.toLowerCase();
            return model.name.toLowerCase().includes(searchText) ||
                   model.provider.toLowerCase().includes(searchText);
        });

        // Group by provider
        const groupedModels = this.groupModelsByProvider(this.filteredModels);

        // Render the grouped options
        this.renderOptions(groupedModels);
    }

    /**
     * Group models by provider
     * @param {Array<LLMModel>} models - Array of model objects
     * @returns {Object<string, Array<LLMModel>>} Grouped models by provider
     */
    groupModelsByProvider(models) {
        const grouped = {};

        models.forEach(model => {
            const provider = model.provider;
            if (!grouped[provider]) {
                grouped[provider] = [];
            }
            grouped[provider].push(model);
        });

        // Sort providers and models within each provider
        const sortedGrouped = {};
        Object.keys(grouped).sort().forEach(provider => {
            sortedGrouped[provider] = grouped[provider].sort((a, b) => a.name.localeCompare(b.name));
        });

        return sortedGrouped;
    }

    /**
     * Render the options dropdown
     * @param {Object<string, Array<LLMModel>>} groupedModels - Models grouped by provider
     * @returns {void}
     */
    renderOptions(groupedModels) {
        const providers = Object.keys(groupedModels);

        if (providers.length === 0) {
            this.optionsContainer.innerHTML = '<div class="llm-selectbox-no-results">No models found</div>';
            return;
        }

        let html = '';
        let optionIndex = 0;

        providers.forEach(provider => {
            const models = groupedModels[provider];
            const providerIcon = this.getProviderIcon(provider);

            // Provider header
            html += `
                <div class="llm-selectbox-provider-header">
                    <span class="llm-provider-icon">${providerIcon}</span>
                    <span class="llm-provider-name">${provider}</span>
                    <span class="llm-provider-count">(${models.length})</span>
                </div>
            `;

            // Provider models
            models.forEach(model => {
                const isSelected = this.selectedModel && this.selectedModel.id === model.id;
                const isFailed = model.status === 'failed';
                const capabilities = model.capabilities.length > 0 ?
                    `<span class="llm-capabilities">${model.capabilities.join(', ')}</span>` : '';

                const contextInfo = model.context_window > 0 ?
                    `<span class="llm-context">${this.formatContextWindow(model.context_window)}</span>` : '';

                const failureReason = isFailed && model.failure_reason ?
                    `<div class="llm-failure-reason">${this.escapeHtml(model.failure_reason)}</div>` : '';

                html += `
                    <div class="llm-selectbox-option ${isSelected ? 'selected' : ''} ${isFailed ? 'failed' : ''}"
                         data-option-index="${optionIndex}"
                         data-model-id="${model.id}">
                        <div class="llm-option-main">
                            <span class="llm-option-name">${model.name}${isFailed ? ' ⚠️' : ''}</span>
                            ${contextInfo}
                        </div>
                        ${capabilities}
                        ${failureReason}
                    </div>
                `;
                optionIndex++;
            });
        });

        this.optionsContainer.innerHTML = html;

        // Setup option click handlers
        this.optionsContainer.querySelectorAll('.llm-selectbox-option').forEach(option => {
            option.addEventListener('click', (e) => {
                e.stopPropagation();
                const modelId = option.dataset.modelId;
                // Convert string "null" back to actual null for None option
                const actualModelId = modelId === "null" ? null : modelId;
                this.selectModel(actualModelId);
            });
        });
    }

    /**
     * Get icon for provider
     * @param {string} provider - Provider name
     * @returns {string} Provider icon emoji
     */
    getProviderIcon(provider) {
        const icons = {
            'Google': '🟢',
            'Groq': '⚡',
            'Ollama': '🤖',
            'OpenAI': '🔴',
            'Anthropic': '🔵',
            'Nvidia': '🟩'
        };
        return icons[provider] || '🔧';
    }

    /**
     * Format context window for display
     * @param {number} contextWindow - Context window size
     * @returns {string} Formatted context window (e.g., "1.5M", "128K")
     */
    formatContextWindow(contextWindow) {
        if (contextWindow >= 1000000) {
            return `${(contextWindow / 1000000).toFixed(1)}M`;
        } else if (contextWindow >= 1000) {
            return `${(contextWindow / 1000).toFixed(0)}K`;
        }
        return contextWindow.toString();
    }

    /**
     * Escape HTML to prevent XSS attacks
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Select a model by ID
     * @param {string|null} modelId - Model ID to select, or null for None option
     * @returns {void}
     */
    selectModel(modelId) {
        const model = this.models.find(m => m.id === modelId);
        if (model) {
            this.selectedModel = model;
            this.label.textContent = model.name;
            this.label.title = `${model.provider}: ${model.name}`;

            // Show/hide failure reason below the selectbox
            if (model.status === 'failed' && model.failure_reason) {
                this.failureDisplay.textContent = model.failure_reason;
                this.failureDisplay.style.display = 'block';
            } else {
                this.failureDisplay.style.display = 'none';
            }

            this.updateSelectboxWidth();
            this.closeDropdown();
            this.options.onSelectionChange(model);
        } else if (modelId === null && this.options.includeNoneOption) {
            // Handle selection of None option
            const noneModel = this.models.find(m => m.isNoneOption);
            if (noneModel) {
                this.selectedModel = noneModel;
                this.label.textContent = noneModel.name;
                this.label.title = `${noneModel.provider}: ${noneModel.name}`;
                this.failureDisplay.style.display = 'none';
                this.updateSelectboxWidth();
                this.closeDropdown();
                this.options.onSelectionChange(noneModel);
            }
        } else if (modelId === null) {
            // Clear selection when modelId is null and no None option exists
            this.selectedModel = null;
            this.label.textContent = this.options.placeholder;
            this.label.title = '';
            this.failureDisplay.style.display = 'none';
            this.updateSelectboxWidth();
            this.closeDropdown();
            this.options.onSelectionChange(null);
        } else {
            // Model not found, store for deferred selection
            console.warn(`Model ${modelId} not found, will try again when models are loaded`);
            this.deferredSelection = modelId;
        }
    }

    /**
     * Update the selectbox width to fit the content
     * Dynamically calculates width based on label text and applies responsive constraints
     * @returns {void}
     */
    updateSelectboxWidth() {
        if (!this.label || !this.selectbox) return;

        // Create a temporary element to measure text width
        const measurer = document.createElement('span');
        measurer.style.visibility = 'hidden';
        measurer.style.position = 'absolute';
        measurer.style.top = '-9999px';
        measurer.style.fontFamily = window.getComputedStyle(this.label).fontFamily;
        measurer.style.fontSize = window.getComputedStyle(this.label).fontSize;
        measurer.style.fontWeight = window.getComputedStyle(this.label).fontWeight;
        measurer.textContent = this.label.textContent;

        document.body.appendChild(measurer);
        const textWidth = measurer.offsetWidth;
        document.body.removeChild(measurer);

        // Add padding for arrow and spacing (arrow: ~12px, padding: ~16px, buffer: ~12px)
        const totalWidth = textWidth + 40;

        // Apply responsive constraints based on screen size
        let minWidth = 90;
        let maxWidth = 250;

        if (window.innerWidth <= 1200) {
            minWidth = 80;
            maxWidth = 180;
        }

        const finalWidth = Math.max(minWidth, Math.min(maxWidth, totalWidth));
        this.selectbox.style.width = `${finalWidth}px`;
    }

    /**
     * Get the currently selected model
     * @returns {LLMModel|null} Selected model or null
     */
    getSelectedModel() {
        return this.selectedModel;
    }

    /**
     * Set the selected model programmatically
     * @param {string} modelId - Model ID to select
     * @returns {void}
     */
    setSelectedModel(modelId) {
        this.selectModel(modelId);
    }

    /**
     * Toggle dropdown open/closed
     * @returns {void}
     */
    toggleDropdown() {
        if (this.isOpen) {
            this.closeDropdown();
        } else {
            this.openDropdown();
        }
    }

    /**
     * Open the dropdown
     * Resets search and focuses search input
     * @returns {void}
     */
    openDropdown() {
        this.isOpen = true;
        this.dropdown.style.display = 'block';
        this.arrow.textContent = '▲';
        this.searchInput.focus();
        this.focusedIndex = -1;

        // Reset search
        this.searchInput.value = '';
        this.searchQuery = '';
        this.filterAndRenderOptions();
    }

    /**
     * Close the dropdown
     * Returns focus to trigger button
     * @returns {void}
     */
    closeDropdown() {
        this.isOpen = false;
        this.dropdown.style.display = 'none';
        this.arrow.textContent = '▼';
        this.focusedIndex = -1;
        this.trigger.focus();
    }

    /**
     * Focus next option in dropdown (keyboard navigation)
     * @returns {void}
     */
    focusNextOption() {
        const options = this.optionsContainer.querySelectorAll('.llm-selectbox-option');
        if (options.length === 0) return;

        this.focusedIndex = Math.min(this.focusedIndex + 1, options.length - 1);
        this.updateFocusedOption(options);
    }

    /**
     * Focus previous option in dropdown (keyboard navigation)
     * @returns {void}
     */
    focusPreviousOption() {
        const options = this.optionsContainer.querySelectorAll('.llm-selectbox-option');
        if (options.length === 0) return;

        this.focusedIndex = Math.max(this.focusedIndex - 1, 0);
        this.updateFocusedOption(options);
    }

    /**
     * Update the visually focused option
     * @param {NodeList} options - List of option elements
     * @returns {void}
     */
    updateFocusedOption(options) {
        // Remove previous focus
        options.forEach(option => option.classList.remove('focused'));

        // Add focus to current option
        if (this.focusedIndex >= 0 && this.focusedIndex < options.length) {
            const focusedOption = options[this.focusedIndex];
            focusedOption.classList.add('focused');
            focusedOption.scrollIntoView({ block: 'nearest' });
        }
    }

    /**
     * Select the currently focused option (keyboard navigation)
     * @returns {void}
     */
    selectFocusedOption() {
        const options = this.optionsContainer.querySelectorAll('.llm-selectbox-option');
        if (this.focusedIndex >= 0 && this.focusedIndex < options.length) {
            const focusedOption = options[this.focusedIndex];
            const modelId = focusedOption.dataset.modelId;
            this.selectModel(modelId);
        }
    }

    /**
     * Show loading state
     * @param {boolean} show - Whether to show loading
     * @returns {void}
     */
    showLoading(show) {
        if (show) {
            this.optionsContainer.innerHTML = '<div class="llm-selectbox-loading">Loading models...</div>';
        }
    }

    /**
     * Show error message
     * @param {string} message - Error message to display
     * @returns {void}
     */
    showError(message) {
        this.optionsContainer.innerHTML = `<div class="llm-selectbox-error">${message}</div>`;
    }

    /**
     * Update model status (called when receiving WebSocket model_failure event)
     * @param {string} modelKey - The model key that failed
     * @param {string} failureReason - The failure reason
     * @returns {void}
     */
    updateModelStatus(modelKey, failureReason) {
        // Find and update the model
        const model = this.models.find(m => m.id === modelKey);
        if (model) {
            model.status = 'failed';
            model.failure_reason = failureReason;

            // Update failure display if this is the currently selected model
            if (this.selectedModel && this.selectedModel.id === modelKey) {
                this.failureDisplay.textContent = failureReason;
                this.failureDisplay.style.display = 'block';
            }

            // Re-render if dropdown is open
            if (this.isOpen) {
                this.filterAndRenderOptions();
            }
        }
    }

    /**
     * Destroy the selectbox and clean up event listeners
     * Removes all event listeners and DOM elements to prevent memory leaks
     * @returns {void}
     */
    destroy() {
        // Remove global event listeners
        if (this.globalClickHandler) {
            document.removeEventListener('click', this.globalClickHandler, { passive: true });
        }
        if (this.resizeHandler) {
            window.removeEventListener('resize', this.resizeHandler);
        }

        // Clear container
        this.container.innerHTML = '';

        // Clear references
        this.models = [];
        this.filteredModels = [];
        this.selectedModel = null;
        this.globalClickHandler = null;
        this.resizeHandler = null;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SearchableLLMSelectbox;
} else if (typeof window !== 'undefined') {
    window.SearchableLLMSelectbox = SearchableLLMSelectbox;
}
