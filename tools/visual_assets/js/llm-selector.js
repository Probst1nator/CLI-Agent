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
class SearchableLLMSelectbox {
    /**
     * Initialize the searchable LLM selectbox
     * @param {HTMLElement} container - The container element to attach the selectbox to
     * @param {Object} options - Configuration options
     * @param {string} options.placeholder - Placeholder text for the selectbox
     * @param {string} options.id - Unique ID for this selectbox instance
     * @param {Function} options.onSelectionChange - Callback when selection changes
     * @param {Function} options.onModelsLoaded - Callback when models are loaded
     * @param {Array} options.initialModels - Initial model list (optional)
     */
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            placeholder: options.placeholder || 'Select LLM...',
            id: options.id || 'llm-selector-' + Date.now(),
            onSelectionChange: options.onSelectionChange || (() => {}),
            onModelsLoaded: options.onModelsLoaded || (() => {}),
            initialModels: options.initialModels || [],
            includeNoneOption: options.includeNoneOption || false
        };

        this.models = [];
        this.filteredModels = [];
        this.selectedModel = null;
        this.isOpen = false;
        this.searchQuery = '';
        this.focusedIndex = -1;

        this.init();
    }

    /**
     * Initialize the selectbox DOM structure and event listeners
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
    }

    /**
     * Setup event listeners for interaction
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
     */
    async loadModelsFromServer() {
        try {
            this.showLoading(true);
            const response = await fetch('/api/llms');
            const data = await response.json();

            if (data.status === 'success') {
                this.setModels(data.models);
                console.log(`Loaded ${data.models.length} LLM models from server`);
                // Store current system configuration for reference
                this.systemConfiguration = data.current_configuration || {};
                console.log('Current system configuration:', this.systemConfiguration);
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
     * @param {Array} models - Array of model objects
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
                this.updateSelectboxWidth();
                console.log(`Successfully restored deferred selection: ${modelId}`);
                // Don't call onSelectionChange for deferred selections to avoid duplicate events
            } else {
                console.warn(`Deferred model ${modelId} still not found after models loaded`);
            }
        }
    }

    /**
     * Filter models based on search query and render options
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
     * @param {Array} models - Array of model objects
     * @returns {Object} Grouped models by provider
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
     * @param {Object} groupedModels - Models grouped by provider
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
                const capabilities = model.capabilities.length > 0 ?
                    `<span class="llm-capabilities">${model.capabilities.join(', ')}</span>` : '';

                const contextInfo = model.context_window > 0 ?
                    `<span class="llm-context">${this.formatContextWindow(model.context_window)}</span>` : '';

                html += `
                    <div class="llm-selectbox-option ${isSelected ? 'selected' : ''}"
                         data-option-index="${optionIndex}"
                         data-model-id="${model.id}">
                        <div class="llm-option-main">
                            <span class="llm-option-name">${model.name}</span>
                            ${contextInfo}
                        </div>
                        ${capabilities}
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
     * @returns {string} Provider icon
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
     * @returns {string} Formatted context window
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
     * Select a model by ID
     * @param {string|null} modelId - Model ID to select, or null for None option
     */
    selectModel(modelId) {
        const model = this.models.find(m => m.id === modelId);
        if (model) {
            this.selectedModel = model;
            this.label.textContent = model.name;
            this.label.title = `${model.provider}: ${model.name}`;
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
                this.updateSelectboxWidth();
                this.closeDropdown();
                this.options.onSelectionChange(noneModel);
            }
        } else if (modelId === null) {
            // Clear selection when modelId is null and no None option exists
            this.selectedModel = null;
            this.label.textContent = this.options.placeholder;
            this.label.title = '';
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
     * @returns {Object|null} Selected model or null
     */
    getSelectedModel() {
        return this.selectedModel;
    }

    /**
     * Set the selected model programmatically
     * @param {string} modelId - Model ID to select
     */
    setSelectedModel(modelId) {
        this.selectModel(modelId);
    }

    /**
     * Toggle dropdown open/closed
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
     */
    closeDropdown() {
        this.isOpen = false;
        this.dropdown.style.display = 'none';
        this.arrow.textContent = '▼';
        this.focusedIndex = -1;
        this.trigger.focus();
    }

    /**
     * Focus next option in dropdown
     */
    focusNextOption() {
        const options = this.optionsContainer.querySelectorAll('.llm-selectbox-option');
        if (options.length === 0) return;

        this.focusedIndex = Math.min(this.focusedIndex + 1, options.length - 1);
        this.updateFocusedOption(options);
    }

    /**
     * Focus previous option in dropdown
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
     * Select the currently focused option
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
     */
    showLoading(show) {
        if (show) {
            this.optionsContainer.innerHTML = '<div class="llm-selectbox-loading">Loading models...</div>';
        }
    }

    /**
     * Show error message
     * @param {string} message - Error message to display
     */
    showError(message) {
        this.optionsContainer.innerHTML = `<div class="llm-selectbox-error">${message}</div>`;
    }

    /**
     * Destroy the selectbox and clean up event listeners
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