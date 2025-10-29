/**
 * Autocomplete Manager
 *
 * Provides intelligent autocomplete suggestions for the input box.
 * Features:
 * - 500ms debounce to reduce API calls
 * - Keyboard navigation (arrow keys, Enter, Esc)
 * - Context-aware suggestions (commands, tools, files)
 * - Click-to-insert behavior
 *
 * @class AutocompleteManager
 * @version 1.0.0
 */

class AutocompleteManager {
    /**
     * Initialize the autocomplete manager
     * @param {HTMLElement} inputElement - The input element to attach autocomplete to
     * @param {Object} options - Configuration options
     */
    constructor(inputElement, options = {}) {
        this.inputElement = inputElement;
        this.options = {
            debounceMs: options.debounceMs || 500,
            maxSuggestions: options.maxSuggestions || 5,
            onSelect: options.onSelect || (() => {})
        };

        this.suggestions = [];
        this.isVisible = false;
        this.selectedIndex = -1;
        this.debounceTimer = null;
        this.lastInput = '';

        // Create UI elements
        this.createUI();
        this.attachEventListeners();

        window.visualLogger.info('[Autocomplete] Initialized');
    }

    /**
     * Create the autocomplete UI elements
     */
    createUI() {
        // Create container for suggestions
        this.container = document.createElement('div');
        this.container.className = 'autocomplete-container';
        this.container.style.display = 'none';

        // Set width to match input field
        this.updateContainerWidth();

        // Insert after input element
        this.inputElement.parentElement.insertBefore(
            this.container,
            this.inputElement.nextSibling
        );
    }

    /**
     * Update container width and position to match input element
     */
    updateContainerWidth() {
        const inputWidth = this.inputElement.offsetWidth;
        const inputLeft = this.inputElement.offsetLeft;

        if (inputWidth > 0) {
            this.container.style.width = inputWidth + 'px';
            this.container.style.left = inputLeft + 'px';
        }
    }

    /**
     * Attach event listeners to input element
     */
    attachEventListeners() {
        // Input event - trigger autocomplete with debounce
        this.inputElement.addEventListener('input', (e) => {
            this.handleInput(e.target.value);
        });

        // Keydown event - handle navigation and selection
        this.inputElement.addEventListener('keydown', (e) => {
            if (!this.isVisible) return;

            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    this.selectNext();
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    this.selectPrevious();
                    break;
                case 'Enter':
                    if (this.selectedIndex >= 0) {
                        e.preventDefault();
                        this.insertSuggestion(this.suggestions[this.selectedIndex]);
                    }
                    break;
                case 'Escape':
                    e.preventDefault();
                    this.hide();
                    break;
                case 'Tab':
                    if (this.selectedIndex >= 0) {
                        e.preventDefault();
                        this.insertSuggestion(this.suggestions[this.selectedIndex]);
                    }
                    break;
            }
        });

        // Click outside to close
        document.addEventListener('click', (e) => {
            if (!this.container.contains(e.target) && e.target !== this.inputElement) {
                this.hide();
            }
        });

        // Focus out - small delay to allow click events to fire
        this.inputElement.addEventListener('blur', () => {
            setTimeout(() => {
                this.hide();
            }, 200);
        });
    }

    /**
     * Handle input changes with debouncing
     * @param {string} value - Current input value
     */
    handleInput(value) {
        // Clear existing timer
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }

        // Store current input
        this.lastInput = value;

        // If input is empty, hide suggestions
        if (!value.trim()) {
            this.hide();
            return;
        }

        // Set debounce timer
        this.debounceTimer = setTimeout(() => {
            this.fetchSuggestions(value);
        }, this.options.debounceMs);
    }

    /**
     * Fetch suggestions from backend
     * @param {string} input - Current input text
     */
    async fetchSuggestions(input) {
        try {
            window.visualLogger.debug(`[Autocomplete] Fetching suggestions for: "${input}"`);

            // Build API URL (handle both local and multi-user modes)
            const url = window.backendConnector
                ? await window.backendConnector.buildURL('/api/autocomplete')
                : '/api/autocomplete';

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-ID': window.sessionId || ''
                },
                body: JSON.stringify({
                    input: input,
                    max_suggestions: this.options.maxSuggestions
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.suggestions = data.suggestions || [];
                window.visualLogger.debug(`[Autocomplete] Received ${this.suggestions.length} suggestions`);

                if (this.suggestions.length > 0) {
                    this.render();
                    this.show();
                } else {
                    this.hide();
                }
            } else {
                window.visualLogger.error('[Autocomplete] API error:', data.error);
                this.hide();
            }
        } catch (error) {
            window.visualLogger.error('[Autocomplete] Fetch error:', error);
            this.hide();
        }
    }

    /**
     * Render suggestions in the UI
     */
    render() {
        this.container.innerHTML = '';
        this.selectedIndex = -1;

        this.suggestions.forEach((suggestion, index) => {
            const item = document.createElement('div');
            item.className = 'autocomplete-item';
            item.dataset.index = index;

            // Add type icon
            const icon = document.createElement('span');
            icon.className = 'autocomplete-icon';
            icon.textContent = this.getIconForType(suggestion.type);
            item.appendChild(icon);

            // Add suggestion text
            const text = document.createElement('span');
            text.className = 'autocomplete-text';
            text.textContent = suggestion.text;
            item.appendChild(text);

            // Add description
            const description = document.createElement('span');
            description.className = 'autocomplete-description';
            description.textContent = suggestion.description;
            item.appendChild(description);

            // Click handler
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                this.insertSuggestion(suggestion);
            });

            // Hover handler
            item.addEventListener('mouseenter', () => {
                this.selectedIndex = index;
                this.updateSelection();
            });

            this.container.appendChild(item);
        });
    }

    /**
     * Get icon for suggestion type
     * @param {string} type - Suggestion type (command, tool, file)
     * @returns {string} Icon character
     */
    getIconForType(type) {
        switch (type) {
            case 'command':
                return '/';
            case 'tool':
                return '⚙';
            case 'file':
                return '📄';
            default:
                return '•';
        }
    }

    /**
     * Show autocomplete dropdown
     */
    show() {
        // Update width to match input (in case it was resized)
        this.updateContainerWidth();
        this.container.style.display = 'block';
        this.isVisible = true;
    }

    /**
     * Hide autocomplete dropdown
     */
    hide() {
        this.container.style.display = 'none';
        this.isVisible = false;
        this.selectedIndex = -1;
    }

    /**
     * Select next suggestion
     */
    selectNext() {
        if (this.suggestions.length === 0) return;

        this.selectedIndex = (this.selectedIndex + 1) % this.suggestions.length;
        this.updateSelection();
    }

    /**
     * Select previous suggestion
     */
    selectPrevious() {
        if (this.suggestions.length === 0) return;

        this.selectedIndex = this.selectedIndex <= 0
            ? this.suggestions.length - 1
            : this.selectedIndex - 1;
        this.updateSelection();
    }

    /**
     * Update visual selection in dropdown
     */
    updateSelection() {
        const items = this.container.querySelectorAll('.autocomplete-item');
        items.forEach((item, index) => {
            if (index === this.selectedIndex) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });

        // Scroll selected item into view
        if (this.selectedIndex >= 0 && items[this.selectedIndex]) {
            items[this.selectedIndex].scrollIntoView({
                block: 'nearest',
                behavior: 'smooth'
            });
        }
    }

    /**
     * Insert selected suggestion into input
     * @param {Object} suggestion - Suggestion to insert
     */
    insertSuggestion(suggestion) {
        window.visualLogger.info(`[Autocomplete] Selected: ${suggestion.text} (${suggestion.type})`);

        // For commands, replace the entire input
        if (suggestion.type === 'command') {
            this.inputElement.value = suggestion.text + ' ';
        } else {
            // For tools and files, append or replace based on context
            this.inputElement.value = suggestion.text;
        }

        // Trigger input event to update any listeners
        this.inputElement.dispatchEvent(new Event('input', { bubbles: true }));

        // Call custom onSelect callback
        this.options.onSelect(suggestion);

        // Hide dropdown and focus input
        this.hide();
        this.inputElement.focus();

        // Move cursor to end
        this.inputElement.setSelectionRange(
            this.inputElement.value.length,
            this.inputElement.value.length
        );
    }

    /**
     * Destroy the autocomplete manager
     */
    destroy() {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        if (this.container && this.container.parentElement) {
            this.container.parentElement.removeChild(this.container);
        }
        window.visualLogger.info('[Autocomplete] Destroyed');
    }
}

// Expose to window for global access
window.AutocompleteManager = AutocompleteManager;
