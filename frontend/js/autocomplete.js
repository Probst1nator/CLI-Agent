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
      onSelect: options.onSelect || (() => {}),
      chatState: options.chatState || null, // Reference to ChatStateManager
      enableCustomSuggestions: options.enableCustomSuggestions !== false, // Default: enabled
    };

    this.suggestions = [];
    this.suggestedPrompts = []; // Store suggested prompts for empty input
    this.customSuggestions = []; // Store custom async LLM-generated suggestions
    this.previousCustomSuggestions = []; // Store previous custom suggestions at score: 50
    this.isVisible = false;
    this.selectedIndex = -1;
    this.debounceTimer = null; // For regular backend autocomplete (500ms)
    this.customSuggestionsTimer = null; // For LLM-based suggestions (1000ms)
    this.lastInput = '';
    this.showingSuggestedPrompts = false; // Track which type is shown
    this.autoShowDropdown = false; // Don't auto-show until user focuses input
    this.inlineCompletionVisible = false; // Track if inline ghost text is shown
    this.currentInlineCompletion = ''; // Store the current inline completion text

    // Abort controller for cancelling in-flight custom suggestion requests
    this.customSuggestionsAbortController = null;

    // Create UI elements
    this.createUI();
    this.attachEventListeners();

    // Load suggested prompts on initialization
    this.fetchSuggestedPrompts();

    // Fetch initial custom autocomplete in background (context-based, empty input)
    // Will be displayed when user focuses the input
    this.triggerInitialCustomSuggestions();

    window.visualLogger.info('[Autocomplete] Initialized');
  }

  /**
   * Trigger initial custom suggestions on page load
   * Called automatically in constructor - generates suggestions using conversation context
   */
  triggerInitialCustomSuggestions() {
    if (this.options.enableCustomSuggestions) {
      window.visualLogger.debug('[Autocomplete] Triggering initial custom suggestions');
      // Fetch suggestions with empty input (context-based from conversation history)
      this.fetchCustomSuggestions('');
    }
  }

  /**
   * Create the autocomplete UI elements
   */
  createUI() {
    // Create container for suggestions
    this.container = document.createElement('div');
    this.container.className = 'autocomplete-container';
    this.container.style.display = 'none';

    // Set width and max-height to match available space
    this.updateContainerSize();

    // Insert after input element
    this.inputElement.parentElement.insertBefore(this.container, this.inputElement.nextSibling);

    // Create inline ghost text element for inline completion
    this.ghostText = document.createElement('div');
    this.ghostText.className = 'autocomplete-inline-ghost';
    this.ghostText.style.display = 'none';

    // Insert ghost text into the input container
    this.inputElement.parentElement.insertBefore(this.ghostText, this.inputElement.nextSibling);
  }

  /**
   * Update container width and max-height to match available space
   * Max-height extends up to the header element
   */
  updateContainerSize() {
    const inputWidth = this.inputElement.offsetWidth;
    const inputLeft = this.inputElement.offsetLeft;

    if (inputWidth > 0) {
      this.container.style.width = inputWidth + 'px';
      this.container.style.left = inputLeft + 'px';
    }

    // Calculate max-height based on available space to header
    const header = document.querySelector('.header');
    if (header) {
      const inputRect = this.inputElement.getBoundingClientRect();
      const headerRect = header.getBoundingClientRect();

      // Available space = distance from top of input to bottom of header
      // Subtract 8px for margin/padding
      const availableHeight = inputRect.top - headerRect.bottom - 8;

      if (availableHeight > 100) {
        // Minimum 100px
        this.container.style.maxHeight = `${availableHeight}px`;
      } else {
        this.container.style.maxHeight = '300px'; // Fallback
      }
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
      // Handle Tab for inline completion (priority over dropdown)
      if (e.key === 'Tab' && this.inlineCompletionVisible) {
        e.preventDefault();
        this.acceptInlineCompletion();
        return;
      }

      // Handle Escape for both inline completion and dropdown
      if (e.key === 'Escape') {
        e.preventDefault();
        this.hideInlineCompletion();
        this.hide();
        return;
      }

      // Rest of the handlers only apply when dropdown is visible
      if (!this.isVisible) return;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          this.selectNext();
          // Update inline completion to match selected item
          this.updateInlineCompletion();
          break;
        case 'ArrowUp':
          e.preventDefault();
          this.selectPrevious();
          // Update inline completion to match selected item
          this.updateInlineCompletion();
          break;
        case 'Enter':
          if (this.selectedIndex >= 0) {
            e.preventDefault();
            this.insertSuggestion(this.suggestions[this.selectedIndex]);
          }
          break;
        case 'Tab':
          // Tab with dropdown visible - insert selected suggestion
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
        this.hideInlineCompletion();
        // Disable auto-show when input loses focus
        this.autoShowDropdown = false;
      }, 200);
    });

    // Focus event - enable auto-show and display suggestions if available
    this.inputElement.addEventListener('focus', () => {
      // Enable auto-showing of suggestions now that input is focused
      this.autoShowDropdown = true;

      // Show suggestions if they're already available
      if (this.inputElement.value.trim().length === 0) {
        if (this.shouldShowSuggestedPrompts()) {
          this.showSuggestedPromptsDropdown();
        } else if (this.customSuggestions.length > 0 || this.previousCustomSuggestions.length > 0) {
          // Show custom suggestions if available
          this.renderMergedSuggestions();
        }
      }
    });
  }

  /**
   * Handle input changes with debouncing
   * Triggers both regular backend suggestions (500ms) and custom LLM suggestions (5000ms)
   * Previous suggestions are always preserved at score: 50
   * @param {string} value - Current input value
   */
  handleInput(value) {
    // Clear existing timers
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    if (this.customSuggestionsTimer) {
      clearTimeout(this.customSuggestionsTimer);
    }

    // Cancel any in-flight custom suggestion requests
    if (this.customSuggestionsAbortController) {
      this.customSuggestionsAbortController.abort();
      this.customSuggestionsAbortController = null;
    }

    // Store current input
    this.lastInput = value;

    // If input is empty, show suggested prompts and keep previous custom suggestions
    if (value.trim().length === 0) {
      // Trigger custom suggestions regeneration with empty input after 5 seconds
      if (this.options.enableCustomSuggestions) {
        this.customSuggestionsTimer = setTimeout(() => {
          this.fetchCustomSuggestions('');
        }, 5000); // 5 seconds idle time
      }

      if (this.shouldShowSuggestedPrompts()) {
        this.showSuggestedPromptsDropdown();
      } else {
        // Show previous custom suggestions even if no suggested prompts
        this.renderMergedSuggestions();
      }
      return;
    }

    // Input has text - trigger both suggestion types with different delays

    // 1. Fast: Regular backend autocomplete (500ms)
    this.debounceTimer = setTimeout(() => {
      this.fetchSuggestions(value);
    }, this.options.debounceMs);

    // 2. Slow: Custom LLM-powered suggestions (5000ms idle)
    if (this.options.enableCustomSuggestions) {
      this.customSuggestionsTimer = setTimeout(() => {
        this.fetchCustomSuggestions(value);
      }, 5000); // 5 seconds idle time
    }
  }

  /**
   * Fetch suggestions from backend
   * @param {string} input - Current input text
   */
  async fetchSuggestions(input) {
    try {
      window.visualLogger.debug(`[Autocomplete] Fetching suggestions for: "${input}"`);

      const response = await fetch('api/autocomplete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': window.sessionId || '',
        },
        body: JSON.stringify({
          input: input,
          max_suggestions: this.options.maxSuggestions,
        }),
      });

      const data = await response.json();

      if (data.status === 'success') {
        this.suggestions = data.suggestions || [];
        this.showingSuggestedPrompts = false; // Regular autocomplete
        window.visualLogger.debug(
          `[Autocomplete] Received ${this.suggestions.length} backend suggestions`
        );

        // Merge with custom suggestions if available
        this.renderMergedSuggestions();
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
   * Fetch custom LLM-powered suggestions (async, triggered after 1s idle)
   * Uses context_chat from backend and LLM routing
   * @param {string} input - Current input text
   */
  async fetchCustomSuggestions(input) {
    try {
      window.visualLogger.debug(`[Autocomplete] Fetching custom LLM suggestions for: "${input}"`);

      // Create new abort controller for this request
      this.customSuggestionsAbortController = new AbortController();

      const response = await fetch('api/autocomplete/custom', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': window.sessionId || '',
        },
        body: JSON.stringify({
          input: input,
          max_suggestions: this.options.maxSuggestions,
        }),
        signal: this.customSuggestionsAbortController.signal,
      });

      const data = await response.json();

      if (data.status === 'success') {
        // Move current suggestions to previous (with reduced score)
        if (this.customSuggestions.length > 0) {
          this.previousCustomSuggestions = this.customSuggestions.map((s) => ({
            ...s,
            score: 50.0, // Reduce score for previous suggestions
            isPrevious: true, // Mark as previous for potential styling
          }));
        }

        // Store new suggestions
        this.customSuggestions = data.suggestions || [];
        window.visualLogger.debug(
          `[Autocomplete] Received ${this.customSuggestions.length} custom LLM suggestions`
        );

        // Merge with backend suggestions and re-render
        this.renderMergedSuggestions();
      } else {
        window.visualLogger.error('[Autocomplete] Custom suggestions API error:', data.error);
      }
    } catch (error) {
      // Ignore abort errors (user kept typing)
      if (error.name === 'AbortError') {
        window.visualLogger.debug('[Autocomplete] Custom suggestions request aborted');
        return;
      }
      window.visualLogger.error('[Autocomplete] Custom suggestions fetch error:', error);
    } finally {
      this.customSuggestionsAbortController = null;
    }
  }

  /**
   * Merge backend suggestions with custom LLM suggestions and render
   * Order: Backend (fast) → Custom LLM (current, score 95) → Previous Custom (score 50)
   */
  renderMergedSuggestions() {
    // Combine all suggestion types
    const mergedSuggestions = [
      ...this.suggestions, // Backend autocomplete (fast)
      ...this.customSuggestions, // Custom LLM (current, score: 95)
      ...this.previousCustomSuggestions, // Previous LLM (score: 50, always visible)
    ];

    if (mergedSuggestions.length > 0) {
      // Temporarily set suggestions to merged list for rendering
      const originalSuggestions = this.suggestions;
      this.suggestions = mergedSuggestions;
      this.render();

      // Update inline completion with first suggestion
      this.updateInlineCompletion();

      // Only show dropdown if input is focused (autoShowDropdown flag)
      if (this.autoShowDropdown) {
        this.show();
      }

      this.suggestions = originalSuggestions; // Restore for next fetch
    } else if (
      this.suggestions.length === 0 &&
      this.customSuggestions.length === 0 &&
      this.previousCustomSuggestions.length === 0
    ) {
      this.hide();
      this.hideInlineCompletion();
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

      // Add score
      if (suggestion.score !== undefined) {
        const score = document.createElement('span');
        score.className = 'autocomplete-score';
        score.textContent = suggestion.score.toFixed(1);
        item.appendChild(score);
      }

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
   * Update inline ghost text completion based on first suggestion or selected suggestion
   * Shows completion text after cursor position with transparency
   * @param {number} suggestionIndex - Optional index of suggestion to use (default: selected or first)
   */
  updateInlineCompletion(suggestionIndex = null) {
    const inputValue = this.inputElement.value;

    // Determine which suggestion to use
    let targetIndex = suggestionIndex;
    if (targetIndex === null) {
      // Use selected suggestion if one is selected, otherwise use first
      targetIndex = this.selectedIndex >= 0 ? this.selectedIndex : 0;
    }

    const targetSuggestion = this.suggestions[targetIndex];

    // Hide if no suggestion or input is empty
    if (!targetSuggestion || inputValue.length === 0) {
      this.hideInlineCompletion();
      return;
    }

    // Get the suggestion text (use fullText for suggestions, text otherwise)
    const suggestionText = targetSuggestion.fullText || targetSuggestion.text;

    // Check if suggestion starts with current input (case-insensitive for better UX)
    if (!suggestionText.toLowerCase().startsWith(inputValue.toLowerCase())) {
      this.hideInlineCompletion();
      return;
    }

    // Calculate the completion part (what comes after current input)
    const completionText = suggestionText.substring(inputValue.length);

    if (completionText.length === 0) {
      this.hideInlineCompletion();
      return;
    }

    // Store the completion for Tab key handler
    this.currentInlineCompletion = completionText;

    // Create a canvas to measure text width for proper positioning
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const inputStyles = window.getComputedStyle(this.inputElement);
    context.font = `${inputStyles.fontSize} ${inputStyles.fontFamily}`;

    // Measure the width of the typed text
    const typedTextWidth = context.measureText(inputValue).width;

    // Position ghost text after the typed text
    this.ghostText.textContent = inputValue + completionText;
    this.ghostText.style.display = 'block';
    this.ghostText.style.paddingLeft = `calc(15px + ${typedTextWidth}px)`;
    this.inlineCompletionVisible = true;

    window.visualLogger.debug(
      `[Autocomplete] Inline completion: "${completionText}" (from: "${suggestionText}")`
    );
  }

  /**
   * Hide inline ghost text completion
   */
  hideInlineCompletion() {
    if (this.ghostText) {
      this.ghostText.style.display = 'none';
      this.inlineCompletionVisible = false;
      this.currentInlineCompletion = '';
    }
  }

  /**
   * Accept the current inline completion by inserting it into the input
   */
  acceptInlineCompletion() {
    if (!this.inlineCompletionVisible || !this.currentInlineCompletion) {
      return false;
    }

    // Insert the completion text
    this.inputElement.value += this.currentInlineCompletion;

    // Trigger input event to update listeners
    this.inputElement.dispatchEvent(new Event('input', { bubbles: true }));

    // Hide the ghost text
    this.hideInlineCompletion();

    // Move cursor to end
    this.inputElement.setSelectionRange(
      this.inputElement.value.length,
      this.inputElement.value.length
    );

    window.visualLogger.info('[Autocomplete] Inline completion accepted');
    return true;
  }

  /**
   * Get icon for suggestion type
   * @param {string} type - Suggestion type (command, tool, file, suggestion)
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
      case 'suggestion':
        return '💡';
      default:
        return '•';
    }
  }

  /**
   * Show autocomplete dropdown
   */
  show() {
    // Update width and max-height (in case window was resized)
    this.updateContainerSize();
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
    // Note: Don't automatically hide inline completion here
    // It should persist independently of the dropdown
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

    this.selectedIndex =
      this.selectedIndex <= 0 ? this.suggestions.length - 1 : this.selectedIndex - 1;
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
        behavior: 'smooth',
      });
    }
  }

  /**
   * Insert selected suggestion into input
   * Populates input field but does NOT auto-submit
   * @param {Object} suggestion - Suggestion to insert
   */
  insertSuggestion(suggestion) {
    window.visualLogger.info(`[Autocomplete] Selected: ${suggestion.text} (${suggestion.type})`);

    // Track click for suggested prompts (type: 'suggestion')
    if (suggestion.type === 'suggestion') {
      this.trackPromptClick(suggestion.fullText || suggestion.text);
    }

    // Set input value based on type
    if (suggestion.type === 'command') {
      this.inputElement.value = suggestion.text + ' ';
    } else if (suggestion.type === 'suggestion') {
      // For suggested prompts, use the full text (not display text)
      this.inputElement.value = suggestion.fullText || suggestion.text;
    } else {
      // For tools and files
      this.inputElement.value = suggestion.text;
    }

    // Trigger input event to update any listeners
    this.inputElement.dispatchEvent(new Event('input', { bubbles: true }));

    // Call custom onSelect callback
    this.options.onSelect(suggestion);

    // Hide dropdown and inline completion
    this.hide();
    this.hideInlineCompletion();
    this.inputElement.focus();

    // Move cursor to end
    this.inputElement.setSelectionRange(
      this.inputElement.value.length,
      this.inputElement.value.length
    );

    // NOTE: Does NOT auto-submit - user must press Enter to send
  }

  /**
   * Track a prompt suggestion click for analytics and future prompt generation
   * @param {string} prompt - The clicked prompt text
   */
  async trackPromptClick(prompt) {
    try {
      window.visualLogger.debug(`[Autocomplete] Tracking prompt click: "${prompt}"`);

      await fetch('api/chat/track-prompt-click', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': window.sessionId || '',
        },
        body: JSON.stringify({
          prompt: prompt,
        }),
      });

      window.visualLogger.debug('[Autocomplete] Prompt click tracked successfully');
    } catch (error) {
      // Fail silently - tracking is non-critical
      window.visualLogger.debug(`[Autocomplete] Failed to track prompt click: ${error}`);
    }
  }

  /**
   * Fetch suggested prompts from backend
   * Called on initialization and when prompts are regenerated
   */
  async fetchSuggestedPrompts() {
    try {
      window.visualLogger.debug('[Autocomplete] Fetching suggested prompts');

      const response = await fetch('api/chat/suggested-prompts', {
        headers: {
          'X-Session-ID': window.sessionId || '',
        },
      });

      const data = await response.json();

      if (data.status === 'success' && data.prompts) {
        this.suggestedPrompts = data.prompts.map((prompt) => {
          // Hide file paths at end (same pattern as card UI)
          const pathPattern = /\s+[/~].+$/;
          const displayText = pathPattern.test(prompt)
            ? prompt.replace(pathPattern, '').trim()
            : prompt;

          return {
            text: displayText,
            fullText: prompt,
            type: 'suggestion',
            description: 'Suggested prompt',
            score: 100,
          };
        });

        window.visualLogger.debug(
          `[Autocomplete] Loaded ${this.suggestedPrompts.length} suggested prompts`
        );
      }
    } catch (error) {
      window.visualLogger.error('[Autocomplete] Failed to fetch suggested prompts:', error);
      this.suggestedPrompts = [];
    }
  }

  /**
   * Check if suggested prompts should be shown
   * @returns {boolean} True if conditions are met (empty input + no real user messages)
   */
  shouldShowSuggestedPrompts() {
    // Check if chat state is available
    if (!this.options.chatState) {
      return false;
    }

    // Check if any non-bootstrap user messages exist in chat
    // Bootstrap messages are training examples and don't count as real conversation
    const hasRealUserMessages = this.options.chatState.messages.some(
      (msg) => msg.role === 'user' && !msg.metadata?.is_bootstrap
    );

    return !hasRealUserMessages && this.suggestedPrompts.length > 0;
  }

  /**
   * Show suggested prompts in dropdown
   */
  showSuggestedPromptsDropdown() {
    if (this.suggestedPrompts.length === 0) {
      this.hide();
      return;
    }

    this.suggestions = this.suggestedPrompts;
    this.showingSuggestedPrompts = true;
    this.render();

    // Update inline completion with first suggested prompt
    this.updateInlineCompletion();

    // Only show dropdown if input is focused (autoShowDropdown flag)
    if (this.autoShowDropdown) {
      this.show();
    }
  }

  /**
   * Update suggested prompts (called from WebSocket event)
   * @param {Array<string>} prompts - New prompt strings
   */
  updateSuggestedPrompts(prompts) {
    window.visualLogger.info('[Autocomplete] Updating suggested prompts from WebSocket');

    this.suggestedPrompts = prompts.map((prompt) => {
      // Hide file paths at end
      const pathPattern = /\s+[/~].+$/;
      const displayText = pathPattern.test(prompt)
        ? prompt.replace(pathPattern, '').trim()
        : prompt;

      return {
        text: displayText,
        fullText: prompt,
        type: 'suggestion',
        description: 'Suggested prompt',
        score: 100,
      };
    });

    // Auto-show regenerated prompts if input is empty (even if conversation exists)
    // This is triggered by user action (clicking regenerate button), so always show
    if (this.inputElement.value.trim().length === 0 && this.suggestedPrompts.length > 0) {
      // Temporarily enable auto-show for user-initiated regeneration
      const wasAutoShow = this.autoShowDropdown;
      this.autoShowDropdown = true;
      this.showSuggestedPromptsDropdown();
      this.autoShowDropdown = wasAutoShow;
    }
  }

  /**
   * Destroy the autocomplete manager
   */
  destroy() {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    if (this.customSuggestionsTimer) {
      clearTimeout(this.customSuggestionsTimer);
    }
    if (this.container && this.container.parentElement) {
      this.container.parentElement.removeChild(this.container);
    }
    if (this.ghostText && this.ghostText.parentElement) {
      this.ghostText.parentElement.removeChild(this.ghostText);
    }
    window.visualLogger.info('[Autocomplete] Destroyed');
  }
}

// Expose to window for global access
window.AutocompleteManager = AutocompleteManager;
