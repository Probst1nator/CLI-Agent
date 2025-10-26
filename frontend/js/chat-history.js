/**
 * Chat History Sidebar Component
 *
 * Manages the chat history sidebar with search, filtering, and context-based discovery.
 * Features BM25 search, fuzzy matching, and sidebar resize functionality.
 *
 * @class ChatHistorySidebar
 * @author CLI-Agent Development Team
 * @version 1.0.0
 */

/**
 * @typedef {Object} ChatItem
 * @property {string} id - Unique chat identifier
 * @property {number} timestamp - Unix timestamp of chat creation
 * @property {string} preview - Chat content preview
 * @property {number} message_count - Number of messages in chat
 * @property {string} [search_snippet] - Highlighted snippet from search (when searching)
 * @property {number} [score] - BM25 relevance score (when searching)
 */

/**
 * @typedef {Object} SearchResult
 * @property {string} status - Status of search ('success' or 'error')
 * @property {Array<ChatItem>} [results] - Array of matching chat items
 * @property {string} [error] - Error message if status is 'error'
 */

/**
 * @typedef {Object} ChatGroup
 * @property {string} label - Group label ('Today', 'Yesterday', 'This Week', 'Older')
 * @property {Array<ChatItem>} chats - Chats in this group
 */

/**
 * @typedef {Object} SidebarState
 * @property {boolean} expanded - Whether sidebar is expanded
 * @property {number} width - Sidebar width in pixels
 */

class ChatHistorySidebar {
    /**
     * Initialize the chat history sidebar
     */
    constructor() {
        /**
         * Sidebar container element
         * @type {HTMLElement|null}
         */
        this.sidebar = document.getElementById('chat-history-sidebar');

        /**
         * Content container element
         * @type {HTMLElement|null}
         */
        this.content = document.getElementById('chat-history-content');

        /**
         * Search input element
         * @type {HTMLInputElement|null}
         */
        this.searchInput = document.getElementById('chat-history-search-input');

        /**
         * Toggle button element
         * @type {HTMLElement|null}
         */
        this.toggleBtn = document.getElementById('chat-history-toggle-btn');

        /**
         * Resize handle element
         * @type {HTMLElement|null}
         */
        this.resizeHandle = document.getElementById('chat-history-resize-handle');

        /**
         * Whether sidebar is expanded
         * @type {boolean}
         */
        this.isExpanded = false;

        /**
         * Current sidebar width in pixels
         * @type {number}
         */
        this.sidebarWidth = 350;

        /**
         * All loaded chats
         * @type {Array<ChatItem>}
         */
        this.chats = [];

        /**
         * Current search query
         * @type {string}
         */
        this.searchQuery = '';

        /**
         * Whether user is currently resizing sidebar
         * @type {boolean}
         */
        this.isResizing = false;

        /**
         * Debounce timer for search input
         * @type {number|null}
         */
        this.searchDebounceTimer = null;

        this.setupEventListeners();
        this.loadState();
        this.loadChats();
    }

    /**
     * Setup event listeners for sidebar interactions
     * Includes toggle, search (with debounce), resize, and new chat button
     * @returns {void}
     */
    setupEventListeners() {
        if (this.toggleBtn) {
            this.toggleBtn.addEventListener('click', () => this.toggle());
        }

        const newChatBtn = document.getElementById('new-chat-btn');
        if (newChatBtn) {
            newChatBtn.addEventListener('click', () => this.startNewChat());
        }

        if (this.searchInput) {
            this.searchInput.addEventListener('input', (e) => {
                this.searchQuery = e.target.value.toLowerCase();

                // Debounce search requests (wait 200ms after user stops typing)
                if (this.searchDebounceTimer) {
                    clearTimeout(this.searchDebounceTimer);
                }

                this.searchDebounceTimer = setTimeout(() => {
                    this.filterAndRenderChats();
                }, 200);
            });
        }

        const contextSearchBtn = document.getElementById('context-search-btn');
        if (contextSearchBtn) {
            contextSearchBtn.addEventListener('click', () => {
                // Clear search input and trigger context-based search
                this.searchInput.value = '';
                this.searchQuery = '';
                this.filterAndRenderChats();
            });
        }

        if (this.resizeHandle) {
            this.resizeHandle.addEventListener('mousedown', (e) => this.startResize(e));
        }

        document.addEventListener('mousemove', (e) => this.handleResize(e));
        document.addEventListener('mouseup', () => this.stopResize());
    }

    /**
     * Start a new chat
     * Triggers chat state manager to create new chat and reload history
     * @async
     * @returns {Promise<void>}
     */
    async startNewChat() {
        if (window.chatState) {
            await window.chatState.startNewChat();

            setTimeout(() => {
                this.loadChats();
            }, 500);
        } else {
            console.error('ChatStateManager not available');
            alert('Chat system not ready. Please refresh the page.');
        }
    }

    /**
     * Load all chats from backend
     * @async
     * @returns {Promise<void>}
     */
    async loadChats() {
        try {
            const url = window.backendConnector ? await window.backendConnector.buildURL('/api/chat/archives') : '/api/chat/archives';
            const response = await fetch(url);
            const data = await response.json();

            if (data.status === 'success') {
                this.chats = data.archives || data.chats || [];
                this.renderChats();
            }
        } catch (error) {
            console.error('Failed to load chat history:', error);
            this.content.innerHTML = '<div class="chat-history-error">Failed to load chat history</div>';
        }
    }

    /**
     * Filter and render chats based on search query or current context
     * Uses BM25 search on backend when query exists, otherwise shows context-based similar chats
     * @async
     * @returns {Promise<void>}
     */
    async filterAndRenderChats() {
        // If no explicit search query, use current chat context
        let searchQuery = this.searchQuery.trim();
        let isContextSearch = false;

        if (!searchQuery) {
            // Extract important terms from current chat
            searchQuery = this.extractCurrentChatContext();
            if (searchQuery) {
                isContextSearch = true;
            } else {
                // No search query and no context - show all chats
                this.renderChats(this.chats);
                return;
            }
        }

        // Show loading indicator
        const loadingMessage = isContextSearch
            ? 'Finding similar chats based on current context...'
            : 'Searching chats...';
        this.content.innerHTML = `<div class="chat-history-loading"><div class="loading-spinner"></div><div class="loading-text">${loadingMessage}</div></div>`;

        try {
            const response = await fetch(`/api/chat/search?q=${encodeURIComponent(searchQuery)}`);
            const data = await response.json();

            if (data.status === 'success') {
                const filtered = data.results || [];
                this.renderChats(filtered, isContextSearch);
            } else {
                console.error('Search failed:', data.error);
                this.renderChats([]);
            }
        } catch (error) {
            console.error('Failed to search chats:', error);
            const filtered = this.chats.filter(chat =>
                chat.preview.toLowerCase().includes(searchQuery)
            );
            this.renderChats(filtered);
        }
    }

    /**
     * Extract context from current chat for similarity search
     * Extracts top 5 most frequent non-common words from recent messages
     * @returns {string} Space-separated context keywords
     */
    extractCurrentChatContext() {
        // Get current chat messages from ChatStateManager
        if (!window.chatState || !window.chatState.messages) {
            return '';
        }

        const messages = window.chatState.messages;
        if (messages.length === 0) {
            return '';
        }

        // Extract key terms from recent messages (last 5)
        const recentMessages = messages.slice(-5);
        const allText = recentMessages
            .map(msg => msg.content || '')
            .join(' ');

        // Extract important words (ignore common words)
        const commonWords = new Set([
            'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'von', 'zu', 'im', 'ist', 'sind',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are',
            'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'you', 'he', 'she', 'it', 'we', 'they',
            'kann', 'kannst', 'können', 'will', 'soll', 'muss', 'can', 'could', 'should', 'would', 'must'
        ]);

        const words = allText
            .toLowerCase()
            .match(/\b\w{4,}\b/g) || []; // Words with 4+ characters

        // Count word frequencies
        const wordFreq = {};
        for (const word of words) {
            if (!commonWords.has(word)) {
                wordFreq[word] = (wordFreq[word] || 0) + 1;
            }
        }

        // Get top 5 most frequent words
        const topWords = Object.entries(wordFreq)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([word]) => word);

        return topWords.join(' ');
    }

    /**
     * Render chats grouped by date
     * @param {Array<ChatItem>} [chatsToRender=this.chats] - Chats to render
     * @param {boolean} [isContextSearch=false] - Whether this is a context-based search
     * @returns {void}
     */
    renderChats(chatsToRender = this.chats, isContextSearch = false) {
        if (chatsToRender.length === 0) {
            this.content.innerHTML = '<div class="chat-history-empty">No chats found</div>';
            return;
        }

        const grouped = this.groupChatsByDate(chatsToRender);
        let html = '';

        // Show context search indicator
        if (isContextSearch) {
            html += '<div class="chat-history-context-notice">📎 Similar to current chat</div>';
        }

        let chatIndex = chatsToRender.length;

        for (const [label, chats] of Object.entries(grouped)) {
            html += `<div class="chat-history-group">`;
            html += `<div class="chat-history-group-label">📅 ${label}</div>`;

            for (const chat of chats) {
                const date = new Date(chat.timestamp * 1000);
                const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                const chatTitle = `Chat #${chatIndex}`;

                // Use search snippet if available (when searching), otherwise show last chars of preview
                let displayPreview;
                if (chat.search_snippet && this.searchQuery) {
                    // Highlight matching terms in snippet
                    displayPreview = this.highlightMatches(chat.search_snippet, this.searchQuery);
                } else {
                    let preview = chat.preview || 'No preview available';

                    // Extract content from <content> tags if present
                    const contentMatch = preview.match(/<content>([\s\S]*?)<\/content>/);
                    if (contentMatch) {
                        preview = contentMatch[1].trim();
                        // Remove CDATA wrapper if present
                        if (preview.startsWith('<![CDATA[') && preview.endsWith(']]>')) {
                            preview = preview.substring(9, preview.length - 3).trim();
                        }
                    }

                    // Show last 100 characters
                    const truncated = preview.length > 100 ? '...' + preview.substring(preview.length - 100) : preview;
                    displayPreview = this.escapeHtml(truncated);
                }

                // Add relevance score indicator if searching
                const scoreIndicator = chat.score !== undefined
                    ? `<span class="chat-history-item-score" title="BM25 relevance score">Relevance: ${chat.score.toFixed(2)}</span>`
                    : '';

                html += `
                    <div class="chat-history-item" data-chat-id="${chat.id}">
                        <div class="chat-history-item-header">
                            <span class="chat-history-item-title">${chatTitle}</span>
                            ${scoreIndicator}
                            <button class="chat-history-item-delete" data-chat-id="${chat.id}" title="Delete chat">🗑️</button>
                        </div>
                        <div class="chat-history-item-time">${timeStr}</div>
                        <div class="chat-history-item-preview">${displayPreview}</div>
                        <div class="chat-history-item-meta">${chat.message_count} messages</div>
                    </div>
                `;
                chatIndex--;
            }

            html += `</div>`;
        }

        this.content.innerHTML = html;

        this.content.querySelectorAll('.chat-history-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (!e.target.classList.contains('chat-history-item-delete')) {
                    this.loadChat(item.dataset.chatId);
                }
            });
        });

        this.content.querySelectorAll('.chat-history-item-delete').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteChat(btn.dataset.chatId);
            });
        });
    }

    /**
     * Highlight matching terms in text using fuzzy matching
     * Uses exact match, stem match, and n-gram fuzzy matching with Jaccard similarity
     * @param {string} text - Text to highlight
     * @param {string} query - Search query
     * @returns {string} HTML string with highlighted matches
     */
    highlightMatches(text, query) {
        if (!query || !text) return this.escapeHtml(text);

        const escaped = this.escapeHtml(text);
        const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 0);

        let highlighted = escaped;

        // Helper function to generate n-grams
        const generateNgrams = (str, n = 3) => {
            if (str.length < n) return [str];
            const grams = [];
            for (let i = 0; i <= str.length - n; i++) {
                grams.push(str.substring(i, i + n));
            }
            return grams;
        };

        // Helper function to calculate Jaccard similarity
        const jaccardSimilarity = (set1, set2) => {
            const intersection = set1.filter(x => set2.includes(x)).length;
            const union = new Set([...set1, ...set2]).size;
            return union === 0 ? 0 : intersection / union;
        };

        // Simple stemming function
        const simpleStem = (word) => {
            const suffixes = ['ung', 'heit', 'keit', 'isch', 'lich', 'en', 'er', 'es', 'e', 's'];
            const lower = word.toLowerCase();
            for (const suffix of suffixes) {
                if (lower.endsWith(suffix) && lower.length > suffix.length + 2) {
                    return lower.slice(0, -suffix.length);
                }
            }
            return lower;
        };

        // Create stems and n-grams for query terms
        const queryStems = terms.map(t => simpleStem(t));
        const queryNgrams = terms.map(t => t.length >= 3 ? generateNgrams(t) : []);

        // Find and highlight matches in text
        const words = escaped.split(/(\s+)/);
        const highlightedWords = words.map(word => {
            const cleanWord = word.replace(/[^\w]/g, '');
            if (!cleanWord) return word;

            const lowerClean = cleanWord.toLowerCase();

            // 1. Exact match
            if (terms.includes(lowerClean)) {
                return word.replace(cleanWord, `<mark class="exact">${cleanWord}</mark>`);
            }

            // 2. Stem match
            const wordStem = simpleStem(cleanWord);
            if (queryStems.includes(wordStem)) {
                return word.replace(cleanWord, `<mark class="stem">${cleanWord}</mark>`);
            }

            // 3. Fuzzy n-gram match
            if (cleanWord.length >= 3) {
                const wordNgrams = generateNgrams(lowerClean);
                for (let i = 0; i < queryNgrams.length; i++) {
                    if (queryNgrams[i].length > 0) {
                        const similarity = jaccardSimilarity(wordNgrams, queryNgrams[i]);
                        if (similarity > 0.5) {
                            return word.replace(cleanWord, `<mark class="fuzzy">${cleanWord}</mark>`);
                        }
                    }
                }
            }

            return word;
        });

        return highlightedWords.join('');
    }

    /**
     * Escape regex special characters
     * @param {string} str - String to escape
     * @returns {string} Escaped string
     */
    escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    /**
     * Group chats by date (Today, Yesterday, This Week, Older)
     * @param {Array<ChatItem>} chats - Chats to group
     * @returns {Object<string, Array<ChatItem>>} Grouped chats
     */
    groupChatsByDate(chats) {
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);
        const weekAgo = new Date(today);
        weekAgo.setDate(weekAgo.getDate() - 7);

        const groups = {
            'Today': [],
            'Yesterday': [],
            'This Week': [],
            'Older': []
        };

        for (const chat of chats) {
            const chatDate = new Date(chat.timestamp * 1000);
            const chatDay = new Date(chatDate.getFullYear(), chatDate.getMonth(), chatDate.getDate());

            if (chatDay.getTime() === today.getTime()) {
                groups['Today'].push(chat);
            } else if (chatDay.getTime() === yesterday.getTime()) {
                groups['Yesterday'].push(chat);
            } else if (chatDate >= weekAgo) {
                groups['This Week'].push(chat);
            } else {
                groups['Older'].push(chat);
            }
        }

        const result = {};
        for (const [label, chats] of Object.entries(groups)) {
            if (chats.length > 0) {
                result[label] = chats;
            }
        }

        return result;
    }

    /**
     * Load a specific chat by ID
     * @async
     * @param {string} chatId - Chat ID to load
     * @returns {Promise<void>}
     */
    async loadChat(chatId) {
        if (window.chatState) {
            await window.chatState.loadChat(chatId);
        } else {
            console.error('ChatStateManager not available');
            alert('Chat system not ready. Please refresh the page.');
        }
    }

    /**
     * Delete a chat by ID
     * @async
     * @param {string} chatId - Chat ID to delete
     * @returns {Promise<void>}
     */
    async deleteChat(chatId) {
        const confirmed = confirm('Delete this chat permanently?');
        if (!confirmed) return;

        try {
            const response = await fetch(`/api/chat/archive/${chatId}`, {
                method: 'DELETE'
            });
            const data = await response.json();

            if (data.status === 'success') {
                await this.loadChats();
            } else {
                alert(`Failed to delete chat: ${data.message}`);
            }
        } catch (error) {
            console.error('Failed to delete chat:', error);
            alert('Failed to delete chat');
        }
    }

    /**
     * Escape HTML to prevent XSS
     * @param {string} text - Text to escape
     * @returns {string} Escaped HTML string
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Toggle sidebar expanded/collapsed
     * @returns {void}
     */
    toggle() {
        if (this.isExpanded) {
            this.collapse();
        } else {
            this.expand();
        }
    }

    /**
     * Expand the sidebar
     * @returns {void}
     */
    expand() {
        this.isExpanded = true;
        this.sidebar.classList.add('expanded');
        this.updateWidth();
        this.saveState();
    }

    /**
     * Collapse the sidebar
     * @returns {void}
     */
    collapse() {
        this.isExpanded = false;
        this.sidebar.classList.remove('expanded');
        this.updateWidth();
        this.saveState();
    }

    /**
     * Update sidebar width based on expanded state
     * @returns {void}
     */
    updateWidth() {
        if (this.isExpanded) {
            this.sidebar.style.width = `${this.sidebarWidth}px`;
        } else {
            this.sidebar.style.width = '60px';
        }
    }

    /**
     * Start resizing sidebar
     * @param {MouseEvent} e - Mouse event
     * @returns {void}
     */
    startResize(e) {
        this.isResizing = true;
        e.preventDefault();
    }

    /**
     * Handle sidebar resize (dragging resize handle)
     * Constrains width between 250px and 600px
     * @param {MouseEvent} e - Mouse event
     * @returns {void}
     */
    handleResize(e) {
        if (!this.isResizing) return;

        const newWidth = e.clientX;
        if (newWidth >= 250 && newWidth <= 600) {
            this.sidebarWidth = newWidth;
            this.sidebar.style.width = `${newWidth}px`;
        }
    }

    /**
     * Stop resizing sidebar
     * @returns {void}
     */
    stopResize() {
        if (this.isResizing) {
            this.isResizing = false;
            this.saveState();
        }
    }

    /**
     * Save sidebar state to localStorage
     * @returns {void}
     */
    saveState() {
        try {
            const state = {
                expanded: this.isExpanded,
                width: this.sidebarWidth
            };
            localStorage.setItem('chat-history-sidebar-state', JSON.stringify(state));
        } catch (e) {
            console.warn('Failed to save chat history sidebar state:', e);
        }
    }

    /**
     * Load sidebar state from localStorage
     * @returns {void}
     */
    loadState() {
        try {
            const stored = localStorage.getItem('chat-history-sidebar-state');
            if (stored) {
                const state = JSON.parse(stored);
                this.sidebarWidth = state.width || 350;
                if (state.expanded) {
                    this.expand();
                }
            }
        } catch (e) {
            console.warn('Failed to load chat history sidebar state:', e);
        }
    }
}

// Export to window for global access
window.ChatHistoryManager = ChatHistorySidebar;

/**
 * Global instance of chat history sidebar
 * @type {ChatHistorySidebar|null}
 */
let chatHistorySidebar = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    chatHistorySidebar = new ChatHistorySidebar();
    window.chatHistorySidebar = chatHistorySidebar;
});
