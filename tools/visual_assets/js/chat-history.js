class ChatHistorySidebar {
    constructor() {
        this.sidebar = document.getElementById('chat-history-sidebar');
        this.content = document.getElementById('chat-history-content');
        this.searchInput = document.getElementById('chat-history-search-input');
        this.toggleBtn = document.getElementById('chat-history-toggle-btn');
        this.resizeHandle = document.getElementById('chat-history-resize-handle');

        this.isExpanded = false;
        this.sidebarWidth = 350;
        this.chats = [];
        this.searchQuery = '';
        this.isResizing = false;

        this.setupEventListeners();
        this.loadState();
        this.loadChats();
    }

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
                this.filterAndRenderChats();
            });
        }

        if (this.resizeHandle) {
            this.resizeHandle.addEventListener('mousedown', (e) => this.startResize(e));
        }

        document.addEventListener('mousemove', (e) => this.handleResize(e));
        document.addEventListener('mouseup', () => this.stopResize());
    }

    startNewChat() {
        if (window.app && window.app.wsManager && window.app.wsManager.isConnected()) {
            window.app.wsManager.sendMessage('/new_chat');

            const contentWrapper = document.getElementById('content-wrapper');
            if (contentWrapper) {
                contentWrapper.innerHTML = '<div class="chat-loaded-message">✅ New chat started! Previous conversation archived.</div>';
            }

            setTimeout(() => {
                this.loadChats();
            }, 500);
        } else {
            alert('WebSocket not connected. Please refresh the page.');
        }
    }

    async loadChats() {
        try {
            const response = await fetch('/api/chat_history');
            const data = await response.json();

            if (data.status === 'success') {
                this.chats = data.chats || [];
                this.renderChats();
            }
        } catch (error) {
            console.error('Failed to load chat history:', error);
            this.content.innerHTML = '<div class="chat-history-error">Failed to load chat history</div>';
        }
    }

    filterAndRenderChats() {
        const filtered = this.searchQuery
            ? this.chats.filter(chat =>
                chat.preview.toLowerCase().includes(this.searchQuery)
              )
            : this.chats;

        this.renderChats(filtered);
    }

    renderChats(chatsToRender = this.chats) {
        if (chatsToRender.length === 0) {
            this.content.innerHTML = '<div class="chat-history-empty">No chats found</div>';
            return;
        }

        const grouped = this.groupChatsByDate(chatsToRender);
        let html = '';
        let chatIndex = chatsToRender.length;

        for (const [label, chats] of Object.entries(grouped)) {
            html += `<div class="chat-history-group">`;
            html += `<div class="chat-history-group-label">📅 ${label}</div>`;

            for (const chat of chats) {
                const date = new Date(chat.timestamp * 1000);
                const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                const chatTitle = `Chat #${chatIndex}`;
                const preview = chat.preview || 'No preview available';
                const truncated = preview.length > 50 ? preview.substring(0, 50) + '...' : preview;

                html += `
                    <div class="chat-history-item" data-chat-id="${chat.id}">
                        <div class="chat-history-item-header">
                            <span class="chat-history-item-title">${chatTitle}</span>
                            <button class="chat-history-item-delete" data-chat-id="${chat.id}" title="Delete chat">🗑️</button>
                        </div>
                        <div class="chat-history-item-time">${timeStr}</div>
                        <div class="chat-history-item-preview">${this.escapeHtml(truncated)}</div>
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

    async loadChat(chatId) {
        const confirmed = confirm('Load this chat? Current conversation will be replaced.');
        if (!confirmed) return;

        try {
            const response = await fetch(`/api/chat_history/${chatId}`);
            const data = await response.json();

            if (data.status === 'success') {
                const contentWrapper = document.getElementById('content-wrapper');
                if (contentWrapper) {
                    contentWrapper.innerHTML = '<div class="chat-loaded-message">✅ Chat loaded! Refresh the page to see the restored conversation.</div>';
                }

                console.log('Chat loaded successfully:', data.chat);
            } else {
                alert(`Failed to load chat: ${data.message}`);
            }
        } catch (error) {
            console.error('Failed to load chat:', error);
            alert('Failed to load chat');
        }
    }

    async deleteChat(chatId) {
        const confirmed = confirm('Delete this chat permanently?');
        if (!confirmed) return;

        try {
            const response = await fetch(`/api/chat_history/${chatId}`, {
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

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    toggle() {
        if (this.isExpanded) {
            this.collapse();
        } else {
            this.expand();
        }
    }

    expand() {
        this.isExpanded = true;
        this.sidebar.classList.add('expanded');
        this.updateWidth();
        this.saveState();
    }

    collapse() {
        this.isExpanded = false;
        this.sidebar.classList.remove('expanded');
        this.updateWidth();
        this.saveState();
    }

    updateWidth() {
        if (this.isExpanded) {
            this.sidebar.style.width = `${this.sidebarWidth}px`;
        } else {
            this.sidebar.style.width = '60px';
        }
    }

    startResize(e) {
        this.isResizing = true;
        e.preventDefault();
    }

    handleResize(e) {
        if (!this.isResizing) return;

        const newWidth = e.clientX;
        if (newWidth >= 250 && newWidth <= 600) {
            this.sidebarWidth = newWidth;
            this.sidebar.style.width = `${newWidth}px`;
        }
    }

    stopResize() {
        if (this.isResizing) {
            this.isResizing = false;
            this.saveState();
        }
    }

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

let chatHistorySidebar = null;

document.addEventListener('DOMContentLoaded', () => {
    chatHistorySidebar = new ChatHistorySidebar();
    window.chatHistorySidebar = chatHistorySidebar;
});
