/**
 * Todos Sidebar Manager
 * Manages the todos sidebar display, including parsing todos from tool calls
 * and auto-showing when todos are updated.
 */

class TodosManager {
    constructor() {
        /** @type {boolean} */
        this.todosVisible = false;

        /** @type {Array<{content: string, status: string, activeForm: string}>} */
        this.currentTodos = [];

        /** @type {number} */
        this.todosWidth = 350; // Default width

        /** @type {boolean} */
        this.isResizing = false;

        /** @type {boolean} */
        this.autoShowEnabled = true;

        this.init();
    }

    init() {
        this.setupToggleButton();
        this.setupResizeHandle();
        this.setupAutoShowToggle();
        this.loadTodosState();
    }

    setupToggleButton() {
        const toggleBtn = document.getElementById('todos-toggle-btn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                this.toggleTodos();
            });
        }
    }

    setupAutoShowToggle() {
        const autoShowCheckbox = document.getElementById('todos-auto-show');
        if (autoShowCheckbox) {
            autoShowCheckbox.addEventListener('change', (e) => {
                this.autoShowEnabled = e.target.checked;
                this.saveTodosState();
            });
        }
    }

    setupResizeHandle() {
        const resizeHandle = document.getElementById('todos-resize-handle');
        if (!resizeHandle) return;

        resizeHandle.addEventListener('mousedown', (e) => {
            this.isResizing = true;
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';

            const onMouseMove = (e) => {
                if (!this.isResizing) return;

                const sidebar = document.getElementById('todos-sidebar');
                if (!sidebar) return;

                // Calculate new width based on mouse position
                const containerRect = sidebar.parentElement.getBoundingClientRect();
                const newWidth = containerRect.right - e.clientX;

                // Apply constraints
                const minWidth = 200;
                const constrainedWidth = Math.max(minWidth, newWidth);

                // Update width
                this.todosWidth = constrainedWidth;
                sidebar.style.width = constrainedWidth + 'px';

                // Save to localStorage
                this.saveTodosState();
            };

            const onMouseUp = () => {
                this.isResizing = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
            };

            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);

            e.preventDefault();
        });
    }

    toggleTodos() {
        this.todosVisible = !this.todosVisible;
        this.updateTodosVisibility();
        this.saveTodosState();
    }

    showTodos() {
        if (!this.todosVisible) {
            this.todosVisible = true;
            this.updateTodosVisibility();
            this.saveTodosState();
        }
    }

    hideTodos() {
        if (this.todosVisible) {
            this.todosVisible = false;
            this.updateTodosVisibility();
            this.saveTodosState();
        }
    }

    updateTodosVisibility() {
        const sidebar = document.getElementById('todos-sidebar');
        const toggleBtn = document.getElementById('todos-toggle-btn');

        if (sidebar) {
            if (this.todosVisible) {
                sidebar.classList.remove('hidden');
                // Apply saved width when showing
                sidebar.style.width = this.todosWidth + 'px';
            } else {
                sidebar.classList.add('hidden');
            }
        }

        if (toggleBtn) {
            if (this.todosVisible) {
                toggleBtn.classList.add('active');
            } else {
                toggleBtn.classList.remove('active');
            }
        }
    }

    /**
     * Parse todos from tool call content
     * Expects XML format with <todos> tag containing array of todo items
     * @param {string} content - XML content from todos tool call
     * @returns {Array<{content: string, status: string, activeForm: string}>}
     */
    parseTodos(content) {
        try {
            // Extract JSON array from content
            // The content should be a JSON array of todo objects
            const jsonMatch = content.match(/\[[\s\S]*\]/);
            if (jsonMatch) {
                const todos = JSON.parse(jsonMatch[0]);
                return todos;
            }
            return [];
        } catch (error) {
            window.logError('Failed to parse todos:', error);
            return [];
        }
    }

    /**
     * Update todos from tool call
     * @param {string} content - XML content from todos tool call
     */
    updateTodos(content) {
        const newTodos = this.parseTodos(content);
        this.currentTodos = newTodos;
        this.renderTodos();

        // Auto-show if enabled and there are todos
        if (this.autoShowEnabled && newTodos.length > 0) {
            this.showTodos();
        }
    }

    /**
     * Render todos in the sidebar
     */
    renderTodos() {
        const content = document.getElementById('todos-content');
        if (!content) return;

        // Clear existing content
        content.innerHTML = '';

        if (this.currentTodos.length === 0) {
            // Show empty state
            content.innerHTML = `
                <div class="todos-empty">
                    <span class="todos-empty-icon">✓</span>
                    <span class="todos-empty-text">No todos yet...</span>
                </div>
            `;
            return;
        }

        // Render each todo
        this.currentTodos.forEach((todo, index) => {
            const todoItem = document.createElement('div');
            todoItem.className = `todo-item ${todo.status}`;

            // Choose icon based on status
            let icon = '○'; // pending
            if (todo.status === 'completed') {
                icon = '✓';
            } else if (todo.status === 'in_progress') {
                icon = '◉';
            }

            // Use activeForm if status is in_progress, otherwise use content
            const displayText = todo.status === 'in_progress' ? todo.activeForm : todo.content;

            todoItem.innerHTML = `
                <span class="todo-status-icon">${icon}</span>
                <div class="todo-content">
                    <div class="todo-text">${this.escapeHtml(displayText)}</div>
                </div>
            `;

            content.appendChild(todoItem);
        });
    }

    /**
     * Escape HTML to prevent XSS
     * @param {string} text
     * @returns {string}
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Save todos state to localStorage
     */
    saveTodosState() {
        try {
            const state = {
                visible: this.todosVisible,
                width: this.todosWidth,
                autoShow: this.autoShowEnabled
            };
            localStorage.setItem('todosState', JSON.stringify(state));
        } catch (error) {
            window.logError('Failed to save todos state:', error);
        }
    }

    /**
     * Load todos state from localStorage
     */
    loadTodosState() {
        try {
            const stateStr = localStorage.getItem('todosState');
            if (stateStr) {
                const state = JSON.parse(stateStr);
                this.todosVisible = state.visible || false;
                this.todosWidth = state.width || 350;
                this.autoShowEnabled = state.autoShow !== undefined ? state.autoShow : true;

                // Update UI
                this.updateTodosVisibility();

                const autoShowCheckbox = document.getElementById('todos-auto-show');
                if (autoShowCheckbox) {
                    autoShowCheckbox.checked = this.autoShowEnabled;
                }
            }
        } catch (error) {
            window.logError('Failed to load todos state:', error);
        }
    }
}

// Make globally available
window.TodosManager = TodosManager;
