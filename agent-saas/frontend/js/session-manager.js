/**
 * Session Manager - Handles agent session list sidebar
 *
 * Data Flow (DAG):
 * - constructor() → init() → loadAvailableAgents() → populateAgentSelect()
 *                         → refreshSessions() → renderSessions()
 * - User action → handleSpawn() → createSession() → refreshSessions() → selectSession()
 * - User action → selectSession() → ChatInterface.setActiveSession()
 * - User action → terminateSession() → refreshSessions()
 *
 * Dependencies:
 * - window.identityManager (IdentityManager): User UUID management
 * - window.chatInterface (ChatInterface): Chat panel updates
 * - Backend API: /api/agents/list, /api/agents/sessions/*
 *
 * @class
 */
class SessionManager {
    /**
     * Initialize SessionManager with DOM references and state
     * @constructor
     */
    constructor() {
        this.sessions = [];
        this.activeSessionId = null;

        // DOM elements
        this.sessionListEl = document.getElementById('session-list');
        this.activeSessionsCountEl = document.getElementById('active-sessions-count');
        this.agentTypeSelectEl = document.getElementById('agent-type-select');
        this.initialPromptEl = document.getElementById('initial-prompt');
        this.spawnSessionBtn = document.getElementById('spawn-session-btn');

        this.availableAgents = [];

        this.init();
    }

    /**
     * Initialize event listeners and load initial data
     * Data Flow: init() → loadAvailableAgents() + refreshSessions()
     * @returns {void}
     */
    init() {
        console.log('[SessionManager] Initializing...');

        // Bind spawn button event
        this.spawnSessionBtn.addEventListener('click', () => this.handleSpawn());

        // Enable/disable spawn button based on agent selection
        this.agentTypeSelectEl.addEventListener('change', () => {
            this.spawnSessionBtn.disabled = !this.agentTypeSelectEl.value;
        });

        // Load available agents
        this.loadAvailableAgents();

        // Initial load
        this.refreshSessions();
    }

    /**
     * Load available agent types from backend
     * Data Flow: loadAvailableAgents() → populateAgentSelect()
     * API: GET /api/agents/list
     * @returns {Promise<void>}
     */
    async loadAvailableAgents() {
        try {
            const response = await fetch('api/agents/list');
            const data = await response.json();

            if (data.status === 'success') {
                this.availableAgents = data.agents;
                console.log(`[SessionManager] Loaded ${data.count} available agents`);
                this.populateAgentSelect();
            }
        } catch (error) {
            console.error('[SessionManager] Failed to load agents:', error);
            this.availableAgents = [];
            this.agentTypeSelectEl.innerHTML = '<option value="">No agents available</option>';
        }
    }

    /**
     * Populate agent type dropdown with available agents
     * Data Flow: Reads this.availableAgents, populates DOM select element
     * @returns {void}
     */
    populateAgentSelect() {
        // Clear existing options
        this.agentTypeSelectEl.innerHTML = '';

        if (this.availableAgents.length === 0) {
            this.agentTypeSelectEl.innerHTML = '<option value="">No agents available</option>';
            this.spawnSessionBtn.disabled = true;
            return;
        }

        // Add placeholder
        const placeholderOption = document.createElement('option');
        placeholderOption.value = '';
        placeholderOption.textContent = 'Select an agent type...';
        this.agentTypeSelectEl.appendChild(placeholderOption);

        // Add agent options
        this.availableAgents.forEach(agent => {
            const option = document.createElement('option');
            option.value = agent.name;
            option.textContent = `${agent.icon} ${agent.display_name}`;
            this.agentTypeSelectEl.appendChild(option);
        });

        this.spawnSessionBtn.disabled = true;
    }

    /**
     * Refresh session list from backend
     * Data Flow: refreshSessions() → renderSessions() + updateSessionCount()
     * API: GET /api/agents/sessions/list?user_uuid={uuid}
     * @returns {Promise<void>}
     */
    async refreshSessions() {
        console.log('[SessionManager] Refreshing sessions...');

        try {
            const userUUID = window.identityManager.getUUID();
            const response = await fetch(`api/agents/sessions/list?user_uuid=${userUUID}`);
            const data = await response.json();

            if (data.status === 'success') {
                this.sessions = data.sessions;
                console.log(`[SessionManager] Loaded ${this.sessions.length} sessions`);
                this.renderSessions();
                this.updateSessionCount();
            }
        } catch (error) {
            console.error('[SessionManager] Failed to refresh sessions:', error);
            this.showError('Failed to load sessions');
        }
    }

    /**
     * Render session list in sidebar
     * Data Flow: Reads this.sessions, creates DOM elements via createSessionElement()
     * @returns {void}
     */
    renderSessions() {
        if (this.sessions.length === 0) {
            this.sessionListEl.innerHTML = `
                <div class="session-list-empty">
                    <p>No active sessions</p>
                    <p class="help-text">Spawn a new session to get started</p>
                </div>
            `;
            return;
        }

        this.sessionListEl.innerHTML = '';

        this.sessions.forEach(session => {
            const sessionEl = this.createSessionElement(session);
            this.sessionListEl.appendChild(sessionEl);
        });
    }

    /**
     * Create DOM element for a single session
     * @param {Object} session - Session object from backend
     * @param {string} session.session_id - Unique session identifier
     * @param {string} session.agent_type - Agent type name
     * @param {string} session.status - Session status (running/stopped/error)
     * @param {string} session.created_at - ISO timestamp
     * @param {string} session.last_activity - ISO timestamp
     * @returns {HTMLDivElement} Session list item element
     */
    createSessionElement(session) {
        const div = document.createElement('div');
        div.className = 'session-item';
        div.dataset.sessionId = session.session_id;

        if (session.session_id === this.activeSessionId) {
            div.classList.add('active');
        }

        const statusClass = session.status === 'running' ? 'running' :
                           session.status === 'stopped' ? 'stopped' : 'error';

        const createdTime = new Date(session.created_at).toLocaleString();
        const lastActivity = new Date(session.last_activity).toLocaleString();

        div.innerHTML = `
            <div class="session-item-header">
                <div class="session-item-title">
                    <span class="session-status ${statusClass}"></span>
                    <span>${this.getAgentIcon(session.agent_type)} ${this.getAgentDisplayName(session.agent_type)}</span>
                </div>
            </div>
            <div class="session-item-meta">
                <div class="session-agent-type">${session.agent_type}</div>
                <div>Created: ${createdTime}</div>
                <div>Active: ${lastActivity}</div>
            </div>
        `;

        div.addEventListener('click', () => {
            this.selectSession(session.session_id);
        });

        return div;
    }

    /**
     * Get icon for agent type
     * @param {string} agentType - Agent type name
     * @returns {string} Emoji icon
     */
    getAgentIcon(agentType) {
        const agent = this.availableAgents.find(a => a.name === agentType);
        return agent ? agent.icon : '🤖';
    }

    /**
     * Get display name for agent type
     * @param {string} agentType - Agent type name
     * @returns {string} Human-readable agent name
     */
    getAgentDisplayName(agentType) {
        const agent = this.availableAgents.find(a => a.name === agentType);
        return agent ? agent.display_name : agentType;
    }

    /**
     * Select a session and notify chat interface
     * Data Flow: selectSession() → ChatInterface.setActiveSession()
     * @param {string} sessionId - Session ID to select
     * @returns {void}
     */
    selectSession(sessionId) {
        console.log(`[SessionManager] Selecting session: ${sessionId}`);

        this.activeSessionId = sessionId;

        // Update UI
        document.querySelectorAll('.session-item').forEach(el => {
            el.classList.remove('active');
        });
        const sessionEl = document.querySelector(`[data-session-id="${sessionId}"]`);
        if (sessionEl) {
            sessionEl.classList.add('active');
        }

        // Notify chat interface
        const session = this.sessions.find(s => s.session_id === sessionId);
        if (session && window.chatInterface) {
            window.chatInterface.setActiveSession(session);
        }
    }

    /**
     * Update session count display in status bar
     * @returns {void}
     */
    updateSessionCount() {
        const runningCount = this.sessions.filter(s => s.status === 'running').length;
        this.activeSessionsCountEl.textContent = `${runningCount} active session${runningCount !== 1 ? 's' : ''}`;
    }

    /**
     * Handle spawn button click
     * Data Flow: handleSpawn() → createSession()
     * @returns {Promise<void>}
     */
    async handleSpawn() {
        const agentType = this.agentTypeSelectEl.value;
        const initialPrompt = this.initialPromptEl.value.trim();

        if (!agentType) {
            alert('Please select an agent type');
            return;
        }

        await this.createSession(agentType, initialPrompt);
    }

    /**
     * Create new agent session
     * Data Flow: createSession() → API → refreshSessions() → selectSession()
     * API: POST /api/agents/sessions/create
     * @param {string} agentType - Agent type to spawn
     * @param {string} [initialPrompt=''] - Optional initial message to send
     * @returns {Promise<void>}
     */
    async createSession(agentType, initialPrompt = '') {
        console.log(`[SessionManager] Creating session for agent: ${agentType}`);

        try {
            const userUUID = window.identityManager.getUUID();

            const requestBody = {
                user_uuid: userUUID,
                agent_type: agentType
            };

            // Add initial prompt if provided
            if (initialPrompt) {
                requestBody.initial_prompt = initialPrompt;
            }

            const response = await fetch('api/agents/sessions/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                console.log(`[SessionManager] Session created: ${data.session.session_id}`);

                // Clear spawn interface
                this.agentTypeSelectEl.selectedIndex = 0;
                this.initialPromptEl.value = '';
                this.spawnSessionBtn.disabled = true;

                await this.refreshSessions();
                this.selectSession(data.session.session_id);
            } else {
                throw new Error(data.detail || 'Session creation failed');
            }
        } catch (error) {
            console.error('[SessionManager] Failed to create session:', error);
            alert(`Failed to create session: ${error.message}`);
        }
    }

    /**
     * Terminate an agent session
     * Data Flow: terminateSession() → API → refreshSessions()
     * API: DELETE /api/agents/sessions/{sessionId}
     * @param {string} sessionId - Session ID to terminate
     * @returns {Promise<void>}
     */
    async terminateSession(sessionId) {
        if (!confirm('Are you sure you want to terminate this session? This cannot be undone.')) {
            return;
        }

        console.log(`[SessionManager] Terminating session: ${sessionId}`);

        try {
            const response = await fetch(`api/agents/sessions/${sessionId}`, {
                method: 'DELETE'
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                console.log(`[SessionManager] Session terminated: ${sessionId}`);

                // Clear active session if it was terminated
                if (sessionId === this.activeSessionId) {
                    this.activeSessionId = null;
                    if (window.chatInterface) {
                        window.chatInterface.clearActiveSession();
                    }
                }

                await this.refreshSessions();
            } else {
                throw new Error(data.detail || 'Termination failed');
            }
        } catch (error) {
            console.error('[SessionManager] Failed to terminate session:', error);
            alert(`Failed to terminate session: ${error.message}`);
        }
    }

    /**
     * Display error message in session list
     * @param {string} message - Error message to display
     * @returns {void}
     */
    showError(message) {
        this.sessionListEl.innerHTML = `
            <div class="error">
                <p>${message}</p>
            </div>
        `;
    }
}

// Export globally
window.SessionManager = SessionManager;
