/**
 * Main Application - Initializes and orchestrates all components
 *
 * Data Flow (DAG):
 * - DOMContentLoaded → Initialize components:
 *   1. IdentityManager → Generate/retrieve UUID
 *   2. SessionManager → Load agents, load sessions
 *   3. ChatInterface → Setup event listeners
 *   4. AgentWebSocket → Ready for connections
 * - Auto-refresh: setInterval() → SessionManager.refreshSessions() (every 30s)
 * - Page unload: beforeunload → AgentWebSocket.disconnect()
 *
 * Component Lifecycle:
 * - IdentityManager: Standalone (provides UUID to others)
 * - SessionManager: Drives UI (calls ChatInterface, AgentWebSocket)
 * - ChatInterface: Responds to SessionManager, calls AgentWebSocket
 * - AgentWebSocket: Responds to ChatInterface, updates ChatInterface
 *
 * Global Error Handling:
 * - window.error: Logs unhandled errors
 * - window.unhandledrejection: Logs promise rejections
 */

console.log('[Main] Agent Management SaaS starting...');

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('[Main] DOM loaded, initializing components...');

    // Initialize identity manager
    window.identityManager = new IdentityManager();
    const userUUID = window.identityManager.getUUID();
    console.log(`[Main] User UUID: ${userUUID}`);

    // Display UUID in header
    const userUUIDEl = document.getElementById('user-uuid');
    if (userUUIDEl) {
        userUUIDEl.textContent = `UUID: ${userUUID.substring(0, 8)}...`;
        userUUIDEl.title = userUUID;
    }

    // Initialize components
    window.sessionManager = new SessionManager();
    window.chatInterface = new ChatInterface();
    window.agentWebSocket = new AgentWebSocket();

    console.log('[Main] All components initialized');

    // Always use dark theme
    document.documentElement.setAttribute('data-theme', 'dark');

    // Auto-refresh sessions every 30 seconds
    setInterval(() => {
        if (window.sessionManager) {
            console.log('[Main] Auto-refreshing sessions...');
            window.sessionManager.refreshSessions();
        }
    }, 30000);

    console.log('[Main] Application ready');
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    console.log('[Main] Page unloading, cleaning up...');

    if (window.agentWebSocket) {
        window.agentWebSocket.disconnect();
    }
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('[Main] Global error:', event.error);
});

// Unhandled promise rejection handler
window.addEventListener('unhandledrejection', (event) => {
    console.error('[Main] Unhandled promise rejection:', event.reason);
});
