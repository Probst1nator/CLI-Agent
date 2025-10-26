/**
 * Common TypeScript-style type definitions for CLI-Agent frontend
 * @fileoverview JSDoc type definitions used across frontend modules
 */

/**
 * @typedef {Object} Message
 * @property {string} role - Message role (user|assistant|system)
 * @property {string} content - Message text content
 * @property {Object} [metadata] - Optional message metadata
 * @property {ToolInfo[]} [metadata.tool_info] - Tool execution information
 * @property {string} [metadata.response_model] - Model used for response
 * @property {string} [type] - Content type for visual responses
 */

/**
 * @typedef {Object} ToolInfo
 * @property {string} name - Tool name
 * @property {string} args - Tool arguments/input
 * @property {string} status - Execution status (success|error|running)
 * @property {number|null} duration_ms - Execution duration
 * @property {string|null} model - Model used by tool
 * @property {string} [context_returned] - Tool output
 * @property {number} [context_length] - Output length
 * @property {string} [error] - Error message if failed
 */

/**
 * @typedef {Object} ChatState
 * @property {Message[]} messages - Chat message history
 * @property {string|null} currentChatId - Active chat ID
 * @property {Object} metadata - Chat metadata
 */

/**
 * @typedef {Object} LLMModel
 * @property {string} id - Model identifier
 * @property {string} name - Model display name
 * @property {string} provider - Provider name (Google, Anthropic, etc.)
 * @property {number} context_window - Context window size
 * @property {string[]} capabilities - Model capabilities
 * @property {string} status - Availability status
 */

/**
 * @typedef {Object} ToolDefinition
 * @property {string} name - Tool name
 * @property {string} description - Tool description
 * @property {boolean} enabled - Whether tool is enabled
 * @property {Object} [metadata] - Additional tool metadata
 */

/**
 * @typedef {Object} WebSocketEvent
 * @property {string} type - Event type
 * @property {*} [data] - Event data payload
 * @property {number} [timestamp] - Event timestamp
 */

/**
 * @typedef {Object} LogEntry
 * @property {string} id - Log entry ID
 * @property {string} sessionId - Session ID
 * @property {number} timestamp - Timestamp (ms)
 * @property {number} relativeTime - Time since logger start (ms)
 * @property {string} level - Log level (DEBUG|INFO|WARN|ERROR)
 * @property {string} message - Log message
 * @property {Object} data - Additional log data
 * @property {string} url - Page URL
 */

// Export types to window for global access
window.Types = {
    Message: {},
    ToolInfo: {},
    ChatState: {},
    LLMModel: {},
    ToolDefinition: {},
    WebSocketEvent: {},
    LogEntry: {}
};
