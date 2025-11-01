/**
 * Common TypeScript-style type definitions for CLI-Agent frontend
 * @fileoverview JSDoc type definitions used across frontend modules
 */

/**
 * Tool execution information stored in message metadata
 * @typedef {Object} ToolInfo
 * @property {string} name - Tool name
 * @property {string} [args] - Tool arguments/input
 * @property {string} [status] - Execution status (success|error|pending)
 * @property {number} [execution_time] - Execution duration in seconds
 * @property {string} [model] - Model used by tool
 * @property {string} [context_returned] - Tool output/context
 * @property {number} [duration_ms] - Execution duration in milliseconds (legacy field)
 * @property {number} [context_length] - Output length (legacy field)
 * @property {string} [error] - Error message if failed
 */

/**
 * Message-level metadata structure
 * @typedef {Object} MessageMetadata
 * @property {boolean} [is_bootstrap] - Whether this is a bootstrap training message
 * @property {string} [timestamp] - ISO 8601 timestamp
 * @property {string} [model] - LLM model used to generate this message
 * @property {ToolInfo[]} [tool_info] - Information about tools executed in this message
 * @property {number} [message_id] - Unique message identifier
 * @property {string} [response_model] - Model used for response (legacy field)
 */

/**
 * Chat-level metadata structure
 * @typedef {Object} ChatMetadata
 * @property {string} [created_at] - ISO 8601 creation timestamp
 * @property {string} [updated_at] - ISO 8601 last update timestamp
 */

/**
 * Message object structure
 * @typedef {Object} Message
 * @property {string} role - Message role (user|assistant|system)
 * @property {string} content - Message text content
 * @property {string[]} [images] - Optional base64-encoded images
 * @property {MessageMetadata} [metadata] - Message metadata
 * @property {string} [type] - Content type for visual responses
 */

/**
 * Chat state structure
 * @typedef {Object} ChatState
 * @property {Message[]} messages - Chat message history
 * @property {string|null} currentChatId - Active chat ID
 * @property {ChatMetadata} metadata - Chat metadata
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
