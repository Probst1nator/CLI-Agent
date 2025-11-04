/**
 * Core type definitions for CLI-Agent frontend
 * Single source of truth for all frontend types
 */

// ============================================================================
// Message Types
// ============================================================================

/**
 * Tool execution information stored in message metadata
 */
export interface ToolInfo {
  /** Tool name */
  name: string
  /** Tool arguments/input */
  args?: string
  /** Execution status */
  status?: 'success' | 'error' | 'pending'
  /** Execution duration in seconds */
  execution_time?: number
  /** Model used by tool */
  model?: string
  /** Tool output/context */
  context_returned?: string
  /** Execution duration in milliseconds (legacy field) */
  duration_ms?: number
  /** Output length (legacy field) */
  context_length?: number
  /** Error message if failed */
  error?: string
}

/**
 * Message-level metadata structure
 */
export interface MessageMetadata {
  /** Whether this is a bootstrap training message */
  is_bootstrap?: boolean
  /** ISO 8601 timestamp */
  timestamp?: string
  /** LLM model used to generate this message */
  model?: string
  /** Information about tools executed in this message */
  tool_info?: ToolInfo[]
  /** Unique message identifier */
  message_id?: number
  /** Model used for response (legacy field) */
  response_model?: string
}

/**
 * Message object structure
 */
export interface Message {
  /** Message role (user|assistant|system) */
  role: 'user' | 'assistant' | 'system'
  /** Message text content */
  content: string
  /** Optional base64-encoded images */
  images?: string[]
  /** Message metadata */
  metadata?: MessageMetadata
  /** Content type for visual responses (html|markdown|image_upload_ack) */
  type?: 'html' | 'markdown' | 'image_upload_ack' | string
  /** Timestamp for message */
  timestamp?: number
}

/**
 * Chat-level metadata structure
 */
export interface ChatMetadata {
  /** ISO 8601 creation timestamp */
  created_at?: string
  /** ISO 8601 last update timestamp */
  updated_at?: string
}

/**
 * Complete chat data structure
 */
export interface ChatData {
  /** Chat identifier */
  id?: string
  /** Chat message history */
  messages: Message[]
  /** Chat metadata */
  metadata?: ChatMetadata
}

/**
 * Chat state structure for frontend state management
 */
export interface ChatState {
  /** Chat message history */
  messages: Message[]
  /** Active chat ID */
  currentChatId: string | null
  /** Chat metadata */
  metadata: ChatMetadata
}

// ============================================================================
// LLM Types
// ============================================================================

/**
 * LLM model information
 */
export interface LLMModel {
  /** Model identifier */
  id: string
  /** Model display name */
  name: string
  /** Provider name (Google, Anthropic, OpenAI, etc.) */
  provider: string
  /** Context window size */
  context_window: number
  /** Model capabilities */
  capabilities: string[]
  /** Availability status */
  status: string
}

/**
 * LLM configuration structure
 */
export interface LLMConfig {
  /** Override LLM (always use this model if set) */
  override_llm?: string
  /** Router LLM for routing decisions */
  router_llm?: string
  /** Planner LLM for first turn planning */
  planner_llm?: string
  /** Default fallback models */
  default_models?: string[]
}

// ============================================================================
// Tool Types
// ============================================================================

/**
 * Tool definition
 */
export interface ToolDefinition {
  /** Tool name */
  name: string
  /** Tool description */
  description: string
  /** Whether tool is enabled */
  enabled: boolean
  /** Additional tool metadata */
  metadata?: Record<string, any>
}

// ============================================================================
// WebSocket Types
// ============================================================================

/**
 * WebSocket event structure
 */
export interface WebSocketEvent<T = any> {
  /** Event type */
  type: string
  /** Event data payload */
  data?: T
  /** Event timestamp */
  timestamp?: number
}

/**
 * Session initialization data
 */
export interface SessionInitData {
  /** Session identifier */
  session_id: string
  /** Additional session data */
  [key: string]: any
}

// ============================================================================
// Logging Types
// ============================================================================

/**
 * Log entry structure for frontend logging
 */
export interface LogEntry {
  /** Log entry ID */
  id: string
  /** Session ID */
  sessionId: string
  /** Timestamp (ms) */
  timestamp: number
  /** Time since logger start (ms) */
  relativeTime: number
  /** Log level */
  level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'
  /** Log message */
  message: string
  /** Additional log data */
  data: Record<string, any>
  /** Page URL */
  url: string
}

/**
 * Logger configuration
 */
export interface LoggerConfig {
  /** Current log level */
  level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'
  /** Whether to send logs to backend */
  sendToBackend: boolean
  /** Maximum log entries to keep in memory */
  maxEntries: number
}

// ============================================================================
// API Response Types
// ============================================================================

/**
 * Standard API response format
 */
export interface APIResponse<T = any> {
  /** Response status */
  status: 'success' | 'error'
  /** Response data (present on success) */
  data?: T
  /** Error message (present on error) */
  error?: string
  /** Error code (present on error) */
  code?: string
}

/**
 * Chat archive info
 */
export interface ChatArchive {
  /** Chat ID */
  id: string
  /** Chat preview/snippet */
  preview: string
  /** ISO 8601 timestamp */
  timestamp: string
  /** Number of messages */
  message_count: number
  /** Chat metadata */
  metadata?: ChatMetadata
}
