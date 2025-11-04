/**
 * WebSocket-specific type definitions
 */

import type { Message, ToolInfo } from './index'

/**
 * Permission request data from backend
 */
export interface PermissionData {
  tool_name: string
  action_description: string
  file_path?: string
  args_display?: string
}

/**
 * Permission request message
 */
export interface PermissionRequest {
  id: string
  permission_data: PermissionData
}

/**
 * WebSocket message types
 */
export type WSMessageType =
  | 'session_init'
  | 'chat_replaced'
  | 'chat_cleared'
  | 'chat_truncated'
  | 'message_added'
  | 'message_deleted'
  | 'message_edited'
  | 'state_changed'
  | 'message_start'
  | 'message_chunk'
  | 'message_end'
  | 'token_stream'
  | 'tool_update'
  | 'prompts_regenerated'
  | 'config'
  | 'permission_request'
  | 'user_message'
  | 'assistant_message'
  | 'console_log'
  | 'screenshot'
  | 'html'
  | 'markdown'
  | 'audio'

/**
 * WebSocket message action types
 */
export type WSMessageAction =
  | 'create'
  | 'update'
  | 'delete'
  | 'clear'
  | 'display'
  | 'connection_init'
  | 'tool_start'
  | 'tool_complete'
  | 'loading_tool_start'
  | 'loading_tool_complete'
  | 'audio_tts'
  | 'error'
  | 'captured'
  | 'failed'

/**
 * Generic WebSocket message structure
 */
export interface WSMessage<T = any> {
  type: WSMessageType
  action?: WSMessageAction
  id?: string
  model?: string
  content?: any
  data?: T
  tool_info?: ToolInfo[]
  tool_name?: string
  tool_args?: string
  tool_status?: 'success' | 'error'
  tool_duration_ms?: number
  tool_result?: string
  tool_error?: string
  message?: string
  line?: string
  level?: 'info' | 'warn' | 'error' | 'debug'
  status?: string
  permission_data?: PermissionData
  continue_mode?: boolean
  clear_history?: boolean
  session_id?: string
  text?: string
  audio_format?: string
  audio_data?: string
  duration?: number
  language?: string
}

/**
 * Event handler callback type
 */
export type EventHandler<T = any> = (data: T) => void

/**
 * Event handlers registry
 */
export type EventHandlersRegistry = Record<string, EventHandler[]>
