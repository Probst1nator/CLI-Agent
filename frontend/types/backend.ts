/**
 * Permission request structure
 */
export interface PermissionRequest {
  /** Request ID */
  id: string
  /** Tool name requesting permission */
  tool_name: string
  /** Tool arguments */
  args: string
  /** Request timestamp */
  timestamp: number
  /** Additional context */
  context?: Record<string, any>
}

/**
 * Permission response from user
 */
export interface PermissionResponse {
  /** Request ID being responded to */
  request_id: string
  /** Whether permission was granted */
  granted: boolean
  /** Whether to remember this decision */
  remember?: boolean
  /** Additional response data */
  data?: Record<string, any>
}
