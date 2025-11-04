/**
 * Backend-specific types for multi-user and container management
 */

/**
 * Container information for multi-user mode
 */
export interface ContainerInfo {
  /** Container UUID */
  uuid: string
  /** Container status */
  status: 'running' | 'stopped' | 'error' | 'pending'
  /** Backend URL for this container */
  backend_url: string
  /** Additional container metadata */
  metadata?: Record<string, any>
}

/**
 * Container health check response
 */
export interface ContainerHealth {
  /** Health status */
  status: 'healthy' | 'unhealthy' | 'unknown'
  /** Container uptime in seconds */
  uptime?: number
  /** Last activity timestamp */
  last_activity?: string
  /** Error message if unhealthy */
  error?: string
}

/**
 * Backend discovery response
 */
export interface BackendDiscovery {
  /** Whether backend was found */
  found: boolean
  /** Backend URL */
  url?: string
  /** Container info */
  container?: ContainerInfo
  /** Error message if not found */
  error?: string
}

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
