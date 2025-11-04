/**
 * ChatStateManager - Frontend state management for chat
 *
 * Provides single source of truth for chat state on the client side.
 * Syncs with backend via API calls and WebSocket events.
 * Enables chat loading without page reload.
 */

import type { Message, ChatData, ChatMetadata } from '../types/index'

declare global {
  interface Window {
    ChatStateManager: typeof ChatStateManager
    showToast?: (message: string, type: 'success' | 'error') => void
    // Note: window.app type is defined in websocket.ts as MainApp
  }
}

interface WebSocketManager {
  on: (event: string, callback: (data: any) => void) => void
}

interface ContentRenderer {
  renderMessage: (message: Message, index: number) => Promise<void>
  clearContent: () => void
}

type EventListener = (data?: any) => void

interface EventListeners {
  chat_loaded: EventListener[]
  chat_cleared: EventListener[]
  message_added: EventListener[]
  state_changed: EventListener[]
}

class ChatStateManager {
  private wsManager: WebSocketManager
  private renderer: ContentRenderer
  private messages: Message[] = []
  private currentChatId: string | null = null
  private metadata: ChatMetadata = {}
  private listeners: EventListeners = {
    'chat_loaded': [],
    'chat_cleared': [],
    'message_added': [],
    'state_changed': []
  }
  private renderingQueue: Promise<void> = Promise.resolve()
  private pendingRender: (() => void) | null = null

  constructor(wsManager: WebSocketManager, renderer: ContentRenderer) {
    this.wsManager = wsManager
    this.renderer = renderer
    this.setupWebSocketHandlers()
    ;(window as any).logInfo?.('ChatStateManager initialized')
  }

  /**
   * Load current chat from backend (on page reload)
   */
  public async loadCurrentChat(): Promise<void> {
    try {
      ;(window as any).logDebug?.('Loading current chat from backend...')
      const url = window.backendConnector ? await window.backendConnector.buildURL('/api/chat/current') : '/api/chat/current'
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 10000)

      try {
        const response = await fetch(url, {
          headers: {
            'X-Session-ID': window.sessionId || ''
          },
          signal: controller.signal
        })
        clearTimeout(timeoutId)

        const data = await response.json()

        if (data.status === 'success' && data.chat) {
          ;(window as any).logDebug?.(`Loaded current chat with ${data.chat.messages ? data.chat.messages.length : 0} messages`)

          this.messages = data.chat.messages || []
          this.currentChatId = data.chat.id || null
          this.metadata = data.chat.metadata || {}

          if (this.messages.length > 0) {
            this.renderAll()
            ;(window as any).logInfo?.('✅ Chat messages rendered on page load')
          } else {
            console.log('No messages to render (new chat)')
          }

          this.emit('chat_loaded', data.chat)
          this.emit('state_changed')
        } else {
          console.log('No active chat or empty response')
        }
      } catch (error) {
        clearTimeout(timeoutId)
        throw error
      }
    } catch (error) {
      console.error('Failed to load current chat:', error)
    }
  }

  /**
   * Load an archived chat by ID
   */
  public async loadChat(chatId: string): Promise<void> {
    try {
      const response = await fetch(`/api/chat/load/${chatId}`, {
        method: 'POST',
        headers: {
          'X-Session-ID': window.sessionId || ''
        }
      })

      const data = await response.json()

      if (data.status === 'success') {
        console.log('Chat load request successful, waiting for sync event...')
      } else {
        throw new Error(data.error || 'Failed to load chat')
      }
    } catch (error) {
      console.error('Failed to load chat:', error)
      this.showError(`Failed to load chat: ${(error as Error).message}`)
      throw error
    }
  }

  /**
   * Start a new chat (archives current conversation)
   */
  public async startNewChat(): Promise<void> {
    try {
      const url = window.backendConnector ? await window.backendConnector.buildURL('/api/chat/new') : '/api/chat/new'
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'X-Session-ID': window.sessionId || ''
        }
      })

      const data = await response.json()

      if (data.status === 'success') {
        console.log('New chat request successful, waiting for sync event...')
      } else {
        throw new Error(data.error || 'Failed to start new chat')
      }
    } catch (error) {
      console.error('Failed to start new chat:', error)
      this.showError(`Failed to start new chat: ${(error as Error).message}`)
      throw error
    }
  }

  /**
   * Add a message to state (called by WebSocket handler)
   */
  public addMessage(message: Message): void {
    this.messages.push(message)
    this.emit('message_added', message)
    this.emit('state_changed')
  }

  /**
   * Replace entire chat state (called by WebSocket handler on chat_replaced event)
   */
  public async replaceChat(chatData: ChatData): Promise<void> {
    ;(window as any).logDebug?.('Replacing chat state with loaded chat:', chatData)

    this.messages = chatData.messages || []
    this.currentChatId = chatData.id || null
    this.metadata = chatData.metadata || {}

    this.emit('chat_loaded', chatData)
    this.emit('state_changed')

    await this.renderAll()

    this.showSuccess('Chat loaded successfully')
  }

  /**
   * Clear chat state (called by WebSocket handler on chat_cleared event)
   */
  public clearChat(): void {
    ;(window as any).logDebug?.('Clearing chat state')

    this.messages = []
    this.currentChatId = null
    this.metadata = {}

    this.emit('chat_cleared')
    this.emit('state_changed')

    this.renderer.clearContent()

    this.showSuccess('New chat started. Previous conversation archived.')
  }

  /**
   * Re-render all messages in UI
   */
  public async renderAll(): Promise<void> {
    ;(window as any).logDebug?.(`Rendering ${this.messages.length} messages`)

    this.renderer.clearContent()

    for (let i = 0; i < this.messages.length; i++) {
      try {
        // Skip system messages
        if (this.messages[i].role === 'system') {
          continue
        }

        // Skip bootstrap messages
        if (this.messages[i].metadata?.is_bootstrap) {
          continue
        }

        // Await to ensure messages render in correct order
        await this.renderer.renderMessage(this.messages[i], i)
      } catch (error) {
        console.error('Failed to render message:', error, this.messages[i])
      }
    }
  }

  /**
   * Queue a render operation to prevent race conditions
   */
  public queueRender(renderFn: () => Promise<void>): Promise<void> {
    this.renderingQueue = this.renderingQueue
      .then(() => renderFn())
      .catch(err => {
        console.error('Render error:', err)
      })
    return this.renderingQueue
  }

  /**
   * Setup WebSocket event handlers for state synchronization
   */
  private setupWebSocketHandlers(): void {
    if (!this.wsManager) {
      console.warn('WebSocket manager not available for ChatStateManager')
      return
    }

    if (typeof this.wsManager.on !== 'function') {
      console.warn('WebSocket manager does not have on() method')
      return
    }

    // Track if we've loaded the chat
    let chatLoaded = false

    // Load chat immediately via HTTP
    this.loadCurrentChat().then(() => {
      chatLoaded = true
      ;(window as any).logInfo?.('✅ Initial chat load complete')
    }).catch((err: Error) => {
      console.warn('Failed to load chat via HTTP, will retry on session_init:', err)
    })

    this.wsManager.on('session_init', async (data: any) => {
      ;(window as any).logInfo?.('Session initialized')
      if (!chatLoaded) {
        await this.loadCurrentChat()
      } else {
        console.log('Chat already loaded, skipping duplicate load')
      }
    })

    this.wsManager.on('chat_replaced', async (data: any) => {
      console.log('chat_replaced event received:', data)
      await this.replaceChat(data.chat)
    })

    this.wsManager.on('chat_cleared', (data: any) => {
      console.log('chat_cleared event received:', data)
      this.clearChat()
    })

    this.wsManager.on('message_added', async (data: any) => {
      console.log('message_added event received:', data)

      const incomingMsg = data.message
      const lastMsg = this.messages[this.messages.length - 1]

      // DEBUG: Log tool_info
      if (incomingMsg.metadata && incomingMsg.metadata.tool_info) {
        console.log('🔍 tool_info received:', incomingMsg.metadata.tool_info)
        incomingMsg.metadata.tool_info.forEach((tool: any, idx: number) => {
          console.log(`  Tool ${idx}: ${tool.name}, model: ${tool.model || 'MISSING'}`)
        })
      }

      // Queue the render to prevent race conditions
      await this.queueRender(async () => {
        // If last message is assistant and incoming is assistant, replace it
        if (lastMsg && lastMsg.role === 'assistant' && incomingMsg.role === 'assistant') {
          console.log('Updating existing assistant message (content/metadata grew)')
          const messageIndex = this.messages.length - 1
          this.messages[messageIndex] = incomingMsg
          this.emit('state_changed')

          // Check if we have an optimistic message to remove
          const contentWrapper = document.getElementById('content-wrapper')
          const optimisticMsgId = window.app?.currentOptimisticMessageId

          if (contentWrapper && optimisticMsgId) {
            const optimisticElement = document.getElementById(optimisticMsgId)
            if (optimisticElement) {
              console.log('💬 Removing optimistic message, real assistant response arrived:', optimisticMsgId)
              optimisticElement.remove()
              if (window.app) {
                window.app.currentOptimisticMessageId = null
              }
            }
          }

          // Remove old DOM element and re-render
          if (contentWrapper) {
            const oldMessageBlock = contentWrapper.querySelector(`[data-message-index="${messageIndex}"]`)
            if (oldMessageBlock) {
              oldMessageBlock.remove()
            }
          }

          // Re-render the updated message
          await this.renderer.renderMessage(incomingMsg, messageIndex)

          // Update interrupt button state
          this.updateInterruptButtonState(incomingMsg)
        } else {
          // New message (user message or new assistant response)
          console.log('Adding new message')

          // Check if we have an optimistic message to remove
          if (incomingMsg.role === 'assistant') {
            const contentWrapper = document.getElementById('content-wrapper')
            const optimisticMsgId = window.app?.currentOptimisticMessageId

            if (contentWrapper && optimisticMsgId) {
              const optimisticElement = document.getElementById(optimisticMsgId)
              if (optimisticElement) {
                console.log('💬 Removing optimistic message, real assistant response arrived:', optimisticMsgId)
                optimisticElement.remove()
                if (window.app) {
                  window.app.currentOptimisticMessageId = null
                }
              }
            }
          }

          this.addMessage(incomingMsg)
          const messageIndex = this.messages.length - 1
          await this.renderer.renderMessage(incomingMsg, messageIndex)

          // Show interrupt button for new assistant messages
          if (incomingMsg.role === 'assistant') {
            this.showInterruptButton()
          }
        }
      })
    })

    this.wsManager.on('chat_truncated', async (data: any) => {
      console.log('chat_truncated event received:', data)
      this.messages = this.messages.slice(0, data.index + 1)
      await this.renderAll()
      this.emit('state_changed')
    })

    this.wsManager.on('message_deleted', async (data: any) => {
      console.log('message_deleted event received:', data)
      if (data.index >= 0 && data.index < this.messages.length) {
        this.messages.splice(data.index, 1)
        await this.renderAll()
        this.emit('state_changed')
      }
    })

    ;(window as any).logDebug?.('WebSocket handlers registered for chat sync')
  }

  /**
   * Update interrupt button state based on message content
   */
  private updateInterruptButtonState(message: Message): void {
    const hasVisualResponse = message.content && (
      message.content.includes('<visualresponse>') ||
      message.content.includes('<response>')
    )

    if (hasVisualResponse) {
      this.hideInterruptButton()
    } else {
      this.showInterruptButton()
    }
  }

  /**
   * Show interrupt button
   */
  private showInterruptButton(): void {
    if (window.app?.showInterruptButton) {
      window.app.showInterruptButton()
    }
  }

  /**
   * Hide interrupt button
   */
  private hideInterruptButton(): void {
    if (window.app?.hideInterruptButton) {
      window.app.hideInterruptButton()
    }
  }

  /**
   * Register an event listener
   */
  public on(event: keyof EventListeners, callback: EventListener): void {
    if (!this.listeners[event]) {
      (this.listeners as any)[event] = []
    }
    this.listeners[event].push(callback)
  }

  /**
   * Emit an event to all registered listeners
   */
  private emit(event: keyof EventListeners, data?: any): void {
    if (this.listeners[event]) {
      for (const callback of this.listeners[event]) {
        try {
          callback(data)
        } catch (error) {
          console.error(`Error in ${event} listener:`, error)
        }
      }
    }
  }

  /**
   * Show success message to user
   */
  private showSuccess(message: string): void {
    if (typeof window.showToast === 'function') {
      window.showToast(message, 'success')
    } else {
      ;(window as any).logInfo?.('✅', message)
    }
  }

  /**
   * Show error message to user
   */
  private showError(message: string): void {
    if (typeof window.showToast === 'function') {
      window.showToast(message, 'error')
    } else {
      console.error('❌', message)
    }
  }
}

window.ChatStateManager = ChatStateManager

export default ChatStateManager
export { ChatStateManager }
