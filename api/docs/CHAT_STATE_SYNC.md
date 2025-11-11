# Chat State Synchronization Architecture

## Overview

The chat state synchronization uses a **growing message pattern** where assistant messages naturally grow via content concatenation. The backend broadcasts after every state change, and the frontend uses simple role-based logic to determine whether to update or add.

## Core Principle

> **"One growing assistant message - backend concatenates, frontend replaces"**

The `Chat.add_message()` method automatically concatenates content when the same role adds consecutively. The frontend simply checks: if last message is assistant and incoming is assistant → replace (content grew), otherwise add new message.

## Backend Architecture

### State Changes That Trigger Broadcast

1. **User message added** → Broadcast full chat
2. **Assistant LLM response added** → Broadcast full chat
3. **Tools execute and add metadata** → Broadcast full chat
4. **Any message modification** → Broadcast full chat

### Broadcast Event Structure

```javascript
{
  type: 'chat_sync',
  chat: {
    messages: [
      {
        role: 'user',
        content: 'message text',
        images: [],
        metadata: {}
      },
      {
        role: 'assistant',
        content: 'response with <tool>...</tool> tags',
        metadata: {
          tool_info: [
            {
              name: 'bash',
              status: 'success',
              duration_ms: 123,
              model: 'gemini-flash-lite-latest',
              args: 'ls',
              context_returned: 'file1.txt\nfile2.txt'
            }
          ]
        }
      }
    ],
    metadata: {
      created_at: "2025-01-30T12:00:00Z",
      updated_at: "2025-01-30T12:05:00Z"
    }
  }
}
```

### Backend Implementation Points

- Single broadcast function: `broadcast_chat_state(chat)`
- Called after any state mutation
- Serializes entire chat to JSON
- Bootstrap messages marked with `metadata.is_bootstrap = true` per-message

## Frontend Architecture

### State Management

The `ChatStateManager` maintains the local chat state and syncs with backend:

```javascript
class ChatStateManager {
  messages: []          // Full message array
  metadata: {}          // Chat-level metadata
}
```

### WebSocket Event Handler

```javascript
wsManager.on('chat_sync', (data) => {
  // Replace entire state
  this.messages = data.chat.messages || [];
  this.metadata = data.chat.metadata || {};

  // Re-render everything
  this.renderAll();
});
```

### Rendering Strategy

The `renderAll()` function:
1. Clears the content wrapper
2. Filters out system messages (`role === 'system'`)
3. Filters out bootstrap messages (`metadata.is_bootstrap === true`)
4. Renders all visible messages
5. Preserves scroll position if needed

### Tool Loading States

Tool execution states are determined by message content:

- **Loading**: `tool_info` exists but no `<context>` tag yet
  - Show tool name with loading spinner
  - Display model and status as "pending"

- **Completed**: `tool_info` exists AND `<context>` tag present
  - Show tool name with checkmark/error icon
  - Display model, duration, and status
  - Allow expanding to see args/results

- **Not Started**: Message contains tool XML but no `tool_info` yet
  - Show tool boxes as "queued"

## Implementation Details

### Backend Changes

**Files to modify:**
- `main.py`: Replace all `broadcast_message_added(message)` calls with `broadcast_chat_state(context_chat)`
- `api/server.py`: Add new `broadcast_chat_state(chat)` function, deprecate `broadcast_message_added`

**Key locations in main.py:**
- Line ~1726: After user message added
- Line ~1755: After assistant response added
- Line ~2281: After tools execute
- Line ~2300: After visualresponse

### Frontend Changes

**Files to modify:**
- `frontend/js/chat-state.js`:
  - Change `message_added` handler to `chat_sync` handler
  - Remove complex matching logic
  - Replace with simple state replacement + renderAll()

**Simplification:**
```javascript
// OLD (complex):
if (lastMsg && lastMsg.role === incomingMsg.role && ...) {
  // Update logic
} else {
  // Add logic
}

// NEW (simple):
this.messages = incomingData.messages;
this.renderAll();
```

### Scroll Position Handling

To prevent jarring scroll jumps during re-renders:

```javascript
const contentWrapper = document.getElementById('content-wrapper');
const wasAtBottom = contentWrapper.scrollHeight - contentWrapper.scrollTop === contentWrapper.clientHeight;

this.renderAll();

if (wasAtBottom) {
  contentWrapper.scrollTop = contentWrapper.scrollHeight;
}
```

## Benefits

1. **Simplicity**: No complex state synchronization logic
2. **Reliability**: Frontend always has correct state from backend
3. **Debuggability**: Just inspect backend chat state to see what should render
4. **Resilience**: Works perfectly after reconnects, page reloads, or dropped messages
5. **Consistency**: Same data structure for HTTP initial load and WebSocket updates

## Performance Considerations

- **Chat length**: Modern browsers easily handle hundreds of messages
- **Re-render cost**: DOM operations are fast for typical chat lengths (< 100 messages)
- **Network payload**: JSON compression makes full state transfer efficient
- **Optimization**: If needed, can add virtual scrolling later without changing architecture

## Migration Notes

### Removing Old Logic

The following can be deleted:
- Content matching logic in `chat-state.js`
- Message index tracking for updates
- Separate `message_added`, `message_deleted`, `message_edited` events (all replaced by `chat_sync`)

### Backward Compatibility

During transition:
- Keep old HTTP endpoint `/api/chat/current` working
- Both return same data structure
- Frontend treats both as full state sync

## Future Enhancements

If performance becomes an issue:
1. Add incremental updates with message IDs
2. Implement virtual scrolling for long chats
3. Add message-level diffing

But start simple - only optimize if needed!
