# Session Sharing

## Architecture

**Single-User Mode** (`python main.py --gui`): All browser tabs share one `g.chat` instance.

```
Process → g.chat ← All WebSocket sessions point to same chat
           ↑
         WS1, WS2, WS3... (multiple browsers/tabs)
```

**SaaS Mode**: Each user's container has its own `g.chat`, but their multiple tabs share it.

```
User A → Container 1 → g.chat A ← WS1, WS2 (User A's tabs)
User B → Container 2 → g.chat B ← WS1 (User B's tab)
```

## Implementation

**Startup** (`main.py`):
```python
g.chat = context_chat  # Main chat instance
```

**WebSocket connects** (`api/server.py:360`):
```python
session_context = g.create_session(session_id, chat=g.chat)  # Share g.chat
```

**Chat property** (`core/globals.py`):
```python
@property
def chat(self):
    """Returns session's chat if exists, else direct storage"""
    return self.get_active_session().chat if session else self._chat

@chat.setter
def chat(self, value):
    """Stores in both _chat and active session"""
    self._chat = value
    if session := self.get_active_session():
        session.chat = value
```

## Message Broadcasting

When messages are added to `g.chat`, all connected browsers receive `message_added` events via WebSocket and update their UI.

## Testing

See `tests/integration/test_session_sharing.py` - verifies multiple sessions share the same chat instance.

## Key Files

- `api/server.py` - Session creation with shared chat
- `core/globals.py` - Chat property with dual storage
- `core/session_context.py` - SessionContext dataclass
- `main.py` - Initial chat creation
