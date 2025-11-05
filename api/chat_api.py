"""
Chat API endpoints for CLI-Agent.

Provides RESTful API for chat operations:
- Loading archived chats
- Creating new chats
- Listing chat history
- Managing chat state
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


async def get_current_chat(request: Any) -> Any:
    """GET /api/chat/current

    Returns the current active chat state.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g

        if hasattr(g, 'chat') and g.chat:
            chat_dict = g.chat._to_dict()
            frontend_chat = convert_chat_to_frontend_format(chat_dict)
            return web.json_response({
                'status': 'success',
                'chat': frontend_chat
            })

        return web.json_response({
            'status': 'success',
            'chat': None
        })

    except Exception as e:
        logger.error(f"Failed to get current chat: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


def convert_chat_to_frontend_format(chat_dict: dict) -> dict:
    """
    Convert chat data to frontend-compatible format.
    Uses persisted metadata for tool_info instead of re-parsing XML.
    """
    frontend_chat = chat_dict.copy()

    if 'messages' in frontend_chat:
        frontend_messages = []
        for msg in frontend_chat['messages']:
            frontend_msg = msg.copy()

            # Use persisted tool_info from metadata if available
            if msg.get('metadata') and 'tool_info' in msg['metadata']:
                frontend_msg['tool_info'] = msg['metadata']['tool_info']

            frontend_messages.append(frontend_msg)
        frontend_chat['messages'] = frontend_messages

    return frontend_chat


async def load_chat(request: Any) -> Any:
    """POST /api/chat/load/{id}

    Loads an archived chat and replaces the current chat.
    Broadcasts 'chat_replaced' event to all WebSocket clients.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        import re

        from core.chat import Chat
        from core.globals import g

        chat_id = request.match_info.get('id')
        if not chat_id:
            return web.json_response({
                'status': 'error',
                'error': 'Chat ID is required',
                'code': 'INVALID_REQUEST'
            }, status=400)

        # Validate chat ID format to prevent path traversal
        # Expected format: chat_TIMESTAMP_HASH (e.g., chat_1730123456_abc1)
        if not re.match(r'^chat_\d{10,}_[a-f0-9]{4}$', chat_id):
            logger.warning(f"Invalid chat ID format attempt: {chat_id}")
            return web.json_response({
                'status': 'error',
                'error': 'Invalid chat ID format',
                'code': 'INVALID_CHAT_ID'
            }, status=400)

        # Additional check for path traversal attempts
        if '..' in chat_id or '/' in chat_id or '\\' in chat_id:
            logger.warning(f"Path traversal attempt detected: {chat_id}")
            return web.json_response({
                'status': 'error',
                'error': 'Invalid chat ID',
                'code': 'INVALID_CHAT_ID'
            }, status=400)

        # Load from archive
        try:
            loaded_chat = Chat.load_from_archive(chat_id)
        except FileNotFoundError:
            return web.json_response({
                'status': 'error',
                'error': f'Chat {chat_id} not found',
                'code': 'CHAT_NOT_FOUND'
            }, status=404)

        # Replace global chat
        g.chat = loaded_chat
        logger.info(f"Loaded chat {chat_id}, replacing current chat")

        # Convert to frontend format with type field
        frontend_chat = convert_chat_to_frontend_format(loaded_chat._to_dict())

        # Broadcast to all connected clients
        from api.server import get_server
        server = get_server()
        if server:
            await server.broadcast({
                'type': 'chat_replaced',
                'chat': frontend_chat,
                'timestamp': time.time()
            })

        return web.json_response({
            'status': 'success',
            'chat': frontend_chat
        })

    except Exception as e:
        logger.error(f"Failed to load chat: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def new_chat(request: Any) -> Any:
    """POST /api/chat/new

    Archives current chat and starts a new one.
    Broadcasts 'chat_cleared' event to all WebSocket clients.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g

        # Archive current chat
        if hasattr(g, 'chat') and g.chat and len(g.chat.messages) > 2:
            try:
                g.chat.save_to_json(also_archive=True)
                logger.info("Archived current chat before creating new one")
            except Exception as e:
                logger.warning(f"Failed to archive current chat: {e}")

        # Create new chat
        try:
            # Import here to avoid circular dependency
            import sys
            main_module = sys.modules.get('__main__')
            if main_module and hasattr(main_module, 'construct_engineered_prompt'):
                construct_engineered_prompt = main_module.construct_engineered_prompt
            else:
                # Fallback - create empty chat
                from core.chat import Chat
                new_chat = Chat()
                logger.warning("construct_engineered_prompt not available, created empty chat")
                g.chat = new_chat

            if hasattr(main_module, 'construct_engineered_prompt'):
                gui_mode = hasattr(g, 'GUI_MODE') and g.GUI_MODE
                # Don't add bootstrap when creating new chat via API
                new_chat = await construct_engineered_prompt("", gui_mode, add_bootstrap=False)
                g.chat = new_chat
                logger.info("Created new chat with engineered prompt")

        except Exception as e:
            logger.error(f"Failed to create new chat: {e}")
            # Create minimal fallback chat
            from core.chat import Chat, Role
            new_chat = Chat()
            new_chat.add_message(Role.SYSTEM, "You are a helpful AI assistant.")
            g.chat = new_chat

        # Save the new chat to recent_chat.json to prevent loading old archived chat
        if hasattr(g, 'chat') and g.chat:
            try:
                g.chat.save_to_json(also_archive=False)
                logger.info("Saved new empty chat to recent_chat.json")
            except Exception as e:
                logger.warning(f"Failed to save new chat: {e}")

        # Set flag to tell main loop this was an intentional chat reset
        g.CHAT_INTENTIONALLY_CLEARED = True

        # Broadcast to all connected clients
        from api.server import get_server
        server = get_server()
        if server:
            await server.broadcast({
                'type': 'chat_cleared',
                'timestamp': time.time()
            })

        return web.json_response({
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Failed to create new chat: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def get_messages(request: Any) -> Any:
    """GET /api/chat/messages

    Returns all messages in current chat.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.globals import g

        if hasattr(g, 'chat') and g.chat:
            messages = [msg.to_dict() if hasattr(msg, 'to_dict') else str(msg)
                        for msg in g.chat.messages]
            return web.json_response({
                'status': 'success',
                'messages': messages
            })

        return web.json_response({
            'status': 'success',
            'messages': []
        })

    except Exception as e:
        logger.error(f"Failed to get messages: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def list_archived_chats(request: Any) -> Any:
    """GET /api/chat/archives

    Lists all archived chats.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.chat import Chat

        chats = Chat.list_archived_chats()
        return web.json_response({
            'status': 'success',
            'chats': chats
        })

    except Exception as e:
        logger.error(f"Failed to list archived chats: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def delete_archived_chat(request: Any) -> Any:
    """DELETE /api/chat/archive/{id}

    Deletes an archived chat.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        import re

        from core.chat import Chat

        chat_id = request.match_info.get('id')
        if not chat_id:
            return web.json_response({
                'status': 'error',
                'error': 'Chat ID is required',
                'code': 'INVALID_REQUEST'
            }, status=400)

        # Validate chat ID format to prevent path traversal
        if not re.match(r'^chat_\d{10,}_[a-f0-9]{4}$', chat_id):
            logger.warning(f"Invalid chat ID format attempt: {chat_id}")
            return web.json_response({
                'status': 'error',
                'error': 'Invalid chat ID format',
                'code': 'INVALID_CHAT_ID'
            }, status=400)

        # Additional check for path traversal attempts
        if '..' in chat_id or '/' in chat_id or '\\' in chat_id:
            logger.warning(f"Path traversal attempt detected: {chat_id}")
            return web.json_response({
                'status': 'error',
                'error': 'Invalid chat ID',
                'code': 'INVALID_CHAT_ID'
            }, status=400)

        success = Chat.delete_from_archive(chat_id)

        if success:
            return web.json_response({
                'status': 'success'
            })
        else:
            return web.json_response({
                'status': 'error',
                'error': f'Chat {chat_id} not found',
                'code': 'CHAT_NOT_FOUND'
            }, status=404)

    except Exception as e:
        logger.error(f"Failed to delete archived chat: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def search_chats(request: Any) -> Any:
    """GET /api/chat/search?q=query

    Searches through all archived chats using BM25 ranking.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        from core.chat import Chat

        query = request.rel_url.query.get('q', '').strip()
        if not query:
            return web.json_response({
                'status': 'error',
                'error': 'Search query is required',
                'code': 'INVALID_REQUEST'
            }, status=400)

        results = Chat.search_chats(query)
        return web.json_response({
            'status': 'success',
            'results': results
        })

    except Exception as e:
        logger.error(f"Failed to search chats: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def edit_message(request: Any) -> Any:
    """POST /api/chat/message/edit

    Edit a message in the current chat.
    Request body: {
        "message_index": int,
        "new_content": str
    }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        data = await request.json()
        message_index = data.get('message_index')
        new_content = data.get('new_content')

        if message_index is None or new_content is None:
            return web.json_response({
                'status': 'error',
                'error': 'message_index and new_content are required',
                'code': 'INVALID_REQUEST'
            }, status=400)

        # Validate message_index is an integer
        if not isinstance(message_index, int):
            return web.json_response({
                'status': 'error',
                'error': 'message_index must be an integer',
                'code': 'INVALID_REQUEST'
            }, status=400)

        # Validate new_content is a string
        if not isinstance(new_content, str):
            return web.json_response({
                'status': 'error',
                'error': 'new_content must be a string',
                'code': 'INVALID_REQUEST'
            }, status=400)

        from core.globals import g

        if not hasattr(g, 'chat') or not g.chat:
            return web.json_response({
                'status': 'error',
                'error': 'No active chat',
                'code': 'NO_CHAT'
            }, status=400)

        success = g.chat.update_message(message_index, new_content)

        if not success:
            return web.json_response({
                'status': 'error',
                'error': 'Failed to update message (invalid index or system message)',
                'code': 'UPDATE_FAILED'
            }, status=400)

        g.chat.save_to_json()

        message_data = {
            'index': message_index,
            'role': g.chat.messages[message_index].role.value,
            'content': new_content
        }

        from api.server import get_server
        server = get_server()
        if server:
            await server.broadcast({
                'type': 'message_edited',
                'message': message_data,
                'timestamp': time.time()
            })

        return web.json_response({
            'status': 'success',
            'message': message_data
        })

    except Exception as e:
        logger.error(f"Failed to edit message: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def delete_message(request: Any) -> Any:
    """POST /api/chat/message/delete/{index}

    Delete a message at the specified index from the chat history.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        # Validate and parse message index
        try:
            message_index = int(request.match_info.get('index'))
        except (ValueError, TypeError):
            return web.json_response({
                'status': 'error',
                'error': 'Invalid message index - must be an integer',
                'code': 'INVALID_REQUEST'
            }, status=400)

        from core.globals import g

        if not hasattr(g, 'chat') or not g.chat:
            return web.json_response({
                'status': 'error',
                'error': 'No active chat',
                'code': 'NO_CHAT'
            }, status=400)

        if message_index < 0 or message_index >= len(g.chat.messages):
            return web.json_response({
                'status': 'error',
                'error': f'Invalid message index: {message_index}',
                'code': 'INVALID_INDEX'
            }, status=400)

        # Don't allow deleting system message at index 0
        if message_index == 0 and g.chat.messages[0].role.value == 'system':
            return web.json_response({
                'status': 'error',
                'error': 'Cannot delete system message',
                'code': 'PROTECTED_MESSAGE'
            }, status=400)

        # Delete the message
        deleted_message = g.chat.messages.pop(message_index)
        logger.info(f"Deleted message at index {message_index}: {deleted_message.role.value}")

        # Save updated chat
        g.chat.save_to_json()

        # Broadcast deletion event to all connected clients
        from api.server import get_server
        server = get_server()
        if server:
            await server.broadcast({
                'type': 'message_deleted',
                'index': message_index,
                'new_length': len(g.chat.messages),
                'timestamp': time.time()
            })

        return web.json_response({
            'status': 'success',
            'index': message_index,
            'new_length': len(g.chat.messages)
        })

    except ValueError:
        return web.json_response({
            'status': 'error',
            'error': 'Invalid index format',
            'code': 'INVALID_REQUEST'
        }, status=400)
    except Exception as e:
        logger.error(f"Failed to delete message: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def truncate_and_continue(request: Any) -> Any:
    """POST /api/chat/truncate/{index}

    Truncates chat history to specified message index and prepares for continuation.
    All messages after the specified index will be removed.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        # Validate and parse message index
        try:
            message_index = int(request.match_info.get('index'))
        except (ValueError, TypeError):
            return web.json_response({
                'status': 'error',
                'error': 'Invalid message index - must be an integer',
                'code': 'INVALID_REQUEST'
            }, status=400)

        from core.globals import g

        if not hasattr(g, 'chat') or not g.chat:
            return web.json_response({
                'status': 'error',
                'error': 'No active chat',
                'code': 'NO_CHAT'
            }, status=400)

        if message_index < 0 or message_index >= len(g.chat.messages):
            return web.json_response({
                'status': 'error',
                'error': f'Invalid message index: {message_index}',
                'code': 'INVALID_INDEX'
            }, status=400)

        # Truncate messages to include only up to and including the specified index
        original_count = len(g.chat.messages)
        g.chat.messages = g.chat.messages[0:message_index + 1]
        truncated_count = original_count - len(g.chat.messages)

        logger.info(f"Truncated chat from {original_count} to {len(g.chat.messages)} messages (removed {truncated_count})")

        # Save truncated chat
        g.chat.save_to_json()

        # Broadcast truncation event to all connected clients
        from api.server import get_server
        server = get_server()
        if server:
            await server.broadcast({
                'type': 'chat_truncated',
                'index': message_index,
                'new_length': len(g.chat.messages),
                'removed_count': truncated_count,
                'timestamp': time.time()
            })

        return web.json_response({
            'status': 'success',
            'index': message_index,
            'new_length': len(g.chat.messages),
            'removed_count': truncated_count
        })

    except ValueError:
        return web.json_response({
            'status': 'error',
            'error': 'Invalid index format',
            'code': 'INVALID_REQUEST'
        }, status=400)
    except Exception as e:
        logger.error(f"Failed to truncate chat: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


def _load_suggested_prompts() -> list[str]:
    """
    Load suggested prompts from JSON config file.

    Falls back to default prompts if file doesn't exist or is invalid.

    Returns:
        List[str]: List of suggested prompt strings
    """
    # Default prompts as fallback
    default_prompts = [
        "What can you help me with?",
        "Show me your capabilities and available tools",
        "Explain how your tool system works",
        "What programming languages and frameworks do you support?",
        "Tell me about your key features"
    ]

    config_path = Path.home() / ".cli-agent" / "suggested_prompts.json"

    try:
        if not config_path.exists():
            logger.info(f"Suggested prompts config not found at {config_path}, using defaults")
            return default_prompts

        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)

        prompts = config.get('prompts', [])

        if not isinstance(prompts, list) or not prompts:
            logger.warning(f"Invalid prompts format in {config_path}, using defaults")
            return default_prompts

        # Validate all prompts are strings
        if not all(isinstance(p, str) for p in prompts):
            logger.warning(f"Non-string prompts found in {config_path}, using defaults")
            return default_prompts

        logger.debug(f"Loaded {len(prompts)} suggested prompts from {config_path}")
        return prompts

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {config_path}: {e}, using defaults")
        return default_prompts
    except Exception as e:
        logger.error(f"Error loading suggested prompts from {config_path}: {e}, using defaults")
        return default_prompts


async def get_suggested_prompts(request: Any) -> Any:
    """GET /api/chat/suggested-prompts

    Returns a list of suggested starting prompts for empty chats.
    These prompts help users get started with the agent.

    Prompts are loaded from ~/.cli-agent/suggested_prompts.json
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        prompts = _load_suggested_prompts()

        return web.json_response({
            'status': 'success',
            'prompts': prompts
        })

    except Exception as e:
        logger.error(f"Failed to get suggested prompts: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def track_prompt_click(request: Any) -> Any:
    """POST /api/chat/track-prompt-click

    Track when a user clicks on a generated prompt suggestion.
    Stores clicked prompts with deduplication for use in future prompt generation.

    Request body:
    {
        "prompt": "The clicked prompt text"
    }

    Response:
    {
        "status": "success",
        "message": "Prompt click tracked"
    }
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        # Parse request body
        try:
            data = await request.json()
        except Exception:
            return web.json_response({
                'status': 'error',
                'error': 'Invalid JSON in request body',
                'code': 'INVALID_REQUEST'
            }, status=400)

        prompt = data.get('prompt', '').strip()
        if not prompt:
            return web.json_response({
                'status': 'error',
                'error': 'Prompt text is required',
                'code': 'INVALID_REQUEST'
            }, status=400)

        # Load existing clicked prompts
        clicked_prompts_path = Path.home() / ".cli-agent" / "clicked_prompts.json"
        clicked_prompts_path.parent.mkdir(parents=True, exist_ok=True)

        clicked_prompts = []
        if clicked_prompts_path.exists():
            try:
                with open(clicked_prompts_path, encoding='utf-8') as f:
                    data = json.load(f)
                    clicked_prompts = data.get('prompts', [])
            except Exception as e:
                logger.warning(f"Failed to load clicked prompts: {e}")

        # Add new prompt to the beginning (most recent first)
        # Remove duplicates - if prompt already exists, remove old occurrence
        clicked_prompts = [p for p in clicked_prompts if p != prompt]
        clicked_prompts.insert(0, prompt)

        # Keep only the most recent 10 (we'll use top 3, but store more for flexibility)
        clicked_prompts = clicked_prompts[:10]

        # Save back to file
        try:
            with open(clicked_prompts_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'prompts': clicked_prompts,
                    'last_updated': time.time()
                }, f, indent=2, ensure_ascii=False)

            logger.info(f"Tracked prompt click: '{prompt[:50]}...'")

            return web.json_response({
                'status': 'success',
                'message': 'Prompt click tracked'
            })

        except Exception as e:
            logger.error(f"Failed to save clicked prompts: {e}")
            return web.json_response({
                'status': 'error',
                'error': 'Failed to save clicked prompt',
                'code': 'INTERNAL_ERROR'
            }, status=500)

    except Exception as e:
        logger.error(f"Failed to track prompt click: {e}")
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


async def regenerate_prompts(request: Any) -> Any:
    """POST /api/chat/regenerate-prompts

    Regenerates suggested prompts by analyzing recent study files using LLM.
    Uses the prompt_regenerator utility to intelligently generate contextual prompts.

    Request body (optional):
    {
        "num_files": 5,        // Number of files to analyze (default: 5)
        "time_frame_days": 30  // Days to look back (default: 30)
    }

    Response:
    {
        "status": "success" | "partial" | "error",
        "prompts": [...],
        "message": "...",
        "files_analyzed": N
    }

    Broadcasts WebSocket event 'prompts_regenerated' when complete.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        # Parse request body for optional parameters
        num_files = 5
        time_frame_days = 30

        if request.body_exists:
            try:
                body = await request.json()
                num_files = body.get('num_files', 5)
                time_frame_days = body.get('time_frame_days', 30)
            except Exception as e:
                logger.warning(f"Failed to parse request body: {e}, using defaults")

        logger.info(f"Regenerating prompts (num_files={num_files}, time_frame_days={time_frame_days})")

        # Import the regenerator (lazy import to avoid startup overhead)
        from shared.utils.prompt_regenerator import regenerate_prompts as regen_fn

        # Run regeneration
        result = await regen_fn(
            num_files=num_files,
            time_frame_days=time_frame_days
        )

        # Broadcast WebSocket event to notify all clients
        from api.server import get_server
        server = get_server()
        if server:
            await server.broadcast({
                'type': 'prompts_regenerated',
                'prompts': result['prompts'],
                'status': result['status'],
                'timestamp': time.time()
            })
            logger.debug("Broadcasted prompts_regenerated event")

        return web.json_response(result)

    except Exception as e:
        logger.error(f"Failed to regenerate prompts: {e}", exc_info=True)
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR',
            'message': 'An error occurred while regenerating prompts'
        }, status=500)


def setup_routes(app: Any):
    """Register all chat API routes"""
    app.router.add_get('/api/chat/current', get_current_chat)
    app.router.add_post('/api/chat/load/{id}', load_chat)
    app.router.add_post('/api/chat/new', new_chat)
    app.router.add_get('/api/chat/messages', get_messages)
    app.router.add_post('/api/chat/message/edit', edit_message)
    app.router.add_post('/api/chat/message/delete/{index}', delete_message)
    app.router.add_post('/api/chat/truncate/{index}', truncate_and_continue)
    app.router.add_get('/api/chat/archives', list_archived_chats)
    app.router.add_get('/api/chat/search', search_chats)
    app.router.add_delete('/api/chat/archive/{id}', delete_archived_chat)
    app.router.add_get('/api/chat/suggested-prompts', get_suggested_prompts)
    app.router.add_post('/api/chat/track-prompt-click', track_prompt_click)
    app.router.add_post('/api/chat/regenerate-prompts', regenerate_prompts)
    logger.info("Chat API routes registered")
