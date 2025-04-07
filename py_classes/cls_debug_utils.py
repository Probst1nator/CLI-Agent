from typing import Optional
from py_classes.cls_chat import Chat

# Constant for debug title format
DEBUG_TITLE_FORMAT = "[{}] "

def get_debug_title_prefix(chat: Chat) -> str:
    """
    Get a formatted prefix string for debug messages that includes the chat's debug title if available.
    
    Args:
        chat (Chat): The chat whose debug_title should be included
        
    Returns:
        str: The formatted prefix string
    """
    return DEBUG_TITLE_FORMAT.format(chat.debug_title) if hasattr(chat, 'debug_title') and chat.debug_title else "" 