"""
Centralized error handling for CLI-Agent API.

Provides production-safe error responses that don't leak sensitive information.
Logs detailed errors server-side while returning generic messages to clients.
"""

import logging
import os
from typing import Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


# Error code mappings to user-friendly messages
ERROR_MESSAGES = {
    # Authentication errors
    'UNAUTHORIZED': 'Authentication required',
    'AUTHENTICATION_FAILED': 'Authentication failed',
    'RATE_LIMIT_EXCEEDED': 'Too many requests - please try again later',

    # Request errors
    'INVALID_REQUEST': 'Invalid request parameters',
    'INVALID_FILE_TYPE': 'Invalid file type',
    'INVALID_FILE_CONTENT': 'Invalid file content',
    'FILE_TOO_LARGE': 'File size exceeds limit',

    # Resource errors
    'NOT_FOUND': 'Resource not found',
    'CHAT_NOT_FOUND': 'Chat session not found',
    'PROFILE_NOT_FOUND': 'Profile not found',
    'CONTAINER_NOT_FOUND': 'Container not found',

    # Operation errors
    'OPERATION_FAILED': 'Operation failed',
    'UPDATE_FAILED': 'Update failed',
    'PROTECTED_MESSAGE': 'Cannot modify protected message',

    # Generic errors
    'INTERNAL_ERROR': 'An internal error occurred',
    'SERVICE_UNAVAILABLE': 'Service temporarily unavailable',
}


def is_production() -> bool:
    """Check if running in production mode."""
    return os.getenv('ENVIRONMENT', 'production') != 'development'


def sanitize_error_message(error: Exception, code: str) -> str:
    """
    Return production-safe error message.

    In production: Returns generic message
    In development: Returns actual error message

    Args:
        error: The exception that occurred
        code: Error code for lookup

    Returns:
        User-facing error message
    """
    if is_production():
        # Production: Return generic message based on code
        return ERROR_MESSAGES.get(code, ERROR_MESSAGES['INTERNAL_ERROR'])
    else:
        # Development: Return actual error for debugging
        return str(error)


def log_error(error: Exception, context: str, **kwargs):
    """
    Log error with full details server-side.

    Args:
        error: The exception that occurred
        context: Context string describing where error occurred
        **kwargs: Additional context to log
    """
    log_data = {
        'context': context,
        'error_type': type(error).__name__,
        'error_message': str(error),
        **kwargs
    }

    logger.error(f"Error in {context}: {error}", extra=log_data, exc_info=True)


def create_error_response(
    error: Exception,
    code: str,
    status: int = 500,
    context: Optional[str] = None,
    **log_kwargs
):
    """
    Create standardized error response.

    Logs detailed error server-side, returns safe message to client.

    Args:
        error: The exception that occurred
        code: Error code (e.g., 'INTERNAL_ERROR')
        status: HTTP status code
        context: Context string for logging
        **log_kwargs: Additional data to log

    Returns:
        aiohttp JSON response
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    # Log full error details server-side
    if context:
        log_error(error, context, code=code, status=status, **log_kwargs)

    # Return sanitized error to client
    user_message = sanitize_error_message(error, code)

    return web.json_response({
        'status': 'error',
        'error': user_message,
        'code': code
    }, status=status)


def create_validation_error(message: str, status: int = 400):
    """
    Create validation error response.

    Validation errors are safe to show to users since they describe
    what the user provided, not internal system state.

    Args:
        message: Validation error message
        status: HTTP status code (default: 400)

    Returns:
        aiohttp JSON response
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    return web.json_response({
        'status': 'error',
        'error': message,
        'code': 'INVALID_REQUEST'
    }, status=status)


def create_success_response(data: dict, status: int = 200):
    """
    Create standardized success response.

    Args:
        data: Response data dictionary
        status: HTTP status code (default: 200)

    Returns:
        aiohttp JSON response
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    response = {
        'status': 'success',
        **data
    }

    return web.json_response(response, status=status)
