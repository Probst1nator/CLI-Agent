"""
Input handling utilities for CLI-Agent.

This module contains input handling functions extracted from main.py to provide
reusable input functionality including multiline input, screenshot capture,
and permission management.

MIGRATION NOTICE: This module is being migrated to use the new unified input
management system. Legacy functions are maintained for backwards compatibility.
"""

import argparse
import asyncio
import threading
from typing import Any, Optional

from termcolor import colored

# Import new unified input system
from agent.core.input_compatibility import (
    get_user_input_with_bindings_unified,
    handle_multiline_input_unified,
    handle_screenshot_capture_unified,
)

# Functions will be imported dynamically to avoid circular imports
# Import command system from refactored module
from core.chat import Chat


def handle_multiline_input() -> str:
    """
    Handles multiline input from the user.

    LEGACY WRAPPER: This function now delegates to the unified input system.
    """
    return handle_multiline_input_unified()


async def get_user_input_with_bindings(
    args: argparse.Namespace,
    context_chat: Optional[Chat],
    prompt: str = colored(" Enter your request: ", 'blue', attrs=["bold"]),
    input_override: Optional[str] = None,
    force_input: bool = False,
    input_event: Optional[asyncio.Event] = None,
    input_lock: Optional[threading.Lock] = None,
    shared_input: Optional[dict[str, Any]] = None
) -> tuple[str, list[str]]:
    """
    Interactively captures user input, including voice for confirmation prompts.

    LEGACY WRAPPER: This function now delegates to the unified input system
    while maintaining the same interface for backwards compatibility.

    This function serves as the primary Read-Eval-Print Loop (REPL). It
    returns the user's text prompt. If voice mode is active and this is a
    confirmation prompt, it will use TTS and STT.
    """
    return await get_user_input_with_bindings_unified(
        args=args,
        context_chat=context_chat,
        prompt=prompt,
        input_override=input_override,
        force_input=force_input,
        input_event=input_event,
        input_lock=input_lock,
        shared_input=shared_input
    )


# Legacy implementation removed - now using unified input system only


# Initialize the stateful flag for the first call.
# get_user_input_with_bindings.is_first_call = True  # Cannot assign to function


async def handle_screenshot_capture() -> list[str]:
    """
    Handles the screenshot capture process.

    LEGACY WRAPPER: This function now delegates to the unified input system.

    Returns:
        List[str]: The base64 encoded image data or an empty list if failed.
    """
    return await handle_screenshot_capture_unified()


async def handle_permission_flags(user_input: str, get_user_input_func, args, input_event, input_lock, shared_input):
    """Handle permission management flags like /permissions, -p, --p."""
    from termcolor import colored

    from core.permissions.permissions_manager import get_permissions_manager

    # Parse the command
    parts = user_input.split()
    command = parts[0]  # /permissions, -p, or --p

    try:
        if len(parts) == 1:
            # Just show the list
            print(colored("\n🔧 CLI-AGENT PERMISSION MANAGER", "yellow", attrs=['bold']))
            print(colored("Using flag-based interface for easy permission management", "white"))
            print(colored("─" * 60, "yellow"))
            manager = get_permissions_manager()
            result = f"Permissions config path: {manager.config_path}\nPermissions data: {manager.permissions}"
            print(result)
            return

        action = parts[1].lower()

        if action == "list":
            manager = get_permissions_manager()
            result = f"Permissions config path: {manager.config_path}\nPermissions data: {manager.permissions}"
            print(result)

        elif action == "remove" and len(parts) >= 3:
            perm_id = parts[2]
            manager = get_permissions_manager()
            manager.remove_permission(perm_id)
            result = f"Permission {perm_id} has been removed."
            print(result)

        elif action == "clear":
            # Ask for confirmation
            print(colored("\n⚠️  WARNING: This will clear ALL saved permissions!", "red", attrs=['bold']))
            print(colored("Type 'yes' to confirm or anything else to cancel:", "yellow"))

            try:
                if get_user_input_func:
                    confirmation, _ = await get_user_input_func(
                        args, None, colored("> ", "cyan"),
                        input_event=input_event,
                        input_lock=input_lock,
                        shared_input=shared_input
                    )
                else:
                    confirmation = input(colored("> ", "cyan"))
            except KeyboardInterrupt:
                # User pressed Ctrl+C during clear confirmation - treat as cancel
                print(colored("\n❌ Clear operation cancelled (Ctrl+C).", "yellow"))
                return

            if confirmation.strip().lower() == 'yes':
                manager = get_permissions_manager()
                manager.clear_all_permissions()
                result = "All permissions have been cleared."
                print(result)
            else:
                print(colored("❌ Clear operation cancelled.", "green"))

        elif action == "interactive" or action == "manage":
            # Open interactive manager
            await interactive_permission_manager(get_user_input_func, args, input_event, input_lock, shared_input)

        else:
            print(colored(f"❓ Unknown permission action: {action}", "red"))
            print(colored("Usage:", "cyan"))
            print(colored("  /permissions list                 # List all permissions", "white"))
            print(colored("  /permissions remove <id>          # Remove permission by ID", "white"))
            print(colored("  /permissions clear                # Clear all permissions", "white"))
            print(colored("  /permissions interactive          # Open interactive manager", "white"))
            print()
            print(colored("Shortcuts:", "cyan"))
            print(colored("  /p list, etc.                    # Same as /permissions", "white"))

    except Exception as e:
        print(colored(f"❌ Error handling permission command: {str(e)}", "red"))


async def interactive_permission_manager(get_user_input_func, args, input_event, input_lock, shared_input):
    """Interactive permission management interface."""
    from termcolor import colored

    from core.permissions.permissions_manager import get_permissions_manager

    print()
    print(colored("🔧 INTERACTIVE PERMISSION MANAGER", "yellow", attrs=['bold']))
    print(colored("═" * 50, "yellow"))

    while True:
        # Show current permissions
        manager = get_permissions_manager()
        result = f"Permissions config path: {manager.config_path}\nPermissions data: {manager.permissions}"
        print(result)

        print()
        print(colored("Commands:", "cyan"))
        print(colored("  list                    # Refresh permission list", "white"))
        print(colored("  remove <id>             # Remove permission by ID", "white"))
        print(colored("  clear                   # Clear all permissions", "white"))
        print(colored("  exit, quit, back        # Return to main CLI", "white"))
        print()

        try:
            if get_user_input_func:
                user_cmd, _ = await get_user_input_func(
                    args, None, colored("Permissions> ", "cyan"),
                    input_event=input_event,
                    input_lock=input_lock,
                    shared_input=shared_input
                )
            else:
                user_cmd = input(colored("Permissions> ", "cyan"))

            user_cmd = user_cmd.strip().lower()
        except KeyboardInterrupt:
            # User pressed Ctrl+C in interactive permission manager - exit gracefully
            print(colored("\n↩️  Returning to main CLI (Ctrl+C)...", "yellow"))
            break

        if user_cmd in ['exit', 'quit', 'back', '']:
            print(colored("↩️  Returning to main CLI...", "green"))
            break

        elif user_cmd == 'list':
            continue  # Will refresh at top of loop

        elif user_cmd.startswith('remove ') and len(user_cmd.split()) >= 2:
            perm_id = user_cmd.split()[1]
            manager = get_permissions_manager()
            manager.remove_permission(perm_id)
            result = f"Permission {perm_id} has been removed."
            print(result)

        elif user_cmd == 'clear':
            print(colored("\n⚠️  WARNING: This will clear ALL saved permissions!", "red", attrs=['bold']))
            print(colored("Type 'yes' to confirm or anything else to cancel:", "yellow"))

            try:
                if get_user_input_func:
                    confirmation, _ = await get_user_input_func(
                        args, None, colored("> ", "cyan"),
                        input_event=input_event,
                        input_lock=input_lock,
                        shared_input=shared_input
                    )
                else:
                    confirmation = input(colored("> ", "cyan"))
            except KeyboardInterrupt:
                # User pressed Ctrl+C during clear confirmation - treat as cancel
                print(colored("\n❌ Clear operation cancelled (Ctrl+C).", "yellow"))
                continue

            if confirmation.strip().lower() == 'yes':
                manager = get_permissions_manager()
                manager.clear_all_permissions()
                result = "All permissions have been cleared."
                print(result)
                break  # Exit after clearing
            else:
                print(colored("❌ Clear operation cancelled.", "green"))

        else:
            print(colored("❓ Unknown command. Use: list, remove <id>, clear, or exit", "yellow"))