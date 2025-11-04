"""
Simplified permission prompt system for CLI-Agent.
Handles user confirmation for tool execution with allow/deny/always options.
"""
import asyncio
import os
import threading
from typing import Any, Optional

from termcolor import colored


async def permission_prompt(
    tool_name: str,
    decision: str,
    reason: str,
    settings_manager,
    file_path: Optional[str] = None,
    command: Optional[str] = None,
    xml_content: Optional[str] = None,
    input_event: Optional[asyncio.Event] = None,
    input_lock: Optional[threading.Lock] = None,
    shared_input: Optional[dict[str, Any]] = None,
    get_user_input_func = None
) -> bool:
    """
    Prompt user for permission to execute a tool.

    This function is called when check_permission() returns "ask" or "default".
    It shows a simple prompt with allow/deny/always options.

    Args:
        tool_name: Name of tool being executed
        decision: Permission decision from settings_manager ("ask" or "default")
        reason: Explanation of decision
        settings_manager: SettingsManager instance for saving permanent choices
        file_path: File path if applicable
        command: Command string if tool is bash
        xml_content: Full XML content of tool call
        input_event: Event for synchronized input
        input_lock: Lock for thread-safe input
        shared_input: Shared input dictionary
        get_user_input_func: Function to get user input

    Returns:
        True if permission granted, False if denied
    """
    # Import here to avoid circular dependencies
    from core.globals import g

    # Check YOLO mode (auto-approve)
    if hasattr(g, 'auto_execute') and g.auto_execute:
        print(colored(f"✅ Auto-approved (YOLO mode): {tool_name}", "green"))
        return True

    # Format action description
    action_desc = _format_action_description(tool_name, file_path, command)

    # Try GUI mode first if available
    if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
        gui_result = await _try_gui_permission_prompt(
            tool_name=tool_name,
            action_desc=action_desc,
            file_path=file_path,
            command=command,
            reason=reason,
            settings_manager=settings_manager
        )
        if gui_result is not None:
            return gui_result

    # Fall back to CLI prompt
    return await _cli_permission_prompt(
        tool_name=tool_name,
        action_desc=action_desc,
        file_path=file_path,
        command=command,
        reason=reason,
        settings_manager=settings_manager,
        input_event=input_event,
        input_lock=input_lock,
        shared_input=shared_input,
        get_user_input_func=get_user_input_func
    )


def _format_action_description(tool_name: str, file_path: Optional[str],
                               command: Optional[str]) -> str:
    """Format human-readable action description."""
    if tool_name == "bash" and command:
        return f"Execute bash command: {command}"
    elif file_path:
        action_verbs = {
            "readfile": "Read",
            "viewfiles": "View",
            "editfile": "Edit",
            "writefile": "Write",
            "appendfile": "Append to",
            "insertline": "Insert line in",
            "deletelines": "Delete lines from"
        }
        verb = action_verbs.get(tool_name, "Access")
        return f"{verb} file: {file_path}"
    else:
        return f"Execute {tool_name}"


async def _try_gui_permission_prompt(
    tool_name: str,
    action_desc: str,
    file_path: Optional[str],
    command: Optional[str],
    reason: str,
    settings_manager
) -> Optional[bool]:
    """
    Try to show permission prompt in GUI mode via WebSocket.

    Returns:
        True/False if GUI prompt succeeded, None if GUI unavailable
    """
    try:
        import time

        from core.globals import g
        from tools.visualresponse import visualresponse

        session = visualresponse._get_session()
        if not session or not session.is_healthy():
            return None

        # Send permission request to GUI
        permission_payload = {
            "id": f"permission-{int(time.time() * 1000)}",
            "action": "create",
            "type": "permission_request",
            "permission_data": {
                "tool_name": tool_name,
                "action_description": action_desc,
                "file_path": file_path or "",
                "command": command or "",
                "reason": reason,
                "options": ["allow", "deny", "always_allow", "always_deny"]
            }
        }

        session.send_update(permission_payload)

        # Wait for response from visual interface
        if hasattr(g, 'visual_input_queue'):
            try:
                timeout_seconds = 60
                response_data = await asyncio.wait_for(
                    g.visual_input_queue.get(),
                    timeout=timeout_seconds
                )

                if response_data and isinstance(response_data, dict):
                    result = response_data.get('result')

                    if result == 'allow':
                        print(colored("✅ Permission granted (GUI)", "green"))
                        return True

                    elif result == 'always_allow':
                        # Add to allow list permanently
                        rule = _create_permission_rule(tool_name, file_path, command)
                        settings_manager.add_permission("allow", rule)
                        settings_manager.save_settings()
                        print(colored(f"✅ Permission granted and saved: {rule}", "green"))
                        return True

                    elif result == 'always_deny':
                        # Add to deny list permanently
                        rule = _create_permission_rule(tool_name, file_path, command)
                        settings_manager.add_permission("deny", rule)
                        settings_manager.save_settings()
                        print(colored(f"❌ Permission denied and saved: {rule}", "red"))
                        return False

                    elif result == 'deny':
                        print(colored("❌ Permission denied (GUI)", "red"))
                        return False

            except TimeoutError:
                print(colored("⚠️  GUI permission timeout, falling back to CLI", "yellow"))
                return None

    except Exception as e:
        print(colored(f"⚠️  GUI permission error: {e}, falling back to CLI", "yellow"))

    return None


async def _cli_permission_prompt(
    tool_name: str,
    action_desc: str,
    file_path: Optional[str],
    command: Optional[str],
    reason: str,
    settings_manager,
    input_event: Optional[asyncio.Event],
    input_lock: Optional[threading.Lock],
    shared_input: Optional[dict[str, Any]],
    get_user_input_func
) -> bool:
    """
    Show permission prompt in CLI mode.

    Returns:
        True if permission granted, False if denied
    """
    # Display permission request
    print("\n" + "=" * 60)
    print(colored("🔐 PERMISSION REQUIRED", "cyan", attrs=['bold']))
    print("=" * 60)
    print(f"\n{action_desc}")
    print(f"\nReason: {reason}")
    print("\nOptions:")
    print("  [y] Allow once")
    print("  [n] Deny once")
    print("  [a] Always allow (add to allow list)")
    print("  [d] Always deny (add to deny list)")
    print("  [?] Show help")
    print("")

    # Get user input
    while True:
        try:
            if get_user_input_func:
                prompt = colored("Choice [y/n/a/d/?]: ", "cyan")
                response = await _get_async_input(
                    prompt,
                    get_user_input_func,
                    input_event,
                    input_lock,
                    shared_input
                )
            else:
                response = input(colored("Choice [y/n/a/d/?]: ", "cyan"))

            response = response.strip().lower()

            if response in ['y', 'yes', '']:
                print(colored("✅ Permission granted (once)", "green"))
                return True

            elif response in ['n', 'no']:
                print(colored("❌ Permission denied (once)", "red"))
                return False

            elif response == 'a':
                # Add to allow list
                rule = _create_permission_rule(tool_name, file_path, command)
                settings_manager.add_permission("allow", rule)
                settings_manager.save_settings()
                print(colored(f"✅ Permission granted and saved: {rule}", "green"))
                return True

            elif response == 'd':
                # Add to deny list
                rule = _create_permission_rule(tool_name, file_path, command)
                settings_manager.add_permission("deny", rule)
                settings_manager.save_settings()
                print(colored(f"❌ Permission denied and saved: {rule}", "red"))
                return False

            elif response == '?':
                _show_help()

            else:
                print(colored("Invalid choice. Please enter y, n, a, d, or ?", "yellow"))

        except (KeyboardInterrupt, EOFError):
            print(colored("\n❌ Permission denied (interrupted)", "red"))
            return False


def _create_permission_rule(tool_name: str, file_path: Optional[str],
                            command: Optional[str]) -> str:
    """
    Create a permission rule string from tool execution details.

    Examples:
        bash + "npm install" → "bash(npm:*)"
        readfile + "./src/main.py" → "readfile(./src/**)"
        readfile + "/home/user/file.txt" → "readfile(/home/user/**)"
    """
    if tool_name == "bash" and command:
        # Extract first word of command
        cmd_word = command.strip().split()[0]
        return f"bash({cmd_word}:*)"

    elif file_path:
        # Create directory pattern from file path
        directory = os.path.dirname(file_path)
        if directory:
            return f"{tool_name}({directory}/**)"
        else:
            return f"{tool_name}(*)"

    else:
        # No specifics, allow all uses of this tool
        return tool_name


async def _get_async_input(
    prompt: str,
    get_user_input_func,
    input_event: Optional[asyncio.Event],
    input_lock: Optional[threading.Lock],
    shared_input: Optional[dict[str, Any]]
) -> str:
    """Get user input asynchronously using the provided input function."""
    if input_event and shared_input is not None:
        # Use event-based input for async compatibility
        input_event.clear()
        if input_lock:
            with input_lock:
                shared_input['prompt'] = prompt
                shared_input['response'] = None
        else:
            shared_input['prompt'] = prompt
            shared_input['response'] = None

        # Wait for input
        await input_event.wait()

        if input_lock:
            with input_lock:
                response = shared_input.get('response', '')
        else:
            response = shared_input.get('response', '')

        return response if response else ''

    else:
        # Fallback to synchronous input
        # Note: We always use plain input() here because get_user_input_func
        # (typically get_user_input_with_bindings) requires args and context_chat
        # which are not available in this fallback path
        return input(prompt)


def _show_help():
    """Show detailed help for permission options."""
    print("\n" + "=" * 60)
    print(colored("PERMISSION SYSTEM HELP", "cyan", attrs=['bold']))
    print("=" * 60)
    print("""
This is the simplified permission system for CLI-Agent.

OPTIONS:
  y - Allow once
      Execute this operation now, but ask again next time

  n - Deny once
      Block this operation now, but ask again next time

  a - Always allow
      Add a rule to settings.json to always allow similar operations
      Example: bash(npm:*) allows all npm commands

  d - Always deny
      Add a rule to settings.json to always block similar operations

CONFIGURATION:
  Edit ~/.cli-agent/settings.json to manage permissions manually

  Format:
    {
      "permissions": {
        "allow": ["bash(npm:*)", "readfile(./src/**)"],
        "deny": ["readfile(.env)", "bash(sudo:*)"],
        "ask": ["bash(git push:*)"]
      }
    }

PRECEDENCE:
  1. Deny rules (always block)
  2. Ask rules (require confirmation)
  3. Allow rules (always permit)
  4. Default (ask for permission)

For more information, see: docs/PERMISSION_SYSTEM.md
""")
    print("=" * 60 + "\n")
