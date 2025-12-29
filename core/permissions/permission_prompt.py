"""
Simplified permission prompt system for CLI-Agent.
Handles user confirmation for tool execution with allow/deny/always options.
"""
import asyncio
import os
import re
import threading
from typing import Any, Optional

from termcolor import colored

from core.permissions.embedding_classifier import tool_call_to_xml


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
    action_desc = _format_action_description(tool_name, file_path, command, xml_content)

    # Try GUI mode first if available
    if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
        gui_result = await _try_gui_permission_prompt(
            tool_name=tool_name,
            action_desc=action_desc,
            file_path=file_path,
            command=command,
            reason=reason,
            settings_manager=settings_manager,
            xml_content=xml_content
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
                               command: Optional[str], xml_content: Optional[str] = None) -> str:
    """Format action description as colorized raw XML tool call."""
    # Get raw XML representation
    raw_xml = tool_call_to_xml(tool_name, file_path, command, xml_content)

    # Apply syntax highlighting with colors
    # Pattern: <tagname>content</tagname> or <tagname><inner>content</inner></tagname>
    def colorize_xml(xml_str: str) -> str:
        """Apply colors to XML: tags in cyan, content in yellow."""
        # Replace opening/closing tags with colored versions
        result = re.sub(
            r'(</?)([\w]+)(>)',
            lambda m: colored(m.group(1), "cyan") + colored(m.group(2), "cyan", attrs=["bold"]) + colored(m.group(3), "cyan"),
            xml_str
        )

        # Color the content between simple tags (non-nested)
        # Find content that's not within angle brackets
        parts = re.split(r'(<[^>]+>)', result)
        colored_parts = []
        for part in parts:
            if part.startswith(colored("<", "cyan")):
                # This is a tag, keep as-is
                colored_parts.append(part)
            elif part.strip():
                # This is content, color it yellow
                colored_parts.append(colored(part, "yellow"))
            else:
                colored_parts.append(part)

        return ''.join(colored_parts)

    return colorize_xml(raw_xml)


async def _try_gui_permission_prompt(
    tool_name: str,
    action_desc: str,
    file_path: Optional[str],
    command: Optional[str],
    reason: str,
    settings_manager,
    xml_content: Optional[str] = None
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

        # Generate rule options for specificity selection
        rule_options = _get_permission_rule_options(tool_name, file_path, command)

        # Format args_display for frontend to show tool call arguments
        args_display = ""
        if tool_name == "bash" and command:
            args_display = command
        elif file_path:
            args_display = f"file: {file_path}"
        elif xml_content:
            # Extract arguments from XML content if available
            try:
                # Simple extraction of XML content between tags
                import re
                content_match = re.search(r'<[^>]+>(.*?)</[^>]+>', xml_content, re.DOTALL)
                if content_match:
                    args_display = content_match.group(1).strip()[:200]  # Limit to 200 chars
            except Exception:
                args_display = f"{tool_name} (arguments not available)"
        else:
            args_display = f"{tool_name}"

        # Get ML classification if enabled
        ml_score = None
        ml_confidence = 0.0
        ml_decision = None
        ml_reason = None
        ml_enabled = False

        try:
            ml_config = settings_manager.get_ml_config()
            ml_enabled = ml_config.get("enabled", False)

            if ml_enabled:
                from core.permissions.autonomous_permission import AutonomousPermission

                # Create autonomous permission instance
                autonomous = AutonomousPermission(
                    settings_manager=settings_manager,
                    ml_enabled=True,
                    ml_model=ml_config.get("model", "embeddinggemma:latest"),
                    threshold_allow=ml_config.get("threshold_allow", 0.01),
                    threshold_deny=ml_config.get("threshold_deny", -0.01)
                )

                # Get ML classification
                decision, ml_reason, confidence, score = await autonomous.check_permission(
                    tool_name=tool_name,
                    file_path=file_path,
                    command=command,
                    xml_content=xml_content
                )

                ml_score = score
                ml_confidence = confidence
                ml_decision = decision

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"ML classification failed in permission prompt: {e}")

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
                "args_display": args_display,
                "reason": reason,
                "options": ["allow", "deny", "always_allow", "always_deny"],
                "rule_options": [
                    {"description": desc, "rule": rule}
                    for desc, rule in rule_options
                ],
                # ML classification data
                "ml_enabled": ml_enabled,
                "ml_score": ml_score,
                "ml_confidence": ml_confidence,
                "ml_decision": ml_decision,
                "ml_reason": ml_reason
            }
        }

        session.send_update(permission_payload)

        # Wait for response from visual interface with timeout to allow deferral
        if hasattr(g, 'permission_response_queue'):
            try:
                # 60 second timeout allows user to defer/switch chats
                response_data = await asyncio.wait_for(
                    g.permission_response_queue.get(),
                    timeout=60.0
                )

                if response_data and isinstance(response_data, dict):
                    result = response_data.get('result')
                    rule_selected = response_data.get('rule')  # Specific rule chosen by user
                    cancelled = response_data.get('cancelled', False)

                    # Handle cancellation (e.g., user switched chats)
                    if cancelled:
                        print(colored(f"⚠️  Permission cancelled: {response_data.get('reason', 'Unknown reason')}", "yellow"))
                        return False

                    if result == 'allow':
                        print(colored("✅ Permission granted (GUI)", "green"))
                        return True

                    elif result == 'always_allow':
                        # If a specific rule was provided, use it; otherwise use default
                        if rule_selected:
                            rule = rule_selected
                        else:
                            rule = _create_permission_rule(tool_name, file_path, command)

                        settings_manager.add_permission("allow", rule)
                        settings_manager.save_settings()
                        print(colored(f"✅ Permission granted and saved: {rule}", "green"))
                        return True

                    elif result == 'always_ask':
                        # Add to ask list permanently (tool-level "ask" permission)
                        if rule_selected:
                            rule = rule_selected
                        else:
                            # For general "ask" permission, just use the tool name
                            rule = tool_name

                        settings_manager.add_permission("ask", rule)
                        settings_manager.save_settings()
                        print(colored(f"ℹ️  Permission set to 'ask' and saved: {rule}", "yellow"))
                        return True  # Allow this instance, but will ask again next time

                    elif result == 'always_deny':
                        # Add to deny list permanently
                        if rule_selected:
                            rule = rule_selected
                        else:
                            rule = _create_permission_rule(tool_name, file_path, command)

                        settings_manager.add_permission("deny", rule)
                        settings_manager.save_settings()
                        print(colored(f"❌ Permission denied and saved: {rule}", "red"))
                        return False

                    elif result == 'deny':
                        print(colored("❌ Permission denied (GUI)", "red"))
                        return False

            except TimeoutError:
                # Timeout means user deferred decision (e.g., switched chats)
                print(colored("⏱️  Permission request timed out - denying by default", "yellow"))
                print(colored("   (User may have switched chats or deferred decision)", "yellow"))
                return False

            except Exception as e:
                print(colored(f"⚠️  GUI permission error: {e} - denying by default", "yellow"))
                return False

    except Exception as e:
        print(colored(f"⚠️  GUI permission error: {e} - denying by default", "yellow"))
        return False

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
    print(f"\nReason: {reason}") if reason else None
    print("\nOptions:")
    print("  [y] Allow once")
    print("  [n] Deny once")
    print("  [a] Allow and remember... (choose specificity)")
    print("  [d] Always deny (add to deny list)")
    print("  [?] Show help")
    print("")

    # Get user input
    while True:
        try:
            if get_user_input_func:
                prompt = colored("Choice [Y/n/a/d/?]: ", "cyan")
                response = await _get_async_input(
                    prompt,
                    get_user_input_func,
                    input_event,
                    input_lock,
                    shared_input
                )
            else:
                response = input(colored("Choice [Y/n/a/d/?]: ", "cyan"))

            response = response.strip().lower()

            if response in ['y', 'yes', '']:
                print(colored("✅ Permission granted (once)", "green"))
                return True

            elif response in ['n', 'no']:
                print(colored("❌ Permission denied (once)", "red"))
                return False

            elif response == 'a':
                # Show specificity submenu
                rule_options = _get_permission_rule_options(tool_name, file_path, command)

                print("\n" + "-" * 60)
                print(colored("Choose permission scope:", "cyan", attrs=['bold']))
                print("-" * 60)

                for idx, (description, rule) in enumerate(rule_options, 1):
                    print(f"  [{idx}] {description}")
                    print(f"      Rule: {colored(rule, 'yellow')}")

                print("  [0] Cancel")
                print("")

                # Get specificity choice
                while True:
                    if get_user_input_func:
                        spec_prompt = colored(f"Choose [0-{len(rule_options)}]: ", "cyan")
                        spec_response = await _get_async_input(
                            spec_prompt,
                            get_user_input_func,
                            input_event,
                            input_lock,
                            shared_input
                        )
                    else:
                        spec_response = input(colored(f"Choose [0-{len(rule_options)}]: ", "cyan"))

                    try:
                        choice = int(spec_response.strip())
                        if choice == 0:
                            print(colored("↩️  Cancelled, returning to permission prompt", "yellow"))
                            break
                        elif 1 <= choice <= len(rule_options):
                            description, rule = rule_options[choice - 1]
                            settings_manager.add_permission("allow", rule)
                            settings_manager.save_settings()
                            print(colored(f"✅ Permission granted and saved: {rule}", "green"))
                            return True
                        else:
                            print(colored(f"Invalid choice. Please enter 0-{len(rule_options)}", "yellow"))
                    except ValueError:
                        print(colored(f"Invalid input. Please enter a number 0-{len(rule_options)}", "yellow"))

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


def _get_permission_rule_options(tool_name: str, file_path: Optional[str],
                                 command: Optional[str]) -> list[tuple[str, str]]:
    """
    Generate multiple permission rule options with different specificity levels.

    Returns a list of (description, rule) tuples ordered from most to least specific.

    Examples:
        bash + "npm install package" →
            [("This exact command", "bash(npm install package)"),
             ("All 'npm install' commands", "bash(npm install:*)"),
             ("All npm commands", "bash(npm:*)"),
             ("All bash commands", "bash")]

        readfile + "/home/user/project/src/main.py" →
            [("This specific file", "readfile(/home/user/project/src/main.py)"),
             ("All files in src/", "readfile(/home/user/project/src/*)"),
             ("All files in src/ tree", "readfile(/home/user/project/src/**)"),
             ("All files in project/", "readfile(/home/user/project/**)"),
             ("All readfile operations", "readfile")]
    """
    options = []

    if tool_name == "bash" and command:
        # Bash command specificity levels
        cmd_parts = command.strip().split()
        cmd_word = cmd_parts[0]

        # Level 1: Exact command
        options.append(("This exact command", f"bash({command})"))

        # Level 2: Command with first N words (if multi-word)
        if len(cmd_parts) >= 2:
            first_two = ' '.join(cmd_parts[:2])
            options.append((f"All '{first_two}' commands", f"bash({first_two}:*)"))

        # Level 3: First word pattern
        options.append((f"All '{cmd_word}' commands", f"bash({cmd_word}:*)"))

        # Level 4: All bash
        options.append(("All bash commands", "bash"))

    elif file_path:
        # File operation specificity levels
        abs_path = os.path.abspath(file_path)
        directory = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)

        # Level 1: Specific file
        options.append((f"This specific file: {filename}", f"{tool_name}({abs_path})"))

        # Level 2+: Directory levels (work up the tree)
        current_dir = directory
        visited_dirs = set()

        while current_dir and current_dir != '/' and current_dir not in visited_dirs:
            visited_dirs.add(current_dir)
            dir_name = os.path.basename(current_dir)

            # Same directory (non-recursive)
            if current_dir == directory:
                options.append((f"All files in {dir_name}/", f"{tool_name}({current_dir}/*)"))

            # Directory tree (recursive)
            options.append((f"All files in {dir_name}/ tree", f"{tool_name}({current_dir}/**)"))

            # Move up one level
            parent = os.path.dirname(current_dir)
            if parent == current_dir:  # Reached root
                break
            current_dir = parent

        # Final level: All uses of this tool
        options.append((f"All {tool_name} operations", tool_name))

    else:
        # Generic tool - only one option
        options.append((f"All {tool_name} operations", tool_name))

    return options


def _create_permission_rule(tool_name: str, file_path: Optional[str],
                            command: Optional[str]) -> str:
    """
    Create a permission rule string from tool execution details.
    Uses the most common middle-ground specificity level.

    DEPRECATED: Use _get_permission_rule_options() for new code.

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
This is the permission system for CLI-Agent.

OPTIONS:
  y - Allow once
      Execute this operation now, but ask again next time

  n - Deny once
      Block this operation now, but ask again next time

  a - Allow and remember...
      Opens a submenu to choose the specificity of the permission rule.
      You can allow:
        - This specific file/command only
        - Files in this directory (non-recursive)
        - Files in this directory tree (recursive)
        - All commands of a specific type (e.g., all npm commands)
        - All uses of this tool

      Examples of rules created:
        bash(npm install:*)           - All 'npm install' commands
        bash(npm:*)                   - All npm commands
        readfile(/path/to/file.txt)   - This specific file only
        readfile(/path/to/dir/*)      - All files in this directory
        readfile(/path/to/dir/**)     - All files in directory tree

  d - Always deny
      Add a broad deny rule to settings.json to block similar operations

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

TIPS:
  - Start with more specific rules and broaden as needed
  - Use 'y' (allow once) to test operations before creating rules
  - Review your rules with: /perms view

For more information, see: docs/PERMISSION_SYSTEM.md
""")
    print("=" * 60 + "\n")
