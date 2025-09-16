# enhanced_permissions.py
"""
Enhanced permission system with always-allow functionality for CLI-Agent.
"""
import asyncio
import os
import re
import threading
from typing import Dict, Any, Optional
from termcolor import colored
from .permissions_manager import get_permissions_manager

def format_xml_args(xml_content: str) -> str:
    """Format XML content to show the tool arguments concisely."""
    if not xml_content:
        return ""
    
    # Remove timeout tags first
    content = re.sub(r'<timeout[^>]*>.*?</timeout>', '', xml_content, flags=re.DOTALL).strip()
    
    # Remove outer tool tags (bash, python, etc.) but keep the content
    content = re.sub(r'^<(bash|python|shell)[^>]*>', '', content, flags=re.MULTILINE).strip()
    content = re.sub(r'</(bash|python|shell)>$', '', content, flags=re.MULTILINE).strip()
    
    # Remove any remaining XML tags
    cleaned = re.sub(r'<[^>]+>', '', content).strip()
    
    # If it's multiline, show first meaningful line
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    if lines:
        return lines[0]
    
    return cleaned if cleaned else ""

def get_path_type_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about the path type for better display formatting.
    
    Args:
        file_path: The file path to analyze
        
    Returns:
        Dictionary with path type information
    """
    if not file_path:
        return {"type": "unknown", "icon": "", "description": ""}
    
    import os
    path_lower = file_path.lower()
    
    # Detect different types of paths
    if path_lower.endswith(('.py', '.pyw')):
        return {"type": "python_source", "icon": "🐍", "description": "Python file"}
    elif path_lower.endswith(('.js', '.ts', '.jsx', '.tsx')):
        return {"type": "javascript", "icon": "📜", "description": "JavaScript/TypeScript file"}
    elif path_lower.endswith(('.json', '.yaml', '.yml', '.toml', '.ini', '.cfg')):
        return {"type": "config", "icon": "⚙️", "description": "Configuration file"}
    elif path_lower.endswith(('.md', '.rst', '.txt', '.doc', '.docx')):
        return {"type": "document", "icon": "📄", "description": "Document file"}
    elif path_lower.endswith(('.sh', '.bash', '.zsh', '.fish', '.bat', '.cmd')):
        return {"type": "script", "icon": "🔧", "description": "Script file"}
    elif os.path.isdir(file_path):
        return {"type": "directory", "icon": "📁", "description": "Directory"}
    elif any(pattern in path_lower for pattern in ['cache', 'tmp', 'temp', '.pyc']):
        return {"type": "cache", "icon": "💾", "description": "Cache/temporary file"}
    elif 'site-packages' in path_lower or 'lib/python' in path_lower:
        return {"type": "library", "icon": "📚", "description": "Library file"}
    else:
        return {"type": "file", "icon": "📄", "description": "File"}

def format_file_path_for_display(file_path: str, max_length: int) -> str:
    """
    Format a file path for display with intelligent truncation.
    Shows the most relevant parts of the path to the user.
    
    Args:
        file_path: The full file path
        max_length: Maximum display length
        
    Returns:
        Formatted path string for display
    """
    if not file_path:
        return ""
        
    if len(file_path) <= max_length:
        return file_path
    
    import os
    
    # Try to show relative path from current working directory
    try:
        cwd = os.getcwd()
        if file_path.startswith(cwd):
            rel_path = os.path.relpath(file_path, cwd)
            if len(rel_path) <= max_length and not rel_path.startswith('..'):
                return rel_path
    except (ValueError, OSError):
        pass
    
    # If relative path doesn't work, show filename + parent directory
    filename = os.path.basename(file_path)
    parent_dir = os.path.basename(os.path.dirname(file_path))
    
    if parent_dir and parent_dir != '/':
        short_format = f"{parent_dir}/{filename}"
        if len(short_format) <= max_length:
            return short_format
    
    # If still too long, show just the filename with indication of truncation
    if len(filename) <= max_length - 4:
        return f".../{filename}"
    
    # Last resort: truncate filename itself
    if max_length > 6:  # Ensure we have room for "..." on both sides
        mid_point = (max_length - 6) // 2
        return f"{filename[:mid_point]}...{filename[-mid_point:]}"
    
    return filename[:max_length]

def get_enhanced_file_display(file_path: str, max_length: int) -> str:
    """
    Get enhanced file display with type information and smart formatting.
    
    Args:
        file_path: The file path to display
        max_length: Maximum display length
        
    Returns:
        Enhanced formatted path with type info
    """
    if not file_path:
        return ""
        
    path_info = get_path_type_info(file_path)
    formatted_path = format_file_path_for_display(file_path, max_length - 2)  # Leave room for icon
    
    # For cache/temp files, add a subtle indicator
    if path_info["type"] in ["cache", "library"]:
        return f"🔗 {formatted_path}"  # Link icon for indirect/system files
    else:
        return formatted_path

def get_terminal_width() -> int:
    """Get the current terminal width, with fallback to 80 columns."""
    try:
        import shutil
        width = shutil.get_terminal_size().columns
        # Ensure minimum width and maximum reasonable width
        return max(80, min(width, 160))
    except:
        return 80

async def enhanced_permission_prompt(
    args, 
    tool_name: str,
    action_description: str,
    file_path: str = None,
    command_hash: str = None,
    command_pattern: str = None,
    has_paths: bool = False,
    xml_content: str = None,
    input_event: Optional[asyncio.Event] = None,
    input_lock: Optional[threading.Lock] = None,
    shared_input: Optional[Dict[str, Any]] = None,
    get_user_input_func = None
) -> bool:
    """
    Enhanced permission prompt with always-allow functionality.
    """
    permissions_manager = get_permissions_manager()
    
    # Get dynamic terminal width for all menu boxes
    BOX_WIDTH = get_terminal_width()
    INNER_WIDTH = BOX_WIDTH - 2
    
    # Check if always denied
    if permissions_manager.is_always_denied(tool_name, file_path):
        def create_denied_operation_box(tool_name: str, action_description: str, file_path: str = None):
            tool_emojis = {"editfile": "✏️", "readfile": "📖", "writefile": "📝", "insertline": "➕", "deletelines": "🗑️", "bash": "🔧", "python": "🐍", "searchweb": "🌐", "showuser": "👁️", "memory": "🧠", "grepsearch": "🔍"}
            tool_emoji = tool_emojis.get(tool_name, "⚙️")
            args_display = format_xml_args(xml_content)
            
            title = "─ PERMISSION DENIED "
            remaining_width = INNER_WIDTH - len(title)
            top_border = f"┌{title}{'─' * remaining_width}┐"
            bottom_border = f"└{'─' * INNER_WIDTH}┘"
            
            print()
            print(colored(top_border, "light_red"))
            print(colored("│", "light_red") + f" ❌ {tool_emoji} {tool_name} operation blocked by saved preferences".ljust(INNER_WIDTH) + colored("│", "light_red"))
            if file_path:
                display_path = get_enhanced_file_display(file_path, INNER_WIDTH - 10)
                print(colored("│", "light_red") + f"    File: {display_path}".ljust(INNER_WIDTH) + colored("│", "light_red"))
            if args_display:
                args_short = (args_display[:(INNER_WIDTH - 9 - 3)] + "...") if len(args_display) > INNER_WIDTH - 9 else args_display
                print(colored("│", "light_red") + f"    Args: {args_short}".ljust(INNER_WIDTH) + colored("│", "light_red"))
            if action_description:
                action_display = (action_description[:(INNER_WIDTH - 11 - 3)] + "...") if len(action_description) > INNER_WIDTH - 11 else action_description
                print(colored("│", "light_red") + f"    Action: {action_display}".ljust(INNER_WIDTH) + colored("│", "light_red"))
            print(colored("│", "light_red") + "".ljust(INNER_WIDTH) + colored("│", "light_red"))
            print(colored("│", "light_red") + " 💡 To manage permissions: /permissions list".ljust(INNER_WIDTH) + colored("│", "light_red"))
            print(colored(bottom_border, "light_red"))
            print()
        
        create_denied_operation_box(tool_name, action_description, file_path)
        return False
    
    # Check if always allowed
    if permissions_manager.is_always_allowed(tool_name, file_path, command_hash, xml_content):
        def create_granted_operation_box(tool_name: str, action_description: str, file_path: str = None):
            tool_emojis = {"editfile": "✏️", "readfile": "📖", "writefile": "📝", "insertline": "➕", "deletelines": "🗑️", "bash": "🔧", "python": "🐍", "searchweb": "🌐", "showuser": "👁️", "memory": "🧠", "grepsearch": "🔍"}
            tool_emoji = tool_emojis.get(tool_name, "⚙️")
            args_display = format_xml_args(xml_content)
            
            title = "─ PERMISSION GRANTED "
            remaining_width = INNER_WIDTH - len(title)
            top_border = f"┌{title}{'─' * remaining_width}┐"
            bottom_border = f"└{'─' * INNER_WIDTH}┘"

            print(colored(top_border, "light_green"))
            print(colored("│", "light_green") + f" ✅ {tool_emoji} {tool_name} operation permitted 💡 To manage permissions: /permissions list".ljust(INNER_WIDTH) + colored("│", "light_green"))
            if file_path:
                display_path = get_enhanced_file_display(file_path, INNER_WIDTH - 10)
                print(colored("│", "light_green") + f"    File: {display_path}".ljust(INNER_WIDTH) + colored("│", "light_green"))
            if args_display:
                args_short = (args_display[:(INNER_WIDTH - 9 - 3)] + "...") if len(args_display) > INNER_WIDTH - 9 else args_display
                print(colored("│", "light_green") + f"    Args: {args_short}".ljust(INNER_WIDTH) + colored("│", "light_green"))
            if action_description:
                action_display = (action_description[:(INNER_WIDTH - 11 - 3)] + "...") if len(action_description) > INNER_WIDTH - 11 else action_description
                print(colored("│", "light_green") + f"    Action: {action_display}".ljust(INNER_WIDTH) + colored("│", "light_green"))
            print(colored("│", "light_green") + "".ljust(INNER_WIDTH) + colored("│", "light_green"))
            print(colored(bottom_border, "light_green"))
        
        create_granted_operation_box(tool_name, action_description, file_path)
        return True
    
    # Get smart quick action
    from .path_detector import PathDetector
    command_pattern = PathDetector.get_command_pattern(tool_name, xml_content or "")
    smart_action = PathDetector.get_smart_quick_action(tool_name, command_pattern, file_path)
    
    def create_tool_operation_box(tool_name: str, action_description: str, file_path: str = None, args_content: str = None):
        top_left, top_right, btm_left, btm_right = "╭", "╮", "╰", "╯"
        horizontal, vertical = "─", "│"
        
        tool_emojis = {"editfile": "✏️", "readfile": "📖", "writefile": "📝", "insertline": "➕", "deletelines": "🗑️", "bash": "🔧", "python": "🐍", "searchweb": "🌐", "showuser": "👁️", "memory": "🧠", "grepsearch": "🔍"}
        tool_emoji = tool_emojis.get(tool_name, "⚙️")
        
        display_path, action_display, args_display = "", "", "(no arguments)"
        if file_path:
            max_len = INNER_WIDTH - len(" File: ")
            display_path = get_enhanced_file_display(file_path, max_len)
        if action_description:
            max_len = INNER_WIDTH - len(" Action: ")
            action_display = (action_description[:max_len-3] + "...") if len(action_description) > max_len else action_description
        if args_content:
            cleaned_args = re.sub(r'\s+', ' ', args_content).strip()
            max_len = INNER_WIDTH - len(" Args: ")
            args_display = (cleaned_args[:max_len-3] + "...") if len(cleaned_args) > max_len else cleaned_args
            
        def format_line(line_content):
            return f"{vertical}{line_content.ljust(INNER_WIDTH)}{vertical}"

        # Create colorized bash-style header
        import getpass
        import socket
        current_user = getpass.getuser()
        hostname = socket.gethostname()
        current_dir = os.getcwd()
        
        user_colored = colored(current_user, 'green', attrs=['bold'])
        host_colored = colored(hostname, 'blue', attrs=['bold'])
        dir_colored = colored(current_dir, 'yellow')
        command_colored = colored(f"{args_display}", 'white', attrs=['bold'])
        
        bash_header = f" 💻 {user_colored}@{host_colored}:{dir_colored}$"
        bash_body = f" ▶️  {command_colored}"

        # Print the permission box with full terminal width
        print(colored(f"{top_left}{horizontal * INNER_WIDTH}{top_right}", "light_yellow"))
        
        # Title line with emoji and text, properly right-padded
        title_text = f" {tool_emoji} TOOL OPERATION PERMISSION - {tool_name}"
        title_padding = " " * max(0, INNER_WIDTH - len(title_text))
        print(colored(f"{vertical}{title_text}{title_padding}{vertical}", "light_yellow"))
        
        # Horizontal separator
        print(colored(f"{vertical}{horizontal * INNER_WIDTH}{vertical}", "light_yellow"))
        
        # Bash-style header line
        header_visual_len = len(re.sub(r'\x1b\[[0-9;]*m', '', bash_header))
        header_padding = " " * max(0, INNER_WIDTH - header_visual_len)
        print(colored(f"{vertical}{bash_header}{header_padding}{vertical}", "light_yellow"))
        # bash style tool
        body_visual_len = len(re.sub(r'\x1b\[[0-9;]*m', '', bash_body))
        body_padding = " " * max(0, INNER_WIDTH - body_visual_len)
        print(colored(f"{vertical}{bash_body}{body_padding}{vertical}"))

        
        # Bottom border
        print(colored(f"{btm_left}{horizontal * INNER_WIDTH}{btm_right}", "light_yellow"))
    
    create_tool_operation_box(tool_name, action_description, file_path, format_xml_args(xml_content))

    # Get permission suggestions and options before potential GUI mode handling
    suggestions = permissions_manager.get_permission_suggestions(tool_name, file_path, command_pattern, has_paths)

    # For bash commands, add a read-only operations quick option at the top
    read_only_option = None
    if tool_name == "bash" and xml_content:
        cwd = os.getcwd()
        from .path_detector import PathDetector
        actual_command = PathDetector.get_command_pattern(tool_name, xml_content).replace("bash: ", "")
        if PathDetector._is_safe_command(actual_command.lower()):
            read_only_option = f"All read-only operations in {os.path.basename(cwd)}"

    # Check for GUI mode and send permission request to visual interface
    from core.globals import g
    if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
        try:
            from tools.visualresponse import visualresponse
            session = visualresponse._get_session()
            if session and session.is_healthy():
                import time
                permission_payload = {
                    "id": f"permission-{int(time.time() * 1000)}",
                    "action": "create",
                    "type": "permission_request",
                    "permission_data": {
                        "tool_name": tool_name,
                        "action_description": action_description,
                        "file_path": file_path,
                        "command_pattern": command_pattern,
                        "xml_content": xml_content or "",
                        "args_display": format_xml_args(xml_content) if xml_content else "",
                        "smart_action": smart_action,
                        "read_only_option": read_only_option,
                        "suggestions": suggestions
                    }
                }

                session.send_update(permission_payload)

                # Wait for response from visual interface
                if hasattr(g, 'visual_input_queue'):
                    try:
                        import json
                        # Wait for permission response (with timeout)
                        timeout_seconds = 60  # 1 minute timeout
                        import multiprocessing
                        try:
                            response_data = g.visual_input_queue.get(timeout=timeout_seconds)
                            response = json.loads(response_data)

                            if response.get('type') == 'permission_response':
                                permission_result = response.get('result')
                                permission_action = response.get('action')

                                # Handle the permission response
                                if permission_result == 'allow':
                                    return True
                                elif permission_result == 'deny':
                                    return False
                                elif permission_result == 'always_allow':
                                    # Handle always allow with scope
                                    scope = permission_action.get('scope', 'exact')
                                    permissions_manager.save_permission(tool_name, file_path, True,
                                                                      command_pattern, scope, permission_action.get('pattern'))
                                    return True
                                elif permission_result == 'always_deny':
                                    # Handle always deny
                                    permissions_manager.save_permission(tool_name, file_path, False,
                                                                      command_pattern, 'exact')
                                    return False
                        except multiprocessing.TimeoutError:
                            print("⏰ Permission request timed out, falling back to CLI prompt")
                        except Exception as e:
                            print(f"❌ Error processing permission response: {e}")
                    except:
                        pass
        except Exception as e:
            # If GUI mode fails, fall back to CLI mode
            print(f"⚠️  GUI permission request failed, using CLI prompt: {e}")

    def format_option_line(text):
        # Calculate visual length without ANSI color codes
        visual_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
        # Account for the space after "│ " that we'll add in the print statement
        # INNER_WIDTH is the content width, so we need to subtract the leading space
        content_width = INNER_WIDTH - 2  # Account for "│ " at start and " │" at end
        padding_needed = max(0, content_width - len(visual_text))
        return text + (" " * padding_needed)

    def prepare_suggestion_for_display(text: str, max_len: int) -> str:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if len(cleaned) > max_len:
            return cleaned[:max_len-3] + "..."
        return cleaned

    options_title = "─ Permission Options "
    remaining_width = INNER_WIDTH - len(options_title)
    top_border = f"┌{options_title}{'─' * remaining_width}┐"
    bottom_border = f"└{'─' * INNER_WIDTH}┘"
    
    print(colored(top_border, "blue"))
    
    # Build the options dynamically to avoid duplicates
    options = []
    used_keys = set()
    
    if hasattr(args, 'voice') and args.voice:
        options.append(f" {colored('Enter', 'green')} or {colored('y', 'green')} (say '{colored('yes', 'green')}') → Confirm this operation")
        options.append(f" {colored('n', 'light_red')} (say '{colored('no', 'light_red')}') → Abort this operation")
        used_keys.update(['enter', 'y', 'n'])
        
        # Add read-only option first if available
        if read_only_option:
            options.append(f" {colored('r', 'green')} (say '{colored('readonly', 'green')}') → {read_only_option}")
            used_keys.add('r')
        
        # Add smart action if available, key not used, and not already permitted
        if smart_action and smart_action['quick_key'] not in used_keys:
            if not permissions_manager.is_smart_action_already_permitted(smart_action, tool_name, file_path):
                options.append(f" {colored(smart_action['quick_key'], 'green')} (say '{colored(smart_action['category'], 'green')}') → Always allow {smart_action['display_name']} for {smart_action['scope_description']}")
                used_keys.add(smart_action['quick_key'])
        
        # Add specific command suggestion for bash
        if tool_name == "bash" and xml_content:
            from .path_detector import PathDetector
            actual_command = PathDetector.get_command_pattern(tool_name, xml_content).replace("bash: ", "")
            base_command = actual_command.split()[0] if actual_command.split() else ""
            if base_command and 'f' not in used_keys:
                options.append(f" {colored('f', 'green')} (say '{colored('find', 'green')}') → Always allow 'bash: {base_command}' commands")
                used_keys.add('f')
        
        # Add remaining suggestions with available keys
        option_keys = ['a', 's', 'p', 't', 'c', 'e']
        suggestion_count = 0
        for key in option_keys:
            if key not in used_keys and suggestion_count < len(suggestions):
                suggestion = suggestions[suggestion_count]
                processed_suggestion = prepare_suggestion_for_display(suggestion, INNER_WIDTH - 24)
                options.append(f" {colored(key, 'green')} (say '{colored(f'always{suggestion_count+1}', 'green')}') → {processed_suggestion}")
                used_keys.add(key)
                suggestion_count += 1
        
        options.append(f" {colored('d', 'red')} (say '{colored('deny', 'red')}') → Always deny this file")
        options.append(f" {colored('D', 'red')} (say '{colored('denyall', 'red')}') → Always deny all {tool_name} operations")
        options.append(f" {colored('m', 'light_yellow')} (say '{colored('manage', 'light_yellow')}') → Manage all saved permissions")
        options.append(f" {colored('?', 'light_yellow')} (say '{colored('help', 'light_yellow')}') → Show detailed help")
    else:
        options.append(f" {colored('Enter', 'green')} or {colored('y', 'green')} → Confirm this operation")
        options.append(f" {colored('n', 'light_red')} → Abort this operation")
        used_keys.update(['enter', 'y', 'n'])
        
        # Add read-only option first if available
        if read_only_option:
            options.append(f" {colored('r', 'green')} → {read_only_option}")
            used_keys.add('r')
        
        # Add smart action if available, key not used, and not already permitted
        if smart_action and smart_action['quick_key'] not in used_keys:
            if not permissions_manager.is_smart_action_already_permitted(smart_action, tool_name, file_path):
                options.append(f" {colored(smart_action['quick_key'], 'green')} → Always allow {smart_action['display_name']} for {smart_action['scope_description']}")
                used_keys.add(smart_action['quick_key'])
        
        # Add specific command suggestion for bash
        if tool_name == "bash" and xml_content:
            from .path_detector import PathDetector
            actual_command = PathDetector.get_command_pattern(tool_name, xml_content).replace("bash: ", "")
            base_command = actual_command.split()[0] if actual_command.split() else ""
            if base_command and 'f' not in used_keys:
                options.append(f" {colored('f', 'green')} → Always allow 'bash: {base_command}' commands")
                used_keys.add('f')
        
        # Add remaining suggestions with available keys
        option_keys = ['a', 's', 'p', 't', 'c', 'e']
        suggestion_count = 0
        for key in option_keys:
            if key not in used_keys and suggestion_count < len(suggestions):
                suggestion = suggestions[suggestion_count]
                processed_suggestion = prepare_suggestion_for_display(suggestion, INNER_WIDTH - 7)
                options.append(f" {colored(key, 'green')} → {processed_suggestion}")
                used_keys.add(key)
                suggestion_count += 1

        options.append(f" {colored('d', 'red')} → Always deny this file")
        options.append(f" {colored('D', 'red')} → Always deny all {tool_name} operations")
        options.append(f" {colored('m', 'light_yellow')} → Manage all saved permissions")
        options.append(f" {colored('?', 'light_yellow')} → Show detailed help")
    
    for line in options:
        formatted_line = format_option_line(line)
        print(colored(f"│ {formatted_line} │", "blue"))
    
    print(colored(bottom_border, "blue"))
    
    while True:
        try:
            if get_user_input_func:
                user_input, _ = await get_user_input_func(
                    args, None, colored("Choose an option: ", "blue"),
                    input_event=input_event, 
                    input_lock=input_lock, 
                    shared_input=shared_input
                )
                user_input = user_input or ""
            else:
                user_input = input(colored(f"Choose an option {colored('y', 'green')}: ", "blue"))
            
            user_input = user_input.strip().lower()
        except KeyboardInterrupt:
            # User pressed Ctrl+C during permission prompt - treat as denial
            print(colored("\n❌ Operation cancelled by user (Ctrl+C).", "yellow"))
            return False
        
        if user_input in ('', 'y', 'yes'):
            return True
        elif user_input in ('n', 'no'):
            tool_emojis = {"editfile": "✏️", "readfile": "📖", "writefile": "📝", "insertline": "➕", "deletelines": "🗑️", "bash": "🔧", "python": "🐍", "searchweb": "🌐", "showuser": "👁️", "memory": "🧠", "grepsearch": "🔍"}
            tool_emoji = tool_emojis.get(tool_name, "⚙️")
            args_display = format_xml_args(xml_content)
            
            title = "─ OPERATION CANCELLED "
            remaining_width = INNER_WIDTH - len(title)
            top_border = f"┌{title}{'─' * remaining_width}┐"
            bottom_border = f"└{'─' * INNER_WIDTH}┘"

            print(colored(top_border, "red"))
            print(colored("│", "red") + f" ❌ {tool_emoji} {tool_name} operation cancelled by user".ljust(INNER_WIDTH) + colored("│", "red"))
            if file_path:
                display_path = get_enhanced_file_display(file_path, INNER_WIDTH - 10)
                print(colored("│", "red") + f"    File: {display_path}".ljust(INNER_WIDTH) + colored("│", "red"))
            if args_display:
                args_short = (args_display[:(INNER_WIDTH - 9 - 3)] + "...") if len(args_display) > INNER_WIDTH - 9 else args_display
                print(colored("│", "red") + f"    Args: {args_short}".ljust(INNER_WIDTH) + colored("│", "red"))
            if action_description:
                action_display = (action_description[:(INNER_WIDTH - 11 - 3)] + "...") if len(action_description) > INNER_WIDTH - 11 else action_description
                print(colored("│", "red") + f"    Action: {action_display}".ljust(INNER_WIDTH) + colored("│", "red"))
            print(colored(bottom_border, "red"))
            return False
        elif user_input == 'r' and read_only_option:
            # Handle read-only operations permission
            cwd = os.getcwd()
            permissions_manager.add_always_allow("bash", dev_tool_category="safe_commands", pattern=f"{cwd}/*")
            print(colored(f"✅ Always allow read-only operations in {os.path.basename(cwd)}", "green"))
            return True
        elif user_input == 'f' and tool_name == "bash" and xml_content:
            # Handle specific bash command permission
            from .path_detector import PathDetector
            actual_command = PathDetector.get_command_pattern(tool_name, xml_content).replace("bash: ", "")
            base_command = actual_command.split()[0] if actual_command.split() else ""
            if base_command:
                permissions_manager.add_always_allow(tool_name, command_pattern=f"{base_command}*", scope_path="*")
                print(colored(f"✅ Always allow 'bash: {base_command}' commands globally", "green"))
                return True
        elif smart_action and (user_input == smart_action['quick_key'] or user_input == smart_action['category']):
            await handle_smart_quick_action(smart_action, tool_name, file_path, permissions_manager, command_pattern)
            return True
        elif user_input in ['a', 's', 'p', 't', 'c', 'e'] and len(suggestions) > 0:
            option_map = {'a': 0, 's': 1, 'p': 2, 't': 3, 'c': 4, 'e': 5}
            if option_map[user_input] < len(suggestions):
                await handle_direct_always_allow(suggestions[option_map[user_input]], tool_name, file_path, permissions_manager, command_hash)
                return True
        elif user_input == 'd':
            if file_path:
                permissions_manager.add_always_deny(tool_name, file_path)
                print(colored(f"❌ Always deny {tool_name} on this file", "red"))
                print(colored(f"   File: {file_path}", "white"))
            else:
                print(colored("❌ No file specified for denial", "yellow"))
            return False
        elif user_input == 'D':
            permissions_manager.add_always_deny(tool_name)
            print(colored(f"❌ Always deny all {tool_name} operations", "red"))
            return False
        elif user_input in ('m', 'manage'):
            await handle_permission_management(get_user_input_func, args, input_event, input_lock, shared_input)
            continue
        elif user_input in ('?', 'help'):
            show_permission_help(tool_name, file_path)
        else:
            valid_options = "[Enter/y], n"
            if read_only_option:
                valid_options += ", r"
            if tool_name == "bash" and xml_content:
                valid_options += ", f"
            if smart_action:
                valid_options += f", {smart_action['quick_key']}"
            valid_options += ", a/s/p/t/c/e, d/D, m, ?"
            print(colored(f"❓ Invalid option. Enter one of: {valid_options}", "yellow"))

async def handle_direct_always_allow(suggestion: str, tool_name: str, file_path: str, permissions_manager, command_hash: str):
    if "* commands in current directory tree" in suggestion:
        # Handle pattern-based permission for current directory tree
        # Extract the command pattern (e.g., "ls" from 'All "ls*" commands in current directory tree')
        import re
        pattern_match = re.search(r'"([^"]+)\*"', suggestion)
        if pattern_match:
            base_command = pattern_match.group(1)
            command_pattern = f"{base_command}*"
            current_dir = os.getcwd()
            scope_path = f"{current_dir}/*"
            permissions_manager.add_always_allow(tool_name, command_pattern=command_pattern, scope_path=scope_path)
            print(colored(f"✅ Pattern permission added: {tool_name}:{command_pattern} for {scope_path}", "green"))
        else:
            print(colored(f"❌ Failed to parse pattern suggestion: {suggestion}", "red"))
    elif "* commands globally" in suggestion:
        # Handle pattern-based permission globally
        import re
        pattern_match = re.search(r'"([^"]+)\*"', suggestion)
        if pattern_match:
            base_command = pattern_match.group(1)
            command_pattern = f"{base_command}*"
            permissions_manager.add_always_allow(tool_name, command_pattern=command_pattern, scope_path="*")
            print(colored(f"✅ Global pattern permission added: {tool_name}:{command_pattern}", "green"))
        else:
            print(colored(f"❌ Failed to parse pattern suggestion: {suggestion}", "red"))
    elif "Exact command:" in suggestion and command_hash:
        permissions_manager.add_always_allow(tool_name, command_hash=command_hash)
        print(colored(f"✅ Permission rule added: {suggestion}", "green"))
    elif "Exact file:" in suggestion:
        permissions_manager.add_always_allow(tool_name, file_path)
        print(colored(f"✅ Permission rule added: {suggestion}", "green"))
    elif "Directory:" in suggestion:
        dir_path = suggestion.split("Directory: ")[1]
        permissions_manager.add_always_allow(tool_name, None, dir_path)
        print(colored(f"✅ Permission rule added: {suggestion}", "green"))
    elif "Project:" in suggestion:
        project_path = suggestion.split("Project: ")[1]
        permissions_manager.add_always_allow(tool_name, None, project_path)
        print(colored(f"✅ Permission rule added: {suggestion}", "green"))
    elif "tools (" in suggestion:
        group_display = suggestion.split("All ")[1].split(" tools")[0]
        group_name = group_display.lower().replace(' ', '_')
        if file_path:
            permissions_manager.add_always_allow(tool_name, file_path, group_name=group_name)
        else:
            permissions_manager.add_always_allow(tool_name, group_name=group_name)
        print(colored(f"✅ Permission rule added: {suggestion}", "green"))
    elif "All " in suggestion:
        permissions_manager.add_always_allow(tool_name)
        print(colored(f"✅ Permission rule added: {suggestion}", "green"))
    else:
        print(colored(f"✅ Permission rule added: {suggestion}", "green"))

async def handle_smart_quick_action(smart_action: Dict[str, Any], tool_name: str, file_path: str, permissions_manager, command_pattern: str):
    category = smart_action['category']
    scope_path = smart_action['scope_path']
    scope_description = smart_action['scope_description']
    display_name = smart_action['display_name']
    
    if smart_action['permission_scope'] == 'global':
        permissions_manager.add_always_allow(tool_name, dev_tool_category=category, pattern=category)
        print(colored(f"✅ Always allow {display_name} {scope_description}", "green"))
        print(colored(f"   Category: {category}", "white"))
    else:
        permissions_manager.add_always_allow(tool_name, dev_tool_category=category, pattern=scope_path)
        print(colored(f"✅ Always allow {display_name} for {scope_description}", "green"))
        print(colored(f"   Pattern: {scope_path}", "white"))

async def handle_permission_management(get_user_input_func, args, input_event, input_lock, shared_input):
    from tools.permissions import permissions
    
    print()
    print(colored("🔧 PERMISSION MANAGEMENT", "yellow", attrs=['bold']))
    print(colored("─" * 50, "yellow"))
    
    result = permissions._list_permissions()
    print(result)
    
    print()
    print(colored("Options:", "cyan"))
    print(colored("  r <id>", "light_blue") + " - Remove permission by ID")
    print(colored("  clear", "red") + " - Clear all permissions")  
    print(colored("  back", "green") + " - Return to permission prompt")
    print()
    
    while True:
        try:
            if get_user_input_func:
                user_input, _ = await get_user_input_func(
                    args, None, colored("Permission Manager> ", "cyan"),
                    input_event=input_event, input_lock=input_lock, shared_input=shared_input
                )
            else:
                user_input = input(colored("Permission Manager> ", "cyan"))
            
            user_input = user_input.strip().lower()
        except KeyboardInterrupt:
            # User pressed Ctrl+C in permission manager - return to prompt
            print(colored("\n↩️  Returning to permission prompt (Ctrl+C)...", "yellow"))
            break
        
        if user_input in ('back', 'exit'):
            print(colored("↩️  Returning to permission prompt...", "green"))
            break
        elif user_input == 'clear':
            result = permissions._clear_all_permissions()
            print(result)
            break
        elif user_input.startswith('r ') and len(user_input) > 2:
            perm_id = user_input[2:].strip()
            result = permissions._remove_permission(perm_id)
            print(result)
            print()
            result = permissions._list_permissions()
            print(result)
        elif user_input == 'list':
            result = permissions._list_permissions()
            print(result)
        else:
            print(colored("❓ Invalid command. Use: r <id>, clear, list, or back", "yellow"))

def show_permission_help(tool_name: str, file_path: str = None):
    print(colored("\n📖 Permission Options Help:", "blue"))
    print(colored("  Enter/y", "light_green") + " - Allow this action once")
    print(colored("  n", "red") + " - Deny this action once")  
    print(colored("  a,s,p,t", "blue") + " - Always allow with a specific scope")
    print(colored("  d/D", "magenta") + " - Always deny this action (file or tool)")
    print(colored("  m", "yellow") + " - Manage all saved permissions")
    print(colored("  ?", "yellow") + " - Show this help")
    print()