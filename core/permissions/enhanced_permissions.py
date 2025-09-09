# enhanced_permissions.py
"""
Enhanced permission system with always-allow functionality for CLI-Agent.
"""
import asyncio
import threading
from typing import Dict, Any, Optional
from termcolor import colored
from .permissions_manager import get_permissions_manager

def format_xml_args(xml_content: str) -> str:
    """Format XML content to show the tool arguments concisely."""
    if not xml_content:
        return ""
    
    # Remove outer tags and clean up the content
    import re
    cleaned = re.sub(r'<[^>]+>', '', xml_content).strip()
    
    # If it's multiline, show first meaningful line
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    if lines:
        return lines[0]
    
    return cleaned if cleaned else ""

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
    
    Args:
        args: Command line arguments
        tool_name: Name of the tool requesting permission
        action_description: Description of the action being performed
        file_path: Optional file path being acted upon
        input_event: Async event for input handling
        input_lock: Thread lock for input coordination
        shared_input: Shared input dictionary
        get_user_input_func: Function to get user input
        
    Returns:
        bool: True if permission granted, False otherwise
    """
    permissions_manager = get_permissions_manager()
    
    # Check if always denied
    if permissions_manager.is_always_denied(tool_name, file_path):
        # Create detailed denied message box
        def create_denied_operation_box(tool_name: str, action_description: str, file_path: str = None):
            tool_emojis = {
                "editfile": "✏️", "readfile": "📖", "writefile": "📝", "insertline": "➕",
                "deletelines": "🗑️", "bash": "🔧", "python": "🐍", "searchweb": "🌐",
                "showuser": "👁️", "memory": "🧠", "search": "🔍"
            }
            tool_emoji = tool_emojis.get(tool_name, "⚙️")
            args_display = format_xml_args(xml_content)
            
            print()
            print(colored("┌─ PERMISSION DENIED ──────────────────────────────────────────────┐", "red"))
            print(colored("│", "red") + f" ❌ {tool_emoji} {tool_name} operation blocked by saved preferences".ljust(65) + colored("│", "red"))
            if file_path:
                display_path = ("..." + file_path[-45:]) if len(file_path) > 45 else file_path
                print(colored("│", "red") + f"    File: {display_path}".ljust(65) + colored("│", "red"))
            if args_display:
                args_short = (args_display[:50] + "...") if len(args_display) > 50 else args_display
                print(colored("│", "red") + f"    Args: {args_short}".ljust(65) + colored("│", "red"))
            if action_description:
                action_display = (action_description[:55] + "...") if len(action_description) > 55 else action_description
                print(colored("│", "red") + f"    Action: {action_display}".ljust(65) + colored("│", "red"))
            print(colored("│", "red") + "".ljust(65) + colored("│", "red"))
            print(colored("│", "red") + " 💡 To manage permissions: -permissions list".ljust(65) + colored("│", "red"))
            print(colored("└───────────────────────────────────────────────────────────────────┘", "red"))
            print()
        
        create_denied_operation_box(tool_name, action_description, file_path)
        return False
    
    # Check if always allowed (with command hash support and sensitive content check)
    if permissions_manager.is_always_allowed(tool_name, file_path, command_hash, xml_content):
        # Create detailed granted message box
        def create_granted_operation_box(tool_name: str, action_description: str, file_path: str = None):
            tool_emojis = {
                "editfile": "✏️", "readfile": "📖", "writefile": "📝", "insertline": "➕", 
                "deletelines": "🗑️", "bash": "🔧", "python": "🐍", "searchweb": "🌐",
                "showuser": "👁️", "memory": "🧠", "search": "🔍"
            }
            tool_emoji = tool_emojis.get(tool_name, "⚙️")
            args_display = format_xml_args(xml_content)
            
            print()
            print(colored("┌─ PERMISSION GRANTED ─────────────────────────────────────────────┐", "green"))
            print(colored("│", "green") + f" ✅ {tool_emoji} {tool_name} operation permitted by saved preferences".ljust(65) + colored("│", "green"))
            if file_path:
                display_path = ("..." + file_path[-45:]) if len(file_path) > 45 else file_path
                print(colored("│", "green") + f"    File: {display_path}".ljust(65) + colored("│", "green"))
            if args_display:
                args_short = (args_display[:50] + "...") if len(args_display) > 50 else args_display
                print(colored("│", "green") + f"    Args: {args_short}".ljust(65) + colored("│", "green"))
            if action_description:
                action_display = (action_description[:55] + "...") if len(action_description) > 55 else action_description
                print(colored("│", "green") + f"    Action: {action_display}".ljust(65) + colored("│", "green"))
            print(colored("│", "green") + "".ljust(65) + colored("│", "green"))
            print(colored("│", "green") + " 💡 To manage permissions: -permissions list".ljust(65) + colored("│", "green"))
            print(colored("└───────────────────────────────────────────────────────────────────┘", "green"))
            print()
        
        create_granted_operation_box(tool_name, action_description, file_path)
        return True
    
    # Get smart quick action for this command
    from .path_detector import PathDetector
    command_pattern = PathDetector.get_command_pattern(tool_name, xml_content or "")
    smart_action = PathDetector.get_smart_quick_action(tool_name, command_pattern, file_path)
    
    # Build enhanced visual permission prompt with box
    def create_tool_operation_box(tool_name: str, action_description: str, file_path: str = None, args_content: str = None):
        """Create a visually appealing box for tool operations."""
        # Box characters
        top_left = "╭"
        top_right = "╮"
        bottom_left = "╰"
        bottom_right = "╯"
        horizontal = "─"
        vertical = "│"
        
        # Tool emoji mapping
        tool_emojis = {
            "editfile": "✏️",
            "readfile": "📖", 
            "writefile": "📝",
            "insertline": "➕",
            "deletelines": "🗑️",
            "bash": "🔧",
            "python": "🐍",
            "searchweb": "🌐",
            "showuser": "👁️",
            "memory": "🧠",
            "search": "🔍"
        }
        
        tool_emoji = tool_emojis.get(tool_name, "⚙️")
        
        # Format file path
        if file_path:
            if len(file_path) > 60:
                display_path = "..." + file_path[-57:]
            else:
                display_path = file_path
            
        # Format action description
        if len(action_description) > 70:
            action_display = action_description[:67] + "..."
        else:
            action_display = action_description
        
        # Format arguments content
        if args_content:
            if len(args_content) > 60:
                args_display = args_content[:57] + "..."
            else:
                args_display = args_content
        else:
            args_display = "(no arguments)"
            
        # Fixed box width for consistent display
        box_width = 75
        
        # Helper function to pad text properly
        def pad_line(text, width):
            return text + " " * (width - len(text))
        
        # Build box
        print(colored(f"{top_left}{horizontal * (box_width - 2)}{top_right}", "yellow"))
        
        # Title line
        title_text = f" {tool_emoji} TOOL OPERATION PERMISSION"
        print(colored(f"{vertical}", "yellow") + colored(pad_line(title_text, box_width - 2), "yellow", attrs=['bold']) + colored(f"{vertical}", "yellow"))
        
        # Separator line
        print(colored(f"{vertical}{horizontal * (box_width - 2)}{vertical}", "yellow"))
        
        # Content lines with proper spacing
        def format_line(label, content, color='white'):
            content_len = len(f" {label}: {content}")
            padding = " " * max(0, box_width - 2 - content_len)
            return f" {label}: {colored(content, color, attrs=['bold'])}{padding}"
        
        # Tool line
        print(colored(f"{vertical}", "yellow") + format_line("Tool", tool_name, 'cyan') + colored(f"{vertical}", "yellow"))
        
        # File line (only if file_path exists)
        if file_path:
            print(colored(f"{vertical}", "yellow") + format_line("File", display_path, 'white') + colored(f"{vertical}", "yellow"))
        
        # Args line
        print(colored(f"{vertical}", "yellow") + format_line("Args", args_display, 'magenta') + colored(f"{vertical}", "yellow"))
        
        # Action line
        print(colored(f"{vertical}", "yellow") + format_line("Action", action_display, 'light_blue') + colored(f"{vertical}", "yellow"))
        
        print(colored(f"{bottom_left}{horizontal * (box_width - 2)}{bottom_right}", "yellow"))
    
    create_tool_operation_box(tool_name, action_description, file_path, format_xml_args(xml_content))
    
    # Build options text with better formatting
    print(colored("┌─ Permission Options ─────────────────────────────────────────────┐", "cyan"))
    
    # Get permission suggestions to show directly
    suggestions = permissions_manager.get_permission_suggestions(tool_name, file_path, command_pattern, has_paths)
    
    # Add voice keywords if voice mode is active
    if hasattr(args, 'voice') and args.voice:
        print(colored("│", "cyan") + f" {colored('Enter', 'light_green', attrs=['bold'])} or {colored('y', 'light_green', attrs=['bold'])} (say '{colored('yes', 'light_green')}' or '{colored('execute', 'light_green')}') → Confirm this operation".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('n', 'red', attrs=['bold'])} (say '{colored('no', 'red')}' or '{colored('abort', 'red')}') → Abort this operation".ljust(65) + colored("│", "cyan"))
        
        # Add smart quick action if available
        if smart_action:
            quick_key = smart_action['quick_key']
            display_name = smart_action['display_name']
            scope_desc = smart_action['scope_description']
            print(colored("│", "cyan") + f" {colored(quick_key, 'light_blue', attrs=['bold'])} (say '{colored(smart_action['category'], 'light_blue')}') → Always allow {display_name} for {scope_desc}".ljust(65) + colored("│", "cyan"))
        
        # Show direct always-allow options
        option_keys = ['a', 's', 'p', 't']
        for i, suggestion in enumerate(suggestions[:4]):  # Show up to 4 direct options
            key = option_keys[i]
            print(colored("│", "cyan") + f" {colored(key, 'blue', attrs=['bold'])} (say '{colored(f'always{i+1}', 'blue')}') → {suggestion}".ljust(65) + colored("│", "cyan"))
        
        # Show direct always-deny options
        print(colored("│", "cyan") + f" {colored('d', 'magenta', attrs=['bold'])} (say '{colored('deny', 'magenta')}') → Always deny this file".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('D', 'magenta', attrs=['bold'])} (say '{colored('denyall', 'magenta')}') → Always deny all {tool_name} operations".ljust(65) + colored("│", "cyan"))
        
        print(colored("│", "cyan") + f" {colored('m', 'yellow', attrs=['bold'])} (say '{colored('manage', 'yellow')}') → Manage all saved permissions".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('?', 'yellow', attrs=['bold'])} (say '{colored('help', 'yellow')}') → Show detailed help".ljust(65) + colored("│", "cyan"))
    else:
        print(colored("│", "cyan") + f" {colored('Enter', 'light_green', attrs=['bold'])} or {colored('y', 'light_green', attrs=['bold'])} → Confirm this operation".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('n', 'red', attrs=['bold'])} → Abort this operation".ljust(65) + colored("│", "cyan"))
        
        # Add smart quick action if available
        if smart_action:
            quick_key = smart_action['quick_key']
            display_name = smart_action['display_name']
            scope_desc = smart_action['scope_description']
            print(colored("│", "cyan") + f" {colored(quick_key, 'light_blue', attrs=['bold'])} → Always allow {display_name} for {scope_desc}".ljust(65) + colored("│", "cyan"))
        
        # Show direct always-allow options
        option_keys = ['a', 's', 'p', 't']
        for i, suggestion in enumerate(suggestions[:4]):  # Show up to 4 direct options
            key = option_keys[i]
            print(colored("│", "cyan") + f" {colored(key, 'blue', attrs=['bold'])} → {suggestion}".ljust(65) + colored("│", "cyan"))
        
        # Show direct always-deny options
        print(colored("│", "cyan") + f" {colored('d', 'magenta', attrs=['bold'])} → Always deny this file".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('D', 'magenta', attrs=['bold'])} → Always deny all {tool_name} operations".ljust(65) + colored("│", "cyan"))
        
        print(colored("│", "cyan") + f" {colored('m', 'yellow', attrs=['bold'])} → Manage all saved permissions".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('?', 'yellow', attrs=['bold'])} → Show detailed help".ljust(65) + colored("│", "cyan"))
    
    print(colored("└───────────────────────────────────────────────────────────────────┘", "cyan"))
    
    while True:
        if get_user_input_func:
            user_input, _ = await get_user_input_func(
                args, None, colored("> ", "cyan"),
                input_event=input_event, 
                input_lock=input_lock, 
                shared_input=shared_input
            )
        else:
            user_input = input(colored("> ", "cyan"))
        
        user_input = user_input.strip().lower()
        
        if user_input == '' or user_input == 'y' or user_input == 'yes':
            return True
        elif user_input == 'n' or user_input == 'no':
            # Get tool emoji for consistency
            tool_emojis = {
                "editfile": "✏️", "readfile": "📖", "writefile": "📝", "insertline": "➕",
                "deletelines": "🗑️", "bash": "🔧", "python": "🐍", "searchweb": "🌐",
                "showuser": "👁️", "memory": "🧠", "search": "🔍"
            }
            tool_emoji = tool_emojis.get(tool_name, "⚙️")
            
            args_display = format_xml_args(xml_content)
            
            print(colored("┌─ OPERATION CANCELLED ────────────────────────────────────────────┐", "red"))
            print(colored("│", "red") + f" ❌ {tool_emoji} {tool_name} operation cancelled by user".ljust(65) + colored("│", "red"))
            if file_path:
                display_path = ("..." + file_path[-45:]) if len(file_path) > 45 else file_path
                print(colored("│", "red") + f"    File: {display_path}".ljust(65) + colored("│", "red"))
            if args_display:
                args_short = (args_display[:50] + "...") if len(args_display) > 50 else args_display
                print(colored("│", "red") + f"    Args: {args_short}".ljust(65) + colored("│", "red"))
            if action_description:
                action_display = (action_description[:55] + "...") if len(action_description) > 55 else action_description
                print(colored("│", "red") + f"    Action: {action_display}".ljust(65) + colored("│", "red"))
            print(colored("└───────────────────────────────────────────────────────────────────┘", "red"))
            return False
        elif smart_action and (user_input == smart_action['quick_key'] or user_input == smart_action['category']):
            # Handle smart quick action
            await handle_smart_quick_action(smart_action, tool_name, file_path, permissions_manager, command_pattern)
            return True
        elif user_input in ['a', 's', 'p', 't'] and len(suggestions) > 0:
            # Handle direct always-allow options
            option_map = {'a': 0, 's': 1, 'p': 2, 't': 3}
            suggestion_index = option_map[user_input]
            if suggestion_index < len(suggestions):
                await handle_direct_always_allow(suggestions[suggestion_index], tool_name, file_path, permissions_manager, command_hash)
                return True
        elif user_input == 'd':
            # Always deny this file
            if file_path:
                permissions_manager.add_always_deny(tool_name, file_path)
                print(colored(f"❌ Always deny {tool_name} on this file", "red"))
                print(colored(f"   File: {file_path}", "white"))
            else:
                print(colored("❌ No file specified for denial", "yellow"))
            return False
        elif user_input == 'D':
            # Always deny all operations for this tool
            permissions_manager.add_always_deny(tool_name)
            print(colored(f"❌ Always deny all {tool_name} operations", "red"))
            return False
        elif user_input == 'm' or user_input == 'manage':
            await handle_permission_management(get_user_input_func, args, input_event, input_lock, shared_input)
            # Continue the loop to ask for permission again after management
            continue
        elif user_input == '?' or user_input == 'help':
            show_permission_help(tool_name, file_path)
        else:
            valid_options = "[Enter/y], n, a/s/p/t, d/D, m, ?"
            if smart_action:
                valid_options = f"[Enter/y], n, {smart_action['quick_key']}, a/s/p/t, d/D, m, ?"
            print(colored(f"❓ Invalid option. Enter one of: {valid_options}", "yellow"))

async def handle_direct_always_allow(suggestion: str, tool_name: str, file_path: str, permissions_manager, command_hash: str):
    """Handle direct always-allow option based on suggestion."""
    
    if "Exact command:" in suggestion and command_hash:
        # Add command-specific permission
        permissions_manager.add_always_allow(tool_name, command_hash=command_hash)
        print(colored(f"✅ {suggestion}", "green"))
    elif "Exact file:" in suggestion:
        permissions_manager.add_always_allow(tool_name, file_path)
        print(colored(f"✅ {suggestion}", "green"))
    elif "Directory:" in suggestion:
        dir_path = suggestion.split("Directory: ")[1]
        permissions_manager.add_always_allow(tool_name, None, dir_path)
        print(colored(f"✅ {suggestion}", "green"))
    elif "Project:" in suggestion:
        project_path = suggestion.split("Project: ")[1]
        permissions_manager.add_always_allow(tool_name, None, project_path)
        print(colored(f"✅ {suggestion}", "green"))
    elif "tools (" in suggestion:
        # Group-based permission
        # Extract group name from "All File Editing tools (editfile, writefile, appendfile)"
        group_display = suggestion.split("All ")[1].split(" tools")[0]
        group_name = group_display.lower().replace(' ', '_')
        
        if file_path:
            permissions_manager.add_always_allow(tool_name, file_path, group_name=group_name)
        else:
            permissions_manager.add_always_allow(tool_name, group_name=group_name)
        print(colored(f"✅ {suggestion}", "green"))
    elif "All " in suggestion:
        permissions_manager.add_always_allow(tool_name)
        print(colored(f"✅ {suggestion}", "green"))

async def handle_smart_quick_action(smart_action: Dict, tool_name: str, file_path: str, permissions_manager, command_pattern: str):
    """Handle the smart quick action by adding appropriate permission rule."""
    category = smart_action['category']
    scope_path = smart_action['scope_path']
    scope_description = smart_action['scope_description']
    display_name = smart_action['display_name']
    
    # Add the permission rule based on the smart action
    if smart_action['permission_scope'] == 'global':
        # Add global permission for this dev tool category
        permissions_manager.add_always_allow(tool_name, dev_tool_category=category, pattern=category)
        print(colored(f"✅ Always allow {display_name} {scope_description}", "green"))
        print(colored(f"   Category: {category}", "white"))
    else:
        # Add path-based permission for this dev tool category
        permissions_manager.add_always_allow(tool_name, dev_tool_category=category, pattern=scope_path)
        print(colored(f"✅ Always allow {display_name} for {scope_description}", "green"))
        print(colored(f"   Pattern: {scope_path}", "white"))

async def handle_always_deny(tool_name: str, file_path: str, permissions_manager, get_user_input_func, args, input_event, input_lock, shared_input):
    """Handle always-deny permission setup."""
    if file_path:
        permissions_manager.add_always_deny(tool_name, file_path)
        print(colored(f"❌ Always deny rule added for {tool_name} on {file_path}", "red"))
    else:
        permissions_manager.add_always_deny(tool_name)
        print(colored(f"❌ Always deny rule added for all {tool_name} operations", "red"))

async def handle_permission_management(get_user_input_func, args, input_event, input_lock, shared_input):
    """Handle interactive permission management."""
    from tools.permissions import permissions
    
    print()
    print(colored("🔧 PERMISSION MANAGEMENT", "yellow", attrs=['bold']))
    print(colored("─" * 50, "yellow"))
    
    # Show current permissions
    result = permissions._list_permissions()
    print(result)
    
    print()
    print(colored("Options:", "cyan"))
    print(colored("  r <id>", "light_blue") + " - Remove permission by ID")
    print(colored("  clear", "red") + " - Clear all permissions")  
    print(colored("  back", "green") + " - Return to permission prompt")
    print()
    
    while True:
        if get_user_input_func:
            user_input, _ = await get_user_input_func(
                args, None, colored("Permission Manager> ", "cyan"),
                input_event=input_event,
                input_lock=input_lock,
                shared_input=shared_input
            )
        else:
            user_input = input(colored("Permission Manager> ", "cyan"))
        
        user_input = user_input.strip().lower()
        
        if user_input == 'back' or user_input == 'exit':
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
            # Refresh the list
            print()
            result = permissions._list_permissions()
            print(result)
        elif user_input == 'list':
            result = permissions._list_permissions()
            print(result)
        else:
            print(colored("❓ Invalid command. Use: r <id>, clear, list, or back", "yellow"))

def show_permission_help(tool_name: str, file_path: str = None):
    """Show help text for permission options."""
    print(colored("\n📖 Permission Options Help:", "blue"))
    print(colored("  Enter/y", "light_green") + " - Allow this action once")
    print(colored("  n", "red") + " - Deny this action once")  
    print(colored("  a", "blue") + " - Always allow (with options for scope)")
    print(colored("  d", "magenta") + " - Always deny this action")
    print(colored("  m", "yellow") + " - Manage all saved permissions")
    print(colored("  ?", "yellow") + " - Show this help")
    print()