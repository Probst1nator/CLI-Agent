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
    
    # Add voice keywords if voice mode is active
    if hasattr(args, 'voice') and args.voice:
        print(colored("│", "cyan") + f" {colored('Enter', 'light_green', attrs=['bold'])} or {colored('y', 'light_green', attrs=['bold'])} (say '{colored('yes', 'light_green')}' or '{colored('execute', 'light_green')}') → Confirm this operation".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('n', 'red', attrs=['bold'])} (say '{colored('no', 'red')}' or '{colored('abort', 'red')}') → Abort this operation".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('a', 'blue', attrs=['bold'])} (say '{colored('always', 'blue')}') → Always permit (save preference)".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('d', 'magenta', attrs=['bold'])} (say '{colored('deny', 'magenta')}') → Always deny (save preference)".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('m', 'yellow', attrs=['bold'])} (say '{colored('manage', 'yellow')}') → Manage all saved permissions".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('?', 'yellow', attrs=['bold'])} (say '{colored('help', 'yellow')}') → Show detailed help".ljust(65) + colored("│", "cyan"))
    else:
        print(colored("│", "cyan") + f" {colored('Enter', 'light_green', attrs=['bold'])} or {colored('y', 'light_green', attrs=['bold'])} → Confirm this operation".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('n', 'red', attrs=['bold'])} → Abort this operation".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('a', 'blue', attrs=['bold'])} → Always permit (save preference)".ljust(65) + colored("│", "cyan"))
        print(colored("│", "cyan") + f" {colored('d', 'magenta', attrs=['bold'])} → Always deny (save preference)".ljust(65) + colored("│", "cyan"))
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
            
            print()
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
            print()
            return False
        elif user_input == 'a' or user_input == 'always':
            await handle_always_allow(tool_name, file_path, permissions_manager, get_user_input_func, args, input_event, input_lock, shared_input, command_hash, command_pattern, has_paths)
            return True
        elif user_input == 'd' or user_input == 'deny':
            await handle_always_deny(tool_name, file_path, permissions_manager, get_user_input_func, args, input_event, input_lock, shared_input)
            return False
        elif user_input == 'm' or user_input == 'manage':
            await handle_permission_management(get_user_input_func, args, input_event, input_lock, shared_input)
            # Continue the loop to ask for permission again after management
            continue
        elif user_input == '?' or user_input == 'help':
            show_permission_help(tool_name, file_path)
        else:
            print(colored("❓ Invalid option. Enter one of: [Enter/y], n, a, d, m, ?", "yellow"))

async def handle_always_allow(tool_name: str, file_path: str, permissions_manager, get_user_input_func, args, input_event, input_lock, shared_input, command_hash: str = None, command_pattern: str = None, has_paths: bool = False):
    """Handle always-allow permission setup."""
    suggestions = permissions_manager.get_permission_suggestions(tool_name, file_path, command_pattern, has_paths)
    
    print(colored("\n📝 Always allow options:", "blue"))
    
    # Voice-friendly keywords for common permission types
    voice_keywords = [
        "exact",      # For exact command matches
        "code",       # For code execution tools  
        "files",      # For file management tools
        "shell",      # For shell/bash operations
        "pattern"     # For pattern-based matches
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        if hasattr(args, 'voice') and args.voice:
            keyword = voice_keywords[i-1] if i-1 < len(voice_keywords) else f"option{i}"
            print(colored(f"  {i}. ", "cyan") + colored(suggestion, "white") + colored(f" (say '{keyword}')", "cyan"))
        else:
            print(colored(f"  {i}. ", "cyan") + colored(suggestion, "white"))
    
    if hasattr(args, 'voice') and args.voice:
        print(colored("  c. Custom pattern", "cyan") + colored(" (say 'custom')", "cyan"))
        print(colored("  x. Cancel", "cyan") + colored(" (say 'cancel')", "cyan"))
    else:
        print(colored("  c. Custom pattern", "cyan"))
        print(colored("  x. Cancel", "cyan"))
    
    while True:
        if get_user_input_func:
            choice, _ = await get_user_input_func(
                args, None, colored("Select option (1-{}, c, x): ".format(len(suggestions)), "cyan"),
                input_event=input_event, input_lock=input_lock, shared_input=shared_input
            )
        else:
            choice = input(colored("Select option (1-{}, c, x): ".format(len(suggestions)), "cyan"))
        
        choice = choice.strip().lower()
        
        if choice == 'x':
            print(colored("Cancelled", "yellow"))
            return
        elif choice == 'c':
            if get_user_input_func:
                custom_pattern, _ = await get_user_input_func(
                    args, None, colored("Enter custom pattern: ", "cyan"),
                    input_event=input_event, input_lock=input_lock, shared_input=shared_input
                )
            else:
                custom_pattern = input(colored("Enter custom pattern: ", "cyan"))
            
            permissions_manager.add_always_allow(tool_name, file_path, custom_pattern.strip())
            print(colored(f"✅ Always allow rule added for pattern: {custom_pattern.strip()}", "green"))
            return
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(suggestions):
                    selected = suggestions[index]
                    if "Exact command:" in selected and command_hash:
                        # Add command-specific permission
                        permissions_manager.add_always_allow(tool_name, command_hash=command_hash)
                        print(colored(f"✅ Always allow rule added for exact command: {command_pattern}", "green"))
                    elif "Exact file:" in selected:
                        permissions_manager.add_always_allow(tool_name, file_path)
                        print(colored(f"✅ Always allow rule added: {selected}", "green"))
                    elif "Directory:" in selected:
                        dir_path = selected.split("Directory: ")[1]
                        permissions_manager.add_always_allow(tool_name, None, dir_path)
                        print(colored(f"✅ Always allow rule added: {selected}", "green"))
                    elif "Project:" in selected:
                        project_path = selected.split("Project: ")[1]
                        permissions_manager.add_always_allow(tool_name, None, project_path)
                        print(colored(f"✅ Always allow rule added: {selected}", "green"))
                    elif "tools (" in selected:
                        # Group-based permission
                        # Extract group name from "All File Editing tools (editfile, writefile, appendfile)"
                        group_display = selected.split("All ")[1].split(" tools")[0]
                        group_name = group_display.lower().replace(' ', '_')
                        
                        if file_path:
                            permissions_manager.add_always_allow(tool_name, file_path, group_name=group_name)
                        else:
                            permissions_manager.add_always_allow(tool_name, group_name=group_name)
                        print(colored(f"✅ Always allow rule added: {selected}", "green"))
                    elif "All " in selected:
                        permissions_manager.add_always_allow(tool_name)
                        print(colored(f"✅ Always allow rule added: {selected}", "green"))
                    
                    return
                else:
                    print(colored("❓ Invalid selection", "yellow"))
            except ValueError:
                print(colored("❓ Invalid selection", "yellow"))

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