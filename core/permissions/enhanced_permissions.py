# enhanced_permissions.py
"""
Enhanced permission system with always-allow functionality for CLI-Agent.
"""
import asyncio
import re
import threading
from typing import Dict, Any, Optional
from termcolor import colored
from .permissions_manager import get_permissions_manager

def format_xml_args(xml_content: str) -> str:
    """Format XML content to show the tool arguments concisely."""
    if not xml_content:
        return ""
    
    # Remove outer tags and clean up the content
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
    """
    permissions_manager = get_permissions_manager()
    
    # Define a consistent width for all menu boxes
    BOX_WIDTH = 80
    INNER_WIDTH = BOX_WIDTH - 2
    
    # Check if always denied
    if permissions_manager.is_always_denied(tool_name, file_path):
        def create_denied_operation_box(tool_name: str, action_description: str, file_path: str = None):
            tool_emojis = {"editfile": "✏️", "readfile": "📖", "writefile": "📝", "insertline": "➕", "deletelines": "🗑️", "bash": "🔧", "python": "🐍", "searchweb": "🌐", "showuser": "👁️", "memory": "🧠", "grepsearch": "🔍"}
            tool_emoji = tool_emojis.get(tool_name, "⚙️")
            args_display = format_xml_args(xml_content)
            
            title = "─ PERMISSION DENIED "
            top_border = f"┌{title}{'─' * (INNER_WIDTH - len(title))}┐"
            bottom_border = f"└{'─' * INNER_WIDTH}┘"
            
            print()
            print(colored(top_border, "light_red"))
            print(colored("│", "light_red") + f" ❌ {tool_emoji} {tool_name} operation blocked by saved preferences".ljust(INNER_WIDTH) + colored("│", "light_red"))
            if file_path:
                display_path = ("..." + file_path[-(INNER_WIDTH - 9 - 3):]) if len(file_path) > INNER_WIDTH - 9 else file_path
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
            top_border = f"┌{title}{'─' * (INNER_WIDTH - len(title))}┐"
            bottom_border = f"└{'─' * INNER_WIDTH}┘"

            print()
            print(colored(top_border, "light_green"))
            print(colored("│", "light_green") + f" ✅ {tool_emoji} {tool_name} operation permitted 💡 To manage permissions: /permissions list".ljust(INNER_WIDTH) + colored("│", "light_green"))
            if file_path:
                display_path = ("..." + file_path[-(INNER_WIDTH - 9 - 3):]) if len(file_path) > INNER_WIDTH - 9 else file_path
                print(colored("│", "light_green") + f"    File: {display_path}".ljust(INNER_WIDTH) + colored("│", "light_green"))
            if args_display:
                args_short = (args_display[:(INNER_WIDTH - 9 - 3)] + "...") if len(args_display) > INNER_WIDTH - 9 else args_display
                print(colored("│", "light_green") + f"    Args: {args_short}".ljust(INNER_WIDTH) + colored("│", "light_green"))
            if action_description:
                action_display = (action_description[:(INNER_WIDTH - 11 - 3)] + "...") if len(action_description) > INNER_WIDTH - 11 else action_description
                print(colored("│", "light_green") + f"    Action: {action_display}".ljust(INNER_WIDTH) + colored("│", "light_green"))
            print(colored("│", "light_green") + "".ljust(INNER_WIDTH) + colored("│", "light_green"))
            print(colored(bottom_border, "light_green"))
            print()
        
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
            display_path = ("..." + file_path[-max_len+3:]) if len(file_path) > max_len else file_path
        if action_description:
            max_len = INNER_WIDTH - len(" Action: ")
            action_display = (action_description[:max_len-3] + "...") if len(action_description) > max_len else action_description
        if args_content:
            cleaned_args = re.sub(r'\s+', ' ', args_content).strip()
            max_len = INNER_WIDTH - len(" Args: ")
            args_display = (cleaned_args[:max_len-3] + "...") if len(cleaned_args) > max_len else cleaned_args
            
        def format_line(line_content):
            return f"{vertical}{line_content.ljust(INNER_WIDTH)}{vertical}"

        print(colored(f"{top_left}{horizontal * INNER_WIDTH}{top_right}", "light_yellow"))
        title_text = f" {tool_emoji} TOOL OPERATION PERMISSION"
        print(colored(format_line(colored(title_text, "light_yellow")), "light_yellow"))
        print(colored(f"{vertical}{horizontal * INNER_WIDTH}{vertical}", "light_yellow"))

        def format_content_line(label, content, color='white'):
            line = f" {label}: {colored(content, color)}"
            visual_len = len(re.sub(r'\x1b\[[0-9;]*m', '', line))
            padding = " " * max(0, INNER_WIDTH - visual_len)
            return colored(f"{vertical}{line}{padding}{vertical}", "light_yellow")

        print(format_content_line("Tool", tool_name, 'light_blue'))
        if file_path:
            print(format_content_line("File", display_path, 'white'))
        print(format_content_line("Args", args_display, 'light_magenta'))
        print(format_content_line("Action", action_display, 'blue'))
        print(colored(f"{btm_left}{horizontal * INNER_WIDTH}{btm_right}", "light_yellow"))
    
    create_tool_operation_box(tool_name, action_description, file_path, format_xml_args(xml_content))
    
    def format_option_line(text):
        visual_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
        padding_needed = max(0, INNER_WIDTH - len(visual_text))
        return text + (" " * padding_needed)

    def prepare_suggestion_for_display(text: str, max_len: int) -> str:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if len(cleaned) > max_len:
            return cleaned[:max_len-3] + "..."
        return cleaned

    options_title = "─ Permission Options "
    top_border = f"┌{options_title}{'─' * (INNER_WIDTH - len(options_title))}┐"
    bottom_border = f"└{'─' * INNER_WIDTH}┘"
    
    print(colored(top_border, "blue"))
    
    suggestions = permissions_manager.get_permission_suggestions(tool_name, file_path, command_pattern, has_paths)
    
    options = []
    if hasattr(args, 'voice') and args.voice:
        options.append(f" {colored('Enter', 'green')} or {colored('y', 'green')} (say '{colored('yes', 'green')}') → Confirm this operation")
        options.append(f" {colored('n', 'light_red')} (say '{colored('no', 'light_red')}') → Abort this operation")
        if smart_action:
            options.append(f" {colored(smart_action['quick_key'], 'light_blue')} (say '{colored(smart_action['category'], 'light_blue')}') → Always allow {smart_action['display_name']} for {smart_action['scope_description']}")
        
        option_keys = ['a', 's', 'p', 't']
        for i, suggestion in enumerate(suggestions[:4]):
            processed_suggestion = prepare_suggestion_for_display(suggestion, INNER_WIDTH - 24)
            options.append(f" {colored(option_keys[i], 'blue')} (say '{colored(f'always{i+1}', 'blue')}') → {processed_suggestion}")
        
        options.append(f" {colored('d', 'light_magenta')} (say '{colored('deny', 'light_magenta')}') → Always deny this file")
        options.append(f" {colored('D', 'light_magenta')} (say '{colored('denyall', 'light_magenta')}') → Always deny all {tool_name} operations")
        options.append(f" {colored('m', 'light_yellow')} (say '{colored('manage', 'light_yellow')}') → Manage all saved permissions")
        options.append(f" {colored('?', 'light_yellow')} (say '{colored('help', 'light_yellow')}') → Show detailed help")
    else:
        options.append(f" {colored('Enter', 'green')} or {colored('y', 'green')} → Confirm this operation")
        options.append(f" {colored('n', 'light_red')} → Abort this operation")
        if smart_action:
            options.append(f" {colored(smart_action['quick_key'], 'light_blue')} → Always allow {smart_action['display_name']} for {smart_action['scope_description']}")
        
        option_keys = ['a', 's', 'p', 't']
        for i, suggestion in enumerate(suggestions[:4]):
            processed_suggestion = prepare_suggestion_for_display(suggestion, INNER_WIDTH - 7)
            options.append(f" {colored(option_keys[i], 'blue')} → {processed_suggestion}")

        options.append(f" {colored('d', 'light_magenta')} → Always deny this file")
        options.append(f" {colored('D', 'light_magenta')} → Always deny all {tool_name} operations")
        options.append(f" {colored('m', 'light_yellow')} → Manage all saved permissions")
        options.append(f" {colored('?', 'light_yellow')} → Show detailed help")
    
    for line in options:
        print(colored("│", "blue") + format_option_line(line) + colored("│", "blue"))
    
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
            top_border = f"┌{title}{'─' * (INNER_WIDTH - len(title))}┐"
            bottom_border = f"└{'─' * INNER_WIDTH}┘"

            print(colored(top_border, "red"))
            print(colored("│", "red") + f" ❌ {tool_emoji} {tool_name} operation cancelled by user".ljust(INNER_WIDTH) + colored("│", "red"))
            if file_path:
                display_path = ("..." + file_path[-(INNER_WIDTH - 9 - 3):]) if len(file_path) > INNER_WIDTH - 9 else file_path
                print(colored("│", "red") + f"    File: {display_path}".ljust(INNER_WIDTH) + colored("│", "red"))
            if args_display:
                args_short = (args_display[:(INNER_WIDTH - 9 - 3)] + "...") if len(args_display) > INNER_WIDTH - 9 else args_display
                print(colored("│", "red") + f"    Args: {args_short}".ljust(INNER_WIDTH) + colored("│", "red"))
            if action_description:
                action_display = (action_description[:(INNER_WIDTH - 11 - 3)] + "...") if len(action_description) > INNER_WIDTH - 11 else action_description
                print(colored("│", "red") + f"    Action: {action_display}".ljust(INNER_WIDTH) + colored("│", "red"))
            print(colored(bottom_border, "red"))
            return False
        elif smart_action and (user_input == smart_action['quick_key'] or user_input == smart_action['category']):
            await handle_smart_quick_action(smart_action, tool_name, file_path, permissions_manager, command_pattern)
            return True
        elif user_input in ['a', 's', 'p', 't'] and len(suggestions) > 0:
            option_map = {'a': 0, 's': 1, 'p': 2, 't': 3}
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
            valid_options = "[Enter/y], n, a/s/p/t, d/D, m, ?"
            if smart_action:
                valid_options = f"[Enter/y], n, {smart_action['quick_key']}, a/s/p/t, d/D, m, ?"
            print(colored(f"❓ Invalid option. Enter one of: {valid_options}", "yellow"))

async def handle_direct_always_allow(suggestion: str, tool_name: str, file_path: str, permissions_manager, command_hash: str):
    if "Exact command:" in suggestion and command_hash:
        permissions_manager.add_always_allow(tool_name, command_hash=command_hash)
    elif "Exact file:" in suggestion:
        permissions_manager.add_always_allow(tool_name, file_path)
    elif "Directory:" in suggestion:
        dir_path = suggestion.split("Directory: ")[1]
        permissions_manager.add_always_allow(tool_name, None, dir_path)
    elif "Project:" in suggestion:
        project_path = suggestion.split("Project: ")[1]
        permissions_manager.add_always_allow(tool_name, None, project_path)
    elif "tools (" in suggestion:
        group_display = suggestion.split("All ")[1].split(" tools")[0]
        group_name = group_display.lower().replace(' ', '_')
        if file_path:
            permissions_manager.add_always_allow(tool_name, file_path, group_name=group_name)
        else:
            permissions_manager.add_always_allow(tool_name, group_name=group_name)
    elif "All " in suggestion:
        permissions_manager.add_always_allow(tool_name)
    
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