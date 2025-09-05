# enhanced_permissions.py
"""
Enhanced permission system with always-allow functionality for CLI-Agent.
"""
import asyncio
import threading
from typing import Dict, Any, Optional
from termcolor import colored
from .permissions_manager import get_permissions_manager

async def enhanced_permission_prompt(
    args, 
    tool_name: str,
    action_description: str,
    file_path: str = None,
    command_hash: str = None,
    command_pattern: str = None,
    has_paths: bool = False,
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
        print(colored(f"❌ {tool_name} action denied by saved permissions", "red"))
        return False
    
    # Check if always allowed (with command hash support)
    if permissions_manager.is_always_allowed(tool_name, file_path, command_hash):
        if command_hash and not file_path:
            print(colored(f"✅ {tool_name} command permitted by saved permissions", "green"))
        else:
            print(colored(f"✅ {tool_name} action permitted by saved permissions", "green"))
        return True
    
    # Build permission prompt
    file_info = f" on {file_path}" if file_path else ""
    prompt_text = colored(f"🔐 {tool_name}{file_info}: ", "yellow") + colored(action_description, "cyan")
    
    # Build options text
    options = []
    options.append(colored("Enter", "light_green") + colored(" to confirm", "cyan"))
    options.append(colored("n", "red") + colored(" to abort", "cyan"))
    options.append(colored("a", "blue") + colored(" to always permit", "cyan"))
    options.append(colored("d", "magenta") + colored(" to always deny", "cyan"))
    options.append(colored("?", "yellow") + colored(" for options", "cyan"))
    
    options_text = "(" + ", ".join(options) + ")"
    
    print(prompt_text)
    print(colored(options_text, "cyan"))
    
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
            print(colored("❌ Action cancelled", "red"))
            return False
        elif user_input == 'a' or user_input == 'always':
            await handle_always_allow(tool_name, file_path, permissions_manager, get_user_input_func, args, input_event, input_lock, shared_input, command_hash, command_pattern, has_paths)
            return True
        elif user_input == 'd' or user_input == 'deny':
            await handle_always_deny(tool_name, file_path, permissions_manager, get_user_input_func, args, input_event, input_lock, shared_input)
            return False
        elif user_input == '?' or user_input == 'help':
            show_permission_help(tool_name, file_path)
        else:
            print(colored("❓ Invalid option. Enter one of: [Enter/y], n, a, d, ?", "yellow"))

async def handle_always_allow(tool_name: str, file_path: str, permissions_manager, get_user_input_func, args, input_event, input_lock, shared_input, command_hash: str = None, command_pattern: str = None, has_paths: bool = False):
    """Handle always-allow permission setup."""
    suggestions = permissions_manager.get_permission_suggestions(tool_name, file_path, command_pattern, has_paths)
    
    print(colored("\n📝 Always allow options:", "blue"))
    for i, suggestion in enumerate(suggestions, 1):
        print(colored(f"  {i}. ", "cyan") + colored(suggestion, "white"))
    
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

def show_permission_help(tool_name: str, file_path: str = None):
    """Show help text for permission options."""
    print(colored("\n📖 Permission Options Help:", "blue"))
    print(colored("  Enter/y", "light_green") + " - Allow this action once")
    print(colored("  n", "red") + " - Deny this action once")  
    print(colored("  a", "blue") + " - Always allow (with options for scope)")
    print(colored("  d", "magenta") + " - Always deny this action")
    print(colored("  ?", "yellow") + " - Show this help")
    print()