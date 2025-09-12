#!/usr/bin/env python3
"""
Demo the colorized bash command header
"""

def demo_colored_header():
    """Show what the colorized header will look like"""
    import getpass
    import socket
    import os
    from termcolor import colored
    
    print("🎨 Colorized Bash Command Header Demo")
    print("=" * 50)
    
    # Get system info
    current_user = getpass.getuser()
    hostname = socket.gethostname()
    current_dir = os.getcwd()
    test_command = "for i in $(seq 1 10); do echo $i; sleep 1; done"
    
    # Create colorized header (same as in bash.py)
    user_colored = colored(current_user, 'green', attrs=['bold'])
    host_colored = colored(hostname, 'blue', attrs=['bold'])
    dir_colored = colored(current_dir, 'yellow')
    prompt_colored = colored('$', 'white', attrs=['bold'])
    command_colored = colored(test_command, 'white', attrs=['bold'])
    
    # Display the colorized header
    header = f"[{user_colored}@{host_colored}:{dir_colored}]{prompt_colored} {command_colored}"
    print(f"💻 {header}")
    
    # Show sample output
    print("⚙️  1")
    print("⚙️  2")
    print("⚙️  3")
    print("...")
    
    print("\n🎨 COLOR SCHEME:")
    print(f"  👤 User: {colored('GREEN + BOLD', 'green', attrs=['bold'])}")
    print(f"  🖥️  Hostname: {colored('BLUE + BOLD', 'blue', attrs=['bold'])}")
    print(f"  📁 Directory: {colored('YELLOW', 'yellow')}")
    print(f"  💲 Prompt: {colored('WHITE + BOLD', 'white', attrs=['bold'])}")
    print(f"  ⚡ Command: {colored('WHITE + BOLD', 'white', attrs=['bold'])}")
    
    print("\n✅ RESULT:")
    print("Professional, readable terminal-style header with clear visual hierarchy!")

if __name__ == "__main__":
    demo_colored_header()