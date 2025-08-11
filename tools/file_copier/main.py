# main.py
import os
import sys
import argparse
import tkinter as tk
from termcolor import colored
import asyncio

# Clean imports using installed packages
# Note: Run 'pip install -e ../../shared -e ../../core' first

# Local imports
from gui import FileCopierApp
from smart_paster import build_clipboard_content, process_smart_request

try:
    import pyperclip
except ImportError:
    pyperclip = None

def run_cli_mode(directory: str, message: str) -> None:
    """Runs the Smart Paster logic in CLI mode."""
    print(colored("🤖 Smart Paster Activated... Running file discovery...", "cyan"))
    
    # 1. Run the main processing function
    found_rel_paths = asyncio.run(process_smart_request(message, directory))

    # 2. Check for results
    if not found_rel_paths:
        print(colored("No valid files found by path or resolved by AI. No action taken.", "yellow"))
        return

    # 3. Build the final output from relative paths
    found_abs_paths = [os.path.join(directory, p.replace('/', os.sep)) for p in found_rel_paths]
    final_output = build_clipboard_content(found_abs_paths, directory)

    # 4. Copy to clipboard and exit
    print("-" * 20)
    if pyperclip:
        pyperclip.copy(final_output)
        size_kb = len(final_output) / 1024
        print(colored(f"\n✅ Content for {len(found_rel_paths)} file(s) copied to clipboard! ({size_kb:.1f} KB)", "green", attrs=["bold"]))
    else:
        print(colored("\n--- Combined Final Output ---", "cyan"))
        print(final_output)

def main() -> None:
    parser = argparse.ArgumentParser(description="GUI/CLI to select and copy file contents.")
    parser.add_argument("directory", nargs="?", default=".", help="The directory to scan (default: current directory).")
    parser.add_argument("-m", "--message", action="store_true", help="Enable Smart Paster CLI mode. Reads from clipboard.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(colored(f"Error: Directory '{args.directory}' not found.", "red"), file=sys.stderr)
        sys.exit(1)

    if args.message:
        if pyperclip is None:
            print(colored("Error: pyperclip is not installed. Please install it: pip install pyperclip", "red"), file=sys.stderr)
            sys.exit(1)
        try:
            clipboard_content = pyperclip.paste()
            if not clipboard_content or not clipboard_content.strip():
                print(colored("Clipboard is empty.", "yellow"))
                sys.exit(1)
            run_cli_mode(args.directory, clipboard_content)
        except Exception as e:
            print(colored(f"An unexpected error occurred: {e}", "red"), file=sys.stderr)
            sys.exit(1)
    else:
        root = tk.Tk()
        FileCopierApp(root, args.directory)
        root.mainloop()

if __name__ == "__main__":
    main()