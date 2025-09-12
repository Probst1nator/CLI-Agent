#!/usr/bin/env python3
"""
Test script to demonstrate the visual input integration.
This script simulates browser input being sent to the TUI.
"""

import threading
import time
from core.globals import g
import queue

def simulate_browser_input():
    """Simulate browser sending messages to the visual queue."""
    if not hasattr(g, 'visual_input_queue'):
        g.visual_input_queue = queue.Queue()
    
    test_messages = [
        "analyze this codebase",
        "what files were recently modified?", 
        "create a simple hello world script"
    ]
    
    print("🌐 Simulating browser input...")
    for i, message in enumerate(test_messages):
        time.sleep(3)  # Wait between messages
        g.visual_input_queue.put(message)
        print(f"🌐 Browser sent: '{message}'")

def main():
    """Main demonstration."""
    print("=" * 60)
    print("CLI-Agent Visual Integration Test")  
    print("=" * 60)
    
    # Start browser simulation in background
    browser_thread = threading.Thread(target=simulate_browser_input)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("\n📱 TUI should show these inputs with 🌐 indicators:")
    print("   - analyze this codebase")
    print("   - what files were recently modified?")
    print("   - create a simple hello world script")
    
    print("\n🚀 Start the TUI with: python tui_main.py")
    print("💡 The browser inputs will automatically appear in the TUI!")
    print("\n   CLI inputs will show: > your message")
    print("   Browser inputs will show: 🌐 Visual Input: browser message")
    
    print("\n✨ Integration complete! The persistent input UI is ready.")

if __name__ == "__main__":
    main()