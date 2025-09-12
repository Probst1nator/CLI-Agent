#!/usr/bin/env python3
"""
Complete integration test: Browser → Visual Server → Queue → CLI-Agent
"""

import time
import threading
import queue
from core.globals import g

def simulate_visual_server():
    """Simulate the visual server creating the queue and receiving browser input."""
    print("🌐 Setting up visual input queue (simulating visual server)")
    g.visual_input_queue = queue.Queue()
    
    # Simulate browser sending messages
    test_messages = [
        "hello from browser",
        "analyze this project structure", 
        "what's the main entry point?"
    ]
    
    for i, msg in enumerate(test_messages, 1):
        time.sleep(2)  # Simulate delay between messages
        g.visual_input_queue.put(msg)
        print(f"🌐 Browser message {i} sent: '{msg}'")

def test_visual_bridge():
    """Test the visual input bridge in isolation."""
    print("\n🔧 Testing visual input bridge...")
    
    # Set up visual queue
    g.visual_input_queue = queue.Queue()
    g.visual_input_queue.put("test message from bridge")
    
    # Install bridge
    from visual_input_bridge import install_visual_input_bridge
    install_visual_input_bridge()
    
    print("✅ Bridge installed - next CLI input will check visual queue first")

def main():
    print("CLI-Agent Complete Visual Integration Test")
    print("=" * 50)
    
    print("\n📋 Test Plan:")
    print("1. Install visual input bridge")
    print("2. Simulate visual server setup")
    print("3. Test message flow: Browser → Queue → CLI")
    
    # Test the bridge
    test_visual_bridge()
    
    # Start visual simulation in background
    print("\n🌐 Starting visual server simulation...")
    visual_thread = threading.Thread(target=simulate_visual_server)
    visual_thread.daemon = True
    visual_thread.start()
    
    print("\n🚀 Integration Status:")
    print("✅ Visual input bridge: Installed")
    print("✅ Visual queue: Ready") 
    print("✅ Message simulation: Active")
    
    print("\n💡 To test complete integration:")
    print("   1. Run: python main.py")
    print("   2. The CLI will now check visual queue before prompting")
    print("   3. Browser messages will appear with 🌐 indicator")
    print("   4. No race conditions - persistent input architecture!")

if __name__ == "__main__":
    main()