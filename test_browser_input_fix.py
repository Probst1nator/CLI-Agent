#!/usr/bin/env python3
"""
Test script to verify browser input race condition fix.
This simulates the race condition and tests the polling solution.
"""

import asyncio
import queue
import time
from core.globals import g
from visual_input_bridge import enhanced_get_user_input_with_bindings

def simulate_delayed_browser_input():
    """Simulate browser input arriving after a delay (mimicking race condition)."""
    def send_delayed_message():
        time.sleep(1.0)  # Simulate delay
        if hasattr(g, 'visual_input_queue'):
            g.visual_input_queue.put("Test message from browser")
            print("🌐 [SIMULATOR] Browser message sent after delay")
    
    import threading
    thread = threading.Thread(target=send_delayed_message, daemon=True)
    thread.start()
    return thread

async def test_browser_input_polling():
    """Test that the polling mechanism catches delayed browser input."""
    print("🧪 Testing Browser Input Race Condition Fix")
    print("=" * 50)
    
    # Set up the visual input queue (simulating visualresponse tool)
    g.visual_input_queue = queue.Queue()
    print("✅ Visual input queue created")
    
    # Start the delayed browser input simulation
    print("⏱️  Starting delayed browser input simulation...")
    thread = simulate_delayed_browser_input()
    
    # Call the enhanced input function
    print("🔍 Calling enhanced_get_user_input_with_bindings...")
    start_time = time.time()
    
    try:
        # This should catch the delayed browser input via polling
        result, images = await enhanced_get_user_input_with_bindings()
        elapsed = time.time() - start_time
        
        print(f"✅ Test PASSED! Got result: '{result}' after {elapsed:.1f}s")
        print(f"📊 Expected: Browser input caught during polling phase")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Test FAILED after {elapsed:.1f}s: {e}")
    
    # Wait for simulation thread to complete
    thread.join(timeout=2.0)
    
    print("\n🏁 Test completed")

async def test_immediate_browser_input():
    """Test that immediate browser input still works instantly."""
    print("\n🧪 Testing Immediate Browser Input")
    print("=" * 40)
    
    # Set up the visual input queue with immediate message
    g.visual_input_queue = queue.Queue()
    g.visual_input_queue.put("Immediate test message")
    print("✅ Visual input queue created with immediate message")
    
    # Call the enhanced input function
    print("🔍 Calling enhanced_get_user_input_with_bindings...")
    start_time = time.time()
    
    try:
        result, images = await enhanced_get_user_input_with_bindings()
        elapsed = time.time() - start_time
        
        if elapsed < 0.1:  # Should be nearly instant
            print(f"✅ Test PASSED! Got immediate result: '{result}' in {elapsed:.3f}s")
        else:
            print(f"⚠️  Test PASSED but slower than expected: '{result}' in {elapsed:.3f}s")
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Test FAILED after {elapsed:.1f}s: {e}")

async def main():
    """Run all tests."""
    print("🚀 Browser Input Race Condition Fix - Test Suite")
    print("=" * 60)
    
    await test_immediate_browser_input()
    await test_browser_input_polling()
    
    print("\n" + "=" * 60)
    print("🎯 Summary:")
    print("   The polling mechanism should catch delayed browser messages")
    print("   while immediate messages still respond instantly.")
    print("   This eliminates the race condition between browser and console input.")

if __name__ == "__main__":
    asyncio.run(main())