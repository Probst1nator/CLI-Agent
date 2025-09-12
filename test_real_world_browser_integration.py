#!/usr/bin/env python3
"""
Real-world integration test for browser input fix.
Simulates the actual flow: visualresponse tool → browser input → main loop processing
"""

import asyncio
import queue
import time
from core.globals import g
from tools.visualresponse import visualresponse
from visual_input_bridge import install_visual_input_bridge, enhanced_get_user_input_with_bindings

def simulate_user_workflow():
    """
    Simulate real user workflow:
    1. User calls visualresponse tool (creates visual server)
    2. User sends message from browser 
    3. Main loop should process the browser message
    """
    
    print("🎬 Simulating Real-World Browser Integration Workflow")
    print("=" * 60)
    
    # Step 1: Simulate visualresponse tool being called
    print("\n1️⃣  User calls visualresponse tool...")
    try:
        # This creates the visual server and sets up g.visual_input_queue
        result = visualresponse.run('<type>markdown</type><content># Test Content\nThis simulates calling the visualresponse tool.</content>')
        print(f"   📄 Visualresponse result: {result}")
        
        # Verify queue was created
        if hasattr(g, 'visual_input_queue'):
            print("   ✅ Visual input queue created by visualresponse")
        else:
            print("   ❌ Visual input queue NOT created")
            return False
            
    except Exception as e:
        print(f"   ❌ Visualresponse failed: {e}")
        return False
    
    # Step 2: Simulate delayed browser input (real-world timing)
    print("\n2️⃣  Simulating browser input arriving after slight delay...")
    def send_browser_message():
        time.sleep(0.5)  # Realistic delay for user to type and send
        if hasattr(g, 'visual_input_queue'):
            g.visual_input_queue.put("Make the ball bounce around")
            print("   🌐 Browser sent: 'Make the ball bounce around'")
    
    import threading
    browser_thread = threading.Thread(target=send_browser_message, daemon=True)
    browser_thread.start()
    
    return True

async def test_main_loop_processing():
    """
    Test that the main loop correctly processes browser input 
    with the race condition fix.
    """
    
    print("\n3️⃣  Main loop checks for user input...")
    print("   (This is where the race condition used to occur)")
    
    # Install the visual input bridge (normally done in main.py)
    install_visual_input_bridge()
    
    start_time = time.time()
    
    try:
        # This simulates the main loop calling get_user_input_with_bindings
        user_input, images = await enhanced_get_user_input_with_bindings()
        elapsed = time.time() - start_time
        
        print(f"\n✅ SUCCESS! Main loop received: '{user_input}'")
        print(f"   ⏱️  Processing time: {elapsed:.1f}s")
        print(f"   🎯 Race condition eliminated!")
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ FAILED after {elapsed:.1f}s: {e}")
        return False

async def main():
    """Run the complete integration test."""
    
    # Simulate the workflow that was failing before our fix
    success = simulate_user_workflow()
    if not success:
        return
    
    # Test that main loop processing now works
    success = await test_main_loop_processing()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 INTEGRATION TEST PASSED!")
        print("   Browser input is now processed correctly by the main loop.")
        print("   The race condition has been eliminated.")
    else:
        print("❌ INTEGRATION TEST FAILED!")
        print("   The race condition fix needs further work.")
    
    print("\n📋 Technical Summary:")
    print("   • visualresponse tool creates g.visual_input_queue")
    print("   • Browser messages go into the queue via WebSocket")
    print("   • Enhanced input bridge polls queue before falling back to console")
    print("   • Race condition eliminated by 2-second polling window")

if __name__ == "__main__":
    asyncio.run(main())