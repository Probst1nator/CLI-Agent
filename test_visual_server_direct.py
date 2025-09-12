#!/usr/bin/env python3
"""
Direct test of the visual server subprocess to capture actual errors
"""

import sys
import traceback

# Add current directory to path to import the visual server
sys.path.insert(0, '~/repos/CLI-Agent')

def test_visual_server_import():
    """Test if we can import the visual server components"""
    print("=== Testing Visual Server Import ===")
    
    try:
        print("1. Testing basic imports...")
        print("   ✓ Basic imports successful")
        
        print("2. Testing aiohttp import...")
        import aiohttp
        print(f"   ✓ aiohttp version {aiohttp.__version__} imported successfully")
        
        print("3. Testing visual server import...")
        from tools.visualresponse import ASSETS_DIR
        print("   ✓ Visual server function imported successfully")
        
        print(f"4. Checking assets directory: {ASSETS_DIR}")
        if ASSETS_DIR.exists():
            print("   ✓ Assets directory exists")
            files = list(ASSETS_DIR.iterdir())
            print(f"   Files: {[f.name for f in files]}")
        else:
            print("   ❌ Assets directory missing!")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_server_function_direct():
    """Test the server function directly"""
    print("\n=== Testing Server Function Direct Call ===")
    
    try:
        # Create mock queues
        from multiprocessing import Queue
        output_queue = Queue()
        input_queue = Queue()
        port = 7860
        
        print("1. Created mock queues")
        
        # Try to call the function directly
        print("2. Calling _server_process_main_loop directly...")
        from tools.visualresponse import _server_process_main_loop
        _server_process_main_loop(output_queue, input_queue, port)
        
    except Exception as e:
        print(f"   ❌ Direct call failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Visual Server Direct Test")
    print("=" * 50)
    
    # Test imports first
    import_success = test_visual_server_import()
    
    if import_success:
        # Test direct function call
        test_server_function_direct()
    else:
        print("Skipping function test due to import failures")
        
    print("\n" + "=" * 50)
    print("Test completed")