#!/usr/bin/env python3
"""
Test script to verify both bash and visual response server fixes
"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_bash_functionality():
    """Test that bash commands work without EOF errors"""
    print("🔧 Testing bash functionality...")
    
    try:
        from tools.bash import bash
        
        # Test simple bash command
        result = bash.run("<timeout>5</timeout>\necho 'Bash test successful'")
        
        if "Bash test successful" in result:
            print("✅ Bash functionality working correctly")
            return True
        else:
            print(f"❌ Bash test failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Bash test failed with exception: {e}")
        return False

def test_visual_response_persistence():
    """Test that visual response servers are created with correct persistence settings"""
    print("\n🌐 Testing visual response server persistence...")
    
    try:
        from tools.visualresponse import visualresponse
        import logging
        
        # Enable debug logging to see the registration messages
        logging.basicConfig(level=logging.DEBUG)
        
        # Test HTML content
        test_content = """<type>html</type><content>
<h1>🧪 Visual Response Server Persistence Test</h1>
<div style="background: lightgreen; padding: 20px; border-radius: 10px; margin: 20px;">
    <h2>✅ Server Persistence Test</h2>
    <p><strong>Expected:</strong> This server should remain active after test completion</p>
    <p><strong>Time:</strong> <span id="current-time"></span></p>
    <script>
        document.getElementById('current-time').textContent = new Date().toLocaleString();
        setInterval(() => {
            document.getElementById('current-time').textContent = new Date().toLocaleString();
        }, 1000);
    </script>
</div>
</content>"""
        
        # Execute the visual response tool
        result = visualresponse.run(test_content)
        print(f"Visual response result: {result}")
        
        # Check if server is running
        if (visualresponse._session and 
            visualresponse._session.process and 
            visualresponse._session.process.is_alive()):
            
            port = visualresponse._session.port
            print(f"✅ Visual server is running on port {port}")
            print(f"🌐 Access test content at: http://127.0.0.1:{port}")
            print("🔍 Server should remain active (check logs for cleanup registration)")
            return True
        else:
            print("❌ Visual server is not running or died immediately")
            return False
            
    except Exception as e:
        print(f"❌ Visual response test failed with exception: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 CLI-Agent Fixes Verification Test")
    print("=" * 60)
    
    # Test results
    bash_success = test_bash_functionality()
    visual_success = test_visual_response_persistence()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"  🔧 Bash EOF Fix:           {'✅ PASS' if bash_success else '❌ FAIL'}")
    print(f"  🌐 Visual Server Persist:  {'✅ PASS' if visual_success else '❌ FAIL'}")
    
    if bash_success and visual_success:
        print("\n🎉 ALL TESTS PASSED - Both critical issues have been resolved!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Issues remain to be fixed")
        return 1

if __name__ == "__main__":
    exit(main())