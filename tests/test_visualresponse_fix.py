#!/usr/bin/env python3
"""
Quick test script to verify visualresponse server persistence fix.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_visualresponse_persistence():
    """Test that visual response server stays alive after tool execution."""
    print("Testing visual response server persistence...")

    # Import the fixed visualresponse tool
    from tools.visualresponse import visualresponse

    # Test content for Mermaid diagram
    test_content = """<type>html</type><content>
<h1>Test Visual Response</h1>
<p>This is a test to verify the server stays alive after content generation.</p>
<div style="background: lightblue; padding: 20px; border-radius: 10px;">
    <h2>Server Persistence Test</h2>
    <p><strong>Expected Result:</strong> This server should remain active after this test completes.</p>
</div>
</content>"""

    # Execute the tool
    result = visualresponse.run(test_content)
    print(f"Tool result: {result}")

    # Check if session exists and is alive
    if visualresponse._session and visualresponse._session.process:
        if visualresponse._session.process.is_alive():
            port = visualresponse._session.port
            print(f"✅ SUCCESS: Visual server is running on port {port}")
            print(f"🌐 Access the content at: http://127.0.0.1:{port}")
            print("🔍 Server should remain active (not auto-cleanup)")
            return True
        else:
            print("❌ FAILED: Visual server process is not alive")
            return False
    else:
        print("❌ FAILED: No visual server session found")
        return False

if __name__ == "__main__":
    test_visualresponse_persistence()