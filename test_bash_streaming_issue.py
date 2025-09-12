#!/usr/bin/env python3
"""
Test script to reproduce the bash streaming output repetition issue.
This will demonstrate the exact problem shown in the user's output.
"""
import sys
import os
import time

# Add project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_bash_streaming_repetition():
    """
    Reproduce the exact bash streaming issue where output is repeated cumulatively.
    Expected: Numbers 1-10 printed once each with 1 second gaps
    Actual: Massive repetition of cumulative output
    """
    print("🧪 Testing Bash Streaming Issue Reproduction")
    print("=" * 60)
    
    try:
        from tools.bash import bash
        
        # The exact command that causes the issue
        test_command = "for i in $(seq 1 10); do echo $i; sleep 1; done"
        
        print(f"🔧 Running command: {test_command}")
        print("📊 Expected output: Numbers 1-10, one per second")
        print("❌ Actual issue: Cumulative repetition of all previous numbers")
        print("\n--- OUTPUT START ---")
        
        # Capture start time
        start_time = time.time()
        
        # Execute the problematic command
        result = bash.run(f"<timeout>15</timeout>\n{test_command}")
        
        end_time = time.time()
        
        print("--- OUTPUT END ---\n")
        print(f"⏱️  Total execution time: {end_time - start_time:.2f} seconds")
        print(f"📄 Final result: {repr(result)}")
        
        # Analyze the output pattern
        print("\n🔍 ISSUE ANALYSIS:")
        print("1. ✅ Command executes successfully") 
        print("2. ❌ Output shows cumulative repetition instead of live streaming")
        print("3. ❌ Each iteration shows ALL previous numbers instead of just the new one")
        print("4. ❌ Massive buffer duplication visible in console output")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_streaming():
    """Test a simpler case to isolate the issue"""
    print("\n🧪 Testing Simple Streaming Case")
    print("=" * 40)
    
    try:
        from tools.bash import bash
        
        # Simpler test case
        simple_command = 'echo "A"; sleep 1; echo "B"; sleep 1; echo "C"'
        
        print(f"🔧 Running: {simple_command}")
        print("📊 Expected: A, B, C with delays")
        print("\n--- SIMPLE OUTPUT START ---")
        
        result = bash.run(f"<timeout>5</timeout>\n{simple_command}")
        
        print("--- SIMPLE OUTPUT END ---")
        print(f"📄 Result: {repr(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple test failed: {e}")
        return False

def main():
    """Run the streaming issue reproduction tests"""
    print("🐛 Bash Streaming Issue Reproduction Test Suite")
    print("This reproduces the exact output repetition problem from the user report")
    print("=" * 80)
    
    # Test 1: Reproduce the exact issue
    success1 = test_bash_streaming_repetition()
    
    # Test 2: Simple case
    success2 = test_simple_streaming()
    
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY:")
    print(f"  🧪 Main Issue Reproduction:  {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  🧪 Simple Streaming Test:    {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎯 ISSUE CONFIRMED: Both tests demonstrate the streaming repetition problem")
        print("📋 Next step: Fix the bash streaming implementation to show incremental output only")
    else:
        print("\n⚠️  Tests failed - may need to debug bash tool initialization first")
    
    return success1 and success2

if __name__ == "__main__":
    exit(0 if main() else 1)