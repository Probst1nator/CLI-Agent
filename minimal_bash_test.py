#!/usr/bin/env python3
"""
Minimal test to reproduce bash streaming issue without dependencies
"""

def test_basic_import():
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        
        print("Testing basic import...")
        from tools.bash import bash
        print("✅ Import successful")
        
        print("Testing simple command...")
        result = bash.run('<timeout>3</timeout>\necho "test1"; sleep 1; echo "test2"')
        print(f"Result: {result}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_import()