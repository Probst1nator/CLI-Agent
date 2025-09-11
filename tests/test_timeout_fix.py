#!/usr/bin/env python3
"""
Test script for the timeout and path detection fixes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.permissions.enhanced_permissions import format_xml_args
from core.permissions.path_detector import PathDetector

def test_timeout_args_parsing():
    """Test that timeout tags are properly handled"""
    
    print("=== TESTING TIMEOUT AND ARGS PARSING ===")
    
    # Test case 1: Command with timeout
    xml_content1 = """<bash><timeout>10</timeout>
ls -d */
</bash>"""
    
    args_display = format_xml_args(xml_content1)
    print(f"Test 1 - XML: {repr(xml_content1)}")
    print(f"Args Display: '{args_display}'")
    print("Expected: 'ls -d */'")
    print(f"✅ Correct: {args_display == 'ls -d */'}")
    print()
    
    # Test case 2: Simple command without timeout
    xml_content2 = """<bash>ls -F</bash>"""
    
    args_display2 = format_xml_args(xml_content2)
    print(f"Test 2 - XML: {repr(xml_content2)}")
    print(f"Args Display: '{args_display2}'")
    print("Expected: 'ls -F'")
    print(f"✅ Correct: {args_display2 == 'ls -F'}")
    print()
    
    # Test case 3: Path detection for glob patterns
    analysis1 = PathDetector.analyze_tool_block("bash", xml_content1)
    print("Test 3 - Path Detection for 'ls -d */':")
    print(f"Paths found: {analysis1['paths']}")
    print(f"Has paths: {analysis1['has_paths']}")
    print(f"Primary path: {analysis1['primary_path']}")
    print(f"✅ Correct (should have no specific file paths): {not analysis1['has_paths']}")
    print()
    
    # Test case 4: Path detection for actual file
    xml_content4 = """<bash>cat /etc/hosts</bash>"""
    analysis4 = PathDetector.analyze_tool_block("bash", xml_content4)
    print("Test 4 - Path Detection for 'cat /etc/hosts':")
    print(f"Paths found: {analysis4['paths']}")
    print(f"Has paths: {analysis4['has_paths']}")
    print(f"Primary path: {analysis4['primary_path']}")
    print(f"✅ Correct (should detect /etc/hosts): {analysis4['has_paths'] and '/etc/hosts' in str(analysis4['primary_path'])}")
    print()

if __name__ == "__main__":
    test_timeout_args_parsing()