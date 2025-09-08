#!/usr/bin/env python3
"""
CLI-Agent Unified Test Runner

A single command to run all tests with clear, actionable output.

Usage:
    python test.py                  # Run core tests (recommended)
    python test.py --quick         # Essential tests only (< 30s)
    python test.py --full          # All tests including autonomous
    python test.py --autonomous    # Autonomous tests only
    python test.py --unit          # Unit tests only  
    python test.py --verbose       # Detailed output
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from termcolor import colored

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from run_tests import TestRunner, TestLevel


def print_banner():
    """Print test runner banner"""
    print()
    print(colored("=" * 60, "cyan"))
    print(colored("🧪 CLI-Agent Unified Test Suite", "cyan", attrs=["bold"]))
    print(colored("=" * 60, "cyan"))
    print()


def print_usage_guide():
    """Print quick usage guide"""
    print(colored("📖 Quick Usage Guide:", "yellow"))
    print("  python test.py           # Core functionality tests")
    print("  python test.py --quick   # Fast essential tests only")  
    print("  python test.py --full    # Complete test suite")
    print("  python test.py --autonomous # End-to-end autonomous tests")
    print("  python test.py --verbose # Show detailed output")
    print()


async def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(
        description="CLI-Agent Unified Test Runner - One command for all testing needs"
    )
    
    # Test modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", "-q", action="store_true",
                           help="Quick essential tests only (< 30s)")
    mode_group.add_argument("--full", "-f", action="store_true", 
                           help="Full test suite including autonomous tests")
    mode_group.add_argument("--autonomous", "-a", action="store_true",
                           help="Autonomous end-to-end tests only")
    mode_group.add_argument("--unit", "-u", action="store_true",
                           help="Unit tests only")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output with detailed information")
    parser.add_argument("--help-usage", action="store_true",
                       help="Show usage guide and exit")
    
    args = parser.parse_args()
    
    if args.help_usage:
        print_usage_guide()
        return 0
    
    print_banner()
    
    # Determine test level based on arguments
    if args.quick:
        level = TestLevel.TOOLS
        test_name = "Quick Essential Tests"
        quick = True
    elif args.full:
        level = TestLevel.UNIT
        test_name = "Full Test Suite"
        quick = False
    elif args.autonomous:
        level = TestLevel.INTEGRATION
        test_name = "Autonomous Framework Tests (Mock)"
        quick = False
    elif args.unit:
        level = TestLevel.UNIT
        test_name = "Unit Tests"
        quick = False
    else:
        # Default: core functionality tests
        level = TestLevel.CORE
        test_name = "Core Functionality Tests"
        quick = False
    
    print(colored(f"🎯 Running: {test_name}", "green"))
    print(colored(f"📊 Test Level: {level.name}", "blue"))
    if quick:
        print(colored("⚡ Quick mode: Essential tests only", "yellow"))
    print()
    
    # Create and run test runner
    start_time = time.time()
    runner = TestRunner(max_level=level, verbose=args.verbose, quick=quick)
    
    try:
        summary = await runner.run_all_tests()
    except KeyboardInterrupt:
        print(colored("\n⚠️  Tests interrupted by user", "yellow"))
        return 1
    except Exception as e:
        print(colored(f"❌ Test runner error: {e}", "red"))
        if args.verbose:
            import traceback
            print(traceback.format_exc())
        return 1
    
    total_time = time.time() - start_time
    
    # Print final summary with actionable information
    print()
    print(colored("=" * 60, "cyan"))
    print(colored("📋 FINAL TEST SUMMARY", "cyan", attrs=["bold"]))
    print(colored("=" * 60, "cyan"))
    
    success_rate = summary['success_rate']
    if success_rate >= 95:
        status_color = "green"
        status_icon = "✅"
        recommendation = "All systems ready for use!"
    elif success_rate >= 80:
        status_color = "yellow" 
        status_icon = "⚠️"
        recommendation = "Most functionality working. Review failed tests."
    else:
        status_color = "red"
        status_icon = "❌"
        recommendation = "Critical issues detected. Fix before using."
    
    print(f"{status_icon} {colored(f'Success Rate: {success_rate:.1f}%', status_color, attrs=['bold'])}")
    print(f"📊 Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
    print(f"⏱️  Total Time: {total_time:.1f}s")
    print(f"💡 {colored(recommendation, status_color)}")
    
    # Show next steps
    print()
    print(colored("🚀 Next Steps:", "blue", attrs=["bold"]))
    
    if summary['failed_tests'] == 0:
        print("  • All tests passed! CLI-Agent is ready to use.")
        print("  • Try: python main.py --help")
    else:
        print(f"  • Fix {summary['failed_tests']} failing test(s) above")
        print("  • Run 'python test.py --verbose' for detailed error info")
        print("  • Check dependencies: pip install -r requirements.txt")
    
    if not args.full and summary['failed_tests'] == 0:
        print("  • Run 'python test.py --full' for complete validation")
    
    print()
    
    # Exit with appropriate code
    return 0 if summary['failed_tests'] == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(colored("\n⚠️  Tests interrupted", "yellow"))
        sys.exit(1)