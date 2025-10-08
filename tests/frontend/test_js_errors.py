"""
Automated frontend JavaScript testing using Playwright.
Tests the web interface for JS errors, captures screenshots, and saves logs.

Usage:
    python tests/frontend/test_js_errors.py

Requirements:
    pip install playwright requests
    playwright install chromium
"""
import asyncio
import json
import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime

try:
    from playwright.async_api import async_playwright, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Page = None  # Define as None for type hints when not installed
    print("WARNING: Playwright not installed. Install with: pip install playwright")
    print("Then run: playwright install chromium")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("WARNING: requests not installed. Install with: pip install requests")


class FrontendTester:
    """Automated frontend testing with screenshot and log capture."""

    def __init__(self, server_url="http://127.0.0.1:7861", output_dir="tests/frontend/test_results"):
        self.server_url = server_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "screenshots").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        self.server_process = None
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "errors": [],
            "warnings": [],
            "console_logs": [],
            "screenshots": [],
            "test_status": "unknown"
        }

    def check_server_running(self):
        """Check if server is already running."""
        if not REQUESTS_AVAILABLE:
            return False
        try:
            response = requests.get(self.server_url, timeout=2)
            return True
        except:
            return False

    async def start_server(self):
        """Start the CLI-Agent server if not already running."""
        if self.check_server_running():
            print(f"✓ Server already running at {self.server_url}")
            return

        print(f"Starting server at {self.server_url}...")
        project_root = Path(__file__).parent.parent.parent
        self.server_process = subprocess.Popen(
            [sys.executable, "main.py", "--gui"],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to be ready
        for i in range(30):
            if self.check_server_running():
                print("✓ Server started successfully")
                return
            time.sleep(1)

        raise Exception("Server failed to start within 30 seconds")

    async def capture_console_logs(self, page: Page):
        """Set up console log capture for the browser page."""
        def handle_console(msg):
            self.test_results["console_logs"].append({
                "type": msg.type,
                "text": msg.text,
                "location": {
                    "url": msg.location.get("url", "unknown"),
                    "lineNumber": msg.location.get("lineNumber", 0)
                } if msg.location else {}
            })

        def handle_page_error(exc):
            self.test_results["errors"].append({
                "type": "page_error",
                "message": str(exc),
                "stack": getattr(exc, 'stack', 'N/A')
            })

        page.on("console", handle_console)
        page.on("pageerror", handle_page_error)

    async def take_screenshot(self, page: Page, name: str):
        """Take a screenshot and save to archive."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = self.output_dir / "screenshots" / f"{name}_{timestamp}.png"
        await page.screenshot(path=screenshot_path, full_page=True)
        self.test_results["screenshots"].append(str(screenshot_path))
        print(f"✓ Screenshot saved: {screenshot_path}")
        return screenshot_path

    async def check_javascript_errors(self, page: Page):
        """Execute JavaScript in browser to detect errors and missing methods."""
        js_errors = await page.evaluate("""
            () => {
                const errors = [];

                // Check if required globals exist
                const requiredGlobals = [
                    'ContentRenderer',
                    'WebSocketManager',
                    'ChatStateManager',
                    'VisualResponseApp',
                    'AudioManager',
                    'ToolManager',
                    'ChatHistoryManager'
                ];

                requiredGlobals.forEach(name => {
                    if (typeof window[name] === 'undefined') {
                        errors.push({
                            type: 'missing_global',
                            message: `${name} is not defined`
                        });
                    }
                });

                // Check ContentRenderer for renderMessage method
                try {
                    if (typeof ContentRenderer !== 'undefined') {
                        const renderer = new ContentRenderer();
                        const proto = Object.getPrototypeOf(renderer);
                        const methods = Object.getOwnPropertyNames(proto);

                        if (typeof renderer.renderMessage !== 'function') {
                            errors.push({
                                type: 'missing_method',
                                message: 'ContentRenderer.renderMessage is not a function',
                                available_methods: methods
                            });
                        } else {
                            // Method exists - this is good!
                            console.log('✓ ContentRenderer.renderMessage exists');
                        }

                        // Check other essential methods
                        const essentialMethods = ['renderContent', 'clearContent'];
                        essentialMethods.forEach(method => {
                            if (typeof renderer[method] !== 'function') {
                                errors.push({
                                    type: 'missing_method',
                                    message: `ContentRenderer.${method} is not a function`
                                });
                            }
                        });
                    }
                } catch (e) {
                    errors.push({
                        type: 'instantiation_error',
                        message: e.message,
                        stack: e.stack
                    });
                }

                // Check ChatStateManager
                try {
                    if (typeof ChatStateManager !== 'undefined' && typeof ContentRenderer !== 'undefined') {
                        const renderer = new ContentRenderer();
                        const chatState = new ChatStateManager(renderer);
                        console.log('✓ ChatStateManager instantiated successfully');
                    }
                } catch (e) {
                    errors.push({
                        type: 'chat_state_error',
                        message: e.message,
                        stack: e.stack
                    });
                }

                return errors;
            }
        """)

        return js_errors

    async def test_message_interaction(self, page: Page):
        """Test sending a message to trigger any runtime errors."""
        try:
            # Check if input exists
            input_exists = await page.is_visible('#chat-input')
            if not input_exists:
                self.test_results["warnings"].append({
                    "type": "missing_element",
                    "message": "Chat input not found"
                })
                return

            # Type and submit message
            await page.fill('#chat-input', 'test message')
            await page.click('button[type="submit"]')

            # Wait for any errors to manifest
            await page.wait_for_timeout(1000)

            print("✓ Message interaction test completed")

        except Exception as e:
            self.test_results["errors"].append({
                "type": "interaction_error",
                "message": str(e),
                "stack": getattr(e, '__traceback__', 'N/A')
            })

    async def test_page_load(self):
        """Main test: Load page, capture errors, take screenshots."""
        if not PLAYWRIGHT_AVAILABLE:
            print("❌ Cannot run tests - Playwright not installed")
            return

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()

            # Set up console capture
            await self.capture_console_logs(page)

            # Navigate to page
            print(f"Navigating to {self.server_url}...")
            try:
                await page.goto(self.server_url, wait_until="networkidle", timeout=30000)
                print("✓ Page loaded")
            except Exception as e:
                self.test_results["errors"].append({
                    "type": "navigation_error",
                    "message": str(e)
                })
                await browser.close()
                return

            # Wait for app initialization
            await page.wait_for_timeout(2000)

            # Take initial screenshot
            await self.take_screenshot(page, "initial_load")

            # Check for JavaScript errors
            print("Checking for JavaScript errors...")
            js_errors = await self.check_javascript_errors(page)
            self.test_results["errors"].extend(js_errors)

            if js_errors:
                print(f"⚠️  Found {len(js_errors)} JavaScript errors")
                for err in js_errors:
                    print(f"   - [{err['type']}] {err['message']}")
            else:
                print("✓ No JavaScript errors detected")

            # Test message interaction
            print("Testing message interaction...")
            await self.test_message_interaction(page)

            # Take screenshot after interaction
            await self.take_screenshot(page, "after_interaction")

            await browser.close()

    async def run_tests(self):
        """Run all frontend tests."""
        print("=" * 60)
        print("Frontend JavaScript Test Suite")
        print("=" * 60)

        if not PLAYWRIGHT_AVAILABLE:
            print("\n❌ Playwright not installed!")
            print("Install with:")
            print("  pip install playwright")
            print("  playwright install chromium")
            self.test_results["test_status"] = "ERROR"
            self.test_results["errors"].append({
                "type": "dependency_missing",
                "message": "Playwright not installed"
            })
            self.save_results()
            return

        try:
            await self.start_server()
            await self.test_page_load()

            # Determine test status
            if self.test_results["errors"]:
                self.test_results["test_status"] = "FAILED"
                print(f"\n❌ Tests FAILED with {len(self.test_results['errors'])} errors")
            else:
                self.test_results["test_status"] = "PASSED"
                print("\n✓ All tests PASSED")

        except Exception as e:
            self.test_results["test_status"] = "ERROR"
            self.test_results["errors"].append({
                "type": "test_suite_error",
                "message": str(e),
                "stack": getattr(e, '__traceback__', 'N/A')
            })
            print(f"\n❌ Test suite error: {e}")

        finally:
            # Save results
            self.save_results()

            # Cleanup
            if self.server_process:
                print("Stopping server...")
                self.server_process.terminate()
                self.server_process.wait()

    def save_results(self):
        """Save test results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / "logs" / f"test_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)

        print(f"\n📄 Test results saved to: {results_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Status: {self.test_results['test_status']}")
        print(f"Errors: {len(self.test_results['errors'])}")
        print(f"Warnings: {len(self.test_results['warnings'])}")
        print(f"Console Logs: {len(self.test_results['console_logs'])}")
        print(f"Screenshots: {len(self.test_results['screenshots'])}")

        if self.test_results["errors"]:
            print("\nERRORS:")
            for i, error in enumerate(self.test_results["errors"], 1):
                print(f"  {i}. [{error['type']}] {error['message']}")

        if self.test_results["screenshots"]:
            print("\nSCREENSHOTS:")
            for screenshot in self.test_results["screenshots"]:
                print(f"  - {screenshot}")


async def main():
    """Main entry point for the test suite."""
    tester = FrontendTester()
    await tester.run_tests()


if __name__ == "__main__":
    asyncio.run(main())
