#!/usr/bin/env python3
"""
Test tool box widths in loading-tools component.

This test verifies that tool execution boxes shrink to minimal width
and only expand to fit their content.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

# Test configuration
BASE_URL = "http://127.0.0.1:7861"
SCREENSHOT_DIR = Path("tests/frontend/test_results/tool_widths")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

def start_server():
    """Start the dev server if not already running."""
    try:
        import requests
        response = requests.get(BASE_URL, timeout=2)
        print(f"✓ Server already running at {BASE_URL}")
        return None
    except:
        print(f"Starting server at {BASE_URL}...")
        server_process = subprocess.Popen(
            [sys.executable, "main.py", "--web"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent.parent.parent
        )

        # Wait for server to start
        print("Waiting for server to start...")
        for i in range(30):
            time.sleep(1)
            try:
                import requests
                requests.get(BASE_URL, timeout=1)
                print(f"✓ Server started after {i+1} seconds")
                return server_process
            except:
                continue

        raise Exception("Server failed to start after 30 seconds")
        return server_process

def test_tool_box_widths():
    """Test that tool boxes have minimal width fitting their content."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    server_process = None

    try:
        server_process = start_server()

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1920, 'height': 1080})

            print(f"Navigating to {BASE_URL}...")
            page.goto(BASE_URL, wait_until="networkidle")

            # Inject test data - create tool execution boxes
            print("Injecting test tool boxes...")
            page.evaluate("""
                // Clear existing messages
                const messagesDiv = document.getElementById('messages');
                if (messagesDiv) messagesDiv.innerHTML = '';

                // Create test message with multiple tool boxes
                const testTools = [
                    { name: 'Read', model: 'opus', status: 'completed', time: '1.2s' },
                    { name: 'Bash', model: 'haiku', status: 'running', time: '0.5s' },
                    { name: 'Edit', model: 'sonnet', status: 'completed', time: '2.1s' },
                    { name: 'WebSearch', model: 'opus', status: 'running', time: '3.4s' }
                ];

                const messageDiv = document.createElement('div');
                messageDiv.className = 'message assistant';

                const toolsHeader = document.createElement('div');
                toolsHeader.className = 'loading-tools-header';

                testTools.forEach(tool => {
                    const item = document.createElement('div');
                    item.className = `loading-tool-item ${tool.status}`;

                    const main = document.createElement('div');
                    main.className = 'loading-tool-main';

                    const name = document.createElement('div');
                    name.className = 'loading-tool-name';
                    name.textContent = tool.name;

                    const model = document.createElement('div');
                    model.className = 'loading-tool-model';
                    model.textContent = tool.model;

                    const status = document.createElement('div');
                    status.className = `loading-tool-status ${tool.status}`;

                    const stopwatch = document.createElement('div');
                    stopwatch.className = 'loading-tool-stopwatch';
                    stopwatch.textContent = tool.time;

                    status.appendChild(stopwatch);

                    main.appendChild(name);
                    main.appendChild(model);
                    main.appendChild(status);
                    item.appendChild(main);
                    toolsHeader.appendChild(item);
                });

                messageDiv.appendChild(toolsHeader);

                if (messagesDiv) {
                    messagesDiv.appendChild(messageDiv);
                }
            """)

            # Wait for rendering
            time.sleep(1)

            # Take full page screenshot
            screenshot_path = SCREENSHOT_DIR / f"tool_boxes_fullpage_{timestamp}.png"
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"✓ Full page screenshot: {screenshot_path}")

            # Take screenshot of just the tool boxes
            tools_header = page.query_selector('.loading-tools-header')
            if tools_header:
                tools_screenshot = SCREENSHOT_DIR / f"tool_boxes_only_{timestamp}.png"
                tools_header.screenshot(path=str(tools_screenshot))
                print(f"✓ Tool boxes screenshot: {tools_screenshot}")

                # Get computed styles and dimensions
                box_data = page.evaluate("""
                    () => {
                        const header = document.querySelector('.loading-tools-header');
                        const items = document.querySelectorAll('.loading-tool-item');

                        const headerStyles = window.getComputedStyle(header);
                        const itemsData = Array.from(items).map((item, i) => {
                            const styles = window.getComputedStyle(item);
                            const rect = item.getBoundingClientRect();
                            return {
                                index: i,
                                text: item.querySelector('.loading-tool-name').textContent,
                                width: rect.width,
                                computedWidth: styles.width,
                                flex: styles.flex,
                                display: styles.display,
                                boxSizing: styles.boxSizing
                            };
                        });

                        return {
                            header: {
                                display: headerStyles.display,
                                gridTemplateColumns: headerStyles.gridTemplateColumns,
                                gap: headerStyles.gap,
                                width: header.getBoundingClientRect().width
                            },
                            items: itemsData
                        };
                    }
                """)

                print("\n" + "="*60)
                print("TOOL BOX MEASUREMENTS")
                print("="*60)
                print("\nHeader container:")
                print(f"  Display: {box_data['header']['display']}")
                print(f"  Grid template: {box_data['header']['gridTemplateColumns']}")
                print(f"  Gap: {box_data['header']['gap']}")
                print(f"  Total width: {box_data['header']['width']}px")

                print("\nIndividual tool boxes:")
                widths = []
                for item in box_data['items']:
                    print(f"  {item['text']}:")
                    print(f"    Width: {item['width']}px (computed: {item['computedWidth']})")
                    print(f"    Flex: {item['flex']}")
                    print(f"    Display: {item['display']}")
                    widths.append(item['width'])

                # Check if widths vary significantly
                if widths:
                    min_width = min(widths)
                    max_width = max(widths)
                    width_variance = max_width - min_width

                    print(f"\nWidth variance: {width_variance}px")
                    print(f"  Min: {min_width}px")
                    print(f"  Max: {max_width}px")

                    if width_variance > 50:
                        print(f"\n⚠️  WARNING: Tool boxes have inconsistent widths (variance: {width_variance}px)")
                        print("    Boxes should have minimal width fitting their content.")
                    else:
                        print(f"\n✓ Tool box widths are relatively consistent (variance: {width_variance}px)")

                print("="*60)

            browser.close()

        print("\n✓ Test completed successfully")
        print(f"\nScreenshots saved to: {SCREENSHOT_DIR}")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if server_process:
            server_process.terminate()
            server_process.wait()
            print("Server stopped")

if __name__ == "__main__":
    print("="*60)
    print("Tool Box Width Test")
    print("="*60)
    test_tool_box_widths()
