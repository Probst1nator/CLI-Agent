# Frontend Testing Suite

Automated browser testing for the CLI-Agent web interface using Playwright.

## Features

- ✅ Automated server startup
- ✅ Headless browser testing
- ✅ JavaScript error detection
- ✅ Console log capture
- ✅ Screenshot archiving
- ✅ JSON test reports
- ✅ Zero manual interaction required

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r tests/frontend/requirements-test.txt
   playwright install chromium
   ```

2. **Run tests**:
   ```bash
   python tests/frontend/test_js_errors.py
   ```

## What Gets Tested

- Page loads successfully
- All required JavaScript globals are defined (ContentRenderer, WebSocketManager, etc.)
- ContentRenderer has essential methods (renderMessage, renderContent, clearContent)
- ChatStateManager can be instantiated
- Message interaction works without errors
- No JavaScript console errors

## Output

Test results are saved to:
- **Screenshots**: `tests/frontend/test_results/screenshots/`
- **Logs**: `tests/frontend/test_results/logs/`

Each test run creates:
- Full-page screenshots (before/after interactions)
- JSON report with errors, warnings, console logs
- Timestamped filenames for historical tracking

## Example Output

```
============================================================
Frontend JavaScript Test Suite
============================================================
✓ Server already running at http://127.0.0.1:7861
Navigating to http://127.0.0.1:7861...
✓ Page loaded
✓ Screenshot saved: tests/frontend/test_results/screenshots/initial_load_20251005_143022.png
Checking for JavaScript errors...
✓ No JavaScript errors detected
Testing message interaction...
✓ Message interaction test completed
✓ Screenshot saved: tests/frontend/test_results/screenshots/after_interaction_20251005_143024.png

✓ All tests PASSED

📄 Test results saved to: tests/frontend/test_results/logs/test_results_20251005_143022.json

============================================================
TEST SUMMARY
============================================================
Status: PASSED
Errors: 0
Warnings: 0
Console Logs: 12
Screenshots: 2

SCREENSHOTS:
  - tests/frontend/test_results/screenshots/initial_load_20251005_143022.png
  - tests/frontend/test_results/screenshots/after_interaction_20251005_143024.png
```

## Integration with CI/CD

Add to your GitHub Actions workflow:

```yaml
- name: Run Frontend Tests
  run: |
    pip install -r tests/frontend/requirements-test.txt
    playwright install chromium
    python tests/frontend/test_js_errors.py
```

## Troubleshooting

**Playwright not installed**:
```bash
pip install playwright
playwright install chromium
```

**Server won't start**:
- Check if port 7861 is already in use
- Ensure main.py is accessible from test location

**Tests hang**:
- Increase timeout in test_page_load()
- Check server logs for startup issues
