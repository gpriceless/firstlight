"""
Playwright configuration for E2E testing.

This configuration sets up Playwright for the FirstLight project,
focusing on testing tile image downloads and validation.
"""

# Playwright pytest configuration is handled via pytest options
# See conftest.py for fixtures

# Default browser settings
BROWSER_OPTIONS = {
    "headless": True,
    "slow_mo": 0,  # Milliseconds between actions (useful for debugging)
}

# Screenshot settings
SCREENSHOT_DIR = "tests/e2e/screenshots"
SCREENSHOT_OPTIONS = {
    "full_page": True,
    "type": "png",
}

# Test server settings
TEST_SERVER_HOST = "127.0.0.1"
TEST_SERVER_PORT = 8765

# Tile validation settings
TILE_VALIDATION = {
    "min_width": 128,
    "min_height": 128,
    "max_blank_percentage": 10,  # Max % of blank/transparent pixels allowed
    "expected_formats": ["png", "tiff", "geotiff"],
}
