"""
Playwright E2E tests for tile image validation.

Tests tile downloads and validates that images:
- Load correctly in the browser
- Have valid dimensions (not 0x0 or too small)
- Are not blank or corrupted
- Display properly in the tile viewer
"""

import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import Page, expect


class TestTileImageValidation:
    """E2E tests for validating tile images using Playwright."""

    @pytest.mark.e2e
    def test_sample_tiles_exist(self, sample_tiles_dir: Path):
        """Verify sample tiles were generated correctly."""
        tiles = list(sample_tiles_dir.glob("*.png"))
        assert len(tiles) >= 3, f"Expected at least 3 sample tiles, found {len(tiles)}"

        for tile in tiles:
            assert tile.exists(), f"Tile file does not exist: {tile}"
            assert tile.stat().st_size > 0, f"Tile file is empty: {tile}"

    @pytest.mark.e2e
    def test_tile_dimensions_valid(self, sample_tiles_dir: Path):
        """Verify tile images have valid dimensions."""
        tiles = list(sample_tiles_dir.glob("*.png"))

        for tile_path in tiles:
            img = Image.open(tile_path)
            width, height = img.size

            assert width >= 128, f"Tile {tile_path.name} width too small: {width}"
            assert height >= 128, f"Tile {tile_path.name} height too small: {height}"
            assert width <= 4096, f"Tile {tile_path.name} width too large: {width}"
            assert height <= 4096, f"Tile {tile_path.name} height too large: {height}"

    @pytest.mark.e2e
    def test_tiles_not_blank(self, sample_tiles_dir: Path):
        """Verify tile images are not blank (all one color or transparent)."""
        tiles = list(sample_tiles_dir.glob("*.png"))

        for tile_path in tiles:
            img = Image.open(tile_path)

            # Convert to numpy array for analysis
            img_array = np.array(img)

            # Check for variation in pixel values
            if img_array.ndim == 3:
                # Color image - check each channel
                std_dev = np.std(img_array)
                assert std_dev > 1, f"Tile {tile_path.name} appears to be blank (std dev: {std_dev:.2f})"
            else:
                # Grayscale
                std_dev = np.std(img_array)
                assert std_dev > 1, f"Tile {tile_path.name} appears to be blank (std dev: {std_dev:.2f})"

    @pytest.mark.e2e
    def test_tile_viewer_loads(self, page: Page, viewer_html_file: Path, screenshots_dir: Path):
        """Test that the tile viewer HTML page loads correctly."""
        # Navigate to the viewer
        page.goto(f"file://{viewer_html_file}")

        # Wait for page to load
        page.wait_for_load_state("networkidle")

        # Take a screenshot
        screenshot_path = screenshots_dir / "tile_viewer_loaded.png"
        page.screenshot(path=str(screenshot_path), full_page=True)

        # Verify the page loaded
        expect(page.locator("h1")).to_contain_text("Tile Validation")

        # Verify tile containers exist
        tile_containers = page.locator(".tile-container")
        assert tile_containers.count() >= 3, "Expected at least 3 tile containers"

    @pytest.mark.e2e
    def test_tile_images_display_in_browser(
        self, page: Page, viewer_html_file: Path, screenshots_dir: Path
    ):
        """Test that tile images display correctly in the browser."""
        page.goto(f"file://{viewer_html_file}")
        page.wait_for_load_state("networkidle")

        # Wait for images to load (JavaScript validation runs on load)
        page.wait_for_timeout(1000)

        # Check each tile image
        tile_images = page.locator(".tile-image")
        count = tile_images.count()

        for i in range(count):
            img = tile_images.nth(i)

            # Verify image is visible
            expect(img).to_be_visible()

            # Get natural dimensions via JavaScript
            dimensions = page.evaluate(
                """(img) => ({
                    width: img.naturalWidth,
                    height: img.naturalHeight,
                    complete: img.complete
                })""",
                img.element_handle(),
            )

            assert dimensions["complete"], f"Image {i} did not finish loading"
            assert dimensions["width"] >= 128, f"Image {i} width too small: {dimensions['width']}"
            assert dimensions["height"] >= 128, f"Image {i} height too small: {dimensions['height']}"

        # Take screenshot of all validated tiles
        screenshot_path = screenshots_dir / "tiles_validated.png"
        page.screenshot(path=str(screenshot_path), full_page=True)

    @pytest.mark.e2e
    def test_tile_validation_status(
        self, page: Page, viewer_html_file: Path, screenshots_dir: Path
    ):
        """Test that tiles show valid status after browser validation."""
        page.goto(f"file://{viewer_html_file}")
        page.wait_for_load_state("networkidle")

        # Wait for JavaScript validation to complete
        page.wait_for_timeout(1500)

        # Check that all tiles are marked as valid
        valid_containers = page.locator(".tile-container.valid")
        invalid_containers = page.locator(".tile-container.invalid")

        valid_count = valid_containers.count()
        invalid_count = invalid_containers.count()

        # Take screenshot before assertions
        screenshot_path = screenshots_dir / "tile_status_check.png"
        page.screenshot(path=str(screenshot_path), full_page=True)

        assert invalid_count == 0, f"Found {invalid_count} invalid tiles"
        assert valid_count >= 3, f"Expected at least 3 valid tiles, found {valid_count}"

        # Verify status text shows dimensions
        status_elements = page.locator(".status.valid")
        for i in range(status_elements.count()):
            status_text = status_elements.nth(i).text_content()
            assert "Valid:" in status_text, f"Status {i} missing 'Valid:' prefix"
            # Check for dimension pattern like "256x256"
            assert re.search(r"\d+x\d+", status_text), f"Status {i} missing dimensions"


class TestTileScreenshots:
    """Tests focused on screenshot capture and comparison."""

    @pytest.mark.e2e
    def test_individual_tile_screenshots(
        self, page: Page, sample_tiles_dir: Path, screenshots_dir: Path
    ):
        """Take individual screenshots of each tile for inspection."""
        tiles = list(sample_tiles_dir.glob("*.png"))

        for tile_path in tiles:
            # Create a simple HTML page for this single tile
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        margin: 0;
                        padding: 20px;
                        background: #222;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }}
                    h2 {{ color: #fff; font-family: Arial; }}
                    img {{
                        border: 2px solid #00d4ff;
                        border-radius: 4px;
                    }}
                </style>
            </head>
            <body>
                <h2>{tile_path.stem}</h2>
                <img src="file://{tile_path}" id="tile" />
            </body>
            </html>
            """

            # Write temporary HTML
            temp_html = sample_tiles_dir / f"view_{tile_path.stem}.html"
            temp_html.write_text(html_content)

            # Navigate and screenshot
            page.goto(f"file://{temp_html}")
            page.wait_for_load_state("networkidle")

            # Wait for image to load
            page.wait_for_selector("#tile")
            page.wait_for_timeout(500)

            # Take screenshot
            screenshot_path = screenshots_dir / f"tile_{tile_path.stem}.png"
            page.screenshot(path=str(screenshot_path))

            # Cleanup temp HTML
            temp_html.unlink()

            # Verify screenshot was created
            assert screenshot_path.exists(), f"Screenshot not created: {screenshot_path}"
            assert screenshot_path.stat().st_size > 0, f"Screenshot is empty: {screenshot_path}"

    @pytest.mark.e2e
    def test_screenshot_not_blank(self, page: Page, viewer_html_file: Path, screenshots_dir: Path):
        """Verify screenshots are not blank images."""
        page.goto(f"file://{viewer_html_file}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1000)

        screenshot_path = screenshots_dir / "blank_check.png"
        page.screenshot(path=str(screenshot_path), full_page=True)

        # Load and analyze the screenshot
        img = Image.open(screenshot_path)
        img_array = np.array(img)

        # Check that screenshot has variation (not solid color)
        std_dev = np.std(img_array)
        assert std_dev > 10, f"Screenshot appears blank (std dev: {std_dev:.2f})"

        # Check that it has reasonable dimensions
        assert img.width >= 800, f"Screenshot too narrow: {img.width}"
        assert img.height >= 400, f"Screenshot too short: {img.height}"
