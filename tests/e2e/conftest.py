"""
Playwright E2E test fixtures and configuration.

Provides fixtures for:
- Browser and page management
- Test server for tile viewing
- Sample tile generation
- Screenshot capture and validation
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from PIL import Image

# Screenshot and test artifact directories
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
SCREENSHOTS_DIR = ARTIFACTS_DIR / "screenshots"


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    """Create and return the artifacts directory."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


@pytest.fixture(scope="session")
def screenshots_dir(artifacts_dir: Path) -> Path:
    """Create and return the screenshots directory."""
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    return SCREENSHOTS_DIR


@pytest.fixture(scope="session")
def sample_tiles_dir() -> Generator[Path, None, None]:
    """Create a temporary directory with sample test tiles."""
    tiles_dir = ARTIFACTS_DIR / "sample_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Generate sample tiles
    _generate_sample_tiles(tiles_dir)

    yield tiles_dir

    # Cleanup is optional - keep artifacts for inspection
    # shutil.rmtree(tiles_dir)


def _generate_sample_tiles(tiles_dir: Path) -> None:
    """Generate sample test tiles with various characteristics."""
    # Tile 1: Valid flood extent tile (blue gradient representing water depth)
    flood_tile = _create_flood_tile(256, 256)
    flood_tile.save(tiles_dir / "tile_flood_001.png")

    # Tile 2: Valid NDWI tile (water index visualization)
    ndwi_tile = _create_ndwi_tile(256, 256)
    ndwi_tile.save(tiles_dir / "tile_ndwi_002.png")

    # Tile 3: Valid burn severity tile (red-yellow gradient)
    burn_tile = _create_burn_severity_tile(256, 256)
    burn_tile.save(tiles_dir / "tile_burn_003.png")


def _create_flood_tile(width: int, height: int) -> Image.Image:
    """Create a sample flood extent tile with blue gradient."""
    # Create RGBA image
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = img.load()

    # Create a gradient with some "flooded" areas
    for y in range(height):
        for x in range(width):
            # Simulate flood pattern (circular gradient from center)
            cx, cy = width // 2, height // 2
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            max_dist = ((width // 2) ** 2 + (height // 2) ** 2) ** 0.5

            # Areas closer to center are "flooded"
            if dist < max_dist * 0.7:
                intensity = int(255 * (1 - dist / (max_dist * 0.7)))
                pixels[x, y] = (0, 100, 200, intensity)  # Blue with alpha
            else:
                pixels[x, y] = (34, 139, 34, 180)  # Forest green (land)

    return img


def _create_ndwi_tile(width: int, height: int) -> Image.Image:
    """Create a sample NDWI (water index) visualization tile."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = img.load()

    # NDWI typically ranges from -1 to 1, visualized as red to blue
    for y in range(height):
        for x in range(width):
            # Create bands of water detection
            ndwi_value = np.sin(x / 20) * np.cos(y / 20)  # Pattern

            if ndwi_value > 0.3:
                # Water (blue)
                pixels[x, y] = (0, 100, 255, 255)
            elif ndwi_value > 0:
                # Possible water (light blue)
                pixels[x, y] = (100, 180, 255, 200)
            else:
                # Land (green/brown)
                pixels[x, y] = (139, 119, 101, 255)

    return img


def _create_burn_severity_tile(width: int, height: int) -> Image.Image:
    """Create a sample burn severity tile."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = img.load()

    # Create burn severity pattern
    for y in range(height):
        for x in range(width):
            # Simulate burn pattern
            burn_intensity = (np.sin(x / 15) ** 2 + np.cos(y / 15) ** 2) / 2

            if burn_intensity > 0.7:
                # Severe burn (dark red/black)
                pixels[x, y] = (80, 0, 0, 255)
            elif burn_intensity > 0.4:
                # Moderate burn (orange)
                pixels[x, y] = (255, 140, 0, 255)
            elif burn_intensity > 0.2:
                # Light burn (yellow)
                pixels[x, y] = (255, 255, 0, 230)
            else:
                # Unburned (green)
                pixels[x, y] = (34, 139, 34, 255)

    return img


@pytest.fixture
def tile_viewer_html(sample_tiles_dir: Path) -> str:
    """Generate HTML for viewing tiles in the browser."""
    tiles = list(sample_tiles_dir.glob("*.png"))

    tile_items = []
    for tile in tiles:
        tile_items.append(f'''
        <div class="tile-container" data-tile="{tile.name}">
            <h3>{tile.stem}</h3>
            <img src="file://{tile}" alt="{tile.name}" class="tile-image" />
            <div class="tile-info">
                <span class="status">Loading...</span>
            </div>
        </div>
        ''')

    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tile Validation Viewer</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 20px;
                background: #1a1a2e;
                color: #eee;
            }}
            h1 {{
                color: #00d4ff;
            }}
            .tiles-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .tile-container {{
                background: #16213e;
                border-radius: 8px;
                padding: 15px;
                border: 2px solid #0f3460;
            }}
            .tile-container.valid {{
                border-color: #00ff88;
            }}
            .tile-container.invalid {{
                border-color: #ff4444;
            }}
            .tile-image {{
                max-width: 100%;
                height: auto;
                border-radius: 4px;
            }}
            .tile-info {{
                margin-top: 10px;
                font-size: 14px;
            }}
            .status {{
                padding: 4px 8px;
                border-radius: 4px;
                background: #333;
            }}
            .status.valid {{
                background: #00ff8833;
                color: #00ff88;
            }}
            .status.invalid {{
                background: #ff444433;
                color: #ff4444;
            }}
            h3 {{
                margin: 0 0 10px 0;
                color: #00d4ff;
            }}
        </style>
    </head>
    <body>
        <h1>FirstLight - Tile Validation</h1>
        <p>Testing {len(tiles)} tiles for validity and data integrity</p>
        <div class="tiles-grid">
            {"".join(tile_items)}
        </div>
        <script>
            // Validate tiles on load
            document.querySelectorAll('.tile-image').forEach(img => {{
                const container = img.closest('.tile-container');
                const status = container.querySelector('.status');

                img.onload = function() {{
                    // Check image dimensions
                    const valid = img.naturalWidth >= 128 && img.naturalHeight >= 128;

                    if (valid) {{
                        container.classList.add('valid');
                        status.classList.add('valid');
                        status.textContent = `Valid: ${{img.naturalWidth}}x${{img.naturalHeight}}`;
                    }} else {{
                        container.classList.add('invalid');
                        status.classList.add('invalid');
                        status.textContent = `Invalid dimensions: ${{img.naturalWidth}}x${{img.naturalHeight}}`;
                    }}
                }};

                img.onerror = function() {{
                    container.classList.add('invalid');
                    status.classList.add('invalid');
                    status.textContent = 'Failed to load image';
                }};
            }});
        </script>
    </body>
    </html>
    '''


@pytest.fixture
def viewer_html_file(tile_viewer_html: str, artifacts_dir: Path) -> Path:
    """Create a temporary HTML file for tile viewing."""
    html_path = artifacts_dir / "tile_viewer.html"
    html_path.write_text(tile_viewer_html)
    return html_path
