#!/usr/bin/env python3
"""
Example: Submit an event to the FirstLight API.

This script demonstrates how to:
1. Submit an event specification
2. Poll for processing status
3. Download products when complete

Requirements:
    pip install requests

Usage:
    export FIRSTLIGHT_API_KEY="your_api_key_here"
    python submit_event.py
"""

import os
import sys
import time
from typing import Dict, Any, Optional

import requests

# Configuration
API_BASE_URL = os.environ.get("FIRSTLIGHT_API_URL", "https://api.firstlight.io/v1")
API_KEY = os.environ.get("FIRSTLIGHT_API_KEY")

# Check for API key
if not API_KEY:
    print("Error: FIRSTLIGHT_API_KEY environment variable not set")
    print("Export your API key: export FIRSTLIGHT_API_KEY='your_key_here'")
    sys.exit(1)


def get_headers() -> Dict[str, str]:
    """Get request headers with authentication."""
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json",
    }


def submit_event(event_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit an event specification for processing.

    Args:
        event_spec: Event specification dictionary

    Returns:
        API response with event_id and status
    """
    response = requests.post(
        f"{API_BASE_URL}/events",
        headers=get_headers(),
        json=event_spec,
    )
    response.raise_for_status()
    return response.json()


def get_status(event_id: str) -> Dict[str, Any]:
    """
    Get processing status for an event.

    Args:
        event_id: Event identifier

    Returns:
        Status response with progress information
    """
    response = requests.get(
        f"{API_BASE_URL}/events/{event_id}/status",
        headers=get_headers(),
    )
    response.raise_for_status()
    return response.json()


def get_progress(event_id: str) -> Dict[str, Any]:
    """
    Get detailed progress for an event.

    Args:
        event_id: Event identifier

    Returns:
        Progress response with phase details
    """
    response = requests.get(
        f"{API_BASE_URL}/events/{event_id}/progress",
        headers=get_headers(),
    )
    response.raise_for_status()
    return response.json()


def wait_for_completion(
    event_id: str,
    poll_interval: int = 10,
    max_wait: int = 3600,
) -> Dict[str, Any]:
    """
    Wait for event processing to complete.

    Args:
        event_id: Event identifier
        poll_interval: Seconds between status checks
        max_wait: Maximum wait time in seconds

    Returns:
        Final status response
    """
    start_time = time.time()

    while time.time() - start_time < max_wait:
        status = get_status(event_id)

        print(f"Status: {status['status']}, Progress: {status['progress']:.0%}")

        if status["status"] == "completed":
            print("Processing completed successfully!")
            return status

        if status["status"] in ("failed", "cancelled"):
            print(f"Processing ended with status: {status['status']}")
            return status

        time.sleep(poll_interval)

    raise TimeoutError(f"Event {event_id} did not complete within {max_wait}s")


def list_products(event_id: str) -> Dict[str, Any]:
    """
    List available products for an event.

    Args:
        event_id: Event identifier

    Returns:
        List of products
    """
    response = requests.get(
        f"{API_BASE_URL}/events/{event_id}/products",
        headers=get_headers(),
    )
    response.raise_for_status()
    return response.json()


def download_product(
    event_id: str,
    product_id: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Download a product file.

    Args:
        event_id: Event identifier
        product_id: Product identifier
        output_path: Optional output file path

    Returns:
        Path to downloaded file
    """
    response = requests.get(
        f"{API_BASE_URL}/events/{event_id}/products/{product_id}/download",
        headers=get_headers(),
        stream=True,
    )
    response.raise_for_status()

    # Get filename from Content-Disposition header or use product_id
    filename = output_path or product_id
    if "Content-Disposition" in response.headers:
        cd = response.headers["Content-Disposition"]
        if "filename=" in cd:
            filename = cd.split("filename=")[1].strip('"')

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded: {filename}")
    return filename


def main():
    """Main example workflow."""
    # Define event specification
    event_spec = {
        "intent": {
            "class": "flood.coastal.storm_surge",
            "source": "explicit",
            "confidence": 0.95,
        },
        "spatial": {
            "type": "Polygon",
            "coordinates": [[
                [-80.3, 25.7],
                [-80.1, 25.7],
                [-80.1, 25.9],
                [-80.3, 25.9],
                [-80.3, 25.7],
            ]],
            "crs": "EPSG:4326",
            "bbox": [-80.3, 25.7, -80.1, 25.9],
        },
        "temporal": {
            "start": "2024-09-15T00:00:00Z",
            "end": "2024-09-20T23:59:59Z",
            "reference_time": "2024-09-17T12:00:00Z",
        },
        "priority": "high",
        "constraints": {
            "max_cloud_cover": 0.4,
            "required_data_types": ["sar", "dem"],
        },
        "metadata": {
            "created_by": "example_script",
            "tags": ["hurricane", "miami", "example"],
        },
    }

    print("Submitting event...")
    result = submit_event(event_spec)
    event_id = result["event_id"]
    print(f"Event submitted: {event_id}")
    print(f"Status: {result['status']}")

    # Wait for completion (comment out for async workflow)
    print("\nWaiting for processing to complete...")
    try:
        final_status = wait_for_completion(event_id, poll_interval=5, max_wait=300)

        if final_status["status"] == "completed":
            # List and download products
            print("\nListing products...")
            products = list_products(event_id)

            for product in products["products"]:
                print(f"  - {product['name']} ({product['format']}, {product['size_bytes']} bytes)")

            # Download first product
            if products["products"]:
                first_product = products["products"][0]
                print(f"\nDownloading {first_product['name']}...")
                download_product(event_id, first_product["id"])

    except TimeoutError as e:
        print(f"Timeout: {e}")
        print("Use webhooks for long-running events")

    print("\nDone!")


if __name__ == "__main__":
    main()
