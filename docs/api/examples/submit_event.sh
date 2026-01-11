#!/bin/bash
#
# Example: Submit an event to the Multiverse Dive API using cURL.
#
# This script demonstrates how to:
# 1. Submit an event specification
# 2. Check processing status
# 3. List available products
#
# Usage:
#   export MULTIVERSE_API_KEY="your_api_key_here"
#   ./submit_event.sh
#

set -e

# Configuration
API_BASE_URL="${MULTIVERSE_API_URL:-https://api.multiverse-dive.io/v1}"
API_KEY="${MULTIVERSE_API_KEY:-}"

# Check for API key
if [ -z "$API_KEY" ]; then
    echo "Error: MULTIVERSE_API_KEY environment variable not set"
    echo "Export your API key: export MULTIVERSE_API_KEY='your_key_here'"
    exit 1
fi

# Submit event
echo "Submitting event..."
RESPONSE=$(curl -s -X POST "${API_BASE_URL}/events" \
    -H "X-API-Key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
        "intent": {
            "class": "flood.coastal.storm_surge",
            "source": "explicit"
        },
        "spatial": {
            "type": "Polygon",
            "coordinates": [[
                [-80.3, 25.7],
                [-80.1, 25.7],
                [-80.1, 25.9],
                [-80.3, 25.9],
                [-80.3, 25.7]
            ]],
            "crs": "EPSG:4326"
        },
        "temporal": {
            "start": "2024-09-15T00:00:00Z",
            "end": "2024-09-20T23:59:59Z"
        },
        "priority": "high",
        "constraints": {
            "max_cloud_cover": 0.4,
            "required_data_types": ["sar", "dem"]
        }
    }')

echo "Response: $RESPONSE"

# Extract event ID
EVENT_ID=$(echo "$RESPONSE" | grep -o '"event_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$EVENT_ID" ]; then
    echo "Error: Failed to get event ID"
    exit 1
fi

echo "Event ID: $EVENT_ID"

# Check status
echo ""
echo "Checking status..."
curl -s -X GET "${API_BASE_URL}/events/${EVENT_ID}/status" \
    -H "X-API-Key: ${API_KEY}" | python3 -m json.tool 2>/dev/null || \
    curl -s -X GET "${API_BASE_URL}/events/${EVENT_ID}/status" \
    -H "X-API-Key: ${API_KEY}"

# Get detailed progress
echo ""
echo "Getting progress..."
curl -s -X GET "${API_BASE_URL}/events/${EVENT_ID}/progress" \
    -H "X-API-Key: ${API_KEY}" | python3 -m json.tool 2>/dev/null || \
    curl -s -X GET "${API_BASE_URL}/events/${EVENT_ID}/progress" \
    -H "X-API-Key: ${API_KEY}"

# List products
echo ""
echo "Listing products..."
curl -s -X GET "${API_BASE_URL}/events/${EVENT_ID}/products" \
    -H "X-API-Key: ${API_KEY}" | python3 -m json.tool 2>/dev/null || \
    curl -s -X GET "${API_BASE_URL}/events/${EVENT_ID}/products" \
    -H "X-API-Key: ${API_KEY}"

echo ""
echo "Done!"
echo ""
echo "To download a product:"
echo "  curl -H 'X-API-Key: ${API_KEY}' -O '${API_BASE_URL}/events/${EVENT_ID}/products/PRODUCT_ID/download'"
