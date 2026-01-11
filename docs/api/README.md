# Multiverse Dive API Documentation

## Overview

The Multiverse Dive API provides RESTful endpoints for the geospatial event intelligence platform. It enables you to:

- Submit event specifications for automated analysis
- Monitor processing status and progress
- Download analysis products (raster data, reports)
- Browse available data sources and algorithms
- Register webhooks for real-time notifications

**Base URL:** `https://api.multiverse-dive.io/v1`

**OpenAPI Spec:** Available at `/openapi.json`

**Interactive Docs:**
- Swagger UI: `/docs`
- ReDoc: `/redoc`

---

## Authentication

All API endpoints (except `/health`) require authentication via API key.

### API Key Header

Include your API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your_api_key_here" https://api.multiverse-dive.io/v1/events
```

### Obtaining an API Key

Contact support@multiverse-dive.io to request API access.

### Rate Limiting

API requests are rate-limited per API key:

| Tier     | Requests/Minute |
|----------|-----------------|
| Free     | 10              |
| Standard | 100             |
| Premium  | 1000            |

When rate limited, you'll receive a `429 Too Many Requests` response with a `Retry-After` header.

---

## Quick Start

### 1. Submit an Event

```bash
curl -X POST https://api.multiverse-dive.io/v1/events \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "class": "flood.coastal.storm_surge",
      "source": "explicit"
    },
    "spatial": {
      "type": "Polygon",
      "coordinates": [[
        [-80.3, 25.7], [-80.1, 25.7],
        [-80.1, 25.9], [-80.3, 25.9],
        [-80.3, 25.7]
      ]],
      "crs": "EPSG:4326"
    },
    "temporal": {
      "start": "2024-09-15T00:00:00Z",
      "end": "2024-09-20T23:59:59Z"
    },
    "priority": "high"
  }'
```

**Response:**
```json
{
  "event_id": "evt_abc123def456",
  "status": "accepted",
  "message": "Event submitted successfully",
  "links": {
    "self": "/events/evt_abc123def456",
    "status": "/events/evt_abc123def456/status",
    "products": "/events/evt_abc123def456/products"
  }
}
```

### 2. Monitor Progress

```bash
curl -H "X-API-Key: your_api_key" \
  https://api.multiverse-dive.io/v1/events/evt_abc123def456/progress
```

**Response:**
```json
{
  "event_id": "evt_abc123def456",
  "status": "processing",
  "progress": 0.65,
  "phases": {
    "discovery": {"status": "completed", "progress": 1.0},
    "ingestion": {"status": "completed", "progress": 1.0},
    "analysis": {"status": "in_progress", "progress": 0.5},
    "quality": {"status": "pending", "progress": 0.0},
    "reporting": {"status": "pending", "progress": 0.0}
  },
  "estimated_completion": "2024-09-17T14:30:00Z"
}
```

### 3. Download Products

```bash
# List available products
curl -H "X-API-Key: your_api_key" \
  https://api.multiverse-dive.io/v1/events/evt_abc123def456/products

# Download a product
curl -H "X-API-Key: your_api_key" \
  -O https://api.multiverse-dive.io/v1/events/evt_abc123def456/products/prod_001/download
```

---

## API Endpoints

### Events

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/events` | Submit a new event |
| GET | `/events` | List events |
| GET | `/events/{id}` | Get event details |
| POST | `/events/{id}/cancel` | Cancel an event |

### Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/events/{id}/status` | Get processing status |
| GET | `/events/{id}/progress` | Get detailed progress |

### Products

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/events/{id}/products` | List products |
| GET | `/events/{id}/products/{pid}` | Get product metadata |
| GET | `/events/{id}/products/{pid}/download` | Download product |

### Catalog

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/catalog/sources` | List data sources |
| GET | `/catalog/algorithms` | List algorithms |
| GET | `/catalog/event-types` | List event types |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Basic health check |
| GET | `/health/ready` | Readiness probe |
| GET | `/health/live` | Liveness probe |

### Webhooks

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/webhooks` | Register webhook |
| GET | `/webhooks` | List webhooks |
| GET | `/webhooks/{id}` | Get webhook details |
| DELETE | `/webhooks/{id}` | Delete webhook |
| POST | `/webhooks/{id}/test` | Test webhook delivery |

---

## Event Specification

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `intent.class` | string | Event class (e.g., `flood.coastal.storm_surge`) |
| `intent.source` | string | `explicit` or `inferred` |
| `spatial.type` | string | `Polygon` or `MultiPolygon` |
| `spatial.coordinates` | array | GeoJSON coordinates |
| `temporal.start` | string | ISO 8601 start time |
| `temporal.end` | string | ISO 8601 end time |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Custom event ID (auto-generated if omitted) |
| `priority` | string | `critical`, `high`, `normal`, `low` |
| `constraints.max_cloud_cover` | number | 0-1, max cloud cover |
| `constraints.required_data_types` | array | Required data types |
| `metadata.tags` | array | Classification tags |

### Supported Event Classes

**Flood:**
- `flood.coastal` - Coastal flooding
- `flood.coastal.storm_surge` - Storm surge flooding
- `flood.riverine` - River flooding
- `flood.pluvial` - Urban/rainfall flooding

**Wildfire:**
- `wildfire.forest` - Forest fires
- `wildfire.grassland` - Grassland fires
- `wildfire.prescribed` - Prescribed burns

**Storm:**
- `storm.tropical_cyclone` - Hurricanes/typhoons
- `storm.tornado` - Tornado damage
- `storm.severe_weather` - Severe storms

---

## Webhooks

Register webhooks to receive real-time notifications when events complete or fail.

### Registering a Webhook

```bash
curl -X POST https://api.multiverse-dive.io/v1/webhooks \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-server.com/webhook",
    "events": ["event.completed", "event.failed"],
    "secret": "your_webhook_secret"
  }'
```

### Webhook Events

| Event Type | Description |
|------------|-------------|
| `event.submitted` | Event accepted for processing |
| `event.started` | Processing started |
| `event.progress` | Progress update |
| `event.completed` | Processing completed |
| `event.failed` | Processing failed |
| `*` | All events |

### Webhook Payload

```json
{
  "event_type": "event.completed",
  "timestamp": "2024-09-17T14:30:00Z",
  "event_id": "evt_abc123def456",
  "payload": {
    "status": "completed",
    "products": ["prod_001", "prod_002"],
    "quality_score": 0.92
  }
}
```

### Verifying Webhook Signatures

If you provided a `secret` during registration, each delivery includes an `X-Webhook-Signature` header:

```python
import hmac
import hashlib

def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Event evt_xyz not found",
  "error_code": "EVENT_NOT_FOUND"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request |
| 401 | Missing authentication |
| 403 | Invalid authentication |
| 404 | Resource not found |
| 422 | Validation error |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

## Examples

See the `examples/` directory for complete code examples:

- `submit_event.py` - Python example using requests
- `submit_event.sh` - cURL example
- `webhook_handler.py` - FastAPI webhook receiver

---

## SDK and Client Libraries

Official client libraries are available for:

- **Python**: `pip install multiverse-dive`
- **JavaScript/TypeScript**: `npm install @multiverse/dive`

---

## Support

- **Email**: support@multiverse-dive.io
- **GitHub Issues**: https://github.com/gpriceless/multiverse_dive/issues
- **Documentation**: https://docs.multiverse-dive.io
