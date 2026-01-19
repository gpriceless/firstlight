# FirstLight - Edge Deployment Guide

This guide covers deploying FirstLight on edge devices for field operations, remote sensing stations, and disconnected environments.

## Overview

Edge deployments enable geospatial analysis in resource-constrained environments where cloud connectivity may be limited or unavailable. The system supports two primary edge platforms:

1. **ARM64 (Raspberry Pi)** - Lightweight deployment for basic analysis
2. **NVIDIA Jetson** - GPU-accelerated deployment for advanced algorithms

## Hardware Requirements

### ARM64 / Raspberry Pi

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Device | Raspberry Pi 4 | Raspberry Pi 5 |
| RAM | 4GB | 8GB |
| Storage | 32GB SD | 128GB NVMe |
| OS | Raspberry Pi OS 64-bit | Ubuntu 22.04 ARM64 |

**Supported algorithms:**
- NDWI (water detection)
- NDVI (vegetation analysis)
- HAND (flood mapping)
- Basic classification

**Not supported:**
- ML/Deep learning models
- GPU-accelerated processing
- Large raster processing (>4GB)

### NVIDIA Jetson

| Component | Jetson Nano | Jetson Xavier NX | Jetson Orin |
|-----------|-------------|------------------|-------------|
| RAM | 4GB | 8-16GB | 32-64GB |
| GPU | 128 CUDA | 384 CUDA | 1024+ CUDA |
| Storage | 64GB | 128GB | 256GB+ |
| JetPack | 4.6+ | 5.x | 6.x |

**Supported algorithms:**
- All ARM64 algorithms
- SAR processing
- ML-based classification
- Object detection
- GPU-accelerated raster operations

## Installation

### Prerequisites

1. Docker installed on the edge device
2. For Jetson: NVIDIA Container Toolkit configured
3. Network access for initial image pull (or pre-loaded images)

### ARM64 Deployment

```bash
# Build the lightweight image (on device or cross-compile)
docker build --platform linux/arm64 \
  -t firstlight-dive/edge-arm64:latest \
  -f deploy/edge/arm64/Dockerfile.lightweight .

# Run the container
docker run -d \
  --name firstlight-dive \
  --restart unless-stopped \
  -p 8080:8080 \
  -v /opt/firstlight/data:/data \
  -v /opt/firstlight/cache:/cache \
  firstlight-dive/edge-arm64:latest

# Check status
docker logs -f firstlight-dive
```

### NVIDIA Jetson Deployment

```bash
# Build the GPU image (on Jetson device)
docker build \
  -t firstlight-dive/edge-jetson:latest \
  -f deploy/edge/nvidia-jetson/Dockerfile.gpu .

# Run with NVIDIA runtime
docker run -d \
  --name firstlight-dive \
  --runtime=nvidia \
  --restart unless-stopped \
  -p 8080:8080 \
  -v /opt/firstlight/data:/data \
  -v /opt/firstlight/cache:/cache \
  -v /opt/firstlight/models:/models \
  firstlight-dive/edge-jetson:latest

# Verify GPU access
docker exec firstlight-dive nvidia-smi
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FIRSTLIGHT_ENV` | edge | Deployment environment |
| `FIRSTLIGHT_DATA_DIR` | /data | Data storage directory |
| `FIRSTLIGHT_CACHE_DIR` | /cache | Cache directory |
| `FIRSTLIGHT_EDGE_MODE` | true | Enable edge optimizations |
| `FIRSTLIGHT_GPU_ENABLED` | false | Enable GPU (Jetson only) |
| `FIRSTLIGHT_OFFLINE_MODE` | false | Disable network features |

### Memory Optimization

For 4GB devices, add these environment variables:

```bash
docker run -d \
  -e FIRSTLIGHT_MAX_WORKERS=1 \
  -e FIRSTLIGHT_CACHE_SIZE_MB=512 \
  -e FIRSTLIGHT_CHUNK_SIZE_MB=256 \
  ...
```

## Performance Expectations

### Processing Times (Approximate)

| Operation | ARM64 4GB | ARM64 8GB | Jetson Nano | Jetson Xavier |
|-----------|-----------|-----------|-------------|---------------|
| NDWI (10km2) | 45s | 30s | 20s | 8s |
| NDVI (10km2) | 40s | 25s | 18s | 6s |
| HAND (10km2) | 120s | 80s | 50s | 15s |
| SAR (10km2) | N/A | N/A | 90s | 25s |

### Concurrent Requests

| Platform | Max Concurrent | Memory per Request |
|----------|---------------|-------------------|
| ARM64 4GB | 1 | ~2GB |
| ARM64 8GB | 2 | ~2GB |
| Jetson Nano | 2 | ~1.5GB |
| Jetson Xavier | 4-6 | ~2GB |

## Offline Operation

Edge deployments can operate without network connectivity:

1. **Pre-load data**: Copy required datasets to `/data` before deployment
2. **Enable offline mode**: Set `FIRSTLIGHT_OFFLINE_MODE=true`
3. **Disable telemetry**: Set `FIRSTLIGHT_TELEMETRY=false`

### Data Synchronization

When connectivity is restored:

```bash
# Sync results to central server
docker exec firstlight-dive python -m core.sync upload --target https://central.example.com

# Pull updated models/data
docker exec firstlight-dive python -m core.sync download --source https://central.example.com
```

## Troubleshooting

### Out of Memory

```bash
# Check memory usage
docker stats firstlight-dive

# Reduce chunk size
docker exec firstlight-dive sed -i 's/CHUNK_SIZE_MB=512/CHUNK_SIZE_MB=128/' /app/config/edge.conf
docker restart firstlight-dive
```

### Slow Processing

1. Check thermal throttling: `vcgencmd measure_temp` (Pi) or `tegrastats` (Jetson)
2. Ensure adequate cooling
3. Reduce concurrent processing

### GPU Not Detected (Jetson)

```bash
# Verify NVIDIA runtime
docker info | grep -i runtime

# Check CUDA
docker exec firstlight-dive nvidia-smi

# Reinstall container toolkit if needed
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Security Considerations

1. **Network isolation**: Edge devices should be on isolated networks
2. **Data encryption**: Enable encryption for sensitive data at rest
3. **Access control**: Use strong authentication for API access
4. **Updates**: Establish secure update procedures for offline devices

## Support Matrix

| Feature | ARM64 | Jetson Nano | Jetson Xavier | Jetson Orin |
|---------|-------|-------------|---------------|-------------|
| Core algorithms | Yes | Yes | Yes | Yes |
| ML inference | No | Limited | Yes | Yes |
| Real-time processing | No | Limited | Yes | Yes |
| Multi-user | No | No | Yes | Yes |
| Offline mode | Yes | Yes | Yes | Yes |
