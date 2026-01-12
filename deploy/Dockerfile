# =============================================================================
# Multiverse Dive - Multi-Stage Production Dockerfile
# =============================================================================
# Combined Dockerfile for building API and Worker images.
#
# Usage:
#   docker build --target api -t multiverse-dive/api .
#   docker build --target worker -t multiverse-dive/worker .
#
# Or with docker-compose:
#   docker-compose up --build
# =============================================================================

# -----------------------------------------------------------------------------
# Base Stage - Common dependencies for all services
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS base

LABEL maintainer="Multiverse Dive Team"
LABEL version="0.1.0"

# Create non-root user for security
RUN groupadd --gid 1000 mdive \
    && useradd --uid 1000 --gid mdive --shell /bin/bash --create-home mdive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Create data directories
RUN mkdir -p /data/products /data/cache /data/tmp \
    && chown -R mdive:mdive /data

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || \
    pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.5.0 \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    shapely>=2.0.0 \
    pyproj>=3.6.0 \
    pystac-client>=0.7.0 \
    python-jose>=3.3.0 \
    passlib>=1.7.4 \
    python-multipart>=0.0.6 \
    aiohttp>=3.9.0 \
    redis>=5.0.0 \
    httpx>=0.25.0 \
    pyyaml>=6.0

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}
ENV PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# API Stage - FastAPI application server
# -----------------------------------------------------------------------------
FROM base AS api

LABEL description="API service image with FastAPI and agent orchestration"

# Copy application code
COPY --chown=mdive:mdive core/ /app/core/
COPY --chown=mdive:mdive openspec/ /app/openspec/
COPY --chown=mdive:mdive agents/ /app/agents/
COPY --chown=mdive:mdive api/ /app/api/
COPY --chown=mdive:mdive cli/ /app/cli/
COPY --chown=mdive:mdive examples/ /app/examples/

# Ensure proper permissions
RUN chown -R mdive:mdive /app

# Switch to non-root user
USER mdive

# API configuration defaults
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_WORKERS=4
ENV LOG_LEVEL=info

# Expose API port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Start API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# -----------------------------------------------------------------------------
# Worker Stage - Background job processing
# -----------------------------------------------------------------------------
FROM base AS worker

LABEL description="Background worker image for distributed processing"

# Copy application code (subset needed for workers)
COPY --chown=mdive:mdive core/ /app/core/
COPY --chown=mdive:mdive openspec/ /app/openspec/
COPY --chown=mdive:mdive agents/ /app/agents/
COPY --chown=mdive:mdive examples/ /app/examples/

# Ensure proper permissions
RUN chown -R mdive:mdive /app

# Switch to non-root user
USER mdive

# Worker configuration defaults
ENV WORKER_CONCURRENCY=4
ENV WORKER_QUEUE=default
ENV REDIS_URL=redis://localhost:6379/0
ENV LOG_LEVEL=info

# No HTTP healthcheck for worker
HEALTHCHECK NONE

# Start worker process
CMD ["python", "-m", "agents.orchestrator.agent"]
