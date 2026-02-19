# =============================================================================
# API Service Image for FirstLight
# =============================================================================
# Full-featured API server with FastAPI and agent orchestration.
# Exposes REST endpoints for event analysis and pipeline management.
# =============================================================================

# Accept base image as build argument for CI/CD flexibility
ARG BASE_IMAGE=firstlight-base:latest
FROM ${BASE_IMAGE} AS api

LABEL maintainer="FirstLight Team"
LABEL description="API service image with FastAPI and agent orchestration"
LABEL version="0.1.0"

# Switch to root for file operations
USER root

# Copy all required modules
COPY --chown=firstlight:firstlight core/ /app/core/
COPY --chown=firstlight:firstlight openspec/ /app/openspec/
COPY --chown=firstlight:firstlight agents/ /app/agents/
COPY --chown=firstlight:firstlight api/ /app/api/
COPY --chown=firstlight:firstlight workers/ /app/workers/
COPY --chown=firstlight:firstlight db/ /app/db/

# Copy configuration files
COPY --chown=firstlight:firstlight examples/ /app/examples/

# Ensure proper permissions
RUN chown -R firstlight:firstlight /app

# Switch back to non-root user
USER firstlight

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# API configuration
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_WORKERS=4
ENV LOG_LEVEL=info

# Expose the API port
EXPOSE 8000

# Healthcheck for API service
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Add pgSTAC migration entrypoint script
COPY --chown=firstlight:firstlight docker/api/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh || true

# Start the API server with uvicorn (entrypoint runs pgSTAC migration first)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
