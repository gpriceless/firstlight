# =============================================================================
# Background Worker Image for FirstLight
# =============================================================================
# Worker image for distributed processing and async task execution.
# Connects to Redis for job queuing and coordinates with agent system.
# =============================================================================

# Accept base image as build argument for CI/CD flexibility
ARG BASE_IMAGE=firstlight-base:latest
FROM ${BASE_IMAGE} AS worker

LABEL maintainer="FirstLight Team"
LABEL description="Background worker image for distributed processing"
LABEL version="0.1.0"

# Switch to root for file operations
USER root

# Copy required modules for worker processes
COPY --chown=firstlight:firstlight core/ /app/core/
COPY --chown=firstlight:firstlight openspec/ /app/openspec/
COPY --chown=firstlight:firstlight agents/ /app/agents/
COPY --chown=firstlight:firstlight workers/ /app/workers/
COPY --chown=firstlight:firstlight api/ /app/api/

# Copy configuration files
COPY --chown=firstlight:firstlight examples/ /app/examples/

# Ensure proper permissions
RUN chown -R firstlight:firstlight /app

# Switch back to non-root user
USER firstlight

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# Worker configuration
ENV WORKER_CONCURRENCY=4
ENV WORKER_QUEUE=default
ENV REDIS_URL=redis://localhost:6379/0

# No HTTP healthcheck for worker - uses Redis ping
HEALTHCHECK NONE

# Start the Taskiq worker process
# Processes tasks from Redis (webhook delivery, maintenance, OGC execution)
CMD ["taskiq", "worker", "workers.taskiq_app:broker", "--workers", "4"]
