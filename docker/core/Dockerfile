# =============================================================================
# Core Processing Image for FirstLight
# =============================================================================
# Lightweight image for batch processing and analysis pipelines.
# Does not include API dependencies - optimized for compute workloads.
# =============================================================================

FROM firstlight-base:latest AS core

LABEL maintainer="FirstLight Team"
LABEL description="Core processing image for batch analysis"
LABEL version="0.1.0"

# Switch to root for file operations
USER root

# Copy core processing modules
COPY --chown=firstlight:firstlight core/ /app/core/
COPY --chown=firstlight:firstlight openspec/ /app/openspec/

# Copy configuration and schema files
COPY --chown=firstlight:firstlight examples/ /app/examples/

# Ensure proper permissions
RUN chown -R firstlight:firstlight /app

# Switch back to non-root user
USER firstlight

# Set Python path to include app directory
ENV PYTHONPATH=/app:${PYTHONPATH}

# No healthcheck needed for batch processing
HEALTHCHECK NONE

# Default command runs a simple validation
CMD ["python", "-c", "from core import *; from openspec import *; print('Core modules loaded successfully')"]
