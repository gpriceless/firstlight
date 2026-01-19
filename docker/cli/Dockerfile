# =============================================================================
# CLI Tools Image for FirstLight
# =============================================================================
# Lightweight image for command-line interface operations.
# Provides the 'flight' CLI tool for local analysis and pipeline execution.
# =============================================================================

FROM firstlight-base:latest AS cli

LABEL maintainer="FirstLight Team"
LABEL description="CLI tools image for firstlight"
LABEL version="0.1.0"

# Switch to root for file operations
USER root

# Copy CLI and core modules (minimal footprint)
COPY --chown=firstlight:firstlight cli/ /app/cli/
COPY --chown=firstlight:firstlight core/ /app/core/
COPY --chown=firstlight:firstlight openspec/ /app/openspec/

# Copy configuration files
COPY --chown=firstlight:firstlight examples/ /app/examples/

# Ensure proper permissions
RUN chown -R firstlight:firstlight /app

# Switch back to non-root user
USER firstlight

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# No healthcheck for CLI tool
HEALTHCHECK NONE

# Set entrypoint to the CLI tool
# Users can run: docker run firstlight-cli analyze --help
ENTRYPOINT ["python", "-m", "cli"]

# Default command shows help
CMD ["--help"]
