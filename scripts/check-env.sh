#!/usr/bin/env bash
# =============================================================================
# Environment Variable Validation Script
# =============================================================================
# Validates that required environment variables are set before starting
# FirstLight services. Aborts with a clear error if any are missing.
#
# Usage:
#   ./scripts/check-env.sh
#   source scripts/check-env.sh  # to abort the calling script on failure
#
# Required variables:
#   POSTGRES_PASSWORD - PostgreSQL database password
#   REDIS_PASSWORD    - Redis authentication password

set -euo pipefail

MISSING=()

if [ -z "${POSTGRES_PASSWORD:-}" ]; then
    MISSING+=("POSTGRES_PASSWORD")
fi

if [ -z "${REDIS_PASSWORD:-}" ]; then
    MISSING+=("REDIS_PASSWORD")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "=========================================================="
    echo "  FATAL: Required environment variables are not set!"
    echo "=========================================================="
    echo ""
    for var in "${MISSING[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "  Set these variables in your .env file or environment"
    echo "  before starting FirstLight services."
    echo ""
    echo "  Example:"
    echo "    export POSTGRES_PASSWORD=your-secure-password"
    echo "    export REDIS_PASSWORD=your-secure-password"
    echo ""
    echo "  Or create a .env file (see .env.example for reference)."
    echo "=========================================================="
    exit 1
fi

echo "Environment check passed: all required variables are set."
