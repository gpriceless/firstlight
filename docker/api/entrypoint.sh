#!/bin/bash
# API entrypoint script that runs pgSTAC migration before starting the server.
#
# Usage: entrypoint.sh [command...]
#
# If DATABASE_URL is set, attempts to run pypgstac migrate before
# starting the main command. If pypgstac is not installed or the
# migration fails, logs a warning and continues.

set -e

echo "FirstLight API entrypoint starting..."

# Run pgSTAC migration if pypgstac is available and DATABASE_URL is set
if [ -n "$DATABASE_URL" ]; then
    echo "Attempting pgSTAC schema migration..."
    if command -v pypgstac &> /dev/null; then
        pypgstac migrate --dsn "$DATABASE_URL" || {
            echo "WARNING: pgSTAC migration failed. STAC endpoints may not work."
        }
    else
        echo "INFO: pypgstac not installed, skipping STAC schema migration."
    fi

    # Run FirstLight SQL migrations
    if command -v psql &> /dev/null; then
        for migration in /app/db/migrations/*.sql; do
            if [ -f "$migration" ]; then
                echo "Running migration: $(basename "$migration")"
                psql "$DATABASE_URL" -f "$migration" 2>/dev/null || true
            fi
        done
    fi
fi

# Execute the main command
exec "$@"
