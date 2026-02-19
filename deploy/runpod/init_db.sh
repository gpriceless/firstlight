#!/usr/bin/env bash
# init_db.sh -- Initialize the FirstLight PostGIS database.
#
# Runs all migrations in order against a running PostgreSQL instance.
# Safe to re-run: all statements use IF NOT EXISTS / CREATE OR REPLACE.
#
# Usage:
#   bash deploy/runpod/init_db.sh
#   # Or with custom credentials:
#   DB_USER=firstlight DB_NAME=firstlight bash deploy/runpod/init_db.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_HOST="${DATABASE_HOST:-localhost}"
DB_PORT="${DATABASE_PORT:-5432}"
DB_NAME="${DATABASE_NAME:-firstlight}"
DB_USER="${DATABASE_USER:-firstlight}"
DB_PASSWORD="${DATABASE_PASSWORD:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

run_sql() {
    local label="$1"
    local file="$2"

    if [ ! -f "$file" ]; then
        warn "Migration file not found, skipping: $file"
        return 0
    fi

    info "Running: $label"
    if PGPASSWORD="$DB_PASSWORD" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -f "$file" \
        --set ON_ERROR_STOP=1 \
        -q; then
        info "  Done: $label"
    else
        error "  Failed: $label"
        error "  File: $file"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Locate project root
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    error "Cannot find project root from ${SCRIPT_DIR}"
    error "Expected pyproject.toml at: ${PROJECT_ROOT}/pyproject.toml"
    exit 1
fi

DB_DIR="${PROJECT_ROOT}/db"
MIGRATIONS_DIR="${DB_DIR}/migrations"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

header "FirstLight Database Initialization"

echo ""
echo "  Host:     ${DB_HOST}:${DB_PORT}"
echo "  Database: ${DB_NAME}"
echo "  User:     ${DB_USER}"
echo ""

# Wait for PostgreSQL to be ready
info "Waiting for PostgreSQL to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=0
until PGPASSWORD="$DB_PASSWORD" psql \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    -c "SELECT 1" \
    -q --tuples-only 2>/dev/null | grep -q 1; do
    ATTEMPT=$((ATTEMPT + 1))
    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
        error "PostgreSQL did not become ready after ${MAX_ATTEMPTS} attempts"
        exit 1
    fi
    echo -n "."
    sleep 2
done
echo ""
info "PostgreSQL is ready"

# Run initialization SQL (PostGIS extensions)
run_sql "Base init (PostGIS extensions)" \
    "${DB_DIR}/init.sql"

# Run migrations in order
run_sql "Migration 000: customer_id column" \
    "${MIGRATIONS_DIR}/000_add_customer_id.sql"

run_sql "Migration 001: control plane schema" \
    "${MIGRATIONS_DIR}/001_control_plane_schema.sql"

run_sql "Migration 002: job events NOTIFY" \
    "${MIGRATIONS_DIR}/002_job_events_notify.sql"

run_sql "Migration 003: webhook tables" \
    "${MIGRATIONS_DIR}/003_webhook_tables.sql"

run_sql "Migration 004: materialized views" \
    "${MIGRATIONS_DIR}/004_materialized_views.sql"

run_sql "Migration 005: partition job events" \
    "${MIGRATIONS_DIR}/005_partition_job_events.sql"

run_sql "Migration 006: pgSTAC init" \
    "${MIGRATIONS_DIR}/006_pgstac_init.sql"

run_sql "Migration 007: context data lakehouse" \
    "${MIGRATIONS_DIR}/007_context_data.sql"

header "Database Initialization Complete"

echo ""
info "All migrations applied successfully."
echo ""
echo "  Database ${DB_NAME} is ready at ${DB_HOST}:${DB_PORT}"
echo ""
