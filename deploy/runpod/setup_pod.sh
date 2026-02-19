#!/usr/bin/env bash
# setup_pod.sh -- Bootstrap a fresh RunPod pod for FirstLight Control Plane.
#
# Installs all system and Python dependencies, sets up PostGIS + Redis,
# clones the repo, runs database migrations, and starts the API and
# Taskiq worker processes in the background.
#
# Safe to re-run: idempotent at every step.
#
# Repo: https://github.com/gpriceless/firstlight.git
#
# Usage:
#   # From a fresh RunPod pod terminal:
#   wget -q https://raw.githubusercontent.com/gpriceless/firstlight/feature/llm-control-plane/deploy/runpod/setup_pod.sh
#   bash setup_pod.sh
#
#   # Or if you already have the repo:
#   bash /path/to/firstlight/deploy/runpod/setup_pod.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIRSTLIGHT_REPO_URL="https://github.com/gpriceless/firstlight.git"
FIRSTLIGHT_BRANCH="feature/llm-control-plane"
FIRSTLIGHT_PROJECT_NAME="firstlight"

# RunPod persistent volume is at /runpod-volume/; fall back to /workspace
if [ -d "/runpod-volume" ]; then
    WORK_DIR="/runpod-volume"
else
    WORK_DIR="/workspace"
fi

API_PORT=8000
DB_NAME="firstlight"
DB_USER="firstlight"

# PID file locations so we can find/kill processes on re-run
API_PID_FILE="${WORK_DIR}/.firstlight_api.pid"
WORKER_PID_FILE="${WORK_DIR}/.firstlight_worker.pid"
LOG_DIR="${WORK_DIR}/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Stop a previously started background process by PID file
stop_if_running() {
    local pid_file="$1"
    local label="$2"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            info "Stopping previous ${label} (PID ${pid})..."
            kill "$pid" || true
            sleep 2
        fi
        rm -f "$pid_file"
    fi
}

# ---------------------------------------------------------------------------
# Step 0: Intro
# ---------------------------------------------------------------------------

header "RunPod Setup: FirstLight Control Plane"

echo ""
echo "  This script installs and starts the FirstLight geospatial event"
echo "  intelligence platform with the LLM Control Plane API."
echo ""
echo "  Work directory: ${WORK_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Check GPU
# ---------------------------------------------------------------------------

header "Step 1/8: Checking GPU"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    info "GPU memory: ${GPU_MEM} MB"

    if [ "$GPU_MEM" -ge 35000 ]; then
        info "A100 / A40 detected -- optimal for all workloads"
    elif [ "$GPU_MEM" -ge 20000 ]; then
        info "RTX 4090 / 24GB GPU detected -- good for raster processing"
    elif [ "$GPU_MEM" -ge 10000 ]; then
        info "10-20GB GPU -- sufficient for standard workloads"
    else
        info "GPU available with ${GPU_MEM}MB VRAM"
    fi
else
    warn "nvidia-smi not found. GPU not detected."
    warn "FirstLight API will run CPU-only (sufficient for control plane demo)."
    GPU_MEM=0
fi

# ---------------------------------------------------------------------------
# Step 2: Install system dependencies
# ---------------------------------------------------------------------------

header "Step 2/8: Installing System Dependencies"

info "Updating apt package list..."
apt-get update -qq

info "Installing PostgreSQL + PostGIS..."
# Install PostgreSQL 15 and PostGIS (common in Ubuntu 22.04-based pods)
apt-get install -y --no-install-recommends \
    postgresql \
    postgresql-contrib \
    postgresql-15-postgis-3 \
    postgresql-15-postgis-3-scripts \
    postgis \
    2>&1 | tail -3 || {
    warn "PostGIS 15 package not found, trying alternative packages..."
    apt-get install -y --no-install-recommends \
        postgresql \
        postgresql-contrib \
        postgis \
        2>&1 | tail -3
}

info "Installing Redis..."
apt-get install -y --no-install-recommends \
    redis-server \
    2>&1 | tail -1

info "Installing build dependencies (GDAL, geospatial libraries)..."
apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-venv \
    gdal-bin \
    libgdal-dev \
    libpq-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    curl \
    git \
    jq \
    2>&1 | tail -3

info "System dependencies installed."

# ---------------------------------------------------------------------------
# Step 3: Clone / update repos
# ---------------------------------------------------------------------------

header "Step 3/8: Setting Up Repositories"

# --- FirstLight ---
if [ -d "${WORK_DIR}/${FIRSTLIGHT_PROJECT_NAME}/.git" ]; then
    info "FirstLight already cloned -- pulling latest ${FIRSTLIGHT_BRANCH}..."
    cd "${WORK_DIR}/${FIRSTLIGHT_PROJECT_NAME}"
    git fetch origin
    git checkout "${FIRSTLIGHT_BRANCH}" 2>/dev/null || \
        git checkout -b "${FIRSTLIGHT_BRANCH}" "origin/${FIRSTLIGHT_BRANCH}"
    git pull origin "${FIRSTLIGHT_BRANCH}" || warn "git pull failed (local changes?)"
else
    info "Cloning FirstLight (branch: ${FIRSTLIGHT_BRANCH})..."
    cd "${WORK_DIR}"
    git clone --branch "${FIRSTLIGHT_BRANCH}" "${FIRSTLIGHT_REPO_URL}" "${FIRSTLIGHT_PROJECT_NAME}"
fi

FIRSTLIGHT_DIR="${WORK_DIR}/${FIRSTLIGHT_PROJECT_NAME}"
info "FirstLight at: ${FIRSTLIGHT_DIR}"

# ---------------------------------------------------------------------------
# Step 4: Install Python dependencies
# ---------------------------------------------------------------------------

header "Step 4/8: Installing Python Dependencies"

info "Upgrading pip..."
pip install --upgrade pip --quiet

# Install FirstLight with control-plane extras
info "Installing FirstLight[control-plane]..."
cd "${FIRSTLIGHT_DIR}"
pip install -e ".[control-plane]" --quiet 2>&1 | tail -5

# Install additional tools
info "Installing additional utilities..."
pip install uvicorn[standard] taskiq taskiq-redis --quiet

# Verify key imports
info "Verifying Python imports..."
python3 -c "
import sys
checks = []

try:
    import fastapi
    checks.append(('fastapi', fastapi.__version__))
except ImportError as e:
    checks.append(('fastapi', f'FAILED: {e}'))

try:
    import uvicorn
    checks.append(('uvicorn', uvicorn.__version__))
except ImportError as e:
    checks.append(('uvicorn', f'FAILED: {e}'))

try:
    import asyncpg
    checks.append(('asyncpg', asyncpg.__version__))
except ImportError as e:
    checks.append(('asyncpg', f'FAILED: {e}'))

try:
    import taskiq
    checks.append(('taskiq', taskiq.__version__))
except ImportError as e:
    checks.append(('taskiq', f'FAILED: {e}'))

try:
    import taskiq_redis
    checks.append(('taskiq_redis', 'OK'))
except ImportError as e:
    checks.append(('taskiq_redis', f'FAILED: {e}'))

try:
    import rasterio
    checks.append(('rasterio', rasterio.__version__))
except ImportError as e:
    checks.append(('rasterio', f'FAILED: {e}'))

for name, status in checks:
    print(f'  {name}: {status}')

failed = [name for name, status in checks if 'FAILED' in str(status)]
if failed:
    print(f'\n  WARNING: {len(failed)} import(s) failed: {failed}')
    sys.exit(1)
"

# ---------------------------------------------------------------------------
# Step 5: Configure environment
# ---------------------------------------------------------------------------

header "Step 5/8: Configuring Environment"

ENV_FILE="${FIRSTLIGHT_DIR}/.env"

if [ ! -f "$ENV_FILE" ]; then
    info "Creating .env from template..."
    cp "${FIRSTLIGHT_DIR}/deploy/runpod/env.template" "$ENV_FILE"

    # Generate random credentials
    DB_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
    REDIS_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
    AUTH_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    DEMO_API_KEY=$(python3 -c "import secrets; print('demo-' + secrets.token_urlsafe(16))")

    sed -i "s/^DATABASE_PASSWORD=$/DATABASE_PASSWORD=${DB_PASSWORD}/" "$ENV_FILE"
    sed -i "s/^REDIS_PASSWORD=$/REDIS_PASSWORD=${REDIS_PASSWORD}/" "$ENV_FILE"
    sed -i "s/^AUTH_SECRET_KEY=$/AUTH_SECRET_KEY=${AUTH_SECRET_KEY}/" "$ENV_FILE"
    sed -i "s/^DEMO_API_KEY=$/DEMO_API_KEY=${DEMO_API_KEY}/" "$ENV_FILE"
    sed -i "s/^AUTH_ALLOWED_API_KEYS=$/AUTH_ALLOWED_API_KEYS=${DEMO_API_KEY}/" "$ENV_FILE"

    # Enable debug mode so Swagger UI is available at /api/docs
    sed -i "s/^FIRSTLIGHT_DEBUG=false/FIRSTLIGHT_DEBUG=true/" "$ENV_FILE"
    sed -i "s/^FIRSTLIGHT_ENVIRONMENT=production/FIRSTLIGHT_ENVIRONMENT=development/" "$ENV_FILE"

    info "Generated credentials and saved to ${ENV_FILE}"
    warn "Keep this file safe -- it contains your database and API secrets."
else
    info ".env already exists -- loading existing credentials."
    # Extract credentials for use below
    DB_PASSWORD=$(grep '^DATABASE_PASSWORD=' "$ENV_FILE" | cut -d= -f2-)
    REDIS_PASSWORD=$(grep '^REDIS_PASSWORD=' "$ENV_FILE" | cut -d= -f2-)
    DEMO_API_KEY=$(grep '^DEMO_API_KEY=' "$ENV_FILE" | cut -d= -f2-)
fi

# Export env vars for use by the rest of this script
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

# ---------------------------------------------------------------------------
# Step 6: Set up PostgreSQL + PostGIS
# ---------------------------------------------------------------------------

header "Step 6/8: Setting Up PostgreSQL + PostGIS"

# Start PostgreSQL service
info "Starting PostgreSQL..."
if command -v pg_ctlcluster &>/dev/null; then
    # Ubuntu-style cluster management
    PG_VERSION=$(pg_lsclusters | awk 'NR>1 {print $1; exit}')
    PG_CLUSTER=$(pg_lsclusters | awk 'NR>1 {print $2; exit}')
    pg_ctlcluster "$PG_VERSION" "$PG_CLUSTER" start || true
elif command -v pg_ctl &>/dev/null; then
    # Direct pg_ctl fallback
    pg_ctl start -D /var/lib/postgresql/data -l /tmp/pg.log || true
else
    service postgresql start || true
fi

# Wait for PostgreSQL to be ready
info "Waiting for PostgreSQL..."
MAX_ATTEMPTS=30
ATTEMPT=0
until pg_isready -U postgres -q 2>/dev/null; do
    ATTEMPT=$((ATTEMPT + 1))
    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
        error "PostgreSQL did not start after ${MAX_ATTEMPTS} attempts"
        exit 1
    fi
    sleep 1
done
info "PostgreSQL is ready."

# Create database role and database (idempotent)
info "Creating database role '${DB_USER}'..."
sudo -u postgres psql -c "
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN
            CREATE ROLE ${DB_USER} WITH LOGIN PASSWORD '${DATABASE_PASSWORD}';
        ELSE
            ALTER ROLE ${DB_USER} WITH PASSWORD '${DATABASE_PASSWORD}';
        END IF;
    END
    \$\$;
" 2>/dev/null || warn "Role creation may have failed (already exists?)"

info "Creating database '${DB_NAME}'..."
sudo -u postgres psql -c "
    SELECT 'already exists' FROM pg_database WHERE datname = '${DB_NAME}'
    UNION ALL
    SELECT pg_catalog.pg_terminate_backend(pid) || 'terminated' FROM pg_stat_activity
    WHERE datname = '${DB_NAME}' AND pid <> pg_backend_pid();
" -q 2>/dev/null || true

sudo -u postgres createdb -O "${DB_USER}" "${DB_NAME}" 2>/dev/null || \
    warn "Database '${DB_NAME}' already exists."

# Grant permissions
sudo -u postgres psql -d "${DB_NAME}" -c "
    GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
    GRANT ALL PRIVILEGES ON SCHEMA public TO ${DB_USER};
" -q 2>/dev/null || warn "Permission grant may have partially failed"

# Run migrations
info "Running database migrations..."
export DATABASE_HOST="localhost"
export DATABASE_PORT="5432"
export DATABASE_NAME="${DB_NAME}"
export DATABASE_USER="${DB_USER}"
export DATABASE_PASSWORD="${DATABASE_PASSWORD}"
bash "${FIRSTLIGHT_DIR}/deploy/runpod/init_db.sh"

# ---------------------------------------------------------------------------
# Step 7: Set up Redis
# ---------------------------------------------------------------------------

header "Step 7/8: Setting Up Redis"

# Configure Redis with password
REDIS_CONF="/etc/redis/redis.conf"
if [ -f "$REDIS_CONF" ]; then
    info "Configuring Redis..."
    # Set password (replace or append)
    if grep -q "^requirepass" "$REDIS_CONF"; then
        sed -i "s/^requirepass .*/requirepass ${REDIS_PASSWORD}/" "$REDIS_CONF"
    else
        echo "requirepass ${REDIS_PASSWORD}" >> "$REDIS_CONF"
    fi
    # Increase maxmemory to 1GB (pods have plenty of RAM)
    if grep -q "^maxmemory " "$REDIS_CONF"; then
        sed -i "s/^maxmemory .*/maxmemory 1gb/" "$REDIS_CONF"
    else
        echo "maxmemory 1gb" >> "$REDIS_CONF"
        echo "maxmemory-policy allkeys-lru" >> "$REDIS_CONF"
    fi
fi

info "Starting Redis..."
service redis-server start || redis-server "${REDIS_CONF}" --daemonize yes || {
    # Last resort: start without config
    redis-server --requirepass "${REDIS_PASSWORD}" --daemonize yes --logfile /tmp/redis.log
}

# Verify Redis is up
info "Verifying Redis connection..."
sleep 2
if redis-cli -a "${REDIS_PASSWORD}" ping 2>/dev/null | grep -q PONG; then
    info "Redis is running."
else
    error "Redis ping failed. Check /tmp/redis.log for errors."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 8: Start FirstLight services
# ---------------------------------------------------------------------------

header "Step 8/8: Starting FirstLight Services"

mkdir -p "$LOG_DIR"
cd "${FIRSTLIGHT_DIR}"

# Stop any previously running processes
stop_if_running "$API_PID_FILE" "API server"
stop_if_running "$WORKER_PID_FILE" "Taskiq worker"

# Build Redis URL for services
REDIS_URL="redis://:${REDIS_PASSWORD}@localhost:6379/0"
DB_URL="postgresql://firstlight:${DATABASE_PASSWORD}@localhost:5432/${DB_NAME}"

# Start uvicorn API server
info "Starting FirstLight API server on port ${API_PORT}..."
FIRSTLIGHT_DEBUG=true \
FIRSTLIGHT_ENVIRONMENT=development \
FIRSTLIGHT_STATE_BACKEND="${FIRSTLIGHT_STATE_BACKEND:-postgis}" \
DATABASE_HOST=localhost \
DATABASE_PORT=5432 \
DATABASE_NAME="${DB_NAME}" \
DATABASE_USER="${DB_USER}" \
DATABASE_PASSWORD="${DATABASE_PASSWORD}" \
REDIS_HOST=localhost \
REDIS_PORT=6379 \
REDIS_PASSWORD="${REDIS_PASSWORD}" \
AUTH_SECRET_KEY="${AUTH_SECRET_KEY}" \
AUTH_ALLOWED_API_KEYS="${AUTH_ALLOWED_API_KEYS:-${DEMO_API_KEY:-}}" \
PYTHONPATH="${FIRSTLIGHT_DIR}" \
nohup uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${API_PORT}" \
    --workers 2 \
    --log-level info \
    > "${LOG_DIR}/api.log" 2>&1 &

API_PID=$!
echo "$API_PID" > "$API_PID_FILE"
info "API server started (PID ${API_PID})"

# Give the API a moment to start
sleep 3

# Start Taskiq worker
info "Starting Taskiq worker..."
FIRSTLIGHT_ENVIRONMENT=development \
FIRSTLIGHT_STATE_BACKEND="${FIRSTLIGHT_STATE_BACKEND:-postgis}" \
DATABASE_HOST=localhost \
DATABASE_PORT=5432 \
DATABASE_NAME="${DB_NAME}" \
DATABASE_USER="${DB_USER}" \
DATABASE_PASSWORD="${DATABASE_PASSWORD}" \
REDIS_HOST=localhost \
REDIS_PORT=6379 \
REDIS_PASSWORD="${REDIS_PASSWORD}" \
AUTH_SECRET_KEY="${AUTH_SECRET_KEY}" \
REDIS_URL="${REDIS_URL}" \
PYTHONPATH="${FIRSTLIGHT_DIR}" \
nohup taskiq worker workers.taskiq_app:broker \
    --workers 2 \
    > "${LOG_DIR}/worker.log" 2>&1 &

WORKER_PID=$!
echo "$WORKER_PID" > "$WORKER_PID_FILE"
info "Taskiq worker started (PID ${WORKER_PID})"

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

info "Waiting for API to become healthy..."
MAX_ATTEMPTS=30
ATTEMPT=0
API_READY=false

until $API_READY; do
    ATTEMPT=$((ATTEMPT + 1))
    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
        warn "API did not become healthy within ${MAX_ATTEMPTS} attempts"
        warn "Check logs: tail -f ${LOG_DIR}/api.log"
        break
    fi
    if curl -sf "http://localhost:${API_PORT}/api/v1/health" > /dev/null 2>&1; then
        API_READY=true
    else
        sleep 3
    fi
done

if $API_READY; then
    info "API health check passed."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

# Get pod public IP (RunPod exposes via env or we detect it)
POD_IP="${RUNPOD_PUBLIC_IP:-$(curl -s https://api.ipify.org 2>/dev/null || echo "<pod-ip>")}"

header "Setup Complete"

echo ""
echo "  Python:       $(python3 --version)"
echo ""
echo "  FirstLight:   ${FIRSTLIGHT_DIR}"
echo "  Logs:         ${LOG_DIR}/"
echo ""
echo "  ============================================================"
echo "  ACCESS URLS"
echo "  ============================================================"
echo ""
echo "  API (local):    http://localhost:${API_PORT}"
echo "  Swagger UI:     http://localhost:${API_PORT}/api/docs"
echo "  Health check:   http://localhost:${API_PORT}/api/v1/health"
echo ""
echo "  If RunPod exposes a public endpoint:"
echo "  API (public):   http://${POD_IP}:${API_PORT}"
echo "  Swagger UI:     http://${POD_IP}:${API_PORT}/api/docs"
echo ""
echo "  ============================================================"
echo "  CREDENTIALS"
echo "  ============================================================"
echo ""
echo "  Demo API key:   ${DEMO_API_KEY:-<see .env file>}"
echo "  Credentials:    ${ENV_FILE}"
echo ""
echo "  ============================================================"
echo "  QUICK TEST"
echo "  ============================================================"
echo ""
echo "  # Check health:"
echo "  curl http://localhost:${API_PORT}/api/v1/health"
echo ""
echo "  # Use the demo API key:"
echo "  curl -H 'X-API-Key: ${DEMO_API_KEY:-<your-demo-key>}' \\"
echo "       http://localhost:${API_PORT}/control/v1/tools"
echo ""
echo "  # Watch API logs:"
echo "  tail -f ${LOG_DIR}/api.log"
echo ""
echo "  # Watch worker logs:"
echo "  tail -f ${LOG_DIR}/worker.log"
echo ""
