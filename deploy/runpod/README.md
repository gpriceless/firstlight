# FirstLight Control Plane - RunPod Deployment

Quick-start guide for running the FirstLight geospatial event intelligence
platform with the LLM Control Plane on a RunPod pod.

---

## Recommended Pod Configuration

| Field | Value |
|-------|-------|
| GPU | A40 (48GB) or RTX 4090 (24GB) |
| vCPU | 8+ |
| RAM | 32 GB+ |
| Disk | 50 GB+ (persistent volume) |
| OS | RunPod PyTorch 2.1+ template |

An A40 or RTX 4090 runs the full stack comfortably. CPU-only pods also work
for the control plane demo (GPU accelerates raster processing).

---

## Option A: Bare-Metal Setup (Recommended)

Most RunPod pods do not have Docker available. This path installs everything
directly on the pod OS.

### Step 1: Create the pod

In the RunPod console:

1. Click **Deploy**
2. Choose a GPU pod (A40 or RTX 4090 recommended)
3. Select **RunPod PyTorch 2.1** or **Ubuntu 22.04** as the template
4. Set disk to at least 50 GB
5. Enable a persistent volume at `/runpod-volume` if you want data to survive
   pod restarts
6. Deploy

### Step 2: Open a terminal

Use the **Web Terminal** button in the RunPod console, or SSH using the
credentials shown after deployment.

### Step 3: Run the setup script

```bash
# If you want to clone the repo first:
git clone -b feature/llm-control-plane https://github.com/gpriceless/firstlight.git
cd firstlight
bash deploy/runpod/setup_pod.sh
```

The script:
- Checks the GPU
- Installs PostGIS, Redis, Python deps via apt and pip
- Clones FirstLight
- Generates random database/Redis/auth credentials and saves them to `.env`
- Runs all database migrations
- Starts the FastAPI server (uvicorn) and Taskiq worker in the background
- Prints access URLs and the demo API key

**The script is idempotent.** Re-running it is safe -- it pulls latest code,
skips already-applied migrations, and restarts processes.

### Step 4: Access the Swagger UI

Once the script finishes, open in your browser:

```
http://<pod-ip>:8000/api/docs
```

The pod IP is printed at the end of the setup script. You can also find it in
the RunPod console under **Connect**.

Use the demo API key printed by the script in the `X-API-Key` header when
testing authenticated endpoints.

### Step 5: Quick health check

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:

```json
{"status": "healthy"}
```

### Step 6: Share with MAIA

Share the Swagger UI URL and demo API key:

```
Swagger UI: http://<pod-ip>:8000/api/docs
API key:    demo-<generated-key>   (printed by setup script, also in .env)
```

---

## Option B: Docker Compose (if Docker is available)

Some RunPod pod templates include Docker. If yours does, you can use the
compose file instead.

```bash
# Clone the repo
git clone -b feature/llm-control-plane https://github.com/gpriceless/firstlight.git
cd firstlight

# Copy and edit the environment template
cp deploy/runpod/env.template .env
nano .env   # Fill in DATABASE_PASSWORD, REDIS_PASSWORD, AUTH_SECRET_KEY

# Start all services
docker compose -f deploy/runpod/docker-compose.runpod.yml up -d

# Follow logs
docker compose -f deploy/runpod/docker-compose.runpod.yml logs -f api

```

Services exposed:

| Port | Service |
|------|---------|
| 8000 | FirstLight API |
| 5432 | PostgreSQL |
| 6379 | Redis |

---

## Credentials

The bare-metal setup script auto-generates credentials and saves them to
`firstlight/.env`. The generated values are:

| Variable | Description |
|----------|-------------|
| `DATABASE_PASSWORD` | PostGIS password for the `firstlight` role |
| `REDIS_PASSWORD` | Redis auth password |
| `AUTH_SECRET_KEY` | JWT signing key (32-byte hex) |
| `DEMO_API_KEY` | API key to share for demo access |

To find your demo key after setup:

```bash
grep DEMO_API_KEY /workspace/firstlight/.env
```

---

## Logs

```bash
# API server
tail -f /workspace/logs/api.log

# Taskiq worker
tail -f /workspace/logs/worker.log
```

---

## Restarting Services

The setup script stops and restarts services on re-run:

```bash
bash /workspace/firstlight/deploy/runpod/setup_pod.sh
```

To restart manually:

```bash
# Find PIDs
cat /workspace/.firstlight_api.pid
cat /workspace/.firstlight_worker.pid

# Restart API
kill $(cat /workspace/.firstlight_api.pid)
cd /workspace/firstlight
source .env
FIRSTLIGHT_DEBUG=true FIRSTLIGHT_ENVIRONMENT=development \
DATABASE_HOST=localhost DATABASE_PORT=5432 \
DATABASE_NAME=firstlight DATABASE_USER=firstlight \
DATABASE_PASSWORD=$DATABASE_PASSWORD \
REDIS_HOST=localhost REDIS_PORT=6379 REDIS_PASSWORD=$REDIS_PASSWORD \
AUTH_SECRET_KEY=$AUTH_SECRET_KEY \
AUTH_ALLOWED_API_KEYS=$AUTH_ALLOWED_API_KEYS \
PYTHONPATH=/workspace/firstlight \
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2 \
    > /workspace/logs/api.log 2>&1 &
```

---

## Database

To run migrations manually or re-initialize the database:

```bash
cd /workspace/firstlight
source .env
bash deploy/runpod/init_db.sh
```

---

## Endpoints Reference

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/health` | Health check |
| `GET /api/docs` | Swagger UI (debug mode only) |
| `POST /api/v1/events` | Submit a geospatial event |
| `GET /api/v1/events/{id}/status` | Check event status |
| `GET /control/v1/tools` | List LLM control plane tools |
| `GET /internal/v1/pipeline/health` | Pipeline health metrics |
| `GET /stac` | STAC catalog |
| `GET /oapi` | OGC API Processes |
