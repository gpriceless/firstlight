# Configuration Management

This directory contains environment configuration files for the FirstLight platform.

## Configuration Files

| File | Purpose |
|------|---------|
| `base.env` | Base configuration with defaults for all environments |
| `production.env` | Production environment overrides |
| `development.env` | Development environment overrides |

## Usage

### Local Development

```bash
# Copy base and development configs
cp deploy/config/base.env .env
cat deploy/config/development.env >> .env

# Or use docker-compose with env_file
docker-compose --env-file deploy/config/development.env up
```

### Production Deployment

```bash
# Use environment-specific config
docker-compose --env-file deploy/config/production.env up

# Or in Kubernetes, create ConfigMaps and Secrets
kubectl create configmap firstlight-config --from-env-file=deploy/config/base.env
```

## Environment Variables Reference

### Application Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `APP_NAME` | No | `firstlight-dive` | Application name for logging and metrics |
| `APP_ENV` | No | `production` | Environment: `development`, `staging`, `production` |
| `APP_VERSION` | No | `0.1.0` | Application version |
| `LOG_LEVEL` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FORMAT` | No | `json` | Log format: `json`, `text` |
| `DEBUG` | No | `false` | Enable debug mode (never in production) |

### API Server

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_HOST` | No | `0.0.0.0` | API bind address |
| `API_PORT` | No | `8000` | API port |
| `API_WORKERS` | No | `4` | Number of worker processes |
| `API_TIMEOUT_SECONDS` | No | `300` | Request timeout in seconds |
| `API_MAX_REQUEST_SIZE_MB` | No | `100` | Maximum request body size |
| `API_CORS_ORIGINS` | No | `*` | Allowed CORS origins (comma-separated) |
| `API_RATE_LIMIT_PER_MINUTE` | No | `100` | Rate limit per client per minute |

### Database

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | **Yes** | - | Database connection URL |
| `DATABASE_POOL_SIZE` | No | `10` | Connection pool size |
| `DATABASE_MAX_OVERFLOW` | No | `20` | Maximum overflow connections |
| `DATABASE_POOL_TIMEOUT` | No | `30` | Connection timeout in seconds |
| `DATABASE_ECHO` | No | `false` | Log SQL queries |

### Cache

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CACHE_BACKEND` | No | `redis` | Backend: `redis`, `memory` |
| `CACHE_URL` | No | `redis://localhost:6379/0` | Redis connection URL |
| `CACHE_TTL_SECONDS` | No | `3600` | Default cache TTL |
| `CACHE_MAX_SIZE_MB` | No | `1024` | Maximum cache size |
| `CACHE_KEY_PREFIX` | No | `flight:` | Cache key prefix |

### Processing

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MAX_CONCURRENT_JOBS` | No | `10` | Maximum concurrent processing jobs |
| `JOB_TIMEOUT_SECONDS` | No | `3600` | Job execution timeout |
| `TILE_SIZE` | No | `512` | Processing tile size in pixels |
| `DEFAULT_PROFILE` | No | `workstation` | Default compute profile: `laptop`, `workstation`, `cloud`, `edge` |
| `MAX_MEMORY_GB` | No | `8` | Maximum memory usage in GB |

### Storage

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STORAGE_BACKEND` | No | `local` | Backend: `local`, `s3` |
| `STORAGE_LOCAL_PATH` | No | `/data/firstlight_dive` | Local storage path |
| `STORAGE_S3_BUCKET` | No | - | S3 bucket name |
| `STORAGE_S3_REGION` | No | `us-east-1` | S3 region |
| `STORAGE_S3_ENDPOINT` | No | - | S3 endpoint (for custom S3-compatible) |

### Feature Flags

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENABLE_DISTRIBUTED` | No | `true` | Enable distributed processing |
| `ENABLE_GPU` | No | `false` | Enable GPU acceleration |
| `ENABLE_ASYNC_PROCESSING` | No | `true` | Enable async job processing |
| `ENABLE_QUALITY_CHECKS` | No | `true` | Enable quality control checks |
| `ENABLE_PROVENANCE_TRACKING` | No | `true` | Enable provenance tracking |
| `ENABLE_WEBHOOKS` | No | `true` | Enable webhook notifications |

### External Services

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STAC_ENDPOINTS` | No | Element84 | STAC catalog endpoints (comma-separated) |
| `PLANETARY_COMPUTER_API_KEY` | No | - | Microsoft Planetary Computer API key |
| `ERA5_API_KEY` | No | - | Copernicus ERA5 API key |
| `GFS_ENDPOINT` | No | NCEP | GFS forecast endpoint |

### Security

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JWT_SECRET_KEY` | **Yes** | - | JWT signing secret (32+ chars) |
| `JWT_ALGORITHM` | No | `HS256` | JWT algorithm |
| `JWT_EXPIRATION_HOURS` | No | `24` | JWT token expiration |
| `API_KEY` | No | - | API key for service auth |

### Monitoring

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENABLE_METRICS` | No | `true` | Enable Prometheus metrics |
| `METRICS_PORT` | No | `9090` | Metrics endpoint port |
| `ENABLE_TRACING` | No | `false` | Enable OpenTelemetry tracing |
| `TRACING_ENDPOINT` | No | - | OTLP collector endpoint |
| `SENTRY_DSN` | No | - | Sentry error tracking DSN |

### Agent Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AGENT_HEARTBEAT_SECONDS` | No | `30` | Agent heartbeat interval |
| `AGENT_MAX_RETRIES` | No | `3` | Maximum retry attempts |
| `AGENT_RETRY_DELAY_SECONDS` | No | `5` | Delay between retries |

## Secrets Management

**Never commit secrets to version control.** Use one of these methods:

### Option 1: Environment Variables

```bash
export JWT_SECRET_KEY="your-secure-secret"
export DATABASE_URL="postgresql://user:pass@host/db"
```

### Option 2: Secrets Manager (AWS)

```python
import boto3

client = boto3.client('secretsmanager')
secret = client.get_secret_value(SecretId='firstlight-dive/production')
```

### Option 3: Kubernetes Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: firstlight-secrets
type: Opaque
stringData:
  JWT_SECRET_KEY: "your-secure-secret"
  DATABASE_URL: "postgresql://user:pass@host/db"
```

### Option 4: HashiCorp Vault

```bash
vault kv get -field=jwt_secret secret/firstlight-dive/production
```

## Configuration Priority

Configuration is loaded in this order (later overrides earlier):

1. `base.env` - Default values
2. Environment-specific file (`production.env`, `development.env`)
3. Environment variables
4. Secrets manager (if configured)

## Validation

The application validates configuration on startup:

- Required variables must be present
- URLs must be valid
- Numeric values must be within expected ranges
- Security settings are enforced in production

Validation errors will prevent the application from starting with clear error messages.
