# QA Deep Cleanse: multiverse_dive → FirstLight Rename

**QA Master Analysis**
**Date:** 2026-01-18
**Scope:** Complete codebase rename impact analysis
**Status:** CRITICAL - 250+ references found across 12 categories

---

## Executive Summary

This rename will impact:
- **92 files** with direct name references
- **70+ environment variables** with MULTIVERSE_ prefix
- **25+ Docker images** and container names
- **15+ Kubernetes resources** (namespaces, services, secrets)
- **20+ AWS resources** (SSM parameters, S3 buckets, SQS queues)
- **11 documentation files** (86+ references)
- **10 CI/CD workflow** references
- **Database names** in 8 different environments
- **API URLs** in 15+ locations
- **Cache/file paths** in 10+ locations

**RISK LEVEL: HIGH** - This rename will break deployments, CI/CD, and external integrations if not executed carefully.

---

## 1. Environment Variables (CRITICAL)

### Pattern: `MULTIVERSE_*` → `FIRSTLIGHT_*`

**Affected Variables:**
```bash
MULTIVERSE_VALIDATION_ENABLED
MULTIVERSE_VALIDATION_SCREENSHOT_DIR
MULTIVERSE_VALIDATION_SAMPLE_RATIO
MULTIVERSE_VALIDATION_MAX_TIME
MULTIVERSE_API_KEY
MULTIVERSE_API_URL
MULTIVERSE_DIVE_ENV
MULTIVERSE_DIVE_DATA_DIR
MULTIVERSE_DIVE_CACHE_DIR
MULTIVERSE_DIVE_MODEL_DIR
MULTIVERSE_DIVE_EDGE_MODE
MULTIVERSE_DIVE_GPU_ENABLED
MULTIVERSE_DIVE_OFFLINE_MODE
MULTIVERSE_DIVE_TELEMETRY
MULTIVERSE_DIVE_MAX_WORKERS
MULTIVERSE_DIVE_CACHE_SIZE_MB
MULTIVERSE_DIVE_CHUNK_SIZE_MB
MULTIVERSE_DIVE_ROOT
MULTIVERSE_VERSION
```

**Files Requiring Changes:**
- `/home/gprice/projects/multiverse_dive/core/data/ingestion/validation/config.py` (15 references)
- `/home/gprice/projects/multiverse_dive/api/config.py` (env_prefix setting)
- `/home/gprice/projects/multiverse_dive/docs/PRODUCTION_IMAGE_VALIDATION_REQUIREMENTS.md`
- `/home/gprice/projects/multiverse_dive/config/ingestion.yaml`
- `/home/gprice/projects/multiverse_dive/tests/integration/test_validation_integration.py`
- `/home/gprice/projects/multiverse_dive/.github/workflows/deploy-staging.yml`
- `/home/gprice/projects/multiverse_dive/.github/workflows/deploy-production.yml`
- `/home/gprice/projects/multiverse_dive/deploy/edge/README.md`
- `/home/gprice/projects/multiverse_dive/deploy/on-prem/cluster/ansible-playbook.yml`
- `/home/gprice/projects/multiverse_dive/deploy/edge/nvidia-jetson/Dockerfile.gpu`
- `/home/gprice/projects/multiverse_dive/deploy/on-prem/standalone/docker-compose.yml`
- `/home/gprice/projects/multiverse_dive/deploy/edge/arm64/Dockerfile.lightweight`
- `/home/gprice/projects/multiverse_dive/core/intent/registry.py`

**GOTCHA:** Runtime environment variable lookups via `os.environ.get()` will silently fail if old names are used!

---

## 2. Docker Images & Containers (CRITICAL)

### Pattern: `multiverse-dive-*` → `firstlight-*`

**Image Names:**
```
multiverse-dive-base:latest → firstlight-base:latest
multiverse-dive-api:latest → firstlight-api:latest
multiverse-dive-worker:latest → firstlight-worker:latest
multiverse-dive-cli:latest → firstlight-cli:latest
multiverse-dive-core:latest → firstlight-core:latest
multiverse-dive/api:* → firstlight/api:*
multiverse-dive/worker:* → firstlight/worker:*
multiverse-dive/edge-arm64:* → firstlight/edge-arm64:*
multiverse-dive/edge-jetson:* → firstlight/edge-jetson:*
```

**Container Names:**
```
mdive-api → firstlight-api
mdive-redis → firstlight-redis
mdive-postgres → firstlight-postgres
mdive-cli → firstlight-cli
mdive-core → firstlight-core
mdive-worker → firstlight-worker
multiverse-dive-api → firstlight-api
multiverse-dive-api-dev → firstlight-api-dev
multiverse-dive-worker-dev → firstlight-worker-dev
multiverse-dive-redis-commander → firstlight-redis-commander
multiverse-dive-pgadmin → firstlight-pgadmin
```

**Files Requiring Changes:**
- `/home/gprice/projects/multiverse_dive/docker-compose.yml` (7 references)
- `/home/gprice/projects/multiverse_dive/docker-compose.minimal.yml` (4 references)
- `/home/gprice/projects/multiverse_dive/deploy/docker-compose.yml` (6 references)
- `/home/gprice/projects/multiverse_dive/deploy/docker-compose.dev.yml` (6 references)
- `/home/gprice/projects/multiverse_dive/deploy/on-prem/standalone/docker-compose.yml` (4 references)
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/deployments/api.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/deployments/worker.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/gcp/cloud-run/service.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/gcp/kubernetes/deployment.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/azure/aks/deployment.yaml`
- `/home/gprice/projects/multiverse_dive/scripts/build-images.sh` (IMAGE_PREFIX variable)
- `/home/gprice/projects/multiverse_dive/scripts/push-images.sh` (IMAGE_PREFIX variable)
- `/home/gprice/projects/multiverse_dive/.github/workflows/deploy-staging.yml` (docker tag commands)
- `/home/gprice/projects/multiverse_dive/.github/workflows/deploy-production.yml` (docker tag commands)

**GOTCHA:** CI/CD pipelines tag images with old names during deployment!

---

## 3. Container Registry Paths (CRITICAL)

### Cloud Provider Registries

**Google Container Registry (GCR):**
```
gcr.io/${PROJECT_ID}/multiverse-dive-api:* → gcr.io/${PROJECT_ID}/firstlight-api:*
```
Files: `deploy/gcp/cloud-run/service.yaml`, `deploy/gcp/kubernetes/deployment.yaml`, `GUIDELINES.md`

**AWS Elastic Container Registry (ECR):**
```
${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/multiverse-dive-worker:*
${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/multiverse-dive-api:*
```
Files: `deploy/aws/batch/job-definition.json`, `deploy/aws/ecs/task-definition.json`

**Azure Container Registry (ACR):**
```
${ACR_LOGIN_SERVER}/multiverse-dive-api:* → ${ACR_LOGIN_SERVER}/firstlight-api:*
```
Files: `deploy/azure/aks/deployment.yaml`, `deploy/azure/container-instances/deployment.json`

**GOTCHA:** Existing images in cloud registries won't be automatically renamed!

---

## 4. Kubernetes Resources (CRITICAL)

### Namespace

**Current:** `multiverse-dive`
**Target:** `firstlight`

Files requiring namespace changes:
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/namespace.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/deployments/api.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/deployments/worker.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/services/api-service.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/services/api-lb.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/ingress.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/hpa.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/configmaps/app-config.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/secrets/app-secrets.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/persistentvolumes/data-pvc.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/gcp/kubernetes/deployment.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/azure/aks/deployment.yaml`

### Service Accounts
```
multiverse-dive-sa → firstlight-sa
multiverse-dive-sa@${PROJECT_ID}.iam.gserviceaccount.com → firstlight-sa@...
```

### ConfigMaps & Secrets
```
multiverse-dive-config → firstlight-config
multiverse-dive-secrets → firstlight-secrets
multiverse-secrets → firstlight-secrets
multiverse-dive-tls → firstlight-tls
```

### PersistentVolumeClaims
```
multiverse-dive-data → firstlight-data
```

### HPA Resources
```
multiverse-dive-api-hpa → firstlight-api-hpa
multiverse-dive-worker-hpa → firstlight-worker-hpa
```

**GOTCHA:** Changing namespaces will orphan existing K8s resources unless migrated!

---

## 5. Database Names (HIGH RISK)

### Pattern: `multiverse_dive` → `first_light`

**Database Names:**
```
multiverse                      (postgres user)
multiverse_dive                 (production DB)
multiverse_dive_dev             (development DB)
multiverse_dive_dev.db          (SQLite)
multiverse_dive_staging         (staging DB)
```

**Connection Strings:**
```
postgresql://multiverse:${POSTGRES_PASSWORD}@postgres:5432/multiverse
postgresql://staging_user:${STAGING_DB_PASSWORD}@postgres:5432/multiverse_dive_staging
postgresql://user:password@prod-db.cluster.region.rds.amazonaws.com:5432/multiverse_dive
sqlite:///./multiverse_dive_dev.db
```

**Files:**
- `/home/gprice/projects/multiverse_dive/docker-compose.yml` (POSTGRES_DB, POSTGRES_USER)
- `/home/gprice/projects/multiverse_dive/deploy/docker-compose.yml`
- `/home/gprice/projects/multiverse_dive/deploy/docker-compose.dev.yml`
- `/home/gprice/projects/multiverse_dive/deploy/on-prem/standalone/docker-compose.yml`
- `/home/gprice/projects/multiverse_dive/deploy/config/base.env`
- `/home/gprice/projects/multiverse_dive/deploy/config/staging.env`
- `/home/gprice/projects/multiverse_dive/deploy/config/development.env`
- `/home/gprice/projects/multiverse_dive/.github/workflows/deploy-production.yml` (backup/restore commands)

**GOTCHA:** Production database backup/restore scripts reference `multiverse-dive-postgres` container and `multiverse_dive` database!

---

## 6. API URLs & Domains (EXTERNAL BREAKING CHANGE)

### Pattern: `multiverse-dive.io` → `firstlight.io` (or similar)

**URLs:**
```
https://api.multiverse-dive.io/v1
https://app.multiverse-dive.io
https://admin.multiverse-dive.io
https://docs.multiverse-dive.io
support@multiverse-dive.io
staging.multiverse-dive.example.com
prod.multiverse-dive.example.com
api.multiverse-dive.example.com
```

**Files:**
- `/home/gprice/projects/multiverse_dive/docs/api/README.md` (15 references)
- `/home/gprice/projects/multiverse_dive/docs/api/examples/submit_event.py`
- `/home/gprice/projects/multiverse_dive/docs/api/examples/submit_event.sh`
- `/home/gprice/projects/multiverse_dive/docs/api/examples/webhook_handler.py`
- `/home/gprice/projects/multiverse_dive/deploy/config/production.env` (CORS origins)
- `/home/gprice/projects/multiverse_dive/api/models/errors.py` (documentation_url)
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/ingress.yaml` (host)
- `/home/gprice/projects/multiverse_dive/.github/workflows/deploy-staging.yml` (STAGING_HOST)
- `/home/gprice/projects/multiverse_dive/.github/workflows/deploy-production.yml` (PRODUCTION_HOST)

**GOTCHA:** Existing API clients with hardcoded URLs will break!

---

## 7. File Paths & Cache Directories (RUNTIME BREAKAGE)

### User Home Directory Paths

**Pattern:** `~/.multiverse_dive` → `~/.firstlight`

```
~/.multiverse_dive/screenshots/
~/.multiverse_dive/config/ingestion.yaml
~/.multiverse_dive/cache.db
~/.multiverse_dive/forecast_cache/
```

**Files:**
- `/home/gprice/projects/multiverse_dive/core/data/ingestion/validation/config.py` (4 references)
- `/home/gprice/projects/multiverse_dive/core/data/cache/manager.py`
- `/home/gprice/projects/multiverse_dive/core/analysis/forecast/ingestion.py`
- `/home/gprice/projects/multiverse_dive/config/ingestion.yaml`
- `/home/gprice/projects/multiverse_dive/docs/PRODUCTION_IMAGE_VALIDATION_REQUIREMENTS.md`
- `/home/gprice/projects/multiverse_dive/docs/IMAGE_VALIDATION_SUMMARY.md`
- `/home/gprice/projects/multiverse_dive/docs/IMAGE_VALIDATION_WORKFLOW.md`

### System Paths

```
/tmp/multiverse_dive/downloads
/tmp/multiverse_dive/products
/data/multiverse_dive
/opt/multiverse-dive
/opt/multiverse/data
/opt/multiverse/cache
/opt/multiverse/models
/etc/multiverse_dive/ingestion.yaml
```

**Files:**
- `/home/gprice/projects/multiverse_dive/agents/discovery/acquisition.py`
- `/home/gprice/projects/multiverse_dive/agents/reporting/main.py`
- `/home/gprice/projects/multiverse_dive/agents/reporting/products.py`
- `/home/gprice/projects/multiverse_dive/deploy/config/base.env` (STORAGE_LOCAL_PATH)
- `/home/gprice/projects/multiverse_dive/deploy/config/staging.env`
- `/home/gprice/projects/multiverse_dive/deploy/edge/README.md`
- `/home/gprice/projects/multiverse_dive/.github/workflows/deploy-staging.yml` (DEPLOY_DIR)

**GOTCHA:** Existing cached data will be orphaned in old directories!

---

## 8. AWS Resources (CRITICAL)

### SSM Parameter Store

**Pattern:** `/multiverse-dive/*` → `/firstlight/*`

```
/multiverse-dive/api-secret-key
/multiverse-dive/database-url
/multiverse-dive/stac-api-key
/multiverse-dive/s3-bucket
/multiverse-dive/redis-url
```

**Files:**
- `/home/gprice/projects/multiverse_dive/deploy/aws/lambda/serverless.yml` (8 references)
- `/home/gprice/projects/multiverse_dive/deploy/aws/batch/job-definition.json`
- `/home/gprice/projects/multiverse_dive/deploy/aws/ecs/task-definition.json`

### S3 Buckets

```
multiverse-dive-${stage} → firstlight-${stage}
```

### SQS Queues

```
multiverse-dive-events-${stage} → firstlight-events-${stage}
multiverse-dive-dlq-${stage} → firstlight-dlq-${stage}
```

### DynamoDB Tables

```
multiverse-dive-cache-${stage} → firstlight-cache-${stage}
```

### Secrets Manager

```
multiverse-dive/production → firstlight/production
```

**GOTCHA:** IAM policies with ARN patterns will need updating!

---

## 9. GitHub & Git References

### Repository URL

```
https://github.com/gpriceless/multiverse_dive
https://github.com/gpriceless/multiverse_dive/issues
```

**Files:**
- `/home/gprice/projects/multiverse_dive/README.md`
- `/home/gprice/projects/multiverse_dive/GUIDELINES.md`
- `/home/gprice/projects/multiverse_dive/docs/api/README.md`

**NOTE:** Actual repository rename is a separate operation!

---

## 10. Package & Module Names

### PyPI Package

**pyproject.toml** already updated:
- `name = "firstlight"` ✓
- `[project.scripts] flight = "cli.main:main"` ✓

**Documentation references:**
```
pip install multiverse-dive → pip install firstlight
npm install @multiverse/dive → npm install @firstlight/client
```

**Files:**
- `/home/gprice/projects/multiverse_dive/docs/api/README.md`

---

## 11. Configuration & Application Names

### Pattern: `multiverse-dive` → `firstlight`

```
APP_NAME=multiverse-dive
service: multiverse-dive
app_name = "multiverse_sedona"
```

**Files:**
- `/home/gprice/projects/multiverse_dive/deploy/config/base.env`
- `/home/gprice/projects/multiverse_dive/deploy/config/staging.env`
- `/home/gprice/projects/multiverse_dive/deploy/config/README.md`
- `/home/gprice/projects/multiverse_dive/deploy/aws/lambda/serverless.yml`
- `/home/gprice/projects/multiverse_dive/core/analysis/execution/sedona_backend.py` (Spark app names)

**Spark Application Names:**
```
multiverse_sedona → firstlight_sedona
multiverse_continental → firstlight_continental
multiverse_databricks → firstlight_databricks
```

---

## 12. Test Assertions & Test Data (TEST BREAKAGE)

### Test Files with Hardcoded Strings

```python
assert "Multiverse Dive" in result.output
assert config.app_name == "multiverse_sedona"
```

**Files:**
- `/home/gprice/projects/multiverse_dive/tests/test_cli.py` (2 assertions)
- `/home/gprice/projects/multiverse_dive/tests/test_sedona_execution.py` (1 assertion)

### Docker Network Names

```
multiverse-dive-network → firstlight-network
multiverse-net → firstlight-net
```

**Files:**
- `/home/gprice/projects/multiverse_dive/docker-compose.yml`
- `/home/gprice/projects/multiverse_dive/deploy/on-prem/cluster/ansible-playbook.yml`

---

## 13. GitIgnore & Test Patterns

### Pattern: `/tmp/mdive_*` → `/tmp/flight_*`

**File:**
- `/home/gprice/projects/multiverse_dive/.gitignore`

---

## 14. Documentation Files (86 references)

**Files requiring updates:**
1. `/home/gprice/projects/multiverse_dive/README.md` (5 refs)
2. `/home/gprice/projects/multiverse_dive/GUIDELINES.md` (9 refs)
3. `/home/gprice/projects/multiverse_dive/ROADMAP.md` (1 ref)
4. `/home/gprice/projects/multiverse_dive/PROJECT_STATUS.md` (2 refs)
5. `/home/gprice/projects/multiverse_dive/docs/api/README.md` (15 refs)
6. `/home/gprice/projects/multiverse_dive/docs/PRODUCTION_IMAGE_VALIDATION_REQUIREMENTS.md` (8 refs)
7. `/home/gprice/projects/multiverse_dive/docs/IMAGE_VALIDATION_SUMMARY.md` (3 refs)
8. `/home/gprice/projects/multiverse_dive/docs/IMAGE_VALIDATION_WORKFLOW.md` (2 refs)
9. `/home/gprice/projects/multiverse_dive/docs/OPENSPEC_ARCHIVE.md` (2 refs)
10. `/home/gprice/projects/multiverse_dive/deploy/config/README.md` (7 refs)
11. `/home/gprice/projects/multiverse_dive/deploy/edge/README.md` (32 refs)

---

## Non-Obvious Gotchas

### 1. Absolute Path in HTML Artifact
```
/home/gprice/projects/multiverse_dive/tests/e2e/artifacts/tile_viewer.html
```
Contains absolute path reference to project directory!

### 2. Deployment Scripts in CI/CD
GitHub Actions workflows contain embedded bash scripts with hardcoded:
- Container names for `docker exec` commands
- Deployment directory paths (`/opt/multiverse-dive`)
- Database backup/restore commands

### 3. Service Discovery
Kubernetes services use DNS like `multiverse-dive-api.multiverse-dive.svc.cluster.local`
Changing namespace breaks internal service discovery!

### 4. Docker Network Names
Services reference Docker networks by name - changing breaks inter-container communication in existing environments.

### 5. Cache Invalidation
Any code looking for cached data in old paths will think cache is empty and refetch everything.

### 6. Log Aggregation
If logs are shipped with `app_name=multiverse-dive` tag, dashboards/queries will break.

### 7. API Keys & Secrets
Environment variable names in secret management systems (AWS Secrets Manager, Vault) won't auto-update.

### 8. Webhook URLs
If external systems send webhooks to old domain, those will fail unless DNS/redirects maintained.

---

## Recommended Rename Strategy

### Phase 1: Prepare (No Breaking Changes)
1. Add NEW env vars alongside old (dual support)
2. Add NEW file paths alongside old (check both)
3. Update documentation
4. Add deprecation warnings for old names

### Phase 2: Infrastructure (Controlled Breakage)
1. Create new cloud resources (S3, SSM, DynamoDB) with new names
2. Deploy new K8s namespace alongside old
3. Build and push images with new names
4. Update CI/CD to tag both old and new images

### Phase 3: Code Cutover
1. Update all code to use new names
2. Update all configs to use new names
3. Keep backward compatibility where possible

### Phase 4: Deployment
1. Blue-green deploy to new resources
2. Migrate data (DB, cache) if needed
3. Update DNS/ingress to point to new services
4. Monitor for 24-48 hours

### Phase 5: Cleanup
1. Remove old environment variable support
2. Deprecate old cloud resources
3. Archive old K8s namespace
4. Remove old Docker images

---

## Testing Checklist

- [ ] Unit tests pass with new names
- [ ] Integration tests pass with new env vars
- [ ] Docker images build with new tags
- [ ] Docker Compose stack starts with new names
- [ ] K8s manifests apply to new namespace
- [ ] CI/CD pipeline runs end-to-end
- [ ] Staging deployment works
- [ ] Production rollback procedure tested
- [ ] Database migrations tested
- [ ] API endpoints respond on new domain
- [ ] Webhook delivery works
- [ ] Log aggregation captures new app names
- [ ] Metrics dashboards updated
- [ ] Documentation updated
- [ ] External integrations notified

---

## Files Requiring Changes: Complete List

### Python Code (15 files)
- `/home/gprice/projects/multiverse_dive/core/data/ingestion/validation/config.py`
- `/home/gprice/projects/multiverse_dive/core/data/cache/manager.py`
- `/home/gprice/projects/multiverse_dive/core/analysis/forecast/ingestion.py`
- `/home/gprice/projects/multiverse_dive/core/intent/registry.py`
- `/home/gprice/projects/multiverse_dive/core/analysis/execution/sedona_backend.py`
- `/home/gprice/projects/multiverse_dive/agents/discovery/acquisition.py`
- `/home/gprice/projects/multiverse_dive/agents/reporting/main.py`
- `/home/gprice/projects/multiverse_dive/agents/reporting/products.py`
- `/home/gprice/projects/multiverse_dive/api/config.py`
- `/home/gprice/projects/multiverse_dive/api/models/errors.py`
- `/home/gprice/projects/multiverse_dive/api/__init__.py`
- `/home/gprice/projects/multiverse_dive/agents/base.py`
- `/home/gprice/projects/multiverse_dive/tests/test_cli.py`
- `/home/gprice/projects/multiverse_dive/tests/test_sedona_execution.py`
- `/home/gprice/projects/multiverse_dive/tests/integration/test_validation_integration.py`

### Docker & Compose (14 files)
- `/home/gprice/projects/multiverse_dive/docker-compose.yml`
- `/home/gprice/projects/multiverse_dive/docker-compose.minimal.yml`
- `/home/gprice/projects/multiverse_dive/deploy/docker-compose.yml`
- `/home/gprice/projects/multiverse_dive/deploy/docker-compose.dev.yml`
- `/home/gprice/projects/multiverse_dive/deploy/on-prem/standalone/docker-compose.yml`
- `/home/gprice/projects/multiverse_dive/docker/base/Dockerfile`
- `/home/gprice/projects/multiverse_dive/docker/api/Dockerfile`
- `/home/gprice/projects/multiverse_dive/docker/worker/Dockerfile`
- `/home/gprice/projects/multiverse_dive/docker/cli/Dockerfile`
- `/home/gprice/projects/multiverse_dive/docker/core/Dockerfile`
- `/home/gprice/projects/multiverse_dive/deploy/Dockerfile`
- `/home/gprice/projects/multiverse_dive/deploy/edge/arm64/Dockerfile.lightweight`
- `/home/gprice/projects/multiverse_dive/deploy/edge/nvidia-jetson/Dockerfile.gpu`
- `/home/gprice/projects/multiverse_dive/.dockerignore`

### Kubernetes (12 files)
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/namespace.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/deployments/api.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/deployments/worker.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/services/api-service.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/services/api-lb.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/ingress.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/hpa.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/configmaps/app-config.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/secrets/app-secrets.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/kubernetes/persistentvolumes/data-pvc.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/gcp/kubernetes/deployment.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/azure/aks/deployment.yaml`

### Cloud Deployments (7 files)
- `/home/gprice/projects/multiverse_dive/deploy/aws/lambda/serverless.yml`
- `/home/gprice/projects/multiverse_dive/deploy/aws/batch/job-definition.json`
- `/home/gprice/projects/multiverse_dive/deploy/aws/ecs/task-definition.json`
- `/home/gprice/projects/multiverse_dive/deploy/gcp/cloud-run/service.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/azure/container-instances/deployment.json`
- `/home/gprice/projects/multiverse_dive/deploy/on-prem/cluster/ansible-playbook.yml`

### CI/CD (3 files)
- `/home/gprice/projects/multiverse_dive/.github/workflows/deploy-staging.yml`
- `/home/gprice/projects/multiverse_dive/.github/workflows/deploy-production.yml`
- `/home/gprice/projects/multiverse_dive/.github/workflows/docker-push.yml`

### Scripts (2 files)
- `/home/gprice/projects/multiverse_dive/scripts/build-images.sh`
- `/home/gprice/projects/multiverse_dive/scripts/push-images.sh`

### Configuration (7 files)
- `/home/gprice/projects/multiverse_dive/config/ingestion.yaml`
- `/home/gprice/projects/multiverse_dive/deploy/config/base.env`
- `/home/gprice/projects/multiverse_dive/deploy/config/staging.env`
- `/home/gprice/projects/multiverse_dive/deploy/config/production.env`
- `/home/gprice/projects/multiverse_dive/deploy/config/development.env`
- `/home/gprice/projects/multiverse_dive/.gitignore`

### Documentation (14 files)
- `/home/gprice/projects/multiverse_dive/README.md`
- `/home/gprice/projects/multiverse_dive/GUIDELINES.md`
- `/home/gprice/projects/multiverse_dive/ROADMAP.md`
- `/home/gprice/projects/multiverse_dive/PROJECT_STATUS.md`
- `/home/gprice/projects/multiverse_dive/docs/api/README.md`
- `/home/gprice/projects/multiverse_dive/docs/api/examples/submit_event.py`
- `/home/gprice/projects/multiverse_dive/docs/api/examples/submit_event.sh`
- `/home/gprice/projects/multiverse_dive/docs/api/examples/webhook_handler.py`
- `/home/gprice/projects/multiverse_dive/docs/PRODUCTION_IMAGE_VALIDATION_REQUIREMENTS.md`
- `/home/gprice/projects/multiverse_dive/docs/IMAGE_VALIDATION_SUMMARY.md`
- `/home/gprice/projects/multiverse_dive/docs/IMAGE_VALIDATION_WORKFLOW.md`
- `/home/gprice/projects/multiverse_dive/docs/OPENSPEC_ARCHIVE.md`
- `/home/gprice/projects/multiverse_dive/deploy/config/README.md`
- `/home/gprice/projects/multiverse_dive/deploy/edge/README.md`

### Test Data (1 file)
- `/home/gprice/projects/multiverse_dive/tests/e2e/artifacts/tile_viewer.html`

---

## TOTAL FILES: 89 unique files requiring changes

---

## Critical Path Items (MUST FIX)

1. **Environment variables in production** - Will break runtime
2. **Docker image names in CI/CD** - Will break deployments
3. **Database connection strings** - Will break data access
4. **K8s namespace** - Will orphan resources
5. **AWS SSM parameters** - Will break secret retrieval
6. **API base URLs** - Will break external integrations

---

## QA Master Recommendation

**STATUS: HOLD RENAME UNTIL MIGRATION PLAN APPROVED**

This rename has extensive blast radius across:
- 12 infrastructure categories
- 89 files
- 250+ individual references
- 5 cloud providers
- Multiple deployment environments

**Recommended approach:**
1. Create detailed migration plan with rollback procedures
2. Set up parallel infrastructure (new names alongside old)
3. Execute rename in non-production environments first
4. Staged production cutover with extensive monitoring
5. Maintain backward compatibility for 30-day transition period

**DO NOT proceed with bulk find-replace without:**
- [ ] Infrastructure team review
- [ ] DevOps approval
- [ ] Rollback plan tested
- [ ] Stakeholder notification
- [ ] Migration runbook prepared

---

**QA Master Sign-off Required Before Proceeding**
