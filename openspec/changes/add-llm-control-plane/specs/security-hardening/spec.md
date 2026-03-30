## ADDED Requirements

### Requirement: Tenant Isolation via Customer Scoping

Every `UserContext` object and every `APIKey` record SHALL carry a `customer_id` field. All database queries that touch job, event, checkpoint, escalation, or webhook records MUST be automatically scoped to the `customer_id` of the authenticated caller. A tenant isolation middleware layer SHALL enforce this scoping before any query reaches the data layer, making per-query manual filtering unnecessary and eliminating cross-tenant data leakage as a class of bug.

#### Scenario: Cross-tenant job access blocked

- **WHEN** an authenticated caller with `customer_id = "acme"` requests a job whose `customer_id = "beta"`
- **THEN** the response is `404 Not Found` (not `403`) so that existence of the resource is not disclosed

#### Scenario: Tenant scope propagates through query layer

- **WHEN** any SQL query is issued against a tenant-scoped table
- **THEN** a `WHERE customer_id = :caller_customer_id` clause is present in the query before execution

#### Scenario: API key carries customer identity

- **WHEN** an API key is created or looked up during authentication
- **THEN** the resolved `customer_id` from the API key record is attached to `UserContext` for the duration of the request

---

### Requirement: Authentication Enabled by Default

`AuthSettings.enabled` SHALL default to `True`. Any deployment where authentication has not been explicitly disabled via environment configuration MUST require a valid API key on every non-public endpoint. This is a **BREAKING** change from the current default of `disabled`. Operators who previously relied on the disabled default MUST set `AUTH_ENABLED=false` explicitly to preserve that behavior during a migration window.

#### Scenario: Unauthenticated request rejected without configuration

- **WHEN** a request arrives with no `Authorization` header and `AuthSettings.enabled` has not been set to `False`
- **THEN** the response is `401 Unauthorized`

#### Scenario: Explicit opt-out still works

- **WHEN** `AUTH_ENABLED=false` is set in the environment
- **THEN** requests without credentials are accepted (preserving backward compatibility for local dev)

#### Scenario: Health and metrics probes exempt

- **WHEN** a request targets a designated liveness or readiness probe path
- **THEN** no authentication is required regardless of `AuthSettings.enabled`

The following paths are exempt from authentication (health and readiness probes):
- `GET /health` — liveness probe; returns `200 OK` with JSON body containing service status
- `GET /ready` — readiness probe; returns `200 OK` if all dependencies (PostgreSQL, Redis) are reachable, or `503 Service Unavailable` if any dependency is unreachable

These exempt paths, combined with the OGC discovery paths below, form the complete auth exemption allowlist. All other paths require authentication when `AuthSettings.enabled` is `True`.

#### Scenario: OGC discovery endpoints exempt

- **WHEN** a request targets a public OGC discovery path (`GET /oapi/`, `GET /oapi/conformance`, `GET /oapi/processes`, `GET /oapi/processes/{id}`)
- **THEN** no authentication is required; these paths are added to the auth exemption allowlist alongside health probes. OGC execution and job management endpoints (`POST /oapi/processes/{id}/execution`, `GET /oapi/jobs/*`) still require authentication as defined in the OGC Auth Boundary requirement in the ogc-integration spec.

---

### Requirement: Infrastructure Credentials via Environment Injection

No default passwords, secret keys, or connection strings SHALL appear as literal values in `docker-compose.yml` or any committed configuration file. All infrastructure credentials (PostgreSQL password, Redis password, secret key material) MUST be supplied via environment variables or a referenced secrets file excluded from version control. The `docker-compose.yml` SHALL reference variable names only, with no fallback default values for secrets.

#### Scenario: Docker Compose contains no literal passwords

- **WHEN** `docker-compose.yml` is inspected
- **THEN** no field under `environment:` contains a literal password string; all secret fields reference `${VARIABLE_NAME}` with no default

#### Scenario: Missing credential variable fails fast

- **WHEN** a required credential environment variable is unset at startup
- **THEN** the service refuses to start and logs a clear message identifying the missing variable

---

### Requirement: Control Plane Permission Entries

The `Permission` enum SHALL include entries for all control plane operations: `state_read`, `state_write`, and `escalation_manage`. These permissions MUST be checked at the route layer before any state read, state transition, or escalation operation is performed. Role-to-permission mappings SHALL be defined such that read-only roles cannot perform state writes or manage escalations.

#### Scenario: State read permitted for read-only role

- **WHEN** a caller with only `state_read` permission requests job state
- **THEN** the request succeeds

#### Scenario: State write blocked for read-only role

- **WHEN** a caller with only `state_read` permission attempts a state transition
- **THEN** the response is `403 Forbidden`

#### Scenario: Escalation management requires escalation_manage

- **WHEN** a caller without `escalation_manage` attempts to create or resolve an escalation
- **THEN** the response is `403 Forbidden`

---

### Requirement: SSRF Protection on Webhook URLs

The webhook registration endpoint SHALL validate every submitted URL before persisting it. A URL MUST be rejected if it: uses any scheme other than `https`, resolves to a loopback address (127.0.0.0/8, ::1), resolves to an RFC 1918 private address range (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16), or resolves to a link-local range (169.254.0.0/16, fe80::/10). HTTP redirects MUST NOT be followed during webhook delivery; a redirect response is treated as a delivery failure.

#### Scenario: Private IP webhook rejected at registration

- **WHEN** a caller submits a webhook URL that resolves to a private RFC 1918 address
- **THEN** the registration request is rejected with `422 Unprocessable Entity` before any network connection is made to that address

#### Scenario: HTTP (non-TLS) webhook rejected

- **WHEN** a caller submits a webhook URL using the `http://` scheme
- **THEN** the registration request is rejected with `422 Unprocessable Entity`

#### Scenario: Redirect during delivery treated as failure

- **WHEN** a webhook delivery attempt receives a `301` or `302` response from the target
- **THEN** the delivery is recorded as failed and queued for retry; no follow-up request is made to the redirect target

---

### Requirement: Atomic State Transition Authorization (TOCTOU Protection)

State transitions MUST be performed as a single atomic `UPDATE ... WHERE state = :expected_state` statement. The system SHALL NOT read the current state in one query and then write the new state in a separate query. A transition that finds the record in an unexpected state (zero rows updated) MUST return `409 Conflict` to the caller. This pattern eliminates time-of-check/time-of-use races when multiple agents or retried requests attempt concurrent transitions on the same job.

#### Scenario: Concurrent transition — one wins, one loses

- **WHEN** two callers simultaneously attempt to transition the same job from `QUEUED` to `DISCOVERING`
- **THEN** exactly one transition succeeds with `200 OK` and the other receives `409 Conflict`

#### Scenario: Transition on already-advanced state rejected

- **WHEN** a caller attempts to transition a job from `QUEUED` to `DISCOVERING` but the job is already `DISCOVERING`
- **THEN** the response is `409 Conflict` with a message indicating the actual current state

#### Scenario: Successful transition returns new state

- **WHEN** an atomic transition update affects exactly one row
- **THEN** the response body includes the job's new `phase` and `status` values

---

### Requirement: GeoJSON Input Validation and Complexity Limits

AOI inputs submitted via the control API MUST be validated before processing. The system SHALL reject any GeoJSON geometry that: exceeds a configurable maximum coordinate pair count per polygon ring, contains coordinate values outside WGS84 bounds (longitude outside [-180, 180] or latitude outside [-90, 90]), or results in a geometry whose bounding box area exceeds a configurable maximum square-kilometer threshold. Validation MUST occur at the API boundary before the geometry is passed to the geospatial processing layer.

#### Scenario: Coordinate out of WGS84 bounds rejected

- **WHEN** an AOI is submitted with a longitude value of `181.0`
- **THEN** the response is `422 Unprocessable Entity` identifying the out-of-bounds coordinate

#### Scenario: Excessive vertex count rejected

- **WHEN** an AOI polygon ring contains more vertices than the configured maximum
- **THEN** the response is `422 Unprocessable Entity` before any database write occurs

#### Scenario: Area cap enforced

- **WHEN** an AOI bounding box exceeds the configured maximum area in square kilometers
- **THEN** the response is `422 Unprocessable Entity` with the actual vs. allowed area included in the error body

#### Scenario: Valid AOI passes through

- **WHEN** a well-formed GeoJSON MultiPolygon with coordinates within bounds and vertex count below the limit is submitted
- **THEN** the geometry is accepted and persisted without modification

---

### Requirement: Reasoning Field Sanitization

Reasoning text submitted by LLM agents to the control API SHALL be validated and sanitized before storage to prevent stored prompt injection. The system MUST enforce a maximum character length on the reasoning field. Content that contains null bytes or characters outside the expected Unicode range MUST be rejected. Stored reasoning values MUST be treated as opaque user-supplied text and MUST NOT be interpolated into any system prompt, log template, or structured command string after retrieval.

#### Scenario: Oversized reasoning payload rejected

- **WHEN** an LLM agent submits a reasoning string exceeding the configured maximum character limit
- **THEN** the response is `422 Unprocessable Entity` indicating the length constraint

#### Scenario: Null bytes in reasoning rejected

- **WHEN** a reasoning submission contains a null byte (`\x00`)
- **THEN** the response is `422 Unprocessable Entity`

#### Scenario: Stored reasoning not interpolated into prompts

- **WHEN** a previously stored reasoning value is retrieved from the database for display or audit
- **THEN** it is returned as a literal string value and is not passed as an instruction to any language model or command interpreter
