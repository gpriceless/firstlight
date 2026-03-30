## ADDED Requirements

### Requirement: pygeoapi ASGI Mount

The system SHALL mount pygeoapi as an ASGI sub-application at the `/oapi` path prefix, adapting it for embedding within the existing FastAPI application using a Starlette-compatible ASGI bridge.

The mounted sub-application MUST be isolated from the host application's routing such that `/oapi/*` paths are fully handled by pygeoapi and do not conflict with existing `/api/v1/*` routes.

#### Scenario: OGC API landing page is reachable

- **WHEN** a client sends `GET /oapi/`
- **THEN** pygeoapi returns a valid OGC API landing page with `200 OK` and a JSON body conforming to the OGC API - Common specification

#### Scenario: Host routes are unaffected by the mount

- **WHEN** pygeoapi is mounted and a client sends `GET /api/v1/status`
- **THEN** the existing FastAPI route handles the request normally without interference from the pygeoapi mount

---

### Requirement: Algorithm-to-Process Mapping

Each algorithm registered in the `AlgorithmRegistry` SHALL be exposed as a distinct OGC API Process by mapping it to a `BaseProcessor` subclass within the pygeoapi process registry.

The mapping MUST be established at application startup by iterating over the `AlgorithmMetadata` entries in the `AlgorithmRegistry` (via `registry.list_all()`) and using their existing fields to populate OGC process descriptions: `AlgorithmMetadata.id` becomes the process `id`, `AlgorithmMetadata.name` becomes the process `title`, `AlgorithmMetadata.description` becomes the process `description`, `AlgorithmMetadata.parameter_schema` and `AlgorithmMetadata.default_parameters` are used to derive process `inputs`, and `AlgorithmMetadata.outputs` populates process `outputs`. No manual process registration configuration file SHALL be required.

#### Scenario: Registered algorithm appears in process list

- **WHEN** a client sends `GET /oapi/processes`
- **THEN** each algorithm in the `AlgorithmRegistry` appears as a process in the response, with `id` matching the algorithm identifier

#### Scenario: Process description reflects algorithm metadata

- **WHEN** a client sends `GET /oapi/processes/{algorithm-id}`
- **THEN** the response contains `title`, `description`, `inputs`, and `outputs` derived from that algorithm's `AlgorithmMetadata` fields (`name`, `description`, `parameter_schema`, `outputs`)

---

### Requirement: Vendor Extension Fields

OGC API Process descriptions for FirstLight algorithms SHALL include vendor extension fields prefixed with `x-firstlight-` to expose LLM-specific metadata without violating OGC API spec compliance.

The following vendor extension fields MUST be present on each process description:

- `x-firstlight-reasoning` — string; describes what reasoning the LLM agent should apply when deciding whether to invoke this algorithm
- `x-firstlight-confidence` — object; describes how the algorithm's confidence score is computed and what thresholds are meaningful
- `x-firstlight-escalation` — boolean; indicates whether this algorithm's output can trigger an escalation to human review

#### Scenario: Vendor fields are present on process description

- **WHEN** a client sends `GET /oapi/processes/{algorithm-id}`
- **THEN** the JSON response contains `x-firstlight-reasoning`, `x-firstlight-confidence`, and `x-firstlight-escalation` fields alongside the standard OGC fields

#### Scenario: Vendor fields do not invalidate OGC conformance

- **WHEN** an OGC-compliant client validates the process description against the OGC API - Processes schema
- **THEN** validation passes, as vendor fields are treated as extension properties and do not conflict with required fields

---

### Requirement: Async Job Execution

The system SHALL support asynchronous OGC API Process execution via the `Prefer: respond-async` request header. Synchronous execution MUST be rejected for long-running algorithms.

The async execution lifecycle MUST implement the full OGC API - Processes async pattern:

1. Client submits job with `Prefer: respond-async`
2. System responds `201 Created` with a `Location` header pointing to the job status endpoint
3. Client polls `GET /oapi/jobs/{job-id}` for status
4. On completion, client retrieves results at `GET /oapi/jobs/{job-id}/results`

Job execution MUST be delegated to the Taskiq task queue and MUST NOT block the HTTP request thread.

#### Scenario: Async job is accepted and a status URL is returned

- **WHEN** a client posts to `POST /oapi/processes/{algorithm-id}/execution` with `Prefer: respond-async`
- **THEN** the system responds `201 Created` with a `Location` header of the form `/oapi/jobs/{job-id}`

#### Scenario: Job status reflects Taskiq execution state

- **WHEN** a client polls `GET /oapi/jobs/{job-id}` while the job is running
- **THEN** the response contains `"status": "running"` and an estimated completion time if available

#### Scenario: Completed job results are retrievable

- **WHEN** a client polls `GET /oapi/jobs/{job-id}` and status is `"successful"`
- **THEN** `GET /oapi/jobs/{job-id}/results` returns the algorithm output as a valid OGC API - Processes result document

#### Scenario: Synchronous execution is rejected for long-running algorithms

- **WHEN** a client posts to execute a long-running algorithm without the `Prefer: respond-async` header
- **THEN** the system responds `400 Bad Request` with a message instructing the client to use async execution

---

### Requirement: LLM Tool Schema Generation

The system SHALL auto-generate a JSON Schema tool definition for each algorithm from its `AlgorithmMetadata` in the `AlgorithmRegistry`. The tool `name` is derived from `AlgorithmMetadata.id`, the `description` from `AlgorithmMetadata.description`, and the `parameters` JSON Schema from `AlgorithmMetadata.parameter_schema`. These schemas MUST be consumable by LLM function-calling interfaces (OpenAI tool-use, Anthropic tool-use) without manual authoring.

The generated schema MUST include the algorithm identifier, a natural-language description, and a typed parameter schema derived from the algorithm's declared inputs and constraints.

#### Scenario: Tool schema is generated for each registered algorithm

- **WHEN** the application starts
- **THEN** each algorithm in the `AlgorithmRegistry` has a corresponding LLM tool schema available in memory, populated from its `AlgorithmMetadata` fields (`id`, `description`, `parameter_schema`)

#### Scenario: Tool schema is valid for LLM function-calling

- **WHEN** an LLM tool schema for an algorithm is submitted to an OpenAI-compatible tool-use API
- **THEN** the schema is accepted without validation errors, as all required fields (`name`, `description`, `parameters`) are present and correctly typed

---

### Requirement: State-Based Tool Filtering

The system SHALL filter the set of algorithm tool schemas presented to an LLM agent based on the current workflow stage of the job. Only algorithms relevant to the active stage SHALL be included in the tool list for any given LLM invocation.

The filtering MUST use the job's current `JobPhase` to select algorithms by their `AlgorithmMetadata.category` field (one of `baseline`, `advanced`, `experimental` from the `AlgorithmCategory` enum) and `AlgorithmMetadata.event_types`. Algorithms outside the active phase's category set MUST be excluded from the tool list.

#### Scenario: Tool list is scoped to the active workflow stage

- **WHEN** a job is in the `ANALYZING` phase and the LLM agent requests available tools
- **THEN** only algorithm tool schemas whose category maps to the analyzing stage are returned; discovery and reporting algorithms are excluded

#### Scenario: Tool list expands as the job advances

- **WHEN** a job transitions from `DISCOVERING` to `ANALYZING` phase
- **THEN** the tool list presented to the LLM agent updates to include analyzing-stage algorithms and exclude discovery-only algorithms

#### Scenario: An empty tool list is returned for unknown phases

- **WHEN** a job is in a phase with no mapped algorithms
- **THEN** the system returns an empty tool list rather than exposing all algorithms as a fallback

---

### Requirement: OGC Auth Boundary
The `/oapi/*` route prefix SHALL follow a split authentication model: discovery endpoints are public, execution endpoints require authentication. This aligns with the standard OGC API pattern where process and collection listings are openly discoverable but job submission and result retrieval are access-controlled.

**Public (no authentication required):**
- `GET /oapi/` — OGC API landing page
- `GET /oapi/conformance` — conformance declaration
- `GET /oapi/processes` — process listing
- `GET /oapi/processes/{id}` — individual process description

**Authenticated (valid API key or bearer token required):**
- `POST /oapi/processes/{id}/execution` — job submission
- `GET /oapi/jobs` — job listing
- `GET /oapi/jobs/{id}` — job status
- `GET /oapi/jobs/{id}/results` — job results
- `DELETE /oapi/jobs/{id}` — job dismissal

The host application's `TenantMiddleware` SHALL be configured with a path-prefix allowlist that exempts the public OGC discovery paths from authentication. pygeoapi's own auth configuration SHALL remain disabled; authentication is handled exclusively by the host middleware. Authenticated OGC endpoints MUST enforce tenant scoping via `customer_id` so that job listing and result retrieval only return records belonging to the authenticated caller.

#### Scenario: Process listing is publicly accessible
- **WHEN** an unauthenticated client sends `GET /oapi/processes`
- **THEN** the response is `200 OK` with the full process listing; no `401` is returned

#### Scenario: Job submission requires authentication
- **WHEN** an unauthenticated client sends `POST /oapi/processes/{id}/execution`
- **THEN** the response is `401 Unauthorized`

#### Scenario: Authenticated job submission is tenant-scoped
- **WHEN** an authenticated client submits a job via OGC execution and another authenticated client queries `GET /oapi/jobs`
- **THEN** each client sees only their own jobs; the OGC job listing is scoped by `customer_id`

#### Scenario: pygeoapi internal auth is disabled
- **WHEN** pygeoapi is mounted as an ASGI sub-application
- **THEN** pygeoapi's own `server.auth` configuration block is absent or set to no-op; all auth enforcement is delegated to the host FastAPI middleware
