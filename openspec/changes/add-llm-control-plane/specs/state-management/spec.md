## ADDED Requirements

### Requirement: Phase-Status State Model

The system SHALL represent job execution state using a two-level model: a coarse `JobPhase` enum (7 values) that maps to the existing `ExecutionStage`, and a fine-grained `JobStatus` that carries per-phase substates. Every job MUST have exactly one `(phase, status)` pair at any point in time. Valid `JobPhase` values are: `QUEUED`, `DISCOVERING`, `INGESTING`, `NORMALIZING`, `ANALYZING`, `REPORTING`, `COMPLETE`. `JobStatus` MUST refine the current phase (e.g., `DISCOVERING` phase carries `DISCOVERED` or `DISCOVERY_FAILED` status). The model SHALL be the single source of truth that replaces the 6 independent state enums currently spread across the codebase.

#### Scenario: LLM agent reads job progress

- **WHEN** an LLM agent polls a job via the control API
- **THEN** the response includes both `phase` and `status` fields, giving the agent coarse positioning (which pipeline stage) and fine-grained detail (what happened within that stage) without requiring knowledge of internal enums

#### Scenario: Existing pipeline stage advances

- **WHEN** the internal orchestrator advances an `ExecutionStage`
- **THEN** the `JobPhase` field updates to reflect the new stage, and `JobStatus` resets to the entry substate of that phase, preserving backward compatibility with existing CLI and agent code

#### Scenario: Phase-level failure

- **WHEN** a pipeline stage fails
- **THEN** `JobStatus` transitions to the failure substate for that phase (e.g., `INGESTION_FAILED`) while `JobPhase` remains at the failing phase, so the agent can identify where failure occurred without inspecting logs

#### ExecutionStage to JobPhase Mapping

The existing `ExecutionStage` enum (10 values) maps to the new `JobPhase` enum (7 values) as follows. The `SQLiteStateBackend` adapter (task 1.2) and the `PostGISStateBackend` (task 1.6) MUST implement this mapping consistently.

| ExecutionStage | JobPhase | Mapping Rationale |
|----------------|----------|-------------------|
| `PENDING` | `QUEUED` | Direct mapping. Job is waiting to start. |
| `VALIDATING` | `QUEUED` | Validation is a substatus of the queued phase. `JobStatus` = `VALIDATING`. |
| `DISCOVERY` | `DISCOVERING` | Direct mapping with verb-form rename. |
| `PIPELINE` | `INGESTING`, `NORMALIZING`, or `ANALYZING` | The coarse PIPELINE stage is split into three phases. The active sub-phase is determined by the pipeline's internal progress reported via `StageProgress.metrics`. Default to `INGESTING` on PIPELINE entry, advance to `NORMALIZING` then `ANALYZING` as sub-stages complete. |
| `QUALITY` | `ANALYZING` | Quality checks are a substatus of the analyzing phase. `JobStatus` = `QUALITY_CHECK`. |
| `REPORTING` | `REPORTING` | Direct mapping. |
| `ASSEMBLY` | `REPORTING` | Assembly is a substatus of the reporting phase (report compilation). `JobStatus` = `ASSEMBLING`. |
| `COMPLETED` | `COMPLETE` | Direct mapping with name normalization. |
| `FAILED` | _(any phase)_ | Terminal status, not a phase. `JobStatus` = `FAILED`. The `JobPhase` remains at the phase where failure occurred. |
| `CANCELLED` | _(any phase)_ | Terminal status, not a phase. `JobStatus` = `CANCELLED`. The `JobPhase` remains at the phase where cancellation occurred. |

#### Per-Phase Substatus Values

Each `JobPhase` has a defined set of valid `JobStatus` values. Any `(phase, status)` pair not in this table MUST be rejected by the state model.

| JobPhase | Valid JobStatus Values | Description |
|----------|----------------------|-------------|
| `QUEUED` | `PENDING`, `VALIDATING`, `VALIDATED`, `VALIDATION_FAILED` | Job is waiting or being validated before pipeline entry. |
| `DISCOVERING` | `DISCOVERING`, `DISCOVERED`, `DISCOVERY_FAILED` | Searching for satellite imagery matching the AOI and time window. |
| `INGESTING` | `INGESTING`, `INGESTED`, `INGESTION_FAILED` | Downloading and staging raw imagery. |
| `NORMALIZING` | `NORMALIZING`, `NORMALIZED`, `NORMALIZATION_FAILED` | Preprocessing, reprojection, and band alignment. |
| `ANALYZING` | `ANALYZING`, `QUALITY_CHECK`, `ANALYZED`, `ANALYSIS_FAILED` | Running detection algorithms and quality gates. |
| `REPORTING` | `REPORTING`, `ASSEMBLING`, `REPORTED`, `REPORTING_FAILED` | Generating products, assembling final deliverables. |
| `COMPLETE` | `COMPLETE` | Terminal success. No further transitions except to `CANCELLED`. |

In addition, `FAILED` and `CANCELLED` are universal terminal statuses valid in any phase. When a job enters `FAILED` or `CANCELLED`, the `JobPhase` is frozen at the phase where the terminal event occurred.

---

### Requirement: StateBackend Abstract Interface

The system SHALL define a `StateBackend` abstract base class that decouples pipeline state operations from the storage implementation. The ABC MUST declare the following methods: `get_state(job_id)`, `set_state(job_id, phase, status)`, `transition(job_id, expected_phase, expected_status, new_phase, new_status)`, `list_jobs(filters)`, and `checkpoint(job_id, payload)`. All methods MUST be async. The `transition` method MUST be atomic — it MUST only succeed if the job's current state matches `(expected_phase, expected_status)` at the moment of write, preventing TOCTOU races. Callers that depend on state consistency MUST use `transition` rather than `set_state`.

#### Scenario: Atomic transition prevents race condition

- **WHEN** two concurrent callers attempt to advance the same job from `(DISCOVERING, DISCOVERING)` to the next state simultaneously
- **THEN** exactly one caller succeeds and the other receives a `StateConflictError`, ensuring the job is never advanced twice

#### Scenario: Swap backend without changing callers

- **WHEN** the storage backend is swapped from `SQLiteStateBackend` to `PostGISStateBackend` via configuration
- **THEN** all orchestrator and CLI code continues to function without modification, because they interact only with the `StateBackend` interface

#### Scenario: List jobs with filters

- **WHEN** an operator queries `list_jobs(phase=ANALYZING, status=ANALYZING)`
- **THEN** only jobs in that exact state are returned, enabling queue inspection without full table scans

---

### Requirement: PostGIS State Backend

The system SHALL implement `PostGISStateBackend`, a concrete `StateBackend` backed by PostgreSQL with PostGIS. The backend MUST use `asyncpg` for all database access. The `transition` method MUST execute as a single atomic `UPDATE ... WHERE phase = :expected_phase AND status = :expected_status` statement — no separate SELECT before UPDATE. On successful transition the backend MUST immediately insert a row into the `job_events` audit table within the same database transaction. The backend MUST be the canonical state store once PostGIS migration is complete.

#### Scenario: Atomic transition in PostGIS

- **WHEN** `transition(job_id, DISCOVERING, DISCOVERING, DISCOVERING, DISCOVERED)` is called
- **THEN** a single SQL statement updates the row only if the current state matches, and on success inserts a `job_events` row recording the old state, new state, timestamp, and actor — all within one transaction

#### Scenario: Concurrent update safety

- **WHEN** the `UPDATE WHERE` matches zero rows (another caller won the race)
- **THEN** `PostGISStateBackend` raises `StateConflictError` and no event row is written

---

### Requirement: DualWrite Migration Backend

The system SHALL implement `DualWriteBackend`, a `StateBackend` that writes to PostGIS as canonical and SQLite as best-effort fallback during the migration period. All reads MUST come from PostGIS. Writes MUST be attempted on PostGIS first; if PostGIS write succeeds, the backend MUST attempt the same write on SQLite and MUST NOT fail the overall operation if the SQLite write fails — it SHALL log a warning instead. If the PostGIS write itself fails, the overall operation MUST fail. The `DualWriteBackend` MUST be selectable via configuration and MUST be removable without changing any caller code once migration is complete.

#### Scenario: PostGIS write succeeds, SQLite write fails

- **WHEN** a state transition is applied and PostGIS accepts it but the SQLite write raises an exception
- **THEN** the transition is considered successful, a warning is logged, and the caller receives a success result — SQLite degradation does not block the operation

#### Scenario: PostGIS write fails

- **WHEN** PostGIS is unavailable during a transition attempt
- **THEN** `DualWriteBackend` raises the PostGIS error immediately without attempting the SQLite write, because SQLite cannot be promoted to canonical

#### Scenario: Removing DualWrite after migration

- **WHEN** migration is complete and configuration is changed to point directly at `PostGISStateBackend`
- **THEN** all orchestrator and CLI code continues to function without modification

---

### Requirement: Job Checkpoint and Resume

The system SHALL support persisting intermediate job state to the `job_checkpoints` table so that a job interrupted mid-phase can resume from its last checkpoint rather than restarting from scratch. The `checkpoint(job_id, payload)` method on `StateBackend` MUST write a timestamped JSON payload keyed by `job_id`. On job resume, the orchestrator MUST read the latest checkpoint for that job and pass it to the pipeline stage as initial context. Checkpoints MUST be immutable once written — new checkpoints append rather than overwrite, and resume always uses the latest row.

#### Scenario: Pipeline restarts mid-analysis

- **WHEN** a job is interrupted during the `ANALYZING` phase and then resumed
- **THEN** the orchestrator reads the latest checkpoint for that job, restores algorithm progress from the payload, and continues from the interrupted point rather than re-running completed bands

#### Scenario: First run with no checkpoint

- **WHEN** a job is started for the first time and no checkpoint exists
- **THEN** the orchestrator starts the pipeline stage from the beginning with no error

#### Scenario: Checkpoint payload is preserved verbatim

- **WHEN** `checkpoint(job_id, {"bands_complete": ["B01", "B02"], "cursor": 1400})` is called
- **THEN** the exact payload is stored and returned unchanged on resume

---

### Requirement: State Transition Audit Trail

The system SHALL record every state transition in an append-only `job_events` table. Each row MUST capture: `job_id`, `event_type` (set to `STATE_TRANSITION`), `from_phase`, `from_status`, `to_phase`, `to_status`, `actor` (the component or agent that triggered the transition), and `occurred_at` (UTC timestamp). Rows MUST NOT be updated or deleted after insertion. The `job_events` table SHALL serve as both the audit log for compliance purposes and the replay buffer for the SSE event stream. Audit rows MUST be written in the same transaction as the state update — partial writes (state updated but no event row) are not permitted.

#### Scenario: Every transition is recorded

- **WHEN** a job advances through three phases during normal execution
- **THEN** three `STATE_TRANSITION` rows exist in `job_events` for that job, one per transition, each with accurate from/to state and timestamp

#### Scenario: Audit row missing on failed write

- **WHEN** the `job_events` INSERT fails (e.g., constraint violation)
- **THEN** the enclosing transaction rolls back, the state update is also reverted, and the job remains in its previous state — there is no state update without a corresponding audit row

#### Scenario: Audit trail supports replay

- **WHEN** an SSE consumer reconnects with a `Last-Event-ID` referencing a past event
- **THEN** the event stream replays all `job_events` rows for that job with `id > Last-Event-ID` in insertion order, using the audit table as the replay source
