# Advanced Calibration Materialization Implementation Plan

> **For agentic workers:** Use the repository-local `subagent-driven-development`
> and `test-driven-development` skills. Do not use any `superpowers:*` skill.
> Execute tasks in order and keep one implementer on shared public files.

**Goal:** Materialize route-specific, source-verified Advanced RUL/SOH calibration
sample ToolResults, resolve them automatically for Agent runs, and expose readiness
through an Owner-facing Next.js UI without accepting business values from clients.

**Architecture:** An ADMIN creates an identity-only materialization request for one
active route. A Celery worker resolves the registered MATR evidence root, verifies
all source hashes, performs label-free inference for the frozen calibration cells,
combines predictions with server-owned supervision, and atomically persists the
sample ToolResults plus a cohort manifest. Agent execution resolves the exact READY
materialization from the current active runtime; the UI only displays status and
evidence summaries.

**Tech Stack:** Python 3.11, Pydantic v2, SQLAlchemy 2, Alembic, Celery/Redis,
FastAPI, PostgreSQL/SQLite migration tests, Next.js 16, React 19, TypeScript,
Vitest, Testing Library, Playwright.

**Approved design:** `docs/superpowers/specs/2026-07-27-advanced-calibration-materialization-design.md`

---

## File Ownership

Shared files are owned by the root Agent:

```text
src/quanxin_life/core/enums.py
src/quanxin_life/core/__init__.py
src/quanxin_life/persistence/models.py
src/quanxin_life/application/assembly.py
src/quanxin_life/api/app.py
src/quanxin_life/agents/supervisor.py
frontend/app/projects/[projectId]/page.tsx
frontend/lib/api-client.ts
frontend/lib/types.ts
frontend/app/globals.css
docs/superpowers/plans/2026-07-19-quanxin-product-completion-and-a100-integration.md
```

Fresh subagents may own new, isolated files only. No two active agents may edit the
same file.

## Task 0: Freeze the Existing Advanced-to-UI Vertical Slice

**Files:**

- Stage the existing Advanced prediction, Conformal, API, Agent, report, UI and
  test files already verified in the previous slice.
- Exclude `.gitignore`, `.playwright-mcp/`, `tmp/`, `output/` and
  `server-results/`.

- [ ] **Step 1: Verify the previous slice before staging**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/unit/application/test_advanced_prediction_services.py `
  tests/unit/application/test_advanced_split_conformal.py `
  tests/unit/tools/test_project_advanced_cycle_life_tool.py `
  tests/unit/tools/test_project_advanced_soh_tool.py `
  tests/unit/tools/test_project_advanced_split_conformal_tool.py `
  tests/unit/tools/test_project_advanced_cell_report_tool.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Stage only the previous slice**

Use explicit `git add -- <paths>` for the source, tests, frontend and main plan
files listed by `git status --short`. Do not use `git add .`.

- [ ] **Step 3: Review the staged boundary**

Run:

```powershell
git diff --cached --name-only
git diff --cached --check
```

Expected: no ignored result directories, `.gitignore`, `.playwright-mcp/` or
`tmp/`.

- [ ] **Step 4: Commit**

```powershell
git commit -m "feat: deliver advanced predictions through the project UI"
```

## Task 1: Add Materialization Status and Persistence Schema

**Files:**

- Modify: `src/quanxin_life/core/enums.py`
- Modify: `src/quanxin_life/core/__init__.py`
- Modify: `src/quanxin_life/persistence/models.py`
- Create: `migrations/versions/0014_advanced_calibration_materializations.py`
- Modify: `tests/unit/persistence/test_models.py`
- Modify: `tests/integration/test_database_migrations.py`
- Create: `tests/integration/test_advanced_calibration_materialization_migration.py`

- [ ] **Step 1: Write failing enum and metadata tests**

Add assertions equivalent to:

```python
assert {
    item.value for item in AdvancedCalibrationMaterializationStatus
} == {"PENDING", "RUNNING", "READY", "FAILED", "STALE"}

tables = Base.metadata.tables
assert "advanced_calibration_materializations" in tables
assert "advanced_calibration_sample_bindings" in tables
```

Verify parent columns include route/runtime/source hashes and state timestamps.
Verify child uniqueness for `(materialization_id, ordinal)`,
`(materialization_id, cell_id)` and `(materialization_id, result_id)`.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/unit/persistence/test_models.py `
  tests/integration/test_advanced_calibration_materialization_migration.py -q
```

Expected: fail because the enum and tables do not exist.

- [ ] **Step 3: Implement the minimal enum and ORM models**

Add:

```python
class AdvancedCalibrationMaterializationStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    READY = "READY"
    FAILED = "FAILED"
    STALE = "STALE"
```

Add `AdvancedCalibrationMaterialization` and
`AdvancedCalibrationSampleBinding` with exact foreign keys, check constraints,
unique constraints and UTC timestamps from the design.

- [ ] **Step 4: Implement migration 0014**

Migration requirements:

```text
revision = "0014"
down_revision = "0013"
```

Upgrade creates both tables and indexes. Downgrade must fail if either table
contains rows; otherwise drop child before parent.

- [ ] **Step 5: Run GREEN and migration round-trip**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/unit/persistence/test_models.py `
  tests/integration/test_advanced_calibration_materialization_migration.py `
  tests/integration/test_database_migrations.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add src/quanxin_life/core src/quanxin_life/persistence/models.py `
  migrations/versions/0014_advanced_calibration_materializations.py `
  tests/unit/persistence/test_models.py `
  tests/integration/test_database_migrations.py `
  tests/integration/test_advanced_calibration_materialization_migration.py
git commit -m "feat: persist advanced calibration materializations"
```

## Task 2: Implement the Trusted MATR Calibration Evidence Reader

**Files:**

- Create: `src/quanxin_life/application/advanced_calibration_evidence.py`
- Create: `tests/unit/application/test_advanced_calibration_evidence.py`

- [ ] **Step 1: Write failing source-validation tests**

Define wished-for contracts:

```python
class AdvancedCalibrationSourceRegistration(ContractModel):
    registration_id: str
    evidence_root: Path
    three_batch_manifest_sha256: str

class AdvancedCalibrationEvidenceResolver(Protocol):
    def resolve(
        self,
        registration_id: str,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
    ) -> AdvancedCalibrationEvidence: ...
```

Tests must prove:

- RUL labels come from non-censored official cycle life evidence.
- SOH observations come from verified supervision Parquet and stop at cycle 500.
- calibration cells exactly match the combined split.
- duplicate/missing cells, path escape, symlink escape, SHA mismatch, duplicate JSON
  keys, unsupported schema and non-finite values are rejected.
- the public evidence object contains immutable source identity hashes, not source
  paths.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/unit/application/test_advanced_calibration_evidence.py -q
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement strict JSON, manifest and Parquet readers**

Reuse structured Pydantic models from:

```text
src/quanxin_life/data/matr_pipeline.py
src/quanxin_life/data/matr_multibatch.py
src/quanxin_life/data/schemas.py
```

Do not import training models or torch. Read Parquet through PyArrow/Pandas only
inside the optional Advanced path. Use resolved paths and require every file to
remain under the registered root.

- [ ] **Step 4: Implement task-specific evidence**

Return immutable records:

```python
AdvancedRULCalibrationEvidence(
    source_identity=...,
    cells=(AdvancedRULObservedCell(cell_id=..., observed_cycle=...),),
)

AdvancedSOHCalibrationEvidence(
    source_identity=...,
    prediction_cycles=(...),
    cells=(AdvancedSOHObservedCell(cell_id=..., observed_soh=(...)),),
)
```

- [ ] **Step 5: Run GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/unit/application/test_advanced_calibration_evidence.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add src/quanxin_life/application/advanced_calibration_evidence.py `
  tests/unit/application/test_advanced_calibration_evidence.py
git commit -m "feat: verify advanced calibration evidence"
```

## Task 3: Produce Samples and Atomically Materialize a Cohort

**Files:**

- Create: `src/quanxin_life/application/advanced_calibration_materialization.py`
- Modify: `src/quanxin_life/audit/project_ledger.py`
- Modify: `src/quanxin_life/audit/sql_project_ledger.py`
- Modify: `src/quanxin_life/tools/advanced_conformal.py`
- Create: `tests/unit/application/test_advanced_calibration_materialization.py`
- Create: `tests/integration/application/test_advanced_calibration_materialization.py`

- [ ] **Step 1: Write failing producer tests**

Use a wished-for request that contains identities only:

```python
AdvancedCalibrationMaterializationRequest(
    project_id=project_id,
    task=AdvancedModelTask.RUL,
    cutoff_cycle=100,
    route_role=AdvancedModelRouteRole.COVERAGE,
    source_registration_id="matr-three-batch-final-v1",
)
```

Tests must reject any extra `cell_ids`, observed/predicted values, path, model
version or SHA fields through Pydantic `extra="forbid"`.

Verify RUL sample ToolResults use:

```text
tool_name = predict_cycle_life
artifact_type = quanxin_life.advanced_rul_calibration_sample.v1
split_partition = calibration
```

Verify SOH samples use the finite trajectory evidence type and axis.

- [ ] **Step 2: Run producer RED**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/unit/application/test_advanced_calibration_materialization.py -q
```

Expected: fail because the producer does not exist.

- [ ] **Step 3: Implement sample production**

`AdvancedCalibrationSampleProducer` must:

1. resolve the exact active runtime;
2. construct label-free input per calibration cell;
3. run the official safetensors runtime;
4. combine output with server-owned supervision;
5. attach OBSERVED and PREDICTED provenance;
6. re-resolve the active runtime after inference;
7. fail if any runtime identity field changed.

- [ ] **Step 4: Write failing atomicity/idempotency tests**

Tests must prove:

- a failure on sample N persists zero ToolResults and zero bindings;
- identical request/idempotency key returns the existing record;
- same exact route identity with different content conflicts;
- concurrent attempts produce one READY materialization;
- active route change marks old evidence unusable;
- project/result binding tampering is rejected.

- [ ] **Step 5: Run integration RED**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/integration/application/test_advanced_calibration_materialization.py -q
```

Expected: fail because atomic materialization is absent.

- [ ] **Step 6: Implement one-transaction persistence**

Add a dedicated SQL ledger method that writes:

```text
sample ToolResults
provenance rows
project-tool-result-binding-v1 rows
sample binding rows
sample manifest hash
READY materialization state
```

in one transaction. Do not loop over public `register_result()` calls.

- [ ] **Step 7: Harden Advanced Conformal sample decoding**

Replace loose dictionary selection with strict Pydantic sample evidence models.
Require exact source identity, outer ToolResult versions, runtime identity and
OBSERVED/PREDICTED provenance.

- [ ] **Step 8: Run GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/unit/application/test_advanced_calibration_materialization.py `
  tests/integration/application/test_advanced_calibration_materialization.py `
  tests/unit/tools/test_project_advanced_split_conformal_tool.py -q
```

Expected: pass.

- [ ] **Step 9: Commit**

```powershell
git add src/quanxin_life/application/advanced_calibration_materialization.py `
  src/quanxin_life/audit/project_ledger.py `
  src/quanxin_life/audit/sql_project_ledger.py `
  src/quanxin_life/tools/advanced_conformal.py `
  tests/unit/application/test_advanced_calibration_materialization.py `
  tests/integration/application/test_advanced_calibration_materialization.py
git commit -m "feat: materialize trusted advanced calibration samples"
```

## Task 4: Add Identity-Only Queueing, ADMIN API and Agent Context Resolution

**Files:**

- Create: `src/quanxin_life/application/advanced_calibration_jobs.py`
- Create: `src/quanxin_life/infrastructure/calibration_queue.py`
- Create: `src/quanxin_life/tasks/advanced_calibration.py`
- Modify: `src/quanxin_life/tasks/__init__.py`
- Create: `src/quanxin_life/api/advanced_calibration.py`
- Create: `src/quanxin_life/application/advanced_agent_execution_context.py`
- Modify: `src/quanxin_life/application/assembly.py`
- Modify: `src/quanxin_life/api/app.py`
- Create: `tests/unit/tasks/test_advanced_calibration_task.py`
- Create: `tests/unit/infrastructure/test_calibration_queue.py`
- Create: `tests/integration/api/test_advanced_calibration_api.py`
- Create: `tests/integration/application/test_advanced_agent_execution_context.py`

- [ ] **Step 1: Write queue and task RED tests**

The Celery payload must contain only:

```python
{"materialization_id": materialization_id}
```

No project, route, cell, label, prediction, SHA or file path may be sent through
Redis. The worker resolves all details from the database.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/unit/infrastructure/test_calibration_queue.py `
  tests/unit/tasks/test_advanced_calibration_task.py -q
```

Expected: fail because queue/task modules do not exist.

- [ ] **Step 3: Implement queue and worker**

Use task identity:

```text
quanxin_life.advanced_calibration.materialize.v1
```

Use `acks_late`, `reject_on_worker_lost`, idempotent claim and bounded retry for an
already-running materialization.

- [ ] **Step 4: Write API RED tests**

Cover:

- POST requires ADMIN, trusted origin and `Idempotency-Key`;
- MEMBER cannot POST;
- ADMIN/MEMBER can GET non-sensitive summaries;
- unknown/cross-project records are hidden;
- revoked session and inactive project are rejected;
- extra observed/predicted/path/hash fields yield 422;
- repeated key returns the existing materialization and queue receipt.

- [ ] **Step 5: Implement API adapter and app wiring**

Request:

```python
class CreateAdvancedCalibrationMaterializationRequest(ContractModel):
    task: AdvancedModelTask
    cutoff_cycle: Literal[20, 50, 100, 150]
    route_role: AdvancedModelRouteRole
    source_registration_id: str | None = None
```

Response exposes status and evidence summaries only.

- [ ] **Step 6: Write Agent resolver RED tests**

The resolver must:

- derive the target cutoff and cell from the bound record batch;
- resolve current RUL coverage and SOH active runtimes;
- find one exact READY materialization;
- revalidate every sample ToolResult/binding;
- reject a target cell in the calibration cohort;
- return stable result ID tuples by ordinal;
- reject missing, duplicate, STALE or tampered evidence.

- [ ] **Step 7: Implement Agent context resolver**

Populate only server-owned context references:

```text
context.rul_calibration_sample_result_ids
context.soh_calibration_sample_result_ids
```

Do not change the fixed Agent plan structure.

- [ ] **Step 8: Run GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/unit/infrastructure/test_calibration_queue.py `
  tests/unit/tasks/test_advanced_calibration_task.py `
  tests/integration/api/test_advanced_calibration_api.py `
  tests/integration/application/test_advanced_agent_execution_context.py `
  tests/unit/agents/test_execution_adapter.py `
  tests/unit/agents/test_supervisor.py -q
```

Expected: pass.

- [ ] **Step 9: Commit**

```powershell
git add src/quanxin_life/application/advanced_calibration_jobs.py `
  src/quanxin_life/application/advanced_agent_execution_context.py `
  src/quanxin_life/infrastructure/calibration_queue.py `
  src/quanxin_life/tasks `
  src/quanxin_life/api/advanced_calibration.py `
  src/quanxin_life/application/assembly.py `
  src/quanxin_life/api/app.py `
  tests/unit/infrastructure/test_calibration_queue.py `
  tests/unit/tasks/test_advanced_calibration_task.py `
  tests/integration/api/test_advanced_calibration_api.py `
  tests/integration/application/test_advanced_agent_execution_context.py
git commit -m "feat: expose advanced calibration readiness"
```

## Task 5: Build the Calibration Management UI

**Files:**

- Create: `frontend/app/projects/[projectId]/calibration/page.tsx`
- Create: `frontend/components/calibration-materialization-panel.tsx`
- Create: `frontend/components/calibration-materialization-contract.ts`
- Modify: `frontend/app/projects/[projectId]/page.tsx`
- Modify: `frontend/lib/api-client.ts`
- Modify: `frontend/lib/types.ts`
- Modify: `frontend/app/globals.css`
- Create: `frontend/tests/components/calibration-materialization-panel.test.tsx`
- Modify: `frontend/tests/lib/api-client.test.ts`

- [ ] **Step 1: Write component RED tests**

Tests must prove:

- READY, RUNNING, FAILED and STALE render distinct restrained states;
- ADMIN can trigger a legal active route;
- MEMBER sees no mutation control;
- request payload contains task/cutoff/route only;
- no observed/predicted values or sample IDs render;
- SHA values wrap and remain selectable;
- polling stops on READY/FAILED/STALE and on unmount;
- API failure does not display fabricated readiness.

- [ ] **Step 2: Run RED**

```powershell
cd frontend
node node_modules\vitest\vitest.mjs run `
  tests/components/calibration-materialization-panel.test.tsx
```

Expected: fail because the component does not exist.

- [ ] **Step 3: Implement typed client and strict decoder**

Add:

```typescript
export type AdvancedCalibrationMaterializationStatus =
  | "PENDING"
  | "RUNNING"
  | "READY"
  | "FAILED"
  | "STALE";
```

The API client must reject malformed status, task, route, SHA or timestamps.

- [ ] **Step 4: Implement the project page**

Use a dense work-focused table on desktop and unframed stacked rows on mobile.
Controls:

- route task/cutoff/role are fixed from server-listed active routes;
- one icon+text action starts missing evidence;
- no free-form business-number fields;
- tooltips explain status and SHA evidence;
- no cards nested inside cards.

- [ ] **Step 5: Run component and frontend gates**

```powershell
cd frontend
node node_modules\vitest\vitest.mjs run
node node_modules\typescript\bin\tsc --noEmit
node node_modules\eslint\bin\eslint.js . --max-warnings=0
node node_modules\next\dist\bin\next build
```

Expected: all pass and the build lists
`/projects/[projectId]/calibration`.

- [ ] **Step 6: Commit**

```powershell
git add frontend/app/projects/[projectId]/calibration `
  frontend/app/projects/[projectId]/page.tsx `
  frontend/components/calibration-materialization-panel.tsx `
  frontend/components/calibration-materialization-contract.ts `
  frontend/lib/api-client.ts frontend/lib/types.ts frontend/app/globals.css `
  frontend/tests/components/calibration-materialization-panel.test.tsx `
  frontend/tests/lib/api-client.test.ts
git commit -m "feat: manage calibration evidence in the project UI"
```

## Task 6: Complete the Vertical E2E and Quality Gates

**Files:**

- Create: `tests/e2e/test_project_advanced_calibration_workflow.py`
- Add or modify browser E2E under the existing frontend/browser test location.
- Modify:
  `docs/superpowers/plans/2026-07-19-quanxin-product-completion-and-a100-integration.md`

- [ ] **Step 1: Write the end-to-end RED test**

Use a persistent test database and source-verified MATR fixture to prove:

```text
ADMIN creates materialization
→ worker materializes RUL/SOH samples
→ Agent context resolves IDs
→ Conformal calibrates/issues
→ report binds results
→ UI API responses contain real result IDs
```

The E2E fixture must be explicitly test-only and must never appear in product UI as
real production evidence.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/e2e/test_project_advanced_calibration_workflow.py -q
```

Expected: fail at the first missing integration boundary.

- [ ] **Step 3: Add only the missing assembly wiring**

Wire the materialization services, queue, worker and Agent context provider through
the existing application factory. Do not add alternate in-memory production paths.

- [ ] **Step 4: Run targeted GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/e2e/test_project_advanced_calibration_workflow.py `
  tests/integration/api/test_advanced_calibration_api.py `
  tests/integration/application/test_advanced_agent_execution_context.py -q
```

Expected: pass.

- [ ] **Step 5: Run browser QA**

Verify desktop 1440x900 and mobile 390x844:

- no overlap or horizontal overflow;
- readable hashes;
- no business values in calibration management;
- READY/FAILED/STALE states are unambiguous;
- single-cell page remains fail-closed without final result IDs.

- [ ] **Step 6: Run full backend gates**

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp D:\qrt-calibration-final
.\.venv\Scripts\python.exe -m pytest tests/leakage -q
.\.venv\Scripts\python.exe -m pytest tests/integration -q
.\.venv\Scripts\python.exe -m pytest tests/e2e -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src\quanxin_life
.\.venv\Scripts\python.exe -m compileall -q src\quanxin_life
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip_audit
```

Report `pip-audit` as failed if the known diskcache/GitPython vulnerabilities remain.

- [ ] **Step 7: Run full frontend gates**

```powershell
cd frontend
node node_modules\vitest\vitest.mjs run
node node_modules\typescript\bin\tsc --noEmit
node node_modules\eslint\bin\eslint.js . --max-warnings=0
node node_modules\next\dist\bin\next build
```

- [ ] **Step 8: Update the main plan**

Mark the trusted calibration producer/importer and calibration-to-UI code chain
complete. Keep real product database migrations, 15-candidate registration, ADMIN
activation and production browser E2E unchecked until actually executed.

- [ ] **Step 9: Final diff and commit**

```powershell
git diff --check
git status --short
git add -- <only Task 6 files>
git commit -m "test: verify advanced calibration through the UI"
```

Do not push.
