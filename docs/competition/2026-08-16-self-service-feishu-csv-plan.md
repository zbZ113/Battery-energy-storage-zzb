# Self-Service Feishu CSV Intake Implementation Plan

**Goal:** Accept self-describing unseen CSV uploads without manual batch preparation and allow atomic multi-parameter Aily scenario changes.

**Architecture:** Extend the existing reviewed CSV normalizer with a strict metadata envelope, derive a deterministic registration for unseen hashes, and provision a frozen project binding from the already verified content batch. Keep the current model route and support checks authoritative. Aily continues to send full typed scenarios, with orchestration instructions allowing any subset of fields to change together.

**Tech Stack:** Python 3.11, Pydantic, SQLAlchemy, FastAPI/MCP, Celery.

## Task 1: Stable canonical identity and self-describing metadata

**Files:**
- Modify: `src/quanxin_life/application/battery_csv_mapping.py`
- Modify: `src/quanxin_life/application/feishu_aily_assembly.py`

- Add failing tests for canonical pass-through identity and unseen self-describing metadata.
- Add a strict metadata contract and normalize only exact reviewed headers/units.
- Build unseen registrations from verified metadata and SHA evidence; retain exact registry precedence.
- Run only the new test nodes.

## Task 2: Automatic project batch provisioning

**Files:**
- Modify: `src/quanxin_life/application/record_batch_bindings.py`
- Modify: `src/quanxin_life/application/feishu_project_models.py`
- Modify: `src/quanxin_life/application/feishu_aily_assembly.py`

- Add a failing test showing an unseen verified content batch currently lacks a frozen project binding.
- Add an idempotent context-authorized provision method that locks the project, reuses an exact binding, or creates and freezes one dataset and binding atomically.
- Invoke provisioning before project model execution; retain exact resolver verification.
- Run only the provisioning and executor test nodes.

## Task 3: Explicit unsupported-domain behavior

**Files:**
- Modify: `src/quanxin_life/application/feishu_project_models.py`
- Modify: `src/quanxin_life/integrations/feishu/jobs.py`

- Add a failing test for a complete non-MATR or unsupported-cutoff upload.
- Return a stable support-gate rejection before numerical model execution and avoid staging numerical sibling analyses for rejected roots.
- Run only the new support-gate tests.

## Task 4: Aily multi-parameter orchestration

**Files:**
- Modify only if required: `src/quanxin_life/api/aily.py`
- External configuration: existing Aily assistant instructions.

- Confirm the current typed request accepts simultaneous changes and up to eight comparisons.
- Add one focused contract test covering multiple changed fields in one request.
- Update Aily instructions to inherit unchanged values, apply all requested changes atomically, and create one new context per request.

## Task 5: Runtime deployment and focused verification

**Files:**
- Runtime only: API/Worker source deployment configuration.

- Deploy without reinstalling dependencies.
- Verify local/public health.
- Upload one registered cutoff CSV and one self-describing unseen-cell CSV.
- Confirm valid ToolResult-backed cards or an explicit support-gate rejection, plus a two-or-more-parameter Aily adjustment.
