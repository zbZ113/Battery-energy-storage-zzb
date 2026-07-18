# MATR Time Tolerance and A100 Runtime Hotfix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow only numerically insignificant MATR timestamp jitter while preserving strict rejection of real time reversal, and make the approved file-backed MLflow/headless A100 runtime automatic.

**Architecture:** The shared data validator remains strict by default and accepts an explicit tolerance parameter. MATR passes `1e-9 s` through its versioned curve configuration and preserves the resulting warning without changing telemetry. The A100 dataset entrypoint exports the approved runtime variables before preparation, preflight, or training.

**Tech Stack:** Python 3.11, Pydantic, pytest, Bash, Ruff, mypy, real MATR Parquet artifacts.

---

### Task 1: Timestamp tolerance contract

**Files:**
- Modify: `src/quanxin_life/data/validation.py`
- Modify: `tests/unit/data/test_validation.py`

- [x] **Step 1: Write failing validation tests**

Add tests constructing one-cell, one-cycle `CycleRecord` sequences that assert:

```python
report = validate_cycle_records(records)
assert "TIME_WITHIN_NUMERIC_TOLERANCE" in {issue.code for issue in report.issues}
assert "NON_MONOTONIC_TIME" not in {issue.code for issue in report.issues}
```

for equal time and a `-2.6e-11 s` reversal, plus:

```python
assert "NON_MONOTONIC_TIME" in {issue.code for issue in report.issues}
```

for a reversal larger than `1e-9 s`.

- [x] **Step 2: Run tests and observe RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/data/test_validation.py -q
```

Expected: tolerance cases fail because the current validator treats every non-increase as `NON_MONOTONIC_TIME`.

- [x] **Step 3: Implement the minimal tolerance**

Add an explicit `time_monotonic_tolerance_s` parameter with a strict `0.0` default. MATR supplies:

```python
TIME_MONOTONIC_TOLERANCE_S = 1e-9
```

When a positive tolerance is supplied, emit `NON_MONOTONIC_TIME` only when:

```python
current < previous - TIME_MONOTONIC_TOLERANCE_S
```

Otherwise, when `current <= previous`, emit warning code `TIME_WITHIN_NUMERIC_TOLERANCE`. With the default zero tolerance, every non-increase remains an error.

- [x] **Step 4: Verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/data/test_validation.py -q
.\.venv\Scripts\ruff.exe check src/quanxin_life/data/validation.py tests/unit/data/test_validation.py
.\.venv\Scripts\mypy.exe src/quanxin_life/data/validation.py
```

Expected: all commands succeed.

### Task 2: Curve tensor warning preservation

**Files:**
- Modify: `src/quanxin_life/features/curve_tensor.py`
- Modify: `tests/unit/features/test_curve_tensor.py`

- [x] **Step 1: Write a failing curve test**

Create valid discharge records with one equal timestamp and assert:

```python
tensor = build_discharge_curve_tensor(records, config=config)
assert "TIME_WITHIN_NUMERIC_TOLERANCE" in tensor.warnings
assert any(tensor.observed_mask)
```

- [x] **Step 2: Run the test and observe RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/features/test_curve_tensor.py -q
```

Expected: extraction no longer blocks after Task 1, but the warning is absent from `CurveTensor.warnings`.

- [x] **Step 3: Preserve the approved warning**

Collect `TIME_WITHIN_NUMERIC_TOLERANCE` from the validation report and add it to the existing tensor warning list. Do not change `time_s`, sample order, voltage, capacity, or interpolation.

- [x] **Step 4: Verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/features/test_curve_tensor.py -q
.\.venv\Scripts\ruff.exe check src/quanxin_life/features/curve_tensor.py tests/unit/features/test_curve_tensor.py
.\.venv\Scripts\mypy.exe src/quanxin_life/features/curve_tensor.py
```

Expected: all commands succeed.

### Task 3: A100 runtime environment

**Files:**
- Modify: `scripts/a100/train_dataset.sh`
- Modify: `tests/integration/test_a100_scripts.py`

- [x] **Step 1: Write a failing shell contract test**

Assert the script contains:

```text
MLFLOW_ALLOW_FILE_STORE=true
MPLBACKEND=Agg
PYTHONUNBUFFERED=1
unset DISPLAY
CUDA_VISIBLE_DEVICES=1
```

and that these declarations appear before `python scripts/run_training_suite.py`.

- [x] **Step 2: Run the test and observe RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_a100_scripts.py -q
```

Expected: fail because the three new runtime controls are not yet in the script.

- [x] **Step 3: Export the runtime controls**

Add before data preparation:

```bash
export MLFLOW_ALLOW_FILE_STORE=true
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
unset DISPLAY
```

Retain the existing physical GPU 1 binding.

- [x] **Step 4: Verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_a100_scripts.py -q
```

Expected: pass.

### Task 4: Real three-batch regression and hotfix archive

**Files:**
- Verify: `data/processed/MATR/*-cutoff150`
- Create generated artifact: `dist/quanxin-a100-smoke-hotfix.zip`

- [x] **Step 1: Run real affected-cell regression**

Scan all three registered batches through cutoff 150, identify every nonzero cycle with equal or slightly reversed time, and run all affected cells through `validate_cycle_records` and `build_discharge_curve_tensor`. All 49 affected cells must emit the tolerance warning, emit no `NON_MONOTONIC_TIME`, and retain at least one observed curve. The complete cohort regression exceeded the local command time limit; final end-to-end cohort assembly remains part of the resumed server Smoke.

- [x] **Step 2: Run focused and full quality gates**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/data/test_validation.py tests/unit/features/test_curve_tensor.py tests/integration/test_a100_scripts.py -q
.\.venv\Scripts\ruff.exe check src/quanxin_life/data/validation.py src/quanxin_life/features/curve_tensor.py tests/unit/data/test_validation.py tests/unit/features/test_curve_tensor.py
.\.venv\Scripts\mypy.exe src/quanxin_life/data/validation.py src/quanxin_life/features/curve_tensor.py
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Expected: all commands succeed.

- [x] **Step 3: Commit the hotfix**

Commit only the approved source, tests, Shell script, and plan updates:

```bash
git add src/quanxin_life/data/validation.py src/quanxin_life/features/curve_tensor.py src/quanxin_life/training/matr_data.py scripts/a100/train_dataset.sh tests/unit/data/test_validation.py tests/unit/features/test_curve_tensor.py tests/integration/test_a100_scripts.py docs/superpowers/plans/2026-07-18-matr-time-tolerance-a100-runtime-hotfix.md
git commit -m "fix: tolerate MATR timestamp precision jitter"
```

- [x] **Step 4: Build the code-only server archive**

Create a ZIP from the committed files:

```powershell
git archive --format=zip --output "dist/quanxin-a100-smoke-hotfix.zip" HEAD src/quanxin_life/data/validation.py src/quanxin_life/features/curve_tensor.py src/quanxin_life/training/matr_data.py scripts/a100/train_dataset.sh
```

Record its SHA-256 with `Get-FileHash`. The archive must contain no data, credentials, model artifacts, checkpoints, or `.git` metadata.

- [ ] **Step 5: Provide server recovery commands**

The server must verify the reported SHA-256, extract the archive over `/data/abd/z/AI-B`, reactivate `quanxin-a100`, and rerun:

```bash
bash scripts/a100/train_dataset.sh matr-three-batch smoke
```

The existing cutoff 20 and 50 results remain in place and are reused.
