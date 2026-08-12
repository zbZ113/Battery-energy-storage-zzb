# Feishu Worker Egress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Feishu analysis worker outbound DNS and HTTPS access while preserving internal database isolation and zero published worker ports.

**Architecture:** Keep the existing `backend` internal network for PostgreSQL and Redis, and attach the worker to the existing `edge` network for outbound Feishu API traffic. Apply the topology consistently to local and production Compose files and lock it with parsed YAML deployment-contract tests.

**Tech Stack:** Docker Compose, Python 3.11, PyYAML, pytest, Ruff

---

### Task 1: Lock the Worker Network Contract

**Files:**
- Modify: `tests/integration/deploy/test_competition_deployment_contract.py`
- Modify: `tests/integration/deploy/test_local_deployment_contract.py`

- [ ] **Step 1: Add the failing production deployment test**

Add `import yaml` and this test to `tests/integration/deploy/test_competition_deployment_contract.py`:

```python
def test_competition_worker_has_outbound_network_without_published_ports() -> None:
    compose = yaml.safe_load(_read("deploy/competition.compose.yaml"))
    worker = compose["services"]["worker"]

    assert set(worker["networks"]) == {"backend", "edge"}
    assert "ports" not in worker
    assert compose["networks"]["backend"]["internal"] is True
```

- [ ] **Step 2: Add the failing local deployment test**

Add `import yaml` and this test to `tests/integration/deploy/test_local_deployment_contract.py`:

```python
def test_local_worker_has_outbound_network_without_published_ports() -> None:
    compose = yaml.safe_load(_read("deploy/local.compose.yaml"))
    worker = compose["services"]["worker"]

    assert set(worker["networks"]) == {"backend", "edge"}
    assert "ports" not in worker
    assert compose["networks"]["backend"]["internal"] is True
```

- [ ] **Step 3: Run the focused tests and observe RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\integration\deploy\test_competition_deployment_contract.py::test_competition_worker_has_outbound_network_without_published_ports `
  tests\integration\deploy\test_local_deployment_contract.py::test_local_worker_has_outbound_network_without_published_ports `
  -q
```

Expected: both tests fail because each worker currently has only `backend`.

### Task 2: Add the Existing Edge Network to Both Workers

**Files:**
- Modify: `deploy/competition.compose.yaml:181`
- Modify: `deploy/local.compose.yaml:180`

- [ ] **Step 1: Make the minimal Compose changes**

Change each worker network list from:

```yaml
    networks:
      - backend
```

to:

```yaml
    networks:
      - backend
      - edge
```

Do not add a `ports` section or modify any other service.

- [ ] **Step 2: Run the focused tests and observe GREEN**

Run the same two-test command from Task 1.

Expected: `2 passed`.

- [ ] **Step 3: Run deployment contract regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\integration\deploy\test_competition_deployment_contract.py `
  tests\integration\deploy\test_local_deployment_contract.py `
  -q
```

Expected: all tests pass.

- [ ] **Step 4: Run static and Compose checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check `
  tests\integration\deploy\test_competition_deployment_contract.py `
  tests\integration\deploy\test_local_deployment_contract.py

docker compose `
  --env-file "D:\QuanxinRuntime\compose\local.env" `
  --env-file "D:\QuanxinRuntime\compose\feishu-aily.env" `
  -f .\deploy\local.compose.yaml `
  -f .\deploy\competition.feishu-aily.override.yaml `
  -f "D:\QuanxinRuntime\compose\full-feishu.override.yaml" `
  config --quiet

git diff --check -- `
  deploy/competition.compose.yaml `
  deploy/local.compose.yaml `
  tests/integration/deploy/test_competition_deployment_contract.py `
  tests/integration/deploy/test_local_deployment_contract.py
```

Expected: all commands exit with code 0.

### Task 3: Verify the Real Worker Egress and Fail-Closed CSV Path

**Files:**
- Runtime-only: `D:\QuanxinRuntime\compose\full-feishu.override.yaml`
- Runtime-only: `D:\QuanxinRuntime\runtime\config\feishu-csv-registrations.json`

- [ ] **Step 1: Remove the temporary runtime-only worker network override if present**

Ensure the runtime override does not define `worker.networks`; the checked-in local Compose now owns the topology.

- [ ] **Step 2: Recreate only the worker**

Run:

```powershell
docker compose `
  --env-file "D:\QuanxinRuntime\compose\local.env" `
  --env-file "D:\QuanxinRuntime\compose\feishu-aily.env" `
  -f .\deploy\local.compose.yaml `
  -f .\deploy\competition.feishu-aily.override.yaml `
  -f "D:\QuanxinRuntime\compose\full-feishu.override.yaml" `
  up -d --no-build worker
```

- [ ] **Step 3: Verify DNS and HTTPS from the worker**

Run:

```powershell
docker exec quanxin-life-local-worker-1 python -c `
  "import socket; print(socket.gethostbyname('open.feishu.cn'))"

docker exec quanxin-life-local-worker-1 python -c `
  "import httpx; print(httpx.get('https://open.feishu.cn', timeout=10).status_code)"
```

Expected: DNS prints an IP address and HTTPS prints an HTTP status instead of a transport exception.

- [ ] **Step 4: Confirm the registration registry remains empty**

Run:

```powershell
.\.venv\Scripts\python.exe -c `
  "from pathlib import Path; from deploy.competition_inputs import load_feishu_csv_registrations; print(len(load_feishu_csv_registrations(Path(r'D:\QuanxinRuntime\runtime\config\feishu-csv-registrations.json'))))"
```

Expected: `0`.

- [ ] **Step 5: Send the controlled connectivity CSV once**

Send `.test-tmp/feishu-upload/canonical-connectivity-test.csv` in the approved Feishu test group without additional text.

Expected: callback returns HTTP 200; Worker downloads and validates the file; exact SHA registration lookup rejects it with `DATA_REGISTRATION_REJECTED`; no prediction or scenario tool executes; the bot sends a rejection card and Bitable receives scalar metadata only.

- [ ] **Step 6: Verify persisted terminal state and logs**

Query the latest `feishu_event_receipts` row and inspect API/Worker logs. Expected terminal state:

```text
job_status=REJECTED
job_stage=REJECTED
job_last_error_code=DATA_REGISTRATION_REJECTED
analysis_result_id=NULL
```

### Task 4: Commit the Reviewed Fix

**Files:**
- Modify: `deploy/competition.compose.yaml`
- Modify: `deploy/local.compose.yaml`
- Modify: `tests/integration/deploy/test_competition_deployment_contract.py`
- Modify: `tests/integration/deploy/test_local_deployment_contract.py`

- [ ] **Step 1: Review the exact diff and stage only four files**

Run:

```powershell
git diff -- `
  deploy/competition.compose.yaml `
  deploy/local.compose.yaml `
  tests/integration/deploy/test_competition_deployment_contract.py `
  tests/integration/deploy/test_local_deployment_contract.py

git add -- `
  deploy/competition.compose.yaml `
  deploy/local.compose.yaml `
  tests/integration/deploy/test_competition_deployment_contract.py `
  tests/integration/deploy/test_local_deployment_contract.py

git diff --cached --check
```

- [ ] **Step 2: Create a local commit without pushing**

Run:

```powershell
git commit -m "fix(deploy): allow audited worker egress"
```

Expected: one local commit containing only the four reviewed files.
