# 泉芯智寿本地云端同构实施计划

> **执行约束：** 禁止使用任何 `superpowers:*` skill。实现阶段使用非 superpowers 的 `test-driven-development`、`executing-plans` 和 `quality-gate-runner`；步骤使用 checkbox 跟踪。未经用户明确授权不提交、不 push。

**Goal:** 在 Windows 本机以 Docker Desktop + WSL2 拉起与未来 ECS 同构的 PostgreSQL、Redis、Celery、Runtime V7、Next.js 和 Gateway，并通过浏览器完成一轮真实 MATR 九步预测及 ToolResult 纵向对账。

**Architecture:** 生产 `competition.compose.yaml`、HTTPS settings 和生产镜像入口保持不变；新增只允许 loopback HTTP 的 local profile、独立 local frontend/gateway 镜像和 `deploy/local.compose.yaml`。运行数据、镜像磁盘和 operator 目录优先放在 `D:\QuanxinRuntime`，样例与模型从现有 SHA-256 核验包构建，服务秘密在仓库外生成且不输出。

**Tech Stack:** Windows 11、WSL2、Docker Desktop、Docker Compose、Python 3.11、FastAPI、SQLAlchemy/Alembic、PostgreSQL 16 + pgvector、Redis 7.4 AOF、Celery、Next.js 16、pnpm 10、Nginx、Playwright。

---

## 文件边界

计划新增：

- `deploy/local_runtime_settings.py`：仅 loopback HTTP 的 settings。
- `deploy/local_api.py`、`deploy/local_worker.py`：显式 local entrypoint。
- `deploy/local.compose.yaml`、`deploy/local.env.example`：本地同构拓扑与非秘密环境模板。
- `deploy/nginx/local.conf`、`deploy/Dockerfile.gateway.local`：无证书、仅 loopback 发布的本地 Gateway。
- `frontend/Dockerfile.local`：显式 local profile 的生产式 Next.js standalone 构建。
- `scripts/local/prepare_runtime.ps1`：D 盘目录、秘密、digest lock 和 Compose 配置准备。
- `scripts/local/prepare_assets.py`：从已核验包构建本地 Runtime V7 数据、registry 和 calibration evidence。
- `scripts/local/bootstrap_admin.ps1`：隐藏输入并清除临时密码文件。
- `scripts/local/install_docker_desktop.ps1`：D 盘优先安装与安装后验证。
- `tests/integration/deploy/test_local_deployment_contract.py`：local/production 隔离和 Compose 静态契约。
- `tests/unit/deploy/test_local_runtime_settings.py`：loopback origin 与 secret/path 契约。
- `tests/unit/deploy/test_local_assets.py`：manifest/delta/样例资产准备。

计划修改：

- `deploy/competition_runtime.py`：增加默认仍为 production 的显式 cookie environment 参数。
- `deploy/Dockerfile.backend`：只在真实 Runtime V7 缺少依赖时补齐，不引入 PyBaMM。
- `frontend/lib/api-client.ts`：local profile + loopback HTTP 双条件。
- `frontend/tests/lib/api-client.test.ts`：生产拒绝 HTTP、本地仅允许 loopback。
- `tests/unit/deploy/test_competition_runtime.py`：production 默认和 local development cookie 断言。
- `tests/integration/deploy/test_competition_deployment_contract.py`：证明生产入口不引用 local profile。
- `.gitignore`：只追加仓库内可能生成的 local env/digest 文件；保留用户已有 `/server-results/` 修改。
- `CODEX_HANDOFF.md`：记录 ECS 已释放、本地验收状态、测试和下一步。

## Task 1：安装前门禁与 Docker Desktop/WSL2

**Files:**
- Create: `scripts/local/install_docker_desktop.ps1`
- Test: read-only PowerShell syntax parse and post-install commands

- [ ] **Step 1: 再次审计系统状态**

Run:

```powershell
Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsArchitecture,HyperVisorPresent
wsl --status
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
Get-PSDrive C,D | Select-Object Name,Free,Used
Get-Command docker -ErrorAction SilentlyContinue
```

Expected: D 盘可用空间不少于 100 GiB；记录哪些 Windows 功能或重启尚缺失，不输出密码。

- [ ] **Step 2: 编写安装脚本并先做语法检查**

脚本必须：

```powershell
param([string]$RuntimeRoot = 'D:\QuanxinRuntime')
$ErrorActionPreference = 'Stop'
$minimumFreeBytes = 100GB
$drive = Get-PSDrive -Name D
if ($drive.Free -lt $minimumFreeBytes) { throw 'D drive requires at least 100 GiB free' }
```

并使用 Docker Desktop 官方 installer 参数：

```text
install --quiet --accept-license
--installation-dir=D:\QuanxinRuntime\docker-desktop
--wsl-default-data-root=D:\QuanxinRuntime\docker-wsl
--windows-containers-default-data-root=D:\QuanxinRuntime\docker-windows
```

Run:

```powershell
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
  'scripts\local\install_docker_desktop.ps1', [ref]$null, [ref]$errors
) | Out-Null
if ($errors.Count) { $errors | Format-List; exit 1 }
```

Expected: exit 0。

- [ ] **Step 3: 启用 WSL2 并安装 Docker Desktop**

Run the script from an elevated prompt. Expected: 需要重启时脚本返回明确 `REBOOT_REQUIRED` 并停止；无需重启时 `docker version` 与 `docker compose version` 成功。

- [ ] **Step 4: 验证实际落盘位置**

Run:

```powershell
docker info
docker system df
Get-ChildItem 'D:\QuanxinRuntime\docker-wsl' -Force
Get-PSDrive C,D | Select-Object Name,Free,Used
```

Expected: Docker WSL 数据位于 D 盘；C 盘仅保留不可避免的 Windows 组件和小型配置。

## Task 2：Local settings 与 Runtime 认证边界

**Files:**
- Create: `deploy/local_runtime_settings.py`
- Create: `tests/unit/deploy/test_local_runtime_settings.py`
- Modify: `deploy/competition_runtime.py`
- Modify: `tests/unit/deploy/test_competition_runtime.py`

- [ ] **Step 1: 写 local origin 失败测试**

新增参数化测试，接受 `http://localhost:8080`、`http://127.0.0.1:8080`、`http://[::1]:8080`，拒绝 `http://192.168.1.2:8080`、`http://example.test`、带路径/用户名/密码的 origin。

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/deploy/test_local_runtime_settings.py -q
```

Expected: FAIL because `LocalCompetitionRuntimeSettings` does not exist。

- [ ] **Step 2: 实现独立 local settings**

实现结构：

```python
@dataclass(frozen=True, slots=True)
class LocalCompetitionRuntimeSettings(CompetitionRuntimeSettings):
    @classmethod
    def from_environment(cls, environment: Mapping[str, str]):
        return _runtime_settings_from_environment(
            cls,
            environment,
            origin_parser=_loopback_http_origin,
        )
```

生产 `CompetitionRuntimeSettings.from_environment` 继续调用 `_trusted_origin`，不得读取 `QUANXIN_RUNTIME_PROFILE`。

- [ ] **Step 3: 测试 production 默认 cookie 与 local development cookie**

为 `create_competition_runtime` 增加：

```python
def create_competition_runtime(
    settings: CompetitionRuntimeSettings,
    *,
    auth_environment: Literal['production', 'development'] = 'production',
) -> CompetitionRuntime:
```

测试 production 默认产生 `__Host-quanxin_session`，local 显式调用产生 `quanxin_dev_session`，且 production origin 仍必须 HTTPS。

- [ ] **Step 4: 运行定向门禁**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/deploy/test_runtime_settings.py tests/unit/deploy/test_local_runtime_settings.py tests/unit/deploy/test_competition_runtime.py -q
```

Expected: PASS。

## Task 3：Local API/Worker entrypoint

**Files:**
- Create: `deploy/local_api.py`
- Create: `deploy/local_worker.py`
- Modify: `tests/integration/deploy/test_competition_deployment_contract.py`

- [ ] **Step 1: 写生产隔离测试**

断言：

```python
assert 'LocalCompetitionRuntimeSettings' not in production_api
assert 'LocalCompetitionRuntimeSettings' not in production_worker
assert "auth_environment='development'" in local_api
assert "auth_environment='development'" in local_worker
```

Run test and confirm FAIL because local entrypoints are absent。

- [ ] **Step 2: 实现 local entrypoint**

两文件均使用：

```python
settings = LocalCompetitionRuntimeSettings.from_environment(os.environ)
runtime = create_competition_runtime(settings, auth_environment='development')
```

API 导出 `app = runtime.http_app`，Worker 导出 `app = runtime.celery_app`。

- [ ] **Step 3: 运行部署入口测试**

```powershell
.venv\Scripts\python.exe -m pytest tests/integration/deploy/test_competition_deployment_contract.py -q
```

Expected: PASS。

## Task 4：Frontend local profile

**Files:**
- Modify: `frontend/lib/api-client.ts`
- Modify: `frontend/tests/lib/api-client.test.ts`
- Create: `frontend/Dockerfile.local`

- [ ] **Step 1: 写 URL 解析失败测试**

目标签名：

```typescript
resolveApiBaseUrl(configuredUrl, environment, runtimeProfile)
```

断言 production + local profile 仍拒绝 HTTP，只有 `environment === 'production'` 且 `runtimeProfile === 'local'` 且 hostname 为 loopback 时允许本地容器构建；非 loopback HTTP 始终拒绝。

- [ ] **Step 2: 最小实现显式 local profile**

默认 profile 读取 `NEXT_PUBLIC_RUNTIME_PROFILE`，允许值仅为空或 `local`。`frontend/Dockerfile.local` 固定：

```dockerfile
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8080
ENV NEXT_PUBLIC_RUNTIME_PROFILE=local
RUN pnpm build
```

生产 `frontend/Dockerfile` 增加拒绝 `NEXT_PUBLIC_RUNTIME_PROFILE=local` 的构建检查。

- [ ] **Step 3: 运行前端定向门禁**

```powershell
pnpm --dir frontend test -- tests/lib/api-client.test.ts
pnpm --dir frontend typecheck
pnpm --dir frontend build
```

Expected: production build PASS；测试证明 local HTTP 例外不影响生产。

## Task 5：Local Gateway 与 Compose

**Files:**
- Create: `deploy/nginx/local.conf`
- Create: `deploy/Dockerfile.gateway.local`
- Create: `deploy/local.compose.yaml`
- Create: `deploy/local.env.example`
- Create: `tests/integration/deploy/test_local_deployment_contract.py`

- [ ] **Step 1: 写静态契约测试**

测试要求：gateway 只发布 `127.0.0.1:8080:80`；PostgreSQL/Redis/API/frontend 不发布端口；Redis 使用 AOF；migration 完成后 API/Worker 启动；local compose 使用 local entrypoints；production compose 不引用 local 文件。

- [ ] **Step 2: 实现 local gateway**

`local.conf` 包含 `/health`、`/v1/` SSE 无缓冲和 `/` frontend 代理，不含 TLS、ACME 或 308 HTTPS 跳转，并设置 `X-Forwarded-Proto http`。

- [ ] **Step 3: 实现 local Compose**

使用：

```yaml
postgres:
  image: pgvector/pgvector:0.8.1-pg16@${POSTGRES_IMAGE_DIGEST}
redis:
  image: redis:7.4.2-alpine@${REDIS_IMAGE_DIGEST}
api:
  build:
    context: ..
    dockerfile: deploy/Dockerfile.backend
  command: [python, -m, uvicorn, deploy.local_api:app, --host, 0.0.0.0, --port, '8000']
worker:
  command: [celery, -A, deploy.local_worker:app, worker, --queues, agent-runs,advanced-calibration, --concurrency, '1', --loglevel, INFO]
gateway:
  ports: ['127.0.0.1:8080:80']
```

- [ ] **Step 4: 验证 Compose 结构**

```powershell
docker compose --env-file D:\QuanxinRuntime\compose\local.env -f deploy\local.compose.yaml config
.venv\Scripts\python.exe -m pytest tests/integration/deploy/test_local_deployment_contract.py -q
```

Expected: config 成功且静态契约 PASS。

## Task 6：秘密、目录与 digest lock

**Files:**
- Create: `scripts/local/prepare_runtime.ps1`
- Modify: `.gitignore`

- [ ] **Step 1: 创建 D 盘 operator 目录并验证 ACL**

脚本创建规格定义的目录，使用 `icacls` 移除继承并仅保留当前用户、SYSTEM、Administrators 所需权限。不得输出 secret 内容。

- [ ] **Step 2: 生成服务秘密**

使用 `RandomNumberGenerator.Fill` 生成 PostgreSQL/Redis 随机凭据，写入 `D:\QuanxinRuntime\secrets`。生成 `database_url`、`redis_url` 和 Redis ACL 时只输出文件路径与 SHA-256，不输出文件内容。

- [ ] **Step 3: 拉取并锁定基础镜像 digest**

```powershell
docker pull pgvector/pgvector:0.8.1-pg16
docker pull redis:7.4.2-alpine
docker image inspect --format '{{index .RepoDigests 0}}' pgvector/pgvector:0.8.1-pg16
docker image inspect --format '{{index .RepoDigests 0}}' redis:7.4.2-alpine
```

只把 digest 写入 `D:\QuanxinRuntime\compose\local.env`，不写秘密。

- [ ] **Step 4: 验证仓库不包含 secret**

```powershell
git status --short
git diff --check
rg -n "postgresql://|redis://.*@|BEGIN PRIVATE KEY|POSTGRES_PASSWORD=" deploy scripts docs --glob '!*.example'
```

Expected: 不出现真实凭据；用户现有 `.gitignore` 内容保留。

## Task 7：准备 Runtime V7 真实资产

**Files:**
- Create: `scripts/local/prepare_assets.py`
- Create: `tests/unit/deploy/test_local_assets.py`

- [ ] **Step 1: 写 manifest 和 delta 失败测试**

测试覆盖：archive SHA 不符、manifest 条目缺失、路径穿越、DELTA weight SHA 不符、registry record SHA 不符、样例 CSV 与 registration `source_sha256` 不符时拒绝。

- [ ] **Step 2: 实现受治理资产准备**

输入固定为：

```text
server-results/deployment-handoff/quanxin-competition-demo-ready-2026.07.28-1.tgz
server-results/deployment-handoff/deployment-registry-provenance-v2
server-results/deployment-handoff/deployment-registry-runtime-v2-delta
```

脚本先验证相邻 SHA/`MANIFEST.sha256`，再安全解包到临时目录；按 `DELTA.json` 从 registry v2 复制或硬链接 15 个 `model.safetensors`，逐个校验 `weight_sha256`，最终安装 registry `c31f62e68faa66e56b16d21ebdd3067d5dea0c8408bb3ad6baa73e05a42824be`。同时安装 calibration evidence、`calibration-sources.json`、四个 `MATR_b3c34` CSV/registration 和 versioned Agent policy。

- [ ] **Step 3: 运行资产测试与真实 dry-run**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/deploy/test_local_assets.py -q
.venv\Scripts\python.exe scripts/local/prepare_assets.py --target D:\QuanxinRuntime\runtime --dry-run
```

Expected: 15 个模型权重、四个 cutoff 和 calibration evidence 全部核验，dry-run 不修改目标目录。

## Task 8：启动基础设施、migration 与管理员

**Files:**
- Create: `scripts/local/bootstrap_admin.ps1`

- [ ] **Step 1: 启动 PostgreSQL/Redis/migrate**

```powershell
docker compose --env-file D:\QuanxinRuntime\compose\local.env -f deploy\local.compose.yaml up -d postgres redis
docker compose --env-file D:\QuanxinRuntime\compose\local.env -f deploy\local.compose.yaml run --rm migrate
```

Expected: 两服务 healthy，migration exit 0，Alembic head 与源码一致。

- [ ] **Step 2: 交互式 bootstrap ADMIN**

脚本用 `Read-Host -AsSecureString` 接收临时密码，短暂写入 ACL 受限文件，运行 `python -m deploy.bootstrap_admin`，`finally` 删除文件。Codex 不读取终端输入或文件内容。

- [ ] **Step 3: 验证幂等与持久化**

重启 PostgreSQL/Redis 后 migration 仍成功；重复 bootstrap 同一未改密账户返回 `created=false`，不同账户拒绝。

## Task 9：样例项目、模型路由与校准物化

**Files:**
- Create: `deploy/bootstrap_local_sample.py`
- Test: `tests/postgres/test_local_sample_bootstrap.py`

- [ ] **Step 1: 写真实 PostgreSQL 幂等引导测试**

测试创建一个项目、一个 MATR 数据集、四个 frozen record batch，绑定同一 `MATR_b3c34` 的 cutoff 20/50/100/150；重复运行不生成重复 identity。不得写入 RUL/SOH/Conformal 结果。

- [ ] **Step 2: 实现引导模块**

模块读取四个 registration JSON 和 CSV，通过现有 canonical ingestion、`ProjectService`、`DatasetService`、`RecordBatchBindingService` 和 model catalog/route service 建立正式记录；不能直接拼 SQL 绕过契约。

- [ ] **Step 3: 启动 API/Worker/frontend/gateway**

```powershell
docker compose --env-file D:\QuanxinRuntime\compose\local.env -f deploy\local.compose.yaml up -d api worker frontend gateway
docker compose --env-file D:\QuanxinRuntime\compose\local.env -f deploy\local.compose.yaml ps
```

Expected: 所有长期服务 healthy，浏览器入口为 `http://127.0.0.1:8080`。

## Task 10：真实九步浏览器验收与纵向对账

**Files:**
- Create: `docs/testing/local-cloud-parity-acceptance-2026-08-03.md`
- Create: `output/playwright/local-cloud-parity-*`（仅在确认后决定是否提交）

- [ ] **Step 1: 浏览器完成真实流程**

登录、强制改密、进入项目、选择 `MATR_b3c34` 和 cutoff、启动全新九步 Agent，观察 Worker 日志与页面状态直到终态。

- [ ] **Step 2: ToolResult 纵向对账**

逐项核对页面、API、PostgreSQL：run ID、9 个 step、结果 ID、输入 SHA、数据/特征/模型版本、RUL/SOH/Conformal/report 上游关系。只记录身份和一致性，不把业务数值抄进计划或日志。

- [ ] **Step 3: 重启恢复检查**

重启 Worker/API/Redis/PostgreSQL，确认已完成 run 和 ToolResult 仍可读取，Redis AOF 无 ACL `NOPERM`，未完成任务有明确恢复或失败状态。

- [ ] **Step 4: 浏览器 QA**

验证 1440、1024、390 宽度下无重叠；SSE 断线重连、刷新、空态、失败态、下载不可用状态均真实。

## Task 11：完整门禁、交接与服务器复用说明

**Files:**
- Modify: `CODEX_HANDOFF.md`
- Create: `docs/deployment/local-to-ecs-release.md`

- [ ] **Step 1: 运行完整门禁**

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src deploy scripts
.venv\Scripts\python.exe -m compileall -q src deploy scripts migrations
pnpm --dir frontend test
pnpm --dir frontend lint
pnpm --dir frontend typecheck
pnpm --dir frontend build
docker compose --env-file D:\QuanxinRuntime\compose\local.env -f deploy\local.compose.yaml config
```

Expected: 全部通过；如存在 skip，逐项说明原因，PyBaMM 不得被误报为已产品化。

- [ ] **Step 2: 写 local-to-ECS 发布说明**

明确服务器不上传 D 盘、local secrets 或 local profile；服务器构建/拉取相同源码 release 的不可变镜像，重新生成 secrets，配置 Linux operator 目录和 TLS，执行 migration、bootstrap/restore、healthcheck、真实浏览器验收和回滚演练。

- [ ] **Step 3: 更新交接并重读**

记录 ECS 已释放、旧 IP 无效、真实本地服务状态、测试结果、已完成/未完成、Task 5/6/7/8 和 PyBaMM 后续边界。重新完整读取，扫描乱码、旧 IP、旧 release 和矛盾。

- [ ] **Step 4: Git 边界检查**

```powershell
git status --short
git diff --stat
git diff
git diff --cached
git diff --check
```

只在用户明确授权后精确 `git add`/commit；始终排除 `.playwright-mcp/`、`tmp/`、`frontend/pnpm-workspace.yaml` 和未确认截图。用户负责 push。
