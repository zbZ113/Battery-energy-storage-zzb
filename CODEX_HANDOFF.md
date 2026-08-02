# Codex 完整交接：泉芯智寿产品化

> 更新时间：2026-08-02 18:01:42 +08:00（Asia/Shanghai）
> 当前实施目标：`docs/superpowers/plans/2026-08-02-quanxin-productization-master-plan.md`
> 当前公网入口：`https://47.99.69.138`
> 本文件只记录可由当前 Git、代码、测试和只读探测支持的事实，不包含密码、私钥、Cookie、连接串或业务数值。

## 0. 一句话状态

Task 0、Task 1 和 Task 2 已完成：可信计算链、固定九步 Agent、B「双栏分析」前端、Runtime V7 修复、真实 PostgreSQL CI 与不可变镜像基线已经形成。提交 `daca47a` 的 Python、PostgreSQL 和前端 CI run `30738774242` 全绿；ACR release `2026.08.02-1` 已由 run `30739080847` 成功发布五张镜像并记录 digest。2026-08-02 ECS 只读审计已经补齐内部事实：migration `0015`、Worker、严格 TLS 和历史九步 ToolResult 可用，但 ECS 仍运行 `2026.07.29-1`、旧 digest、空 OCI 身份和 Runtime V7/Gateway bind mount，Redis AOF 还有 ACL `NOPERM` 风险。因此当前 release 仍是 `Published`，不是 `Deployed/Demonstrated`。下一实施任务是 Task 3 的项目、数据和分析目录 API。

## 1. 当前 Git 与工作区快照

| 项目 | 当前事实 |
| --- | --- |
| 工作区 | `D:\guet_learning\26 AI acting\Battery-energy-storage-zzb` |
| 分支 | `codex/quanxin-full` |
| 当前 HEAD | `45a38887dda785a74a6cad7acc015a44c88c5b7c` |
| HEAD 摘要 | `docs: record immutable ACR release evidence` |
| 当前 release 源码 | `daca47a7d0a2f8e82a7c549b6b97d0f25b5596a7` |
| 远端关系 | 用户已 push，当前工作区无受跟踪差异；仅保留明确保护的未跟踪内容 |
| 当前 release | `2026.08.02-1`，基于 `daca47a...`，状态 `Published` |
| 历史 release | `2026.07.28-1`，基于旧提交 `912f8ae...` 和旧 IP，仅保留为历史证据 |
| 当前任务 | Task 0、Task 1、Task 2 已完成；准备进入 Task 3，ECS 部署与运营闭环归入 Task 8 |

Task 2 实现与 CI 修复提交：

```text
aa5aa40 chore: ignore local server results
2b6e786 build: enforce immutable releases and postgres CI
ec4cd8b fix(agent): persist auditable results and enforce frozen batches
7afedaf feat(frontend): implement option B analysis workspace
7907be8 fix: harden Runtime V7 calibration artifacts
41fcd1d docs: record productization plan and current handoff
68ab29f test: order postgres principal fixture inserts
daca47a test: assemble complete postgres agent runtime
```

### 1.1 当前 `git status --short`

以下是本轮 ECS 审计文档提交完成后的预期状态：

```text
?? .playwright-mcp/
?? frontend/pnpm-workspace.yaml
?? output/playwright/option-b-login-1024.png
?? output/playwright/option-b-login-1440.png
?? output/playwright/option-b-login-390.png
?? output/playwright/option-b-workspace-1440.png
?? output/playwright/option-b-workspace-390-menu.png
?? output/playwright/option-b-workspace-390.png
?? tmp/
```

除本轮 ECS audit/status/handoff 文档外，所有受跟踪产品修改已经按审查边界提交；没有业务源码差异。`.gitignore` 的用户修改已按原样独立提交。`frontend/pnpm-workspace.yaml` 是工具运行期间意外生成的未跟踪占位文件，继续保留。

### 1.2 本地提交边界

本轮没有使用 `git add .`、`git add -A` 或 GitHub Desktop 的全选提交。提交边界为：

- `7907be8`：既有 Runtime V7 calibration、conformal 和 artifact 身份修复；
- `7afedaf`：B「双栏分析」前端、测试、设计规范和 7 张最终 QA 截图；
- `ec4cd8b`：Agent/普通项目 ToolResult 外键顺序、canonical input 恢复和 frozen record batch 范围；
- `2b6e786`：不可变发布、Gateway 镜像、PostgreSQL CI 和部署契约；
- `aa5aa40`：用户已有 `/server-results/` 忽略规则；
- `41fcd1d`：产品化总计划和 Task 0/Task 1 交接；
- `45a3888`：当前 release、公开状态、ECS 手册和不可变 ACR 证据；
- 本轮 ECS 审计文档提交：VNC 只读对账、公开状态、ECS 手册和本交接的实际证据更新。

### 1.3 受跟踪工作区

本轮 ECS 审计文档提交完成后，`git diff` 与 `git diff --cached` 应为空。发布源码、测试、配置和 release 证据 `45a3888` 已 push 到 `origin/codex/quanxin-full`；新增 ECS 审计文档由根 Agent 本地提交后，仍需用户通过 GitHub Desktop push。

### 1.4 未跟踪内容

- `.playwright-mcp/`：用户已有的浏览器日志与页面快照，保留，不删除、不提交。
- `tmp/`：用户已有的数据库、日志、论文文本和过程证据，保留，不删除、不提交，也不在交接中输出内容。
- `frontend/pnpm-workspace.yaml`：工具生成的意外未跟踪占位文件，保留待后续人工决定。
- `output/playwright/` 中 6 张未被 `design-qa.md` 引用的中间截图：保留，不删除、不提交；7 张最终证据已随前端提交。

每个提交前均执行了 `git diff --cached --name-only` 和 `git diff --cached --check`；没有提交密码、连接串、`.playwright-mcp/`、`tmp/` 或意外占位文件。用户只负责在 GitHub Desktop 中执行 push，根 Agent 负责精确 add/commit。

## 2. 总体宏图与当前路线

产品目标是让普通用户不接触 UUID、数据库或服务器命令，完成：

```text
登录
→ 进入项目
→ 选择内置样例或上传单/多电芯数据
→ 选择电芯和 cutoff
→ 启动一个全新的固定九步 Agent
→ 查看实时进度与失败原因
→ 查看 RUL / SOH / Conformal 及证据
→ 阅读同批 ToolResult 的审计报告
→ 下载 PDF / Word / Markdown / JSON / 派生 CSV / ToolResult ZIP
```

技术路线保持已有可信计算链，不重写算法：

```text
Next.js 产品工作台
→ FastAPI 产品编排与目录 API
→ PostgreSQL 项目/数据/运行事实
→ MinIO 隔离上传与 Celery 导入队列
→ 固定九步 Agent
→ Advanced Input / RUL / SOH / Conformal 工具
→ 可审计 ToolResult
→ 报告与导出
```

所有业务数值只能来自有效 `ToolResult`。LLM 只编排工具并解释结果；UI、提示词、设计效果图和演示不得填入虚构 SOH、RUL、区间、阈值比较或经营指标。

## 3. 当前真实完成度

### 3.1 已完成或当前代码已具备

- Python 3.11、`src/quanxin_life` 包布局和公共 `quanxin_life.core` 契约。
- PostgreSQL system of record、Redis/Celery 协调、FastAPI、Next.js、Nginx 的单机 Compose 架构。
- 安全模型制品路线、manifest/SHA-256 校验和对不可信可执行反序列化制品的拒绝边界。
- 按 `cell_id` 隔离的数据/实验约束和 PyBaMM 仅作短期参考的边界。
- Advanced Input、RUL、SOH、Split Conformal、审计 ledger、报告的代码链。
- 固定九步 Agent 计划：Advanced Input、两个 RUL 分支、SOH、两个 calibration、RUL interval、SOH band、报告。
- Agent run 的持久化、SSE、审批、fenced claim、exact step、失败关闭与恢复逻辑。
- 登录、改密、项目列表/详情、通用 Agent 创建、SSE 时间线、单结果页、单电芯结果页和 calibration 管理页。
- 单电芯结果页只显示服务端 ToolResult，并核验 result/run/project scope、身份、版本、SHA 和警告；错误时停止展示业务数值。
- B「双栏分析」主工作台、顶部导航、六阶段流程条、固定九步 Tracker、运行审计双栏、可信结果等待态和响应式登录/改密页面。
- 前端唯一主操作为“启动全新九步 Agent”；内置样例、上传、目录和导出未接入时均显示禁用或等待服务端状态，不伪装为可用能力。
- 九步 Tracker 仅在后端 `plan.steps` 精确匹配 Runtime V7 九个 step ID 时显示；SSE 对 `run_id`、sequence、事件名、带时区时间和 payload 结构做运行时校验，切换 run 时卸载旧会话。
- 可重试失败显示“等待重试”，终态失败显示服务端 `failure_code`；浏览器不拼装报告，不制造结果数值。
- Runtime V7 的外键父行顺序已覆盖 Agent run/event、普通项目 ToolResult、项目 Agent ToolResult、非项目 Agent ToolResult 和 calibration materialization；非项目路径发现的遗漏已修复。
- 高级单电芯 run 现在拒绝父 dataset ID，要求恰好一个同项目且父数据集已冻结的 record batch，错误时不派发队列任务。
- 发布 workflow 在任何构建前检查五个仓库的 release tag，已存在或检查结果不明确时 fail closed；backend/frontend/gateway 镜像携带 OCI source revision 和 release version 标签。
- Gateway Nginx 配置由 `deploy/Dockerfile.gateway` 固化进镜像，Competition Compose 不再 bind mount 宿主机 `competition.conf`。
- CI 已增加 `pgvector/pg16` 的 passwordless loopback PostgreSQL job，并在 run `30738774242` 实际通过 Alembic `0015`、schema check、固定九步 Agent、9 个 ToolResult/provenance 和重复投递回归。
- 同一 run 的 `python-quality`、`postgres-quality`、`frontend-quality` 全部成功；mypy 对 207 个源文件零问题。
- ACR release `2026.08.02-1` 已由 workflow run `30739080847` 从 `daca47a...` 发布成功，五张镜像 digest 已写入 `docs/deployment/releases/2026.08.02-1.md`。
- backend/frontend/gateway build summary 均确认 `org.opencontainers.image.revision=daca47a...` 与 `org.opencontainers.image.version=2026.08.02-1`；frontend 还确认内置 public origin 为 `https://47.99.69.138`。
- 基线提交 `e91ce7c` 已修复旧 0013 PostgreSQL Boolean 可移植性、0013/0015 PostgreSQL native alter 路径，并补齐 backend `llm` extra；Task 2 的 Runtime、Agent、前端和发布提交位于其后。旧交接中的 migration blocker 对当前源码已经过时。
- 公网 `47.99.69.138` 的严格 TLS、Gateway、Frontend 和受认证 API 路由可达，详见第 7 节。

### 3.2 部分完成

- Agent 产品入口：视觉与唯一主操作已完成，但目录 API 尚未提供可选择的数据集、电芯和 cutoff，底层表单仍依赖现有 UUID/参数契约，尚不能完成普通用户的一键闭环。
- 实时进度：固定九步名称、序号、可信失败/重试状态、SSE 校验、重连和审计事件已具备；步骤耗时、自动结果发现和真实纵向运行仍待后续目录 API 与 E2E。
- 单电芯结果：RUL/SOH/Conformal/报告组件存在，但调用者必须提供 record batch 和多个 result ID，运行结果不可自动发现。
- 报告导出：通用后端具备 JSON、Markdown、PDF、DOCX 能力；competition runtime 未确认注入 report exporter，前端没有下载中心。
- 上传：后端只有单电芯、单 cutoff、JSON base64 canonical CSV 的底层接口和内容寻址对象存储；没有产品上传会话。
- ECS：公网 edge、Compose、migration `0015`、Worker、历史九步 ToolResult、备份和证书 timer 已只读验证；但 ECS 仍运行 `2026.07.29-1` 和可变热修复绑定，当前 release 尚未部署。

### 3.3 未完成

- Task 3 的目录/编排 API 尚未实现：
  - `GET /v1/projects/{project_id}/analysis-inputs`
  - `GET /v1/projects/{project_id}/datasets`
  - `GET /v1/datasets/{dataset_id}/batches`
  - `GET /v1/projects/{project_id}/agent/runs`
  - `GET /v1/agent/runs/{run_id}/results`
  - `POST /v1/projects/{project_id}/advanced-analyses`
- 项目首页中的内置样例、电芯/cutoff、最近运行、数据状态和结果目录。
- CSV/Parquet/多文件/ZIP、`metadata.csv`、16 MiB 分块、暂停/重试/续传、质量确认、异步导入和四 cutoff 自动生成。
- 由真实目录 API 驱动的完整工作台、整合结果页、证据抽屉、报告阅读页和下载中心；当前 B 方案实现的是可信布局与状态骨架。
- 派生 CSV、完整 ToolResult 集合、带 manifest/SHA 的异步 ZIP；ZIP 必须排除原始上传数据。
- 当前 release `2026.08.02-1` 的 ECS 部署、RepoDigest/OCI 对账、无 bind mount 验收、正式 route/calibration 和发布后真实九步 E2E。
- Redis AOF ACL `NOPERM` 专项核验、报告表关联修复、备份恢复与回滚演练。
- 严格浏览器纵向 E2E、备份恢复、回滚、证书自动续期和磁盘/队列运维演练。

### 3.4 已放弃或明确否决

- 让 LLM 或 UI 生成业务数值。
- 按行或周期随机拆分导致同一电芯跨集合。
- 未核验 pickle/joblib/`.pt`/`.pth` 等制品。
- 用 PyBaMM 生成长期退化标签。
- 用 SQLite 通过替代 PostgreSQL migration/写入验收。
- 用 `curl -k` 替代严格 TLS 验收。
- 把 Runtime V7 bind mount 当最终发布。
- 用 `DROP CASCADE`、清库或 `stamp head` 绕过 migration。
- 把历史 release `2026.07.28-1` 的记录直接篡改成新 IP/新提交；它应作为历史制品身份保留并标为 superseded。
- 当前阶段引入 Kubernetes/企业 HA 阻塞个人比赛演示。
- 使用任何 superpowers 系列 skill。

## 4. 前端与上传真实状态

### 4.1 页面结构

当前路由：

```text
/
/login
/login/change-password
/projects
/projects/[projectId]
/projects/[projectId]/calibration
/projects/[projectId]/cells/[recordBatchId]
/agent/runs/[runId]
/results/[resultId]
```

当前主要组件：`app-shell`、`workflow-rail`、`analysis-workspace`、`nine-step-tracker`、`project-overview`、`create-agent-run-form`、`agent-timeline`、`approval-panel`、`single-cell-analysis`、`soh-trajectory-chart`、`evidence-panel`、`calibration-materialization-panel`。

用户已选择 B「双栏分析」。当前视觉采用人民币 50 元启发的暖白、墨绿、玉绿、灰豆绿与小面积暗金/橄榄；桌面为操作/状态与结果/审计双栏，平板和手机将操作区排在结果区之前。顶部导航在窄屏使用可访问菜单，流程条局部横向滚动，登录/改密在手机端表单优先，并支持 `prefers-reduced-motion`。

可信边界：项目结果区无 ToolResult 时只显示等待态；内置样例、上传和导出入口保持禁用；流程条不根据 URL 或阶段位置推断此前步骤已完成。

### 4.2 上传边界

现有正式接口：

```text
POST /v1/datasets/{dataset_id}/batches/canonical-csv
```

事实：

- 请求体是 JSON 内的 base64 CSV，不是 multipart 或分块上传；
- 一次只绑定一个电芯和一个 cutoff；
- 只允许授权用户向 ACTIVE project 的 DRAFT dataset 写入；
- 服务端派生 project/dataset/content ID 和哈希，调用方不能注入；
- 解码后 CSV 上限为 25 MiB，要求 UTF-8、固定字段、已登记 metadata/version/provenance；
- dataset 只有创建、单项读取和冻结，没有项目 datasets 列表或 batch 列表；
- MinIO adapter 有内容寻址和读取时 SHA 校验，但没有隔离区、upload session 或异步导入产品流程。

## 5. Runtime V7 与 Agent 关键模块

- `src/quanxin_life/application/advanced_calibration_materialization.py`：校准样本生产、SOH 稀疏轴对齐。
- `src/quanxin_life/application/advanced_deployment_bundles.py`：v1/v2 artifact 构建、恢复与身份隔离。
- `src/quanxin_life/tools/advanced_conformal.py`：RUL interval/SOH band 的工具边界和校准样本验证。
- `src/quanxin_life/application/agent_runs.py`：run 创建、scope、幂等、事件和状态。
- `src/quanxin_life/application/agent_run_execution.py`：exact step 执行、恢复、canonical input hash。
- `src/quanxin_life/audit/sql_project_ledger.py`：ToolResult、provenance、project binding、Agent result 和 calibration binding 的原子持久化。
- `src/quanxin_life/agents/supervisor.py`：固定九步计划。
- `src/quanxin_life/application/advanced_agent_execution_context.py`：active route、READY calibration 和 record batch 的可信上下文解析。
- `src/quanxin_life/api/agent_runs.py`、`tasks/agent_runs.py`：HTTP/SSE 与 Celery 入口。

## 6. 部署与发布事实

Compose 包含 7 个服务：PostgreSQL、Redis、一次性 `migrate`、API、Worker、Frontend、Gateway。只有 Gateway 暴露 80/443；后端网络 internal；API/Worker 依赖 migration 成功；使用文件型 secrets、只读容器和 `no-new-privileges`。Gateway 配置已改为镜像内置，宿主机只挂载证书和 ACME webroot。

当前发布链：

```text
历史 release 2026.07.28-1
→ source commit 912f8ae...
→ 历史公网 IP 47.98.37.232
→ superseded，仅保留历史证据

当前 release 2026.08.02-1
→ source commit daca47a7d0a2f8e82a7c549b6b97d0f25b5596a7
→ CI run 30738774242 全绿
→ ACR run 30739080847 成功
→ 五张镜像 digest 已记录
→ backend/frontend/gateway OCI revision/version 已确认
→ public origin https://47.99.69.138
→ Published；ECS 已审计但仍运行旧 `2026.07.29-1`
```

因此：

- 旧 release 记录是历史证据，已标注由 `2026.08.02-1` supersede，原身份不得改写；
- 当前源码已经形成可信不可变 release；ECS 受信只读审计已确认公网运行的是旧 `2026.07.29-1`、旧 digest 和空 OCI 身份，不是当前 release；
- migration `0015`、Worker、旧数据库和历史九步 Agent 已对账，但这些事实不能证明 `2026.08.02-1`；
- 部署当前 release 时仍必须逐项比较 release 文档中的五个 digest，并检查自建镜像 revision/version；
- Compose 仍通过 `${RELEASE_TAG}` 引用镜像，不是直接 pin digest，不可变性依赖发布记录和 RepoDigest 验证。
- workflow 已在本次发布前检查五个仓库并拒绝已存在 tag；检查异常 fail closed。自建 backend/frontend/gateway 镜像已记录 `org.opencontainers.image.revision` 与 `org.opencontainers.image.version`。

本轮文档已同步：

- `docs/status.md` 已更新到当前 IP、当前 release 和 `Published`/`Not deployed` 边界；
- `docs/deployment/competition-ecs.md` 已更新当前发布基线和 ECS 旧部署事实；
- `docs/deployment/ecs-audit-2026-08-02.md` 记录系统、容器、migration、Worker、TLS、备份和 Agent 只读证据；
- `docs/deployment/releases/2026.08.02-1.md` 记录本次 run、source、origin、五个 digest 和 OCI 身份；
- `docs/deployment/releases/2026.07.28-1.md` 保留原身份并标注 superseded。

## 7. 本轮只读公网与 ECS 检查

2026-08-02 公网与 ECS 实测：

- `http://47.99.69.138/`：`308` 跳转 HTTPS；
- `https://47.99.69.138/`：严格 TLS 校验成功，Nginx/Next.js 返回 `307` 到 `/projects`；
- `https://47.99.69.138/health`：严格 TLS 下 `200`；
- `https://47.99.69.138/login`：`200`；
- `https://47.99.69.138/v1/projects`：未认证请求返回 `401`，证明受认证 API 路由在公网暴露；
- TCP 80/443 可达；
- 安全组临时放行审计电脑 `/32` 后 TCP 22 可达；
- 现有项目公钥对 `root`、`ubuntu`、`ecs-user` 均不匹配，未修改 SSH 配置或注入密钥；
- VNC 只读审计确认 Ubuntu 22.04.5、4 vCPU、约 7.1 GiB 内存、40 GiB 根盘、2 GiB swap；
- 根盘约 72%，Docker 镜像 21.1 GB，其中约 9.921 GB 可回收，本轮未清理；
- Compose 可解析，API/Frontend/Gateway/PostgreSQL/Redis healthy，Worker 返回 `pong`；
- migrate `exit=0`，日志包含 `DATABASE_MIGRATIONS_OK`，Alembic 为 `0015`；
- 所有运行服务仍使用 `2026.07.29-1`，OCI revision/version 为空；
- API/Worker 仍绑定 `/srv/quanxin/hotfixes/calibration-runtime-v7/` 单文件，Gateway 仍绑定宿主机 `competition.conf`；
- 历史唯一 Agent run 为 9 steps / 9 ToolResult，关联 23 条 provenance，但 `reports` 表关联为 0；
- Redis 近 24 小时出现 AOF loading client ACL `NOPERM` CRITICAL 记录，持久化完整性未证明；
- snap certbot timer active，2026-08-02 两次运行成功；当前证书截止 `2026-08-08 02:50:38 UTC`；
- PostgreSQL dump 共 36 份、约 17 MB，最新可见时间 `2026-08-01 20:25`。

能证明：新 IP 的 Gateway、严格 TLS、Frontend、认证 API edge、旧部署内部服务、migration、Worker 和历史九步 ToolResult 可用。
不能证明：当前 release 已部署、Redis AOF 完整恢复、正式 route/calibration、报告表闭环、发布后真实九步 E2E、备份恢复或回滚演练。完整证据见 `docs/deployment/ecs-audit-2026-08-02.md`。

## 8. 本轮验证结果

所有 Python 命令使用项目 `.venv` 的 Python 3.11；系统默认 `python` 不是可信项目解释器。

### 8.1 Python 完整门禁

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

结果：`1777 passed, 9 skipped, 50 warnings`，exit 0，耗时 9 分 35.26 秒。警告为 Starlette/httpx 弃用和 PyTorch Transformer nested tensor 提示。新增 2 个 skip 是真实 PostgreSQL 专用测试在本机没有显式 passwordless loopback DSN 时的预期结果。

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

结果：`All checks passed!`

```powershell
.\.venv\Scripts\python.exe -m mypy
```

结果：`Success: no issues found in 207 source files`

```powershell
.\.venv\Scripts\python.exe -m compileall -q src workbench deploy migrations
```

结果：exit 0，无错误输出。

### 8.2 前端完整门禁

直接调用现存 `node_modules/.bin`，未安装或修改依赖：

```powershell
frontend\node_modules\.bin\vitest.cmd run
frontend\node_modules\.bin\eslint.cmd . --max-warnings=0
frontend\node_modules\.bin\tsc.cmd --noEmit
frontend\node_modules\.bin\next.cmd build
```

结果：

- Vitest：21 files、`81 passed`；
- ESLint：exit 0；
- TypeScript：exit 0；
- Next.js 16 production build：exit 0，`/icon.svg` 与全部当前路由编译/预渲染成功；
- build 仅提示 `baseline-browser-mapping` 数据较旧，不是构建失败。

### 8.3 Task 2 定向验证

- Runtime V7/Agent 聚焦回归：`120 passed`；补充外键与 scope 后两文件回归：`26 passed`。
- 发布、Compose、CI、运行文档和 PostgreSQL 测试收集：`37 passed, 2 skipped`。
- `.github/workflows/ci.yml`、`.github/workflows/publish-acr.yml` 与 `deploy/competition.compose.yaml` 可由现有 YAML 解析器读取。
- `git diff --check`、`git diff --cached --check`：exit 0；仅报告既有 Windows LF/CRLF 转换提示。

生产预览使用 `NEXT_PUBLIC_API_BASE_URL=https://qa.quanxin.invalid` 构建，当前本地入口为 `http://127.0.0.1:3011`。

### 8.4 GitHub Actions 与不可变发布

- CI run `30738774242`，source `daca47a...`：`python-quality`、`postgres-quality`、`frontend-quality` 全部成功；
- `python-quality`：6 分 2 秒，pytest、Ruff、mypy 和 compileall 全部成功；
- `postgres-quality`：2 分 33 秒，真实 `pgvector/pg16` 的 migration/schema/固定九步与 ToolResult 回归成功；
- `frontend-quality`：47 秒，前端完整门禁成功；
- ACR run `30739080847`：10 分 13 秒成功，release tag `2026.08.02-1`；
- 五个不可变 digest 与三张自建镜像 OCI 标签见 `docs/deployment/releases/2026.08.02-1.md`。

GitHub Actions 的 Node.js 20 弃用注释来自 action 运行时被强制切到 Node.js 24，不是本次 CI 或发布失败，也不改变镜像 source identity。

### 8.5 浏览器 Design QA

- 登录页已在 `1440×1024`、`1024×768`、`390×844` 核验；手机端表单先于品牌说明，且无整页横向溢出。
- 项目工作台已在三种视口核验；桌面保持约 62/38 双栏，平板/手机操作与进度区先于结果与审计区，流程条只在自身区域滚动。
- Agent run 页已在桌面和手机核验；九步状态与原始审计事件顺序正确，无页面横向溢出。
- 单电芯结果等待页未出现业务指标，只显示等待服务端签发结果和待服务端提供导出。
- 最终生产页面控制台为 `0 errors / 0 warnings`；缺失 favicon 已由 `frontend/app/icon.svg` 修复。
- `prefers-reduced-motion` 下过渡时长为 `0.01ms`、transform 为 `none`。
- 完整证据见 `design-qa.md` 和 `output/playwright/`，`design-qa.md` 最终结论为 `passed`。

一次直接 `pnpm` 调用因 Codex 运行时包装器尝试执行 install，并被 ignored build scripts 策略拒绝；未执行 `pnpm approve-builds`，未安装依赖。随后使用现存本地二进制完成上述门禁。该过程留下 `frontend/pnpm-workspace.yaml` 占位文件，当前保留。

### 8.6 仍缺少的验证

- 本机没有 Docker/Podman/nerdctl，未运行本地 `docker compose config` 或容器 smoke；真实镜像 build 已由 GitHub Actions 完成；
- 当前 release 尚未部署；ECS 仍运行旧 tag、旧 digest 和热修复 bind mount；
- 已完成 mock 边界内的布局与可信状态浏览器 QA，但未进行真实登录、真实 PostgreSQL/Worker/ToolResult 的纵向 E2E；
- Redis AOF ACL 错误、报告表关联和备份恢复尚未完成专项验收。

## 9. 当前最大风险

最大风险不是算法或 registry，而是“公网正在运行的仍是旧 release 和可变热修复覆盖”：当前源码已经从全绿 CI 构建为 `2026.08.02-1`，但 ECS 实测仍是 `2026.07.29-1`，没有新 OCI 身份，且 API/Worker/Gateway 依赖宿主机 bind mount。任何继续演示若忽略这个断层，都可能把 `Published` 误写成 `Deployed/Demonstrated`。第二个高风险是 Redis AOF 回放 ACL `NOPERM`，当前健康状态不能替代持久化完整性证明。

Task 2 已降低未来发布继续产生该断层的概率：release tag 不可覆盖，自建镜像带 source revision/version，gateway 配置不再依赖宿主机文件，并且真实 PostgreSQL CI 与五个 ACR digest 已形成证据。当前剩余断层属于 Task 8 的 ECS 部署与运营验收。

第二层风险：

- 本轮 release 文档提交仍需用户 push；如果 GitHub Desktop 误选全部未跟踪文件，会混入浏览器日志、临时数据库、论文过程文件和意外 pnpm 占位配置；
- 产品目录 API 尚不存在，前端若先编码会继续依赖 UUID 和临时查询参数；
- v1/v2 artifact 目录若同时存在会 fail closed，发布前必须核验 catalog 和 ECS artifact 目录；
- migration 对不兼容历史审计行应 fail closed，上线前需要真实 DB 预检和备份；
- 上传是外部不可信输入，未来必须覆盖大小、哈希、ZIP 路径穿越、单位、重复电芯、周期不足、部分失败和幂等。
- SSE 已有运行时 decoder，但 HTTP `AgentRunRecord` 仍由通用 `apiRequest<T>` 类型断言接收，前端接口也比后端 Pydantic 契约宽松；这是后续目录 API/契约加固时需要收紧的风险。

## 10. 已确认的工程与产品决策

- PostgreSQL 是业务事实源；Redis 只做队列和短期协调。
- 第一版每次只分析一个选中的电芯和 cutoff；多电芯先做上传、管理和选择。
- 每次主动点击创建新运行；网络重试复用同一幂等键。
- 上传通过后自动冻结，但绝不自动启动 Agent。
- 报告必须有 PDF 和 Word，同时保留 Markdown、JSON；允许派生 CSV，禁止原始上传数据下载。
- 状态必须同时用文字、图标和颜色；错误、不确定、缺测和域外输入必须显式降级。
- 视觉采用人民币 50 元启发的暖白、墨绿、玉绿、灰豆绿和小面积暗金/橄榄，不复制钞票图案或防伪纹理。
- 用户已选择 B「双栏分析」作为前端视觉真相，规范文件为 `docs/superpowers/specs/2026-08-02-quanxin-frontend-option-b-design.md` 与同名 PNG。
- 前端主操作始终只有一个：“启动全新九步 Agent”。
- 用户选定三套视觉方案之一后才允许前端视觉编码；选定图是实现的视觉真相。
- 密码只由用户在交互提示中输入，不读、不输出、不保存。

## 11. Task 2 当前状态与下一步

Task 0、Task 1、Task 2 已完成。Runtime V7 差异已审计并合入正常源码；外键父记录顺序、Agent scope、canonical input、固定九步和共享 `AuditLedger` 已有回归；Python、真实 PostgreSQL 和前端 CI 全绿；backend、frontend、gateway 已作为不可变镜像构建，Gateway 不再依赖配置 bind mount；release `2026.08.02-1` 已发布并记录五个 digest 和三张自建镜像的 OCI revision/version。

下一步进入 Task 3：先实现项目、数据和分析目录 API，消除普通用户手填 UUID。ECS 只读审计已完成，确认旧部署可用但当前 release 未部署；新 release 部署、Redis 恢复专项、route/calibration、报告链和发布后九步 E2E 归入 Task 8，不阻塞 Task 3 源码开发。任何公网演示仍必须标明运行的是旧 release，直到 `2026.08.02-1` 完成 RepoDigest/OCI/无 bind mount 对账。

## 12. 后续实施顺序

按总计划依赖顺序继续：

1. Task 2：已完成，包含 Runtime V7 整理、全量门禁、真实 PostgreSQL CI 和不可变 release `2026.08.02-1`。
2. Task 3：当前下一任务，先实现目录/编排 API，避免前端继续手填 UUID。
3. Task 4：在 B 方案可信布局骨架上接入目录 API，形成完整产品工作台。
4. Task 5：实现分块上传、隔离校验、异步导入和多电芯管理。
5. Task 6：实现完整报告与 ToolResult 导出。
6. Task 7：真实浏览器纵向验收，并逐项对账 ToolResult。
7. Task 8：新 release、严格 TLS、备份恢复、回滚和运维闭环。

## 13. 用户协作偏好

- 中文、直接、工程化；先核验真实状态，再下结论。
- 长任务持续汇报；说明完成、阻塞、风险和下一步。
- 能从代码、Git、测试、配置或只读 ECS 检查发现的事实不询问用户。
- 只有产品功能或展示取舍会实质改变结果时才用小白能理解的话询问。
- 子 Agent 只做边界清楚、互不冲突的审计；根 Agent 负责契约、集成和复验。
- 不覆盖已有修改，不删除 `.playwright-mcp/`、`tmp/` 或未确认文件。
- 不执行 `git reset --hard`、`git checkout --` 或工作区清理。
- 不提交、不 push，除非用户明确授权。
- 不使用任何 superpowers 系列 skill。

## 14. 新对话启动步骤

1. 完整阅读根目录 `AGENTS.md`、本文件和 `docs/superpowers/plans/2026-08-02-quanxin-productization-master-plan.md`。
2. 重新执行 `git status --short`、`git diff --stat`、`git diff`、`git diff --cached`、`git log -5 --oneline`、分支和 HEAD 检查。
3. 保护 `.playwright-mcp/`、`tmp/`、`frontend/pnpm-workspace.yaml` 和 6 张未提交的中间截图；不要在 GitHub Desktop 中全选未跟踪文件。
4. 不把旧 IP、旧 release 或旧 migration 阻塞当成当前事实；公网 IP 是 `47.99.69.138`。
5. Task 1 已完成且用户已确认 B「双栏分析」；不得回退到重新选方案，也不得把未来目录、上传或导出能力写成当前已接通。
6. Task 2 已完成并发布为 `2026.08.02-1`；release 证据 `45a3888` 已 push，本轮 ECS 审计文档提交后仍由用户在 GitHub Desktop push。下一任务是 Task 3。
7. 任何生产结论必须有当前命令、ToolResult、API、数据库、镜像 digest 或浏览器证据；无法核验就明确写 `unverified`。
