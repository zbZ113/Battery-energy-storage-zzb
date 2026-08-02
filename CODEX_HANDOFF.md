# Codex 完整交接：泉芯智寿产品化

> 更新时间：2026-08-02 15:28:15 +08:00（Asia/Shanghai）
> 当前实施目标：`docs/superpowers/plans/2026-08-02-quanxin-productization-master-plan.md`
> 当前公网入口：`https://47.99.69.138`
> 本文件只记录可由当前 Git、代码、测试和只读探测支持的事实，不包含密码、私钥、Cookie、连接串或业务数值。

## 0. 一句话状态

可信计算链、固定九步 Agent、B「双栏分析」前端和 Task 2 的源码级不可变发布约束已经形成，本地完整工程门禁通过，并已拆分为 5 个本地审查提交。Gateway 配置已固化进镜像，release workflow 会拒绝复用 tag，并为自建镜像写入 source revision/version 标签；同时增加了 passwordless loopback PostgreSQL CI 验收。用户已授权本地 add/commit，并明确由用户在 GitHub Desktop 执行 push；在 push、真实 PostgreSQL CI、镜像构建、digest 和 ECS 对账完成前，Task 2 仍只能标为部分完成，不得称为 `Published/Deployed/Demonstrated`。

## 1. 当前 Git 与工作区快照

| 项目 | 当前事实 |
| --- | --- |
| 工作区 | `D:\guet_learning\26 AI acting\Battery-energy-storage-zzb` |
| 分支 | `codex/quanxin-full` |
| 交接写入前 HEAD | `aa5aa4051b352aae16d6210646fc53eaa1eb511f` |
| HEAD 摘要 | `chore: ignore local server results` |
| 远端关系 | 本地分支比 `origin/codex/quanxin-full` 领先 5 个实现提交；本交接提交完成后将领先 6 个提交 |
| 最近历史 release | `2026.07.28-1`，基于旧提交 `912f8ae...`，不是当前源码发布 |
| 当前任务 | Task 0、Task 1 已完成；Task 2 本地源码、契约和本地提交完成，待用户 push、真实 PostgreSQL CI、ACR 和 ECS 对账 |

本轮 5 个本地实现提交：

```text
aa5aa40 chore: ignore local server results
2b6e786 build: enforce immutable releases and postgres CI
ec4cd8b fix(agent): persist auditable results and enforce frozen batches
7afedaf feat(frontend): implement option B analysis workspace
7907be8 fix: harden Runtime V7 calibration artifacts
```

### 1.1 当前 `git status --short`

以下是本交接与总计划纳入最终文档提交后的预期状态；提交前两份文档自身仍显示为未跟踪：

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

所有受跟踪产品修改已经按审查边界提交；没有 staged、unstaged 或 `MM` 文件。`.gitignore` 的用户修改已按原样独立提交。`frontend/pnpm-workspace.yaml` 是工具运行期间意外生成的未跟踪占位文件，继续保留。

### 1.2 本地提交边界

本轮没有使用 `git add .`、`git add -A` 或 GitHub Desktop 的全选提交。提交边界为：

- `7907be8`：既有 Runtime V7 calibration、conformal 和 artifact 身份修复；
- `7afedaf`：B「双栏分析」前端、测试、设计规范和 7 张最终 QA 截图；
- `ec4cd8b`：Agent/普通项目 ToolResult 外键顺序、canonical input 恢复和 frozen record batch 范围；
- `2b6e786`：不可变发布、Gateway 镜像、PostgreSQL CI 和部署契约；
- `aa5aa40`：用户已有 `/server-results/` 忽略规则；
- 最终文档提交：产品化总计划和本交接。

### 1.3 受跟踪工作区

最终文档提交完成后，`git diff` 与 `git diff --cached` 应为空。所有产品源码、测试、配置和受审文档均已进入本地提交；尚未 push。

### 1.4 未跟踪内容

- `.playwright-mcp/`：用户已有的浏览器日志与页面快照，保留，不删除、不提交。
- `tmp/`：用户已有的数据库、日志、论文文本和过程证据，保留，不删除、不提交，也不在交接中输出内容。
- `frontend/pnpm-workspace.yaml`：工具生成的意外未跟踪占位文件，保留待后续人工决定。
- `output/playwright/` 中 6 张未被 `design-qa.md` 引用的中间截图：保留，不删除、不提交；7 张最终证据已随前端提交。

每个提交前均执行了 `git diff --cached --name-only` 和 `git diff --cached --check`；没有提交密码、连接串、`.playwright-mcp/`、`tmp/` 或意外占位文件。用户将在 GitHub Desktop 中执行 push。

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
- CI 已增加 `pgvector/pg16` 的 passwordless loopback PostgreSQL job，计划执行 Alembic `0015`、schema check、固定九步 Agent、9 个 ToolResult/provenance 和重复投递回归。
- 基线提交 `e91ce7c` 已修复旧 0013 PostgreSQL Boolean 可移植性、0013/0015 PostgreSQL native alter 路径，并补齐 backend `llm` extra；本轮 5 个本地实现提交位于其后。旧交接中的 migration blocker 对当前源码已经过时。
- 公网 `47.99.69.138` 的严格 TLS、Gateway、Frontend 和受认证 API 路由可达，详见第 7 节。

### 3.2 部分完成

- Runtime V7 / Task 2：源码修复、回归、发布契约、本地完整门禁和审查提交已完成；分支尚未由用户 push，真实 PostgreSQL CI 尚未运行，也没有从受审提交构建新不可变镜像。
- Agent 产品入口：视觉与唯一主操作已完成，但目录 API 尚未提供可选择的数据集、电芯和 cutoff，底层表单仍依赖现有 UUID/参数契约，尚不能完成普通用户的一键闭环。
- 实时进度：固定九步名称、序号、可信失败/重试状态、SSE 校验、重连和审计事件已具备；步骤耗时、自动结果发现和真实纵向运行仍待后续目录 API 与 E2E。
- 单电芯结果：RUL/SOH/Conformal/报告组件存在，但调用者必须提供 record batch 和多个 result ID，运行结果不可自动发现。
- 报告导出：通用后端具备 JSON、Markdown、PDF、DOCX 能力；competition runtime 未确认注入 report exporter，前端没有下载中心。
- 上传：后端只有单电芯、单 cutoff、JSON base64 canonical CSV 的底层接口和内容寻址对象存储；没有产品上传会话。
- ECS：公网 edge 可验证，内部运行状态与发布来源无法验证。
- 不可变发布：源码和本地提交已经拒绝 tag 覆盖、固化 gateway 配置并写入 revision 标签，但尚未 push、workflow dispatch，也没有 ACR digest 或 ECS RepoDigest 证据。

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
- 当前源码的新不可变 backend/frontend/gateway release、真实构建、digest 对账和 ECS 发布。
- ECS 内部 migration head、容器状态、镜像 digest、Worker 队列、ADMIN、route、calibration materialization 和真实九步 ToolResult 闭环。
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

当前发布断层：

```text
历史 release 2026.07.28-1
→ source commit 912f8ae...
→ 历史公网 IP 47.98.37.232

当前仓库
→ 远端基线 e91ce7c
→ 本地实现链 aa5aa40（另有本交接文档提交）
→ 已包含后续 PostgreSQL migration、backend extra、Runtime V7、B 方案和不可变发布修复
→ 当前公网 IP 47.99.69.138
```

因此：

- 旧 release 记录是历史证据，不能伪改为当前 release；
- 当前公网正在运行的镜像 tag/digest/source commit 尚未通过 SSH 或其他受信证据对账；
- 不能根据公网 200/401 推断 migration head、Worker、数据库、route 或真实 Agent 已完成；
- 下一次发布必须从完成审查的源码构建新 tag，记录五个镜像 digest，并在 ECS 逐项对账；
- Compose 仍通过 `${RELEASE_TAG}` 引用镜像，不是直接 pin digest，不可变性依赖发布记录和 RepoDigest 验证。
- 新 workflow 会在 build/push 前检查五个仓库并拒绝已存在 tag；检查异常也 fail closed。自建 backend/frontend/gateway 镜像会记录 `org.opencontainers.image.revision` 与 `org.opencontainers.image.version`。

过时文档：

- `docs/status.md` 仍记录 2026-07-28、旧 IP 和旧部署阶段；
- `docs/deployment/competition-ecs.md` 仍保留旧 release 的历史基线和 `Not deployed` 状态，同时已补充新 Gateway 镜像构建约束；
- `docs/deployment/releases/2026.07.28-1.md` 是历史 release 记录，应保留原身份并补充 superseded 说明，而不是替换旧值。

`docs/status.md` 和历史 release 记录尚未更新；必须等新 release 身份、五个 digest 和 ECS 对账确定后再统一更新，不能预先写成已部署。

## 7. 本轮只读公网与 ECS 检查

2026-08-02 实测：

- `http://47.99.69.138/`：`308` 跳转 HTTPS；
- `https://47.99.69.138/`：严格 TLS 校验成功，Nginx/Next.js 返回 `307` 到 `/projects`；
- `https://47.99.69.138/health`：严格 TLS 下 `200`；
- `https://47.99.69.138/login`：`200`；
- `https://47.99.69.138/v1/projects`：未认证请求返回 `401`，证明受认证 API 路由在公网暴露；
- TCP 80/443 可达；
- 本机已有 `47.99.69.138` 的 SSH 主机记录；
- 使用现有密钥路径、`BatchMode`、严格 host key 的 SSH 22 连接超时。

能证明：新 IP 的 Gateway、严格 TLS、Frontend 和认证 API edge 可达。
不能证明：ECS 内部 Compose、镜像 digest、migration、数据库、Redis、Worker、ADMIN、route、calibration、ToolResult、Agent、报告、备份、恢复或续期演练状态。

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

### 8.4 浏览器 Design QA

- 登录页已在 `1440×1024`、`1024×768`、`390×844` 核验；手机端表单先于品牌说明，且无整页横向溢出。
- 项目工作台已在三种视口核验；桌面保持约 62/38 双栏，平板/手机操作与进度区先于结果与审计区，流程条只在自身区域滚动。
- Agent run 页已在桌面和手机核验；九步状态与原始审计事件顺序正确，无页面横向溢出。
- 单电芯结果等待页未出现业务指标，只显示等待服务端签发结果和待服务端提供导出。
- 最终生产页面控制台为 `0 errors / 0 warnings`；缺失 favicon 已由 `frontend/app/icon.svg` 修复。
- `prefers-reduced-motion` 下过渡时长为 `0.01ms`、transform 为 `none`。
- 完整证据见 `design-qa.md` 和 `output/playwright/`，`design-qa.md` 最终结论为 `passed`。

一次直接 `pnpm` 调用因 Codex 运行时包装器尝试执行 install，并被 ignored build scripts 策略拒绝；未执行 `pnpm approve-builds`，未安装依赖。随后使用现存本地二进制完成上述门禁。该过程留下 `frontend/pnpm-workspace.yaml` 占位文件，当前保留。

### 8.5 仍缺少的验证

- 本机没有 Docker/Podman/nerdctl，未运行真实 `docker compose config`、backend/frontend/gateway 镜像 build 或容器 smoke；
- 真实 PostgreSQL migration、外键写入和固定九步测试已进入本地提交，但分支尚未 push，CI 尚未执行；
- SSH 超时，未完成 ECS 内部检查；
- 已完成 mock 边界内的布局与可信状态浏览器 QA，但未进行真实登录、真实 PostgreSQL/Worker/ToolResult 的纵向 E2E；
- 未校验当前公网运行镜像与 HEAD 的来源一致性。

## 9. 当前最大风险

最大风险不是算法，而是“公网正在运行的东西与当前仓库源码无法对账”：历史不可变 release 基于旧提交，本地分支又包含后续修复，但 SSH 不通，无法确认 ECS 的 revision、镜像 digest、Gateway 配置和真实业务闭环。任何继续开发或演示若忽略这个断层，都可能把本地通过误写成生产通过。

Task 2 已降低未来发布继续产生该断层的概率：release tag 不可覆盖，自建镜像带 source revision/version，gateway 配置不再依赖宿主机文件。这些约束已进入本地提交，但在 push、CI、ACR 和 ECS 实际执行前，当前公网断层仍没有被修复。

第二层风险：

- 本地分支已有 6 个待 push 提交；如果 GitHub Desktop 误选全部未跟踪文件，会混入浏览器日志、临时数据库、论文过程文件和意外 pnpm 占位配置；
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

Task 0、Task 1 已完成。Task 2 的本地源码和提交部分已完成：Runtime V7 差异完成审计，补齐非项目 ToolResult FK 顺序和高级 record batch scope；发布链路拒绝 tag 覆盖、固化 gateway 配置、写入 OCI revision/version；真实 PostgreSQL 固定九步验收已加入 CI。所有本地门禁通过，提交边界已复核。

用户已授权执行 Task 2 收尾，并明确由用户在 GitHub Desktop 执行 push。下一步是用户 push 当前 6 个本地提交；随后根 Agent 检查 `postgres-quality` 证据，再手动 dispatch ACR workflow，获得五个新 digest，最后在 ECS 对账 revision/RepoDigest。Task 2 闭环前不进入 Task 3 的业务 API 修改。

## 12. 后续实施顺序

按总计划依赖顺序继续：

1. Task 2：本地源码、契约和提交已完成；待用户 push、真实 PostgreSQL CI、ACR build/digest 和 ECS 对账。
2. Task 3：先实现目录/编排 API，避免前端继续手填 UUID。
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
6. Task 2 本地提交完成但尚未 push；用户负责 GitHub Desktop push，根 Agent 在 push 后检查 CI，再按已授权范围执行 workflow dispatch 与 ECS 对账。
7. 任何生产结论必须有当前命令、ToolResult、API、数据库、镜像 digest 或浏览器证据；无法核验就明确写 `unverified`。
