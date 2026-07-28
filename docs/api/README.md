# API 文档

本文描述泉芯智寿 HTTP API 的当前代码事实、认证边界和比赛运行时实际暴露的能力。请求与响应的最终机器事实源是运行中 FastAPI 生成的 OpenAPI，而不是本文中的手写示例。

## 1. 权威来源与适用范围

API 有三层事实源：

1. Pydantic 契约与 FastAPI 路由：`src/quanxin_life/api/`；
2. 应用装配：`src/quanxin_life/application/assembly.py`；
3. 比赛生产装配：`deploy/competition_runtime.py`。

FastAPI 默认生成：

- `/openapi.json`：机器可读 OpenAPI；
- `/docs`：Swagger UI；
- `/redoc`：ReDoc。

比赛 Nginx 网关只把 `/health` 和 `/v1/` 转发给 API，`/docs`、`/redoc` 和 `/openapi.json` **不作为公网接口发布**。在本地开发环境可直接访问这些地址；在 ECS 上应从 API 容器内部导出 OpenAPI，不能因为公网看不到 Swagger 就另写一套 DTO。

## 2. Foundation 与 competition runtime

### 2.1 Foundation API 工厂

`src/quanxin_life/api/app.py:create_fastapi_app` 是通用 HTTP adapter 工厂。它可按调用方提供的 adapter 组合认证、项目、数据集、实验、知识库、工业沙箱、模型治理、Agent、报告和工具调用等能力。

`src/quanxin_life/application/http_application.py:create_competition_fastapi_app` 是较宽的基础应用装配，服务于本地开发和组合测试。它不等于当前 ECS 实际进程。

### 2.2 ECS competition runtime

`deploy/competition_runtime.py:create_competition_runtime` 是比赛单机部署的权威 composition root。当前只装配：

- Cookie 认证；
- 项目；
- 数据集与 canonical CSV record batch；
- Advanced 模型制品目录；
- Advanced active route；
- Advanced calibration materialization；
- 项目级 Advanced 工具；
- 持久化 Agent run 与 SSE；
- PostgreSQL audit ledger；
- Redis/Celery 的 `agent-runs`、`advanced-calibration` 队列。

知识库、飞书、工业接入、实验注册和全局基础模型工具虽有独立 adapter，但未被当前 competition runtime 装配，不能在比赛说明中写成“公网已提供”。

## 3. 认证、安全与角色

### 3.1 Session Cookie

生产环境使用 `__Host-quanxin_session`：

- `Secure`；
- `HttpOnly`；
- `SameSite=Strict`；
- `Path=/`；
- 响应带 `Cache-Control: no-store`。

前端 `frontend/lib/api-client.ts` 对所有请求设置 `credentials: "include"`，浏览器脚本不读取原始 token。

### 3.2 Origin

所有浏览器状态变更都要求请求 `Origin` 精确匹配服务端 `QUANXIN_TRUSTED_ORIGIN`。通配符被禁止；生产 origin 必须是 HTTPS。

Origin 校验不是登录替代品。状态变更通常同时要求：

1. 有效 Session Cookie；
2. 已完成首次密码修改；
3. 角色允许；
4. 可信 Origin；
5. 对象属于当前用户可见项目。

### 3.3 角色

| 角色 | 当前 HTTP 能力 |
|---|---|
| `ADMIN` | 全部比赛操作；模型制品注册、route 激活/回退、calibration 创建仅限 ADMIN |
| `MEMBER` | 项目内日常操作、Agent run、项目工具；可读取 calibration |
| `JUDGE` | 已完成密码修改后只读查看其可见项目、结果与 active route；不是 operator |

首次初始化账号若 `must_change_password=true`，除 `/v1/auth/me` 和改密流程外，业务接口返回 `403 credential_change_required`。

## 4. 幂等与并发控制

以下请求必须携带 `Idempotency-Key`：

| 接口 | 目的 |
|---|---|
| `POST /v1/agent/runs` | 相同请求重试返回同一 run；不同请求复用 key 返回 `409 idempotency_conflict` |
| `POST /v1/admin/model-routes/activations` | 防止重复追加 route 决策 |
| `POST /v1/admin/model-routes/rollbacks` | 防止重复回退 |
| `POST /v1/projects/{project_id}/advanced-calibration/materializations` | 防止重复创建物化任务 |

Route 变更还要求 `expected_previous_event_sha256`。它是乐观并发控制：调用方必须基于刚读取的 ledger head 提交；活动 route 已变化时返回 `409 model_route_state_conflict`。

不要把密码、文件路径、样本数组、模型权重或哈希清单放入 Redis。Agent 队列只传 `run_id` 和 `plan_hash`；calibration 队列只传 `materialization_id`。

## 5. Competition API 总表

### 5.1 健康与契约

| 方法 | 路径 | 权限 | 说明 |
|---|---|---|---|
| GET | `/health` | 无 | 进程级存活检查，不证明 DB、Worker、模型 route 已 READY |
| GET | `/v1/tools` | 已登录且完成改密 | 返回当前 registry 的版本化工具 Schema |

### 5.2 认证

| 方法 | 路径 | 权限 | 说明 |
|---|---|---|---|
| POST | `/v1/auth/login` | 可信 Origin | 创建 HttpOnly session |
| GET | `/v1/auth/me` | 有效 session | 返回当前 principal，可读取 `must_change_password` |
| POST | `/v1/auth/change-password` | 有效 session、可信 Origin | 修改密码并轮换 session |
| POST | `/v1/auth/logout` | 可信 Origin | 撤销 session，返回 204 |

登录失败统一返回 `401 authentication_failed`，不暴露“用户名不存在”或“密码错误”的差异。

### 5.3 项目、数据集与 record batch

| 方法 | 路径 | 权限 | 说明 |
|---|---|---|---|
| POST | `/v1/projects` | ADMIN/MEMBER、可信 Origin | 创建项目 |
| GET | `/v1/projects` | ready user | 列出当前 principal 可见项目 |
| GET | `/v1/projects/{project_id}` | ready user | 读取一个可见项目 |
| POST | `/v1/datasets` | ADMIN/MEMBER、可信 Origin | 在项目中登记数据集版本与 Schema 版本 |
| GET | `/v1/datasets/{dataset_id}` | ready user | 读取数据集 |
| POST | `/v1/datasets/{dataset_id}/freeze` | ADMIN/MEMBER、可信 Origin | 冻结数据集；后续业务绑定依赖冻结状态 |
| POST | `/v1/datasets/{dataset_id}/batches/canonical-csv` | ADMIN/MEMBER、可信 Origin | 上传 base64 canonical CSV 并创建项目级 record batch |

Canonical CSV 只接受 `payload_base64` 和 `CanonicalCsvBatchRegistration`。服务端负责 base64、大小、Schema、来源与绑定验证，客户端不能直接声明“已验证”状态。

### 5.4 Advanced 模型目录与 route

| 方法 | 路径 | 权限 | 说明 |
|---|---|---|---|
| POST | `/v1/admin/model-artifacts` | ADMIN、可信 Origin | 以 `project_id + artifact_id` 注册一个服务端已知制品 |
| POST | `/v1/admin/model-artifacts/advanced-candidates` | ADMIN、可信 Origin | 从服务端 deployment registry 注册冻结候选集合 |
| GET | `/v1/model-artifacts` | ready user | 按项目、kind、format、dataset、cutoff 过滤 |
| GET | `/v1/model-artifacts/{artifact_id}` | ready user | 读取无本地路径泄露的目录记录 |
| POST | `/v1/admin/model-routes/activations` | ADMIN、可信 Origin、幂等 key | 追加激活决策 |
| POST | `/v1/admin/model-routes/rollbacks` | ADMIN、可信 Origin、幂等 key | 追加回退决策 |
| GET | `/v1/projects/{project_id}/model-routes/active` | ready user | 返回已验证 active route 摘要 |
| GET | `/v1/model-routes/activation-events` | ready user | 按 project/task/cutoff/role 查询 append-only 决策流 |

制品注册 API 不接受路径、制品状态或客户端提供的 SHA-256。路径、manifest、模型版本、数据版本、split、feature 与 normalization SHA 均从服务器 registry 解析。

### 5.5 Advanced calibration materialization

| 方法 | 路径 | 权限 | 说明 |
|---|---|---|---|
| POST | `/v1/projects/{project_id}/advanced-calibration/materializations` | ADMIN、可信 Origin、幂等 key | 创建并派发物化任务，返回 202 |
| GET | 同上 | ADMIN/MEMBER | 列出项目内物化记录 |
| GET | `.../materializations/{materialization_id}` | ADMIN/MEMBER | 读取单条状态 |

创建 payload 仅包含：

```json
{
  "task": "RUL",
  "cutoff_cycle": 150,
  "route_role": "COVERAGE",
  "source_registration_id": "matr-three-batch-final-v1"
}
```

`source_registration_id` 可省略并使用服务端默认值。客户端不能提交 calibration 样本、路径、结果 ID、manifest SHA 或 active artifact 身份。

状态由 Worker 和数据库驱动，包括 `PENDING`、`RUNNING`、`READY`、`FAILED`、`STALE`。active route 改变后，旧 READY 证据读取时会被识别为 STALE，不能静默复用。

### 5.6 项目工具与结果

Competition runtime 注册五个项目级工具，名称沿用公共 `StandardToolName`：

| 工具名 | 当前 Advanced 语义 |
|---|---|
| `extract_early_cycle_features` | 从已绑定 record batch 生成受控 Advanced 输入 `ToolResult` |
| `predict_cycle_life` | 使用 active RUL route 输出 MATR official cycle-life |
| `predict_soh_trajectory` | 使用 active SOH route 输出截至 cycle 500 的有限时域轨迹 |
| `calibrate_prediction_interval` | 对 RUL 或 SOH 执行 route-specific Split Conformal |
| `generate_audited_report` | 只引用已登记 RUL/SOH/Conformal 结果生成报告 |

| 方法 | 路径 | 权限 | 说明 |
|---|---|---|---|
| POST | `/v1/projects/{project_id}/tools/{tool_name}` | ADMIN/MEMBER、可信 Origin | 在实时项目上下文调用工具 |
| GET | `/v1/projects/{project_id}/results/{result_id}` | ready user | 按对象权限解析登记结果 |

返回统一 `ToolResult`，核心字段包括：

```text
result_id
tool_name / tool_version
input_hash
values
uncertainty
provenance[]
warnings[]
model_version / data_version / feature_version
created_at
```

客户端不得自行构造这些业务数值或把 UI 缓存当成权威结果。

### 5.7 Agent run 与 SSE

| 方法 | 路径 | 权限 | 说明 |
|---|---|---|---|
| POST | `/v1/agent/runs` | ADMIN/MEMBER、可信 Origin、幂等 key | 持久化 intent/plan/step 并派发 Worker，返回 202 |
| GET | `/v1/agent/runs/{run_id}` | ready user | 读取 run 与审批状态 |
| GET | `/v1/agent/runs/{run_id}/results/{result_id}` | ready user | 读取属于该 run 的结果 |
| POST | `/v1/agent/runs/{run_id}/cancel` | ADMIN/MEMBER、可信 Origin | 取消非终态 run |
| POST | `/v1/agent/runs/{run_id}/approve` | ADMIN/MEMBER、可信 Origin | 批准待审批步骤并重新派发 |
| POST | `/v1/agent/runs/{run_id}/reject` | ADMIN/MEMBER、可信 Origin | 拒绝待审批步骤 |
| GET | `/v1/agent/runs/{run_id}/events` | ready user | SSE 事件流 |

创建请求：

```json
{
  "project_id": "项目 UUID",
  "user_goal": "分析该电芯的 RUL、SOH 与可信区间并生成审计报告",
  "dataset_ids": ["冻结数据集 UUID"],
  "requested_outputs": ["cycle_life"]
}
```

SSE 支持 `Last-Event-ID`。其值必须是非负整数 sequence；断线重连后服务端只返回更大的事件。终态为 `COMPLETED`、`FAILED` 或 `CANCELLED` 时流结束，空闲期发送 heartbeat。

### 5.8 通用工厂保留但不属于 competition golden path 的路由

由于 competition app 复用通用 `create_fastapi_app`，OpenAPI 仍可能列出：

- `POST /v1/tools/{tool_name}`；
- `/v1/analyses/quality` 等全局工具快捷路由；
- `GET /v1/results/{result_id}`；
- `GET /v1/reports/{result_id}`。

当前 competition registry 中的五个工具全部是 `PROJECT` scope。它们经全局 `/v1/tools/...` 调用会被授权层拒绝；全局 result/report route 也没有配置 global audit ledger。比赛客户端必须使用：

```text
/v1/projects/{project_id}/tools/{tool_name}
/v1/projects/{project_id}/results/{result_id}
/v1/agent/runs/{run_id}/results/{result_id}
```

这些兼容路由的存在不代表当前部署承诺了全局执行能力。后续若收紧 OpenAPI，应在通用工厂中显式按 runtime profile 隐藏，而不是绕过 scope 校验让它们“能调用”。

## 6. 通用错误语义

| HTTP | 含义 |
|---:|---|
| 400 | 通常不使用；契约错误归入 422 |
| 401 | Session 缺失、过期、撤销或登录失败 |
| 403 | Origin、角色、首次改密或项目访问被拒绝 |
| 404 | 对象不存在，或为避免跨项目枚举而隐藏 |
| 409 | 幂等冲突、状态冲突、route head 变化、证据不可用 |
| 422 | Pydantic 输入、枚举、过滤条件或业务标识格式无效 |
| 500 | 持久化状态破坏或工具执行违反内部契约 |
| 503 | 当前运行时没有该能力，或 project audit storage 暂不可用 |

前端只向用户展示有限错误码，详见 `frontend/lib/api-client.ts`。后端不得把异常栈、文件路径、密钥、数据库 URL 或原始敏感 payload 放进 `detail`。

## 7. 列表、过滤与分页现状

当前比赛规模下，列表端点返回项目权限过滤后的完整数组；模型制品和实验等列表支持查询过滤，但**尚无通用 `limit/offset/cursor` 分页契约**。SSE 的 `Last-Event-ID` 是事件恢复游标，不是 REST 分页。

如果未来扩展到多租户或大规模历史运行，应先在公共契约中设计稳定 cursor，再同时修改 API、service、前端和 OpenAPI，不能单独在数据库层偷偷截断。

## 8. Golden path

推荐的正式调用顺序：

```text
login
→ 首次改密
→ create project
→ create/freeze dataset
→ upload canonical CSV record batch
→ ADMIN 注册 Advanced candidates
→ ADMIN 激活 RUL/SOH routes
→ ADMIN 创建 calibration materializations
→ 等待 RUL/SOH 所需记录 READY
→ create Agent run
→ SSE 观察 step
→ 如需要则 approve/reject
→ 按 result_id 读取 ToolResult
→ UI 展示 RUL/SOH/Conformal/报告与 provenance
```

详细代码路径见[代码走读](../development/code-walkthrough.md)；部署命令以[运行与部署说明](../runtime-setup.md)为准。

## 9. 维护规则

任何公共 API 变更必须同步：

1. Pydantic 契约；
2. FastAPI adapter；
3. application service；
4. OpenAPI snapshot 或接口测试；
5. `frontend/lib/api-client.ts` 与解码器；
6. 本文对应表格；
7. 若改变信任边界，新增或更新 ADR。

不得把本文当成实现输入。发生差异时先以运行时 OpenAPI 和代码为准，再修正文档。
