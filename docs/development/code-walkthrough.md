# 代码走读：从 HTTPS 请求到可审计电池结果

本文按比赛运行时的一条真实纵向链路走读代码。目标不是逐文件复述，而是说明一个
数值如何从 Next.js 进入 FastAPI，再经过 active route、模型、Conformal、Worker
和 Audit Ledger，最终只以 `ToolResult` 返回 API 与 UI。

## 1. 从哪里开始读

建议按下列顺序阅读：

1. `deploy/competition_runtime.py`：比赛运行时 composition root；
2. `src/quanxin_life/api/app.py`：HTTP 总入口；
3. `src/quanxin_life/application/assembly.py`：项目工具和 calibration 组件装配；
4. `src/quanxin_life/tools/`：受控数值工具；
5. `src/quanxin_life/application/agent_run_execution.py`：可恢复 Agent Worker；
6. `src/quanxin_life/audit/sql_project_ledger.py`：原子结果提交与读取；
7. `frontend/lib/api-client.ts`：浏览器唯一 API client；
8. `frontend/app/` 与 `frontend/components/`：页面和可信结果呈现。

不要从某个神经网络类直接推断产品行为。模型能否被调用，还取决于项目权限、冻结数据、active route、artifact manifest、calibration READY 和 Agent step 绑定。

## 2. 进程装配：没有隐式全局单例

### 2.1 API 与 Worker 共用的持久化边界

`deploy/competition_runtime.py:create_competition_runtime` 显式创建：

- SQLAlchemy engine 与 `SessionFactory`；
- `ProjectInvocationContextService`；
- `SqlProjectAuditLedger`；
- `AuthService`；
- 文件系统 record batch store；
- Advanced deployment registry；
- model artifact catalog 与 route service；
- active-route-aware runtime resolver；
- 项目级 ToolInvocationService；
- Celery app；
- Agent run service 与执行 Worker；
- calibration service、Worker、HTTP adapter；
- FastAPI app。

模块导入不会自动连接数据库、Redis、网络或加载模型。真实连接与模型解析发生在显式运行时路径中。

### 2.2 Foundation 与 competition

`src/quanxin_life/api/app.py:create_fastapi_app` 接受多个可选 adapter，是通用 HTTP 外壳。`deploy/competition_runtime.py` 只传入比赛必须的 adapter，因此“仓库中有路由文件”不代表公网服务已装配它。

competition runtime 的项目工具来自：

```python
create_project_prediction_tool_invocation_service(
    ProjectPredictionToolDependencies(...)
)
```

Registry 只注册 Advanced 正式链需要的五个 PROJECT 工具。工具仍复用公共 `StandardToolName`，避免另造含义相同的 API DTO。

## 3. 登录与项目权限

### 3.1 浏览器请求

`frontend/lib/api-client.ts:apiRequest`：

- 从 `NEXT_PUBLIC_API_BASE_URL` 解析 API；
- 生产或非本机地址强制 HTTPS；
- 设置 `credentials: "include"`；
- 禁用 fetch cache；
- 只暴露有限的服务端错误码。

### 3.2 Session 创建

`POST /v1/auth/login` 进入：

```text
api/auth.py
→ AuthService.login
→ SQLAlchemy transaction
→ Argon2id 密码验证
→ HttpOnly Cookie
```

Cookie 不返回给业务 JavaScript。生产 cookie 使用 `Secure + HttpOnly + SameSite=Strict`。

### 3.3 每次业务访问都重建作用域

项目端点不会相信前端声称的 owner 或 role。`ProjectInvocationContextService.resolve_http` 使用数据库中的 session、用户、角色和项目状态构造 `VerifiedProjectInvocationContext`。项目不存在、用户无权访问或账号失效时，请求在进入数值工具之前失败。

## 4. 数据进入：登记、冻结、绑定

### 4.1 数据集登记

`POST /v1/datasets` 只登记：

- `project_id`；
- `name`；
- `data_version`；
- `schema_version`。

随后 `POST /v1/datasets/{dataset_id}/freeze` 冻结可变元数据。下游不能把仍可任意修改的数据集当成正式输入。

### 4.2 Canonical CSV

`POST /v1/datasets/{dataset_id}/batches/canonical-csv` 调用：

```text
api/record_batches.py
→ CanonicalCsvUploadRequest.decoded_payload
→ RecordBatchBindingService.register_canonical_csv
→ FileSystemVerifiedEarlyCycleBatchStore
→ 项目/数据集/record batch 持久绑定
```

传输层只负责 base64 和大小限制。Service 再验证项目作用域、冻结状态、canonical schema、数据来源与绑定冲突。正式 split 继续按 `cell_id`，不能把逐 cycle 行随机分到不同集合。

## 5. 模型治理：制品先登记，再激活 route

### 5.1 服务端 deployment registry

`AdvancedDeploymentBundleRegistry` 从部署目录读取 Advanced bundle v2。`AdvancedDeepModelArtifactCatalogSource` 只解析登记在 bundle 中且通过 manifest/SHA-256 约束的候选。

`POST /v1/admin/model-artifacts/advanced-candidates` 不接收文件路径或客户端 SHA；它只接收 `project_id`，服务端解析冻结候选集合并写入项目 catalog。

### 5.2 Append-only route

`POST /v1/admin/model-routes/activations` 进入：

```text
api/model_routes.py
→ ModelRouteActivationService.activate
→ 校验 ADMIN、项目、artifact/task/cutoff/role
→ 校验 Idempotency-Key
→ 比较 expected_previous_event_sha256
→ 追加 activation event
→ 更新可验证 ledger head
```

回退同样追加新事件，不覆盖历史。`ActiveAdvancedRuntimeResolver` 在每次正式推理或 calibration 准备时重新解析当前 active route；不能仅凭上一次 UI 缓存继续加载旧模型。

### 5.3 安全加载

`ManagedAdvancedRuntimeProvider` 依据服务器 registry 定位模型，校验 artifact manifest、模型版本、数据/split/feature 版本和 normalization SHA。正式制品采用 safetensors；不从请求或数据库字符串加载 pickle、joblib、`.pt` 或 `.pth`。

## 6. Calibration：从身份请求到 READY 证据

### 6.1 创建

Calibration 页面通过：

```text
frontend/components/calibration-materialization-panel.tsx
→ frontend/lib/api-client.ts
→ POST /v1/projects/{project_id}/advanced-calibration/materializations
```

请求只包含 task、cutoff、route role 和可选 source registration ID。`AdvancedCalibrationMaterializationService.create` 在数据库中创建幂等 `PENDING` 记录。

API 随后调用 `CeleryAdvancedCalibrationQueue.enqueue`。Redis 消息只有：

```json
{"materialization_id": "UUID"}
```

样本、路径、模型哈希、project context 和审批信息全部由 Worker 从可信持久化状态重建。

### 6.2 Claim、租约和恢复

`AdvancedCalibrationMaterializationWorker.execute`：

1. 锁定 materialization 记录；
2. 对 `PENDING`、可重试 `FAILED` 或租约过期 `RUNNING` 获取 fenced claim；
3. 对仍有有效租约的 `RUNNING` 返回 Busy，让 Celery 延迟重试；
4. 从数据库重建 ADMIN project context；
5. 重新解析 active route、artifact 和 evidence registration；
6. 读取已登记 calibration cohort；
7. 对每个合法样本运行受控预测工具；
8. 构造 manifest 和结果集合；
9. 通过 ledger 原子提交；
10. 状态变为 `READY`。

Claim token 不以明文持久化，只存 SHA-256。旧 Worker 即使在崩溃后恢复，也不能凭过期 claim 把新 Worker 的结果覆盖为 READY 或 FAILED。

### 6.3 原子提交

`SqlProjectAuditLedger.commit_advanced_calibration_materialization` 在同一数据库事务中校验：

- materialization identity；
- 当前 claim attempt 与 lease；
- 当前 active route head；
- artifact manifest；
- source identity；
- 样本结果数量与契约；
- RUL/SOH 对应工具、版本与 provenance；
- sample manifest SHA。

只有这些条件同时成立，样本 `ToolResult` 和 materialization READY 状态才一起提交。Route 已变化时旧结果不会被“补记”为当前证据。

## 7. Agent run：计划不是执行权限

### 7.1 Intent 与 plan 持久化

`POST /v1/agent/runs` 接收 `SupervisorPlanningRequest`。`SupervisorPlanner` 可以使用 LLM 生成结构草案，但当前 competition runtime 传入 `gateway=None`，使用确定性 fallback。

无论草案来自哪里，都必须满足：

- 最多 12 个 step；
- tool 在当前 PROJECT registry 中；
- role/tool allowlist；
- input reference 白名单；
- 依赖拓扑合法；
- approval 与 failure policy 合法；
- `plan_hash` 与规范化内容一致。

`AgentRunService.create_run` 在数据库中持久化 exact intent、plan 和每个 `AgentStep`。LLM 文本本身不携带模型路径、结果数值或执行权限。

### 7.2 Outbox 与 identity-only queue

创建后 `dispatch_pending` 发送：

```json
{
  "run_id": "UUID",
  "plan_hash": "64 位小写 SHA-256"
}
```

`src/quanxin_life/infrastructure/celery_queue.py` 强制 JSON serializer、UTC、late ack、worker lost reject 和 prefetch 1。Redis 中不放用户完整 prompt、数据、密码、ToolResult 或模型制品。

### 7.3 Worker 逐 step 执行

`AgentRunExecutionWorker.execute`：

1. 按 `run_id + plan_hash` 加载持久 intent/plan/step；
2. 验证 exact step rows 与 plan 一致；
3. 找到第一个未完成 step；
4. 对 RUNNING step 检查租约；
5. 获取新的 fenced claim；
6. `AdvancedAgentExecutionContextResolver` 从持久 intent 推导目标 batch；
7. 重新解析 RUL/SOH active route；
8. 查找与 task/cutoff/role/artifact 精确匹配的 READY calibration；
9. 拒绝目标电芯出现在 calibration cohort；
10. 将前序 `ToolResult` 作为 result reference 解析；
11. 调用 `ToolInvocationService.invoke_in_project`；
12. 原子提交结果并把 exact step 标记 COMPLETED。

请求取消、拒绝审批或 claim 过期都会在数据库状态机中显式表现，不靠 Celery 内存推测。

## 8. 五步 Advanced 数值链

一条完整目标电芯分析通常为：

```text
extract_early_cycle_features
→ predict_cycle_life
→ predict_soh_trajectory
→ calibrate_prediction_interval（RUL/SOH）
→ generate_audited_report
```

### 8.1 Advanced 输入

`tools/advanced_input.py` 读取项目绑定的 record batch，输出版本化输入 `ToolResult`。目标 cell、cutoff、数据版本和来源在这里冻结，后续模型不直接重新解释上传文件。

### 8.2 RUL

`tools/advanced_cycle_life_prediction.py` 调用 `AdvancedRULPredictionService`：

```text
输入 ToolResult
→ target-aware batch
→ 当前 RUL route
→ safetensors runtime
→ MATR official cycle-life
→ RUL ToolResult
```

这里的 target 是官方 MATR cycle life，不被 UI 或 Agent 改写为统一 EOL80。

### 8.3 SOH

`tools/advanced_soh_prediction.py` 使用当前 SOH route 输出有限预测时域。正式协议截至 cycle 500；未跨越阈值不能被描述成“已可靠预测无限寿命”。

### 8.4 Split Conformal

`tools/advanced_conformal.py` 有两阶段操作：

1. 从与当前 route 精确匹配的 calibration sample results 计算 calibration；
2. 将已登记 prediction result 与 calibration result 结合签发 interval。

RUL 与 SOH 使用不同任务、route role 和样本契约。旧 route、不同 cutoff、不同 artifact 或包含目标 cell 的 calibration 会被拒绝。

### 8.5 报告

`tools/advanced_project_report.py` 只接受 ledger 中已登记的 RUL、SOH 和 Conformal result IDs。报告中的业务数字从这些 `ToolResult` 引用，不由 LLM 或 Markdown 模板生成。

## 9. Ledger 为什么是数值防火墙

`SqlProjectAuditLedger` 不只是日志表。它负责：

- 对 `ToolResult` 做规范化与哈希验证；
- 将结果绑定到 project、run、exact Agent step；
- 验证 step 处于正确 RUNNING claim；
- 将 result 写入与 step COMPLETED 合并为同一事务；
- 拒绝跨项目 result reference；
- 保存模型、数据、feature、provenance 和 warnings；
- 为 GET API 提供对象授权后的权威读取。

因此“模型函数返回了一个 Python 数字”不等于“产品可展示结果”。只有完成 ledger 注册并得到 `result_id`，前端和报告才可使用。

## 10. API 到 UI

前端没有第二套数值计算：

- `frontend/lib/api-client.ts` 负责认证 Cookie、错误码与 REST 调用；
- `frontend/components/calibration-materialization-contract.ts` 对 calibration 和 active route 做运行时解码；
- `frontend/app/agent/runs/[runId]/page.tsx` 消费 run 和 SSE；
- `frontend/components/single-cell-analysis.tsx` 以 `result_id` 加载 RUL/SOH；
- `frontend/components/soh-trajectory-chart.tsx` 只绘制 ToolResult 中的轨迹；
- `frontend/components/evidence-panel.tsx` 呈现 provenance、版本与 warning；
- `frontend/app/results/[resultId]/page.tsx` 提供可定位结果页。

前端可以格式化、排序、画图和解释状态，但不能补点、平滑、外推或硬编码演示数值。

## 11. 断线、重启与失败路径

### 11.1 浏览器断线

Agent 事件通过 SSE 输出。客户端用 `Last-Event-ID` 重连，服务端从持久 event sequence 继续；浏览器断开不会取消 Worker。

### 11.2 Worker 崩溃

Celery late ack 与 `reject_on_worker_lost` 负责重新投递，数据库 lease/fence 决定旧任务是否仍有写权限。两者缺一不可。

### 11.3 Redis 重启

Redis 只承载任务身份，不承载业务真相。已提交的 run、step、materialization 和 ToolResult 在 PostgreSQL；未成功派发的 Agent run 保留 PENDING outbox，可再次派发。

### 11.4 Route 变化

Route 变化会使不再匹配的 READY calibration 变为 STALE。新 Agent run 必须先创建对应 route 的 materialization，不能继续使用旧半径。

## 12. 调试时从哪里查

| 症状 | 第一检查点 |
|---|---|
| `401 authentication_failed` | Cookie、session 是否过期/撤销 |
| `403 credential_change_required` | 首次密码是否已修改 |
| `403 origin_not_allowed` | 浏览器 Origin 与 `QUANXIN_TRUSTED_ORIGIN` |
| 数据集无法上传 | 是否冻结、canonical CSV、base64 与大小 |
| active route 为空 | 候选是否已注册、activation ledger 是否完整 |
| calibration 长时间 PENDING | `advanced-calibration` 队列与 Worker 日志 |
| calibration STALE | 当前 route head 是否已变化 |
| Agent run PENDING | outbox 是否派发、`agent-runs` 队列 |
| Agent run RUNNING 不动 | step lease、Busy retry、Worker 日志 |
| 项目结果 404 | result 是否属于当前 project/run |
| UI 有结果但证据为空 | 视为缺陷；检查 ToolResult provenance 与解码器 |

不要先修改数据库状态。应从 API 返回码、PostgreSQL 状态、Celery task identity、active route 和 ledger result binding 顺序定位。

## 13. 一个最小阅读练习

若要验证“UI 中的 RUL 是否真来自模型”，沿以下路径逐一对照：

```text
frontend/app/projects/[projectId]/cells/[recordBatchId]/page.tsx
→ frontend/components/single-cell-analysis.tsx
→ frontend/lib/api-client.ts:invokeProjectTool / getProjectResult
→ src/quanxin_life/api/app.py:invoke_project_tool
→ src/quanxin_life/api/service.py:ToolInvocationService.invoke_in_project
→ src/quanxin_life/tools/advanced_cycle_life_prediction.py
→ src/quanxin_life/application/advanced_prediction.py
→ src/quanxin_life/application/advanced_runtime.py
→ safetensors artifact + verified manifest
→ src/quanxin_life/audit/sql_project_ledger.py
→ ToolResult.result_id
→ UI evidence panel
```

任一环节缺少项目作用域、active route、manifest SHA、result ID 或 provenance，都不应被视为正式业务输出。

## 14. 相关文档

- [API 文档](../api/README.md)
- [系统架构](../../ARCHITECTURE.md)
- [数据契约](../../DATA_CONTRACT.md)
- [实验协议](../../EXPERIMENT_PROTOCOL.md)
- [复现说明](../reproducibility.md)
- [运行环境](../runtime-setup.md)
- [当前状态](../status.md)
