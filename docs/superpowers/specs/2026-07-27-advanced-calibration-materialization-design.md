# Advanced Calibration Materialization 设计

日期：2026-07-27  
状态：已批准，待实施  
对应计划：[2026-07-19-quanxin-product-completion-and-a100-integration.md](../plans/2026-07-19-quanxin-product-completion-and-a100-integration.md) Task 3

## 1. 目标

为正式 Advanced RUL/SOH 推理链提供可信、项目级、route-specific 的
calibration sample ToolResult，使 Split Conformal、Agent、报告和 Next.js UI
不再依赖人工提供的 sample result ID。

本设计只覆盖 MATR 已批准模型和已冻结 calibration split，不扩展为跨域
recalibration，不改变现有模型晋级结论，也不重新训练 A100。

完成后的纵向链为：

```text
ADMIN 准备 calibration evidence
→ 服务端解析冻结 calibration split
→ 解析 exact active route
→ 生成 route-specific calibration sample ToolResult
→ 原子持久化 materialization
→ Agent 服务端解析 sample result IDs
→ Split Conformal
→ 报告
→ 单电芯 UI
```

## 2. 不可违反的边界

调用方、LLM、Agent 和浏览器均不得提交或修改：

- calibration cell ID；
- observed cycle life；
- observed SOH；
- predicted cycle life 或 predicted SOH；
- calibration sample result ID；
- 文件路径；
- 模型、数据、split、feature 或 normalizer 版本；
- artifact、manifest 或来源 SHA-256。

所有业务数值和身份必须由服务端从冻结证据、active route 和持久
ToolResult 派生。目标电芯推理继续保持 label-free。

## 3. 权威数据来源

### 3.1 RUL

`observed_cycle` 只能来自 SHA-256 已验证的三批 MATR 转换证据中的
`MatrCellConversionEvidence.official_life_label`，并满足：

- `official_life_right_censored=False`；
- label 为有限正整数；
- label 大于 cutoff；
- cell 属于冻结 `SplitManifest.calibration`。

不得从 EOL80、SOH 轨迹、目标 record batch 或预测结果反推 RUL 标签。

### 3.2 SOH

`observed_soh` 只能来自 `MatrSupervisionArtifact` 指向且 SHA-256 已验证的
supervision Parquet，并满足：

- cell 属于冻结 `SplitManifest.calibration`；
- 只取 `cutoff + 1` 到 cycle 500 的真实周期；
- cycle 唯一、连续且与正式模型输出轴一致；
- SOH 由真实 discharge capacity / reference capacity 得到；
- 所有数值有限并位于有效范围；
- 不进行插值补标签，不外推到 cycle 500 之后。

### 3.3 Split 与来源闭包

服务端必须复验：

- three-batch manifest；
- combined/component split manifest；
- conversion evidence；
- supervision report 和 Parquet；
- training input bundle SHA-256；
- data、split 和 feature version。

已知本地 reconstructed input bundle 与 A100 training input bundle 字节哈希不同，
必须在来源身份中显式保留，不能静默声称一致。

## 4. 方案选择

采用项目级异步预物化方案。

不采用以下方案：

1. 单电芯 Agent 临时计算整个 calibration cohort：会重复计算、超过合理步骤数，
   并把固定 calibration 证据与目标电芯错误耦合。
2. 浏览器上传 calibration CSV/Parquet：会产生自报数值和来源注入风险。
3. 直接复用 legacy `CalibrationCohort`：该表不绑定 Advanced task、route、
   artifact、normalizer、sample ToolResult 和激活账本身份。

离线结果包以后可以作为服务端登记的可信 source adapter，但不能成为客户端
上传业务数值的旁路。

## 5. 持久化设计

新增 migration：

```text
migrations/versions/0014_advanced_calibration_materializations.py
```

新增父表 `advanced_calibration_materializations`，至少包含：

- `id`；
- `project_id`；
- `task`；
- `cutoff_cycle`；
- `route_role`；
- `status`：`PENDING / RUNNING / READY / FAILED / STALE`；
- data、split、feature version；
- artifact ID、artifact manifest SHA-256；
- normalizer SHA-256；
- activation decision event、sequence 和 ledger head；
- source identity SHA-256；
- sample manifest SHA-256；
- sample count；
- created/started/completed timestamps；
- created by user；
- 非敏感失败代码。

新增子表 `advanced_calibration_sample_bindings`，至少包含：

- `materialization_id`；
- 稳定 `ordinal`；
- `cell_id`；
- `result_id`；
- sample ToolResult SHA-256。

必须具有：

- materialization 内 ordinal、cell 和 result 的唯一约束；
- exact project/task/cutoff/route/runtime identity 的幂等唯一约束；
- SHA-256 长度约束；
- task/route 合法组合约束；
- 样本 ToolResult、project binding、sample binding 和 READY 状态原子提交。

中途失败不得留下可被 Agent 解析的部分 cohort。

## 6. 应用服务

### 6.1 Source Resolver

`AdvancedCalibrationEvidenceResolver` 负责：

- 只从服务器登记的 evidence root 解析来源；
- 防止路径逃逸和符号链接逃逸；
- 逐层验证 schema 与 SHA-256；
- 返回完整 calibration partition 和监督证据；
- 不加载 pickle、joblib、`.pt` 或 `.pth`。

产品路径应使用轻量 reader，不直接依赖会初始化训练模型的完整 final data loader。

### 6.2 Sample Producer

`AdvancedCalibrationSampleProducer` 负责：

- 根据 task、cutoff 和 route 解析 exact active runtime；
- 为 calibration cell 构建 label-free early sequence；
- 使用正式 safetensors runtime 推理；
- 从可信来源合并 observed supervision；
- 生成现有 evidence type：
  - `quanxin_life.advanced_rul_calibration_sample.v1`
  - `quanxin_life.advanced_soh_calibration_sample.v1`
- 执行前后重新验证 active route，防止 TOCTOU。

### 6.3 Materialization Service

`AdvancedCalibrationMaterializationService` 负责：

- 创建或幂等解析 materialization；
- 限制 calibration cell 数和 SOH 总轨迹长度；
- 生成全部 sample ToolResult；
- 构建稳定 sample manifest；
- 在单一数据库事务中持久化结果；
- 查询状态并在 route 变化时返回 `STALE`；
- 不向 UI 返回逐电芯 observed/predicted 数组。

## 7. API

新增项目级管理端点：

```text
POST /v1/projects/{project_id}/advanced-calibration/materializations
GET  /v1/projects/{project_id}/advanced-calibration/materializations
GET  /v1/projects/{project_id}/advanced-calibration/materializations/{materialization_id}
```

POST 请求只允许：

- `task`；
- `cutoff_cycle`；
- `route_role`；
- 可选的服务端登记 `source_registration_id`。

POST 要求：

- ADMIN；
- 认证 Cookie；
- trusted origin；
- `Idempotency-Key`；
- ACTIVE project。

GET 可供同项目 ADMIN/MEMBER 查看非敏感摘要。跨项目、撤销 session、未知记录和
非 ACTIVE project 均失败关闭。

## 8. Agent 接入

固定单电芯 Agent 计划结构保持不变。新增生产级服务端 execution context resolver，
在 dispatch/step 输入冻结前：

1. 解析目标 record batch；
2. 冻结 project、dataset、cutoff、cell 和版本；
3. 解析当前 RUL coverage route 与 SOH route；
4. 查找 exact READY materialization；
5. 逐个重验 sample ToolResult、provenance 和 project binding；
6. 拒绝目标 cell 属于 calibration cohort；
7. 按持久 ordinal 返回唯一 UUID tuple；
8. 将 tuple 写入：
   - `context.rul_calibration_sample_result_ids`
   - `context.soh_calibration_sample_result_ids`

sample result ID 不得来自 intent、Planner、query string 或 UI local state。

Conformal step 继续通过 exact AgentStep、resolved input hash、fenced claim 和
原子 ledger commit 持久化。

## 9. UI

新增项目页面：

```text
/projects/{projectId}/calibration
```

项目页新增“校准证据”入口。

页面使用紧凑表格展示：

- task；
- cutoff；
- route；
- active model/artifact；
- `PENDING / RUNNING / READY / FAILED / STALE`；
- sample count；
- data/split/feature version；
- artifact、normalizer 和 sample manifest SHA-256；
- 创建与完成时间；
- 非敏感失败原因。

ADMIN 可对合法 active route 点击“准备校准证据”；MEMBER 只读。页面不提供自由文本
数值输入，不展示逐电芯 observed label 或完整预测数组。

异步状态通过有界轮询刷新；按钮具有 loading、disabled、幂等重试和错误状态。
移动端将表格转换为不嵌套的紧凑条目，确保 SHA 文本换行且无横向溢出。

现有单电芯页面不增加 calibration 数值，只继续展示最终 RUL/SOH、Conformal
interval/band、报告和来源证据。

## 10. 错误处理

以下情况必须失败关闭：

- 请求包含任何业务数值、cell ID、路径、自报版本或 SHA；
- source manifest、split、supervision 或 artifact SHA 不一致；
- calibration cell 缺失、重复或跨 split；
- RUL label censored、非法或不大于 cutoff；
- SOH cycle 缺失、重复、非有限、超范围或超过 cycle 500；
- task/route/cutoff 非法；
- active route 在生产过程中变化；
- project、dataset、runtime、artifact 或 normalizer 身份不一致；
- 并发导入产生不同内容；
- sample 数量无法支持请求 coverage；
- ToolResult、provenance 或 binding 被篡改；
- target cell 属于 calibration cohort。

日志只记录标识符、状态和摘要，不记录完整标签、预测轨迹或用户提示。

## 11. 测试

所有实现按测试先行完成。

后端至少覆盖：

- 请求模型拒绝自报数值和路径；
- source SHA/schema/split 验证；
- RUL official label；
- finite-horizon SOH；
- route identity 与 TOCTOU；
- 原子回滚和幂等；
- 跨项目、撤销 session 和权限；
- materialization READY/STALE；
- Agent 服务端解析 sample IDs；
- sample/result/binding 篡改；
- finite-sample coverage 拒绝；
- API 到 Agent/报告的集成链。

前端至少覆盖：

- ADMIN 创建；
- MEMBER 只读；
- READY、RUNNING、FAILED、STALE；
- 请求不包含业务数值；
- API 错误和重试；
- 缺少证据时不展示业务数字；
- 手机与桌面布局。

最终门禁包括 pytest、leakage、integration、e2e、Ruff、mypy、compileall、
前端测试、TypeScript、ESLint、Next.js build 和浏览器视觉检查。

`pip-audit` 当前已知存在 `diskcache` 和 `GitPython` 漏洞，除非本切片同时完成
依赖升级与兼容性验证，否则必须继续作为未通过门禁报告。

## 12. 完成定义

本切片只有同时满足以下条件才完成：

- ADMIN 可以为合法 active route 创建可信 materialization；
- 所有 sample 来自服务器冻结来源和正式 runtime；
- sample 与 manifest 原子持久化；
- Agent 自动解析 sample result IDs；
- Split Conformal、报告和单电芯 UI 不再依赖人工 sample IDs；
- calibration 管理 UI 可显示完整 readiness 和 SHA 证据；
- 缺失、过期、篡改或非法输入均失败关闭；
- 后端、前端和浏览器门禁通过；
- 未伪造生产数据库、ADMIN、route 激活或业务数值。
