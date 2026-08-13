# 飞书/Aily 与现有电池模型工具联动

## 状态与范围

本文件描述第一阶段接入及其 BLAST-Lite 情景扩展。交互层是飞书机器人、飞书卡片、多维表格、受审计报告和 Aily；独立 Web 前端不在本阶段范围。本阶段不训练模型，不运行 PBT/MAGNet，也不自动改变任何模型 activation 状态。

当前 Advanced 路由仍受服务器端路由、制品清单、SHA-256、支持域和人工审批约束。`CONDITIONAL` 或 `NOT_ACTIVATED` 路由不能展示预测数值。PBT 与 MAGNet 完成训练、独立评估、制品核验和人工晋级前，不会出现在可执行路由中。

BLAST-Lite 数值路由独立于上述深度学习 activation。当前两个参考路由均为 `REGISTERED_CANDIDATE`，证据等级为 `PHYSICS_REFERENCE`；只有显式启用候选情景执行与候选结果展示时，才允许在 manifest 支持范围内生成和展示 ToolResult。它们不是海辰产品模型，也不构成 15～25 年真实寿命验证。

第一阶段附件仅接受当前数据层已审查的 canonical CSV。Parquet、XLSX、ZIP、pickle、joblib、`.pt`、`.pth` 和可执行文件均拒绝。后续格式只能在数据层契约、内容检查和测试完整后启用。

## 架构

```text
飞书回调 / Aily Bearer 连接器
  -> 验签、解密、verification token、重放窗口、receipt 租约
  -> 只保留 chat/user/message/file/run 等机器引用
  -> 下载 CSV、大小/MIME/magic/UTF-8/SHA-256 检查
  -> 现有 RecordBatch 登记与 validate_battery_data
  -> 固定业务工具白名单与服务器端模型路由
  -> ToolRegistry -> ToolResult -> AuditLedger
  -> Audited Card / Bitable 摘要 / Audited Report Artifact
  -> 飞书消息、卡片、文件和 Aily 只读解释
```

飞书和 Aily 只看到稳定业务工具名：

| 业务工具 | 第一阶段行为 |
|---|---|
| `validate_battery_data` | 可用；所有预测的硬前置门禁 |
| `predict_cycle_life` | 仅在服务器端存在已激活、域内且经人工批准的路由时可用 |
| `predict_soh_trajectory` | 仅在服务器端存在已激活、域内且经人工批准的路由时可用 |
| `compare_operation_scenarios` | 受支持范围约束的 BLAST 参考情景；候选执行与展示默认关闭 |
| `project_storage_lifetime` | 1～25 年自然时间参考情景；候选执行与展示默认关闭 |
| `ingest_observed_soh` | 映射现有观测 SOH 接入契约 |
| `update_trajectory` | 映射现有在线更新契约 |
| `generate_audited_report` | 仅消费账本中有效的 `result_id` 和字段路径 |

模型选择只发生在服务器端。CyclePatch、Hybrid、PBT、MAGNet 等内部名称不作为面向用户的
飞书/Aily 业务工具名；卡片和报告的审计证据层仍可展示实际 route 与 model metadata。

## BLAST-Lite 情景边界

受控 vendor 来源固定为上游 commit `b093495b47dc40dd96dba865d91f553619501e94`，许可证和来源见 `THIRD_PARTY_NOTICES.md` 与 wheel 内的 vendor LICENSE。`Lfp_Gr_SonyMurata3Ah_Battery` 只用于 Naumann LFP/石墨数据的一致性和工况方向核验，不能缩放为大型方形电芯。`Lfp_Gr_250AhPrismatic` 是大型方形 LFP 储能电芯参考模型，不得称为海辰 280Ah 专属模型。

场景支持范围来自版本化 route manifest 和验证数据 manifest，不在 Aily、卡片或提示词中手写。温度、SOC、DoD、充放电倍率、每年 EFC、静置时间、1～25 年 horizon 和连续 `ScenarioSegment` 都在工具入口校验。域内允许情景计算，边界附近保留警告；超界、非 LFP、格式不匹配、3Ah/250Ah 误路由、缺字段、时间空洞或冲突均拒绝。

长期曲线是时间、SOC、温度和循环序列驱动的确定性物理参考情景。15/20/25 年 milestone 只在 ToolResult 中对应时间点存在时展示。当前不把 sensitivity envelope、scenario range 或 model disagreement 称为统计置信区间，也不支持任意当前 SOH 状态注入 BLAST 内部状态。

## 飞书应用

在飞书开放平台创建企业自建应用并启用机器人、事件订阅和交互卡片。不要把 App Secret、verification token 或 encrypt key 写入仓库、卡片、日志或多维表格。

建议从以下最小权限开始，并在飞书控制台按当前权限名称复核后逐项审批：

- 以应用机器人身份发送消息；
- 接收群聊中提及机器人的消息和私聊消息；
- 读取消息附件资源；
- 上传消息文件和图片；
- 更新机器人发送的交互卡片；
- 对指定多维表格执行记录查询、创建和更新。

不要授予云文档全盘访问、通讯录全量读取或管理员级权限。应用可见范围只包含批准的测试群和生产项目成员。

事件回调地址为：

```text
https://YOUR_REVIEWED_HOST/v1/integrations/feishu/events
```

订阅消息接收事件和卡片操作事件。启用加密回调后，生产装配必须提供经审查的 decryptor。普通事件先对原始字节验签，再解密、检查 verification token、解析机器引用并原子领取 receipt。飞书 URL verification 的加密握手可能不携带签名头；该分支只能在解密后明确识别为 `url_verification` 且 verification token 匹配时返回 challenge。其他无签名加密事件必须拒绝。耗时任务必须在持久化队列成功入队后快速回执；失败时释放租约以允许有限重试。

事件表和日志不保存完整消息正文。持久化映射只包含必要的 `event_id`、`chat_id`、`user_id`、`message_id`、`file_key`、项目、数据批次和 `run_id`。

## 附件与数据

附件下载后按以下顺序处理：

1. 文件大小、basename、扩展名和 content type；
2. executable、archive、Parquet 和危险模型制品 magic bytes；
3. UTF-8 与 canonical CSV 边界；
4. 计算 SHA-256，并与声明值核对；
5. 使用现有数据层登记 RecordBatch 和来源；
6. 调用 `validate_battery_data`；
7. 只有未阻断的验证结果才能进入预测工具。

缺测、无 `cell_id`、单位不明、域外、循环范围不足或数据验证失败时明确拒绝。不会静默填补。数据集训练和评估仍必须按 `cell_id` 分割。

生产装配还要求运维预先登记每一份获准 canonical CSV。复制
`deploy/feishu-csv-registrations.example.json` 为
`feishu-csv-registrations.json`，计算文件实际 SHA-256，并在一条 registration 中保持
以下三处完全相同：

1. 顶层 entry 的 `payload_sha256`；
2. `registration.metadata.source_sha256`（即 `metadata.source_sha256`）；
3. 至少一条 `registration.provenance` 中 `source_kind=OBSERVED` 记录的 `sha256`。

登记还必须包含已核验的 `cell_id`、LFP chemistry、标称容量、来源 URI、数据/切分版本和
早期循环特征配置。Worker 按下载后字节的精确 SHA 查找登记；空注册表、unknown payload、
三处哈希不一致或没有 `OBSERVED` 来源都会拒绝，不从文件名、群聊文本或 Aily 提示中补写。
正式 runtime 通过 `QUANXIN_FEISHU_CSV_REGISTRATIONS_FILE` 读取该只读文件，Compose
override 固定为 `/srv/quanxin/config/feishu-csv-registrations.json`。

## 卡片与审计

状态卡片支持 received、data-check、queued、running、success、rejected/degraded 和 report-ready。状态卡片只含机器引用。

结果卡片只能以 `run_id` 和 `result_id` 构建。Builder 从审计账本重建 `ToolResult`，按工具名和工具版本选择固定 JSON 路径，并在展示前重新检查路由授权。每个展示值旁保留：

```text
result_id#values.approved.path
```

卡片同时显示 route、activation、模型版本、数据版本、特征版本、证据等级、支持域和警告。结果不存在、版本未知、来源无效、路径不在白名单或路由未激活时返回拒绝卡片，不显示数值。调用方不能把任意数值参数传给结果卡片。

## 多维表格

使用 `run_id` 作为幂等键。建议字段如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `run_id` | 文本，唯一 | 幂等键 |
| `task_type` | 文本/单选 | 稳定业务工具名 |
| `task_status` | 文本/单选 | 任务状态 |
| `data_batch_id` | 文本 | RecordBatch 引用 |
| `input_file_sha256` | 文本 | 输入文件 SHA-256 |
| `cell_reference` | 文本 | cell 或批次引用，不写大数组 |
| `scenario_context_id` | 文本 | 受控场景输入引用 |
| `scenario_id` | 文本 | 基准或主场景标识 |
| `scenario_version` | 文本 | 场景版本 |
| `primary_result_id` | 文本 | 主要 ToolResult 引用 |
| `model_route` | 文本 | 服务器端路由引用 |
| `model_version` | 文本 | 模型版本 |
| `data_version` | 文本 | 数据版本 |
| `feature_version` | 文本 | 特征版本 |
| `evidence_level` | 文本/单选 | 证据等级 |
| `warnings` | 多行文本 | 原样警告 |
| `report_link` | URL | 受控报告链接 |
| `created_at_utc` | 日期时间 | UTC |
| `updated_at_utc` | 日期时间 | UTC |

Writer 先查询 `run_id`：无记录则创建，一条则更新，多条或分页无法证明唯一性时拒绝。单进程内使用锁避免并发覆盖；多副本生产部署还应在持久化任务层按 `run_id` 串行化，并配置飞书端唯一性治理。轨迹数组、预测数组和大体积制品保存在审计制品存储中，表格只写引用。

## 报告交付

正式报告必须先调用现有 `generate_audited_report` 工具。该工具只接收受控报告类型、`result_id` 和白名单字段路径；数值由 `AuditLedger` 解析。交付层再使用 `AuditedReportArtifactExporter` 从账本按报告结果 ID 导出 Markdown、JSON 或已安装的受审查格式。

飞书上传前重新计算制品 SHA-256。上传成功后只发送 `file_key`，交付收据只保存报告 ToolResult、格式、文件名、SHA、file key 和 message ID。PDF 必须使用经 SHA 核验的字体；可选渲染依赖缺失时明确失败，不降级为 LLM 文本。

## Aily

Aily 通过受保护 HTTP façade 接入，不直接调用模型，也不连接当前无连接器鉴权的通用 MCP Host。OpenAPI 文件：

```text
docs/integrations/aily-connector-openapi.yaml
```

稳定操作为：创建受控场景上下文、创建分析任务、查询任务状态、读取 run-bound ToolResult、下载 run-bound 审计报告。情景请求先提交 `OperationScenario`，服务端从已验证 batch 解析 chemistry、capacity、数据版本和来源并返回 `scenario_context_id`；随后创建任务时只能提交该引用，不能再附带业务结果数字。所有请求使用：

```http
Authorization: Bearer ${FEISHU_CONNECTOR_API_KEY}
```

服务端使用常量时间比较。生产环境只允许 TLS，API Gateway/负载均衡层还应限制来源网络、请求速率和凭证轮换，并将调用身份、`run_id`、`result_id`、结果状态和 UTC 时间写入审计日志。禁止把受保护端点或通用 MCP 无鉴权暴露到公网。

Aily 系统提示词位于：

```text
docs/integrations/aily-system-prompt.md
```

提示词要求只调用工具、逐项引用 ToolResult、不得补写数值、不得改写拒绝原因、不得把循环寿命自动换算为自然年，并在温度、倍率、DoD、SOC、EFC、静置时间、阶段或 horizon 缺失时向用户补问。

## 环境变量

### 正式 competition runtime

正式 API/Worker 使用 `deploy.competition_api:app` 与 `deploy.competition_worker:app`，
基础 `deploy/competition.compose.yaml` 默认不挂载任何 Feishu/Aily secret。只有显式合并
`deploy/competition.feishu-aily.override.yaml` 才会统一装配 migrate、API 与 Worker。先执行
Alembic migration 到唯一 head `0022`，再启动 API 与 Worker。启用时设置以下非秘密身份：

```text
FEISHU_APP_ID
FEISHU_BITABLE_APP_TOKEN
FEISHU_BITABLE_TABLE_ID
EXTERNAL_HTTPS_BASE_URL
ALLOW_CANDIDATE_SCENARIO_EXECUTION=true|false
ALLOW_CANDIDATE_SCENARIO_RESULTS=true|false
```

四个秘密只通过容器 secret file 读取：

```text
feishu_app_secret
feishu_verification_token
feishu_encrypt_key
aily_connector_api_key
```

它们在容器内分别映射为 `QUANXIN_FEISHU_APP_SECRET_FILE`、
`QUANXIN_FEISHU_VERIFICATION_TOKEN_FILE`、`QUANXIN_FEISHU_ENCRYPT_KEY_FILE` 和
`QUANXIN_AILY_CONNECTOR_API_KEY_FILE`。override 还将
`ALLOW_CANDIDATE_SCENARIO_EXECUTION` 和 `ALLOW_CANDIDATE_SCENARIO_RESULTS` 映射为
`QUANXIN_ALLOW_CANDIDATE_SCENARIO_EXECUTION` 与
`QUANXIN_ALLOW_CANDIDATE_SCENARIO_RESULTS`。两个 candidate 开关必须独立审批：第一个
只允许 `REGISTERED_CANDIDATE` 路由执行，第二个才允许卡片、Aily 报告和 Bitable 发布
该结果。任一开关关闭都不能被另一开关绕过。

同一 override 还要求 `${CONFIG_ROOT}/feishu-csv-registrations.json` 已由
`deploy/feishu-csv-registrations.example.json` 创建并填入经审查的精确 SHA 登记；未登记
附件不会进入 validation-first workflow。

启动命令：

```bash
docker compose \
  --env-file competition.env \
  -f competition.compose.yaml \
  -f competition.feishu-aily.override.yaml \
  up -d
```

生产装配挂载：

```text
POST /v1/integrations/feishu/events
POST /v1/aily/scenario-contexts
POST /v1/aily/analysis-tasks
GET  /v1/aily/analysis-tasks/{run_id}
GET  /v1/aily/analysis-tasks/{run_id}/results/{result_id}
GET  /v1/aily/analysis-tasks/{run_id}/reports/{report_result_id}
```

### 本地 callback runner

以下变量只用于独立本地 callback runner 与 preflight，不代替 production secret files：

```text
FEISHU_APP_ID
FEISHU_APP_SECRET
FEISHU_VERIFICATION_TOKEN
FEISHU_ENCRYPT_KEY
FEISHU_TEST_CHAT_ID
FEISHU_BITABLE_APP_TOKEN
FEISHU_BITABLE_TABLE_ID
FEISHU_CONNECTOR_API_KEY
QUANXIN_EXTERNAL_HTTPS_BASE_URL
```

`QUANXIN_EXTERNAL_HTTPS_BASE_URL` 是经批准的对外 HTTPS 基础地址。生产 preflight 只返回已配置/缺失的变量名和错误码，不回显值。

## 当前验证边界

自动化 Fake Feishu scenario E2E 已验证真实 BLAST 数值、ToolResult、曲线卡片、报告和
scalar-only Bitable。Fake Aily scenario E2E 已通过生产 assembly 创建受控场景、持久化
任务、运行共享 Worker、读取 run-bound ToolResult/报告并写入 scalar-only Bitable，且
没有飞书聊天或文件副作用。目标租户已经真实执行飞书 CSV 到项目级 CyclePatch，以及独立
BLAST-Lite 到曲线、卡片、报告和 Bitable 的两条交付链。当前运行记录尚无 `job_origin=AILY`
证据，因此不能声称 Aily 已经通过自然语言完成同一纵向编排；Fake Aily E2E 也不能描述为
真实租户的 Aily 验证。

## 本地运行

### 本地真实回调 runner

本地 runner 是独立的 callback surface，不复用或暴露
`deploy.foundation_api:app`。它只挂载：

```text
GET  /health
POST /v1/integrations/feishu/events
```

它复用现有 `FeishuEventProcessor`、`create_feishu_http_adapter`、
`SqlAlchemyFeishuReceiptStore`、`SqlAlchemyFeishuJobStore` 和
`SqlAlchemyFeishuJobRouter`。receipt 与 sanitized durable job 默认写入本地 SQLite；只有已经持久化的 job UUID 进入进程内 identity-only 队列。完整消息正文不落库，callback 不运行模型或业务 worker。因此这个入口适合验证 URL verification、签名、verification token、防重、durable job 入队和快速 ACK，不是完整生产任务执行器。

本地 runner 已装配经审查的 Feishu AES-CBC decryptor，健康检查会显示
`LOCAL_ENCRYPTED`。本地隧道联调可以保留飞书事件加密；runner 会使用
`SHA-256(FEISHU_ENCRYPT_KEY)` 作为 AES-256 key，按飞书官方 Python SDK 约定将 Base64 解码后的前 16 字节作为随机 IV，并解密其余 PKCS#7 密文得到 JSON。生产环境仍不得直接把本地 runner 当作完整生产后端，必须使用持久化 receipt、受保护入口和正式部署审查。
在仓库根目录启动：

```powershell
.\.venv\Scripts\python.exe scripts\run_feishu_callback.py `
    --host 127.0.0.1 `
    --port 8787 `
    --default-file-task predict_cycle_life
```

另开一个 PowerShell 做健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8787/health
```

预期 `service` 为 `feishu-callback-runner`，`security_mode` 为
`LOCAL_ENCRYPTED`。runner 强制绑定 loopback，不能直接监听公网网卡。

### 临时 HTTPS 隧道

本地健康检查通过后，才创建临时 HTTPS 隧道。无需先购买域名。若本机已安装
Cloudflare Tunnel，可在另一个 PowerShell 运行：

```powershell
cloudflared tunnel --url http://127.0.0.1:8787
```

Cloudflare 会返回临时 `https://...trycloudflare.com` 地址。仅将基础地址写入当前进程和
Windows 用户环境变量：

```powershell
$publicBase = "https://实际返回的临时主机名"
$env:QUANXIN_EXTERNAL_HTTPS_BASE_URL = $publicBase
[Environment]::SetEnvironmentVariable(
    "QUANXIN_EXTERNAL_HTTPS_BASE_URL",
    $publicBase,
    "User"
)
Remove-Variable publicBase
```

飞书开放平台中的回调 URL 填写：

```text
https://实际返回的临时主机名/v1/integrations/feishu/events
```

隧道 URL 每次可能变化。隧道关闭后应视为失效；不要把临时主机名当作生产地址。配置 URL
verification 前确认 `FEISHU_ENCRYPT_KEY` 和 `FEISHU_VERIFICATION_TOKEN` 与当前应用一致；本地 runner 支持加密 URL verification，不要求关闭飞书事件加密。

2026-08-08 已使用 Cloudflare Quick Tunnel 与目标企业飞书应用完成 callback receipt 链的受控验证：加密 URL verification、群聊文本消息和 CSV 文件消息均到达 `/v1/integrations/feishu/events` 并返回 HTTP 200。此后目标租户已真实执行 durable job、项目级 CyclePatch，以及独立 BLAST-Lite 的曲线、卡片、报告与 Bitable 交付。Aily 自然语言创建和编排这些任务仍缺少真实 `job_origin=AILY` 运行证据。临时隧道主机名不构成生产部署证据。

### 真实 Bitable metadata smoke

2026-08-08 已在目标企业多维表格执行一次受控 metadata-only 验证：创建一条
`feishu_callback_metadata_smoke` 记录，按同一 `run_id` 查询并更新状态，再次查询只匹配一条
记录。该记录没有 SOH、RUL、EOL、区间、模型指标或其他预测数值。飞书查询 API 对文本字段
返回富文本片段数组，writer 只依赖稳定的 `record_id` 执行幂等更新。

启动本地 Fake Feishu Sandbox：

```powershell
.\.venv\Scripts\python.exe scripts\run_fake_feishu_sandbox.py --host 127.0.0.1 --port 8765
```

Sandbox token 固定为本地协议测试用途，只能绑定 loopback，不得部署到公网。它模拟 token、消息、卡片、文件、资源下载和 Bitable；不运行模型，也不声称完成真实飞书配置。

运行 Sandbox preflight：

```powershell
.\.venv\Scripts\python.exe scripts\feishu_preflight.py --sandbox
```

运行生产配置 preflight：

```powershell
.\.venv\Scripts\python.exe scripts\feishu_preflight.py
```

运行本次接入测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\integrations\feishu tests\integration\api\test_feishu_api.py tests\integration\api\test_aily_api.py tests\integration\test_fake_feishu_sandbox.py tests\e2e\test_fake_feishu_scenario_delivery.py -q
```

`Fake Feishu scenario E2E` 使用真实 validation-first workflow、BLAST 数值工具、ToolResult、审计报告工厂、PNG 曲线渲染、卡片、文件交付和 scalar-only Bitable metadata；测试不包含硬编码 SOH/EOL 输出。

## 正式部署检查

- 回调公网地址具有有效 TLS 证书，反向代理保留签名头和原始请求体；
- App ID/Secret、verification token、encrypt key 和 connector key 来自秘密管理器；
- receipt、任务队列、Agent run、ToolResult、路由审批和报告制品使用持久化存储；
- BLAST route 保持 `REGISTERED_CANDIDATE`，候选执行和候选展示分别由服务器端显式授权；
- 附件对象存储具有租户/项目访问控制、保留期和 SHA 校验；
- Advanced route 的 manifest、来源和 artifact SHA 均通过，activation ledger 有人工审批；
- 多维表格 `run_id` 唯一性、写入频率和单写者策略已配置；
- Aily 端点经过 API Gateway 的 TLS、来源限制、速率限制和审计；
- 日志脱敏，不记录 Secret、Token、原始附件、完整用户提示或完整消息正文；
- Fake Sandbox 与生产部署包隔离；
- 相关 pytest、Ruff、mypy、compile/build 在目标提交上产生最新输出。

## 故障恢复

签名、token、解密或重放检查失败时拒绝回调。路由入队失败时释放 receipt 租约，飞书重试后仍按 event ID 幂等。Feishu API 的限流和服务错误只做有限指数退避；永久错误不重试。

数据验证、route、artifact、SHA、账本绑定或卡片路径检查失败时停止数值交付，发送拒绝/降级状态。Bitable 出现重复 `run_id` 时停止覆盖并由运维合并记录。报告制品过期、哈希不一致或渲染失败时重新从有效 ToolResult 生成，不能修改旧制品内容。

审计调查至少关联 event ID、任务 ID、`run_id`、数据批次、输入 SHA、`result_id`、route ID、版本、activation 决策、卡片消息 ID、Bitable record ID 和报告 SHA。敏感原文不进入该关联表。

## PBT/MAGNet 后续接入

训练结束后只能执行以下路由内步骤：验证训练结果与独立测试指标；构建安全部署制品和 manifest；核对来源与 SHA-256；注册 candidate route；完成人工晋级；在工具内部切换 route。

PBT 只能作为 `predict_cycle_life` 的候选内部路由。MAGNet 保持独立研究候选，不替代当前 manifest-bounded BLAST 情景工具，也不能绕过其支持范围和证据等级。两者接入不得修改飞书事件、卡片、Bitable 字段、Aily 提示词或连接器 API。

## 科学边界

现有 MATR 结果是内部数据集和既定协议下的深度学习模型证据，不是目标工业电芯长期自然年寿命的精确验证。Naumann 最新外部验证结果和逐测试点明细位于 `reports/experiments/blast_naumann_v1/validation-v2/`：其中 19 条温度、DoD、倍率 `leave-one-condition` 指标是固定上游参数的 held-condition 诊断，`parameter_refit=false`，不等于本项目训练或独立训练 holdout。280Ah 方形 LFP 的观测范围内尺度核验位于 `reports/experiments/blast_280ah_v1/validation-v1/`。约 700 循环范围内的核验不能外推成 25 年真实验证。最新长期图表和 CSV 源数据位于 `reports/experiments/blast_scenarios_v1/scenario-v3/`，只能称为物理参考情景。

`LFP_FIELD_160AH` 当前只建立了系统级来源索引和 `no_health_target` 现场监测视图，时间戳语义仍是 `LOCAL_TIME_TIMEZONE_UNRESOLVED`。它可以支持自然时间跨度、遥测异常和弱单体偏离检查，但没有受审查的 SOH/RUL 标签，因此当前不能生成 SOH 精度、无偏总体寿命分布或 15～25 年真实验证结论。

当前链路不支持把循环数直接描述为自然年，也不支持未经受审计情景工具的长期经营结论。PyBaMM 仅可用于短时滚动物理核验和敏感性参考，不能制造长期退化标签。

因此，本阶段交付的是可追溯、安全拒绝的情景工具链，不是海辰电芯 15～25 年真实寿命结论。真实飞书交付仍依赖有效凭证、目标租户配置与人工候选授权；Aily 自然语言编排还需要真实调用证据。PBT/MAGNet activation 状态没有改变。
