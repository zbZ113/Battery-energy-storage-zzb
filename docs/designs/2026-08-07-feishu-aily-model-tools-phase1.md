# 飞书/Aily 电池模型工具联动第一阶段设计

## 范围

第一阶段把飞书和 Aily 接到现有 `ToolRegistry`、Agent、审计账本、RecordBatch、
模型路由与报告链。它不训练模型，不修改 PBT/MAGNet 任务，不激活任何
`CONDITIONAL` 或 `NOT_ACTIVATED` 路由，也不恢复独立 Web 前端开发。

上传格式只启用当前数据层已经完整支持的 canonical CSV。Parquet、XLSX 和 ZIP
在出现经过审查的 RecordBatch 输入契约前明确拒绝，不以文件后缀或临时解析器冒充
支持。

## 架构

```text
Feishu callback / Aily connector
  -> authenticated integration facade
  -> sanitized event/job identity
  -> attachment policy and SHA-256 verification
  -> existing RecordBatch registration
  -> validate_battery_data gate
  -> fixed allowlisted ToolRegistry workflow
  -> existing AuditLedger / SqlProjectAuditLedger
  -> audited card / Bitable summary / audited report artifact
  -> Feishu outbound delivery
```

飞书和 Aily 只看到稳定业务工具名。内部映射如下：

| 外部业务名 | 内部现有工具 | 第一阶段状态 |
|---|---|---|
| `validate_battery_data` | `VALIDATE_BATTERY_DATA` | 可用 |
| `predict_cycle_life` | `PREDICT_CYCLE_LIFE` | 仅服务器端已激活路由可用 |
| `predict_soh_trajectory` | `PREDICT_SOH_TRAJECTORY` | 仅服务器端已激活路由可用 |
| `compare_operation_scenarios` | 无等价正式工具 | 明确不可用；等待 MAGNet 评估和晋级 |
| `ingest_observed_soh` | `INGEST_NEWLY_OBSERVED_SOH` | 映射现有契约 |
| `update_trajectory` | `UPDATE_CELL_PARAMETERS` | 映射现有契约 |
| `generate_audited_report` | `GENERATE_AUDITED_REPORT` | 可用；仅消费有效结果 ID |

## 组件

### 飞书出站客户端

新增依赖注入式 transport、tenant token 缓存、超时、有限重试、指数退避、错误分类，
以及消息、回复、卡片更新、文件/图片上传、附件下载和多维表格 API。测试使用 Fake
Transport，不需要真实凭证。

### 入站事件与任务

保留现有验签、token、时间窗、event receipt 和租约。事件解析只产生经过长度限制的
`chat_id`、`user_id`、`message_id`、`file_key`、action value 等机器引用；不保存完整
消息正文。新路由协议版本接收 sanitized event，旧三参数 router 保持兼容。

耗时工作必须先持久化任务身份再回执。生产装配复用数据库和现有 Agent/队列；Fake
Sandbox 使用同一协议的内存实现。

### 附件安全

附件下载后先执行大小、文件名、扩展名、content type、magic bytes、UTF-8、SHA-256
和危险格式检查。第一阶段只允许 `.csv`，显式拒绝 archive、pickle、joblib、`.pt`、
`.pth`、可执行文件和伪装的 Parquet。通过后交给现有 canonical CSV 注册与数据验证。

### 审计卡片

状态卡片可以只使用非数值身份。结果卡片的公开入口只接受 `run_id` 和 `result_id`。
Builder 从审计账本重新解析 ToolResult，按工具版本配置白名单 `values.*` 路径，并在
显示任何数值前重新检查来源、版本和 route activation。无结果、未知来源、路径不在
白名单或模型未激活时生成拒绝/降级卡片，不显示数值。

### 多维表格

使用 `run_id` 作为幂等键。表格保存任务身份、状态、batch/SHA、主要 result ID、
版本、证据等级、警告、报告引用和 UTC 时间；轨迹数组和大型制品只保存受审计引用。

### Aily

新增受保护 HTTP façade 的 OpenAPI 文档。连接器只提供创建任务、查询状态、读取受审计
结果、读取报告四类操作；Bearer/API Key 必须校验，远程部署要求 TLS、访问控制和审计。
系统提示词禁止补写工具未返回的数字、改写拒绝原因或把循环数换算为自然年。

## 验证边界

Fake Sandbox 验证 transport、事件、附件、验证门、ToolResult、卡片、多维表和报告交付
的协议链。成功预测测试只能使用显式 test-only、工具生成且写入审计账本的结果，不把
测试 fixture 宣称为模型成果。真实 Advanced 候选另有拒绝测试，确认
`CONDITIONAL/NOT_ACTIVATED` 不能展示数值。

真实飞书联调需要用户提供凭证；真实 CyclePatch/Hybrid 路由调用需要目标环境完成
人工激活和校准证据；PBT/MAGNet 只在训练、评估、制品核验和人工晋级后新增内部路由。
