# 数据质检工具协议 v1

`validate_battery_data` 是共享工具注册表中的确定性数据质检入口。它将
`CycleRecord` 原样交给 `validate_cycle_records`，输出携带完整来源链的
`ToolResult`；不清洗、补全、重排或预测输入数据。

## 输入与边界

- 输入必须包含原始采样点记录、数据版本、特征版本、至少一条来源记录及带时区的验证时间。
- 空记录是有效的待质检输入，会返回阻断性 `EMPTY_CELL` 证据，不会被静默忽略。
- 混合电芯、混合数据集、重复样本、非单调时间、循环缺口与温度缺失均由既有规则引擎如实报告。
- `quality_score` 只是结构化数据可用性分诊分数，不是 SOH、RUL、模型性能或产业指标。

## 审计输出

每次调用生成新的 `ToolResult`，其中：

- `model_version` 固定为 `data-quality-rule-engine-v1`；
- `input_hash` 对已验证的完整输入做规范 JSON 哈希；
- `values` 保留数据集标识、阻断状态、分诊分数、问题数量及完整问题清单；
- `warnings` 只来自质检规则的真实问题代码；
- `provenance` 完整保留输入来源。

Agent、API、MCP 与 Next.js 必须通过同一注册表调用该工具。任何后续
特征、模型或决策步骤都应使用此工具返回的 `result_id` 作为上游证据，而不是把 UI
中的示例状态当作质检结论。
