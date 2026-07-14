# 审计报告工具协议 v1

`generate_audited_report` 是正式报告的唯一数值渲染入口。它不预测 SOH、RUL、区间、退化率或任何工程数值；其职责仅是将已经登记的 `ToolResult` 中、经审计台账复核的数值证据渲染为 Markdown 报告。

## 输入边界

工具输入只有以下字段：

- `report_kind`：受控报告类型；
- `claims`：一个或多个受控 `AuditedReportClaimReference`；
- `upstream_result_ids`：本次报告允许引用的、互不重复的 UUID 型上游结果标识。

输入不接受 `title`、`narrative`、`claim_id`、`reported_value`、`provenance`、调用方时间戳、调用方模型版本、调用方数据版本、调用方特征版本或任意独立数值字段。Pydantic 契约设置为 `extra="forbid"`，这些字段必须被拒绝。

报告标题和声明叙述仅由系统维护的 `ReportKind` 与 `ReportClaimKind` 模板生成；调用方只能选择受控键，不能提交任何可渲染文本。因此中文数字、全角数字、URL 内数字和任何其他书写形式都不能绕过数值审计边界。每个 `NumericEvidenceReference` 只包含 `result_id` 与 `json_path`；证据等级由 `ReportClaimKind` 的受控映射确定，调用方不能伪造。执行器从已登记的 `ToolResult` 读取实际数值后，才在内部构造 `NumericEvidence` 并渲染。

## 证据与来源链

执行器注入 `AuditLedger`，按 `upstream_result_ids` 逐一调用 `resolve_registered_result`。任一未登记 ID 都会立即失败。随后：

1. 所有 `NumericEvidenceReference.result_id` 的集合必须与声明的 ID 集合精确相同，不能夹带无关来源；
2. 台账从已登记 `ToolResult.values` 的 JSON 路径读取有限数值；
3. 执行器仅使用该已解析数值在内部构造 `NumericEvidence`，随后 `build_audited_markdown_report` 再调用台账的 `verify_numeric_evidence`；
4. 仅从本次解析出的上游结果收集 `ProvenanceRecord`，按规范化记录哈希去重后写入报告工具的 `ToolResult.provenance`。

调用方提供的 provenance 不存在于接口中，因而不能改变报告输出来源。未声明、未登记、路径不存在、非数值或被篡改的证据均不能生成 Markdown。

## 输出契约

返回的 `ToolResult.values` 只包含报告制品和已登记上下文：

- `report_id`；
- `rendering_version`；
- `markdown`；
- `claim_ids`；
- `upstream_result_ids`；
- `upstream_context`（每个已解析结果的工具、模型、数据、特征和输入哈希元数据）。

报告工具不独立计算工程数值。公共 `ToolResult` 所需的 `model_version` 标记为 Markdown 渲染器版本；`data_version` 与 `feature_version` 标记为台账绑定报告路径版本。它们不是电芯模型或数据集结论；精确的上游模型、数据和特征版本只保留在 `upstream_context` 与来源链中。

Markdown 中由系统生成的 UTC 时间戳、`ToolResult` UUID 和经过 `NumericEvidence` 核验的证据条目可能含有数字。这些不是调用方可在标题或叙述中伪造的定量主张。

## 时间与可复现性

`created_at` 与 Markdown 的生成时间只取自注入的、带时区的执行时钟，并归一化为 UTC。接口不存在调用方可控的报告生成时间字段。输入哈希始终是通过契约验证后的完整输入规范化 JSON 的 SHA-256，兼容共享 `ToolRegistry` 的结果校验。

## 注册与失败语义

工具由 `register_generate_audited_report_tool` 注册到共享 `ToolRegistry`，工具名为 `generate_audited_report`。注册表仍负责输入哈希、工具名、工具版本、版本字段和 provenance 非空性校验。

以下情况必须失败而不得降级为“示例报告”：重复或非 UUID 的上游 ID、额外字段、未登记上游结果、未声明的证据结果、数值证据与台账不一致、无来源的上游结果，以及非 UTC 感知的执行时钟。
