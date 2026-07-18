# 审计 Markdown 报告协议 v1

`audited-markdown-v1` 是正式 PDF、Word、飞书文档和 Web 驾驶舱之前的唯一文本报告底座。报告构建器先把每个 `NumericEvidence` 送入数值来源防火墙，只有全部验证通过后，才将已解析的原始工具数值写进 Markdown。

每个报告 Claim 必须包含：唯一 `claim_id`、不冒充数值结论的叙述、至少一条 `NumericEvidence`。每条数值证据记录 `ToolResult` UUID、`values.*` 路径、实际报告数值和证据等级；导出的 Markdown 会原样显示路径、值、等级和结果 ID。

数值不匹配、无来源、未知路径、重复 Claim、缺时区或任一工具结果无效时，构建过程直接失败且不输出半成品报告。Markdown 构建模块本身只生成结构化对象与 Markdown，不调用 LLM、不创建 PDF/Word、不修改任何 ToolResult。下游 `AuditedReportArtifactExporter` 只能从 `AuditLedger` 重新解析已登记的 `generate_audited_report` 结果，并将同一份 Markdown 导出为 JSON、Markdown、PDF 或 DOCX；它不接受新的标题、叙述或数值，也不形成新的模型结论。PDF 字体必须通过路径和 SHA-256 复验。
