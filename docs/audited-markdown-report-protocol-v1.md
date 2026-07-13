# 审计 Markdown 报告协议 v1

`audited-markdown-v1` 是正式 PDF、Word、飞书文档和 Web 驾驶舱之前的唯一文本报告底座。报告构建器先把每个 `NumericEvidence` 送入数值来源防火墙，只有全部验证通过后，才将已解析的原始工具数值写进 Markdown。

每个报告 Claim 必须包含：唯一 `claim_id`、不冒充数值结论的叙述、至少一条 `NumericEvidence`。每条数值证据记录 `ToolResult` UUID、`values.*` 路径、实际报告数值和证据等级；导出的 Markdown 会原样显示路径、值、等级和结果 ID。

数值不匹配、无来源、未知路径、重复 Claim、缺时区或任一工具结果无效时，构建过程直接失败且不输出半成品报告。当前模块只生成结构化对象与 Markdown，不调用 LLM、不创建 PDF/Word、不修改任何 ToolResult；后续格式导出器只能消费已经构建完成的 `AuditedMarkdownReport`。
