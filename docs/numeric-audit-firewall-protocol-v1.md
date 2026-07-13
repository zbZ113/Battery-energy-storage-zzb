# 数值来源防火墙协议 v1

项目中的正式报告不能让 LLM 或 UI 直接写入 SOH、EOL80、寿命区间、退化率、试验成本或质量分流数值。`numeric-firewall-v1` 通过 `AuditLedger` 和 `NumericEvidence` 将每个报告数值绑定到一个已经验证的 `ToolResult`。

## 规则

1. `AuditLedger` 只接收重新通过 `ToolResult` Pydantic 契约检查的结果，拒绝 `model_construct` 绕过、重复 `result_id`、无来源或非 JSON 可审计的结果。
2. 每项 `NumericEvidence` 必须指定 UUID `result_id`、仅指向 `values.*` 的 JSON 路径、精确的有限 `reported_value` 和证据等级。
3. 防火墙从重验证的工具结果解析该路径；值不存在、不是数值、非有限或与报告数值不一致时，报告构建必须停止。
4. 工具结果本身还需要名称、版本、输入哈希、模型/数据/特征版本和来源记录；防火墙不会补造这些字段。

这是一条结构化证据链，不是对自然语言文本“猜测是否含数字”的脆弱正则。后续 Markdown/PDF/Word 报告构建器必须先验证全部 `NumericEvidence`，再把对应的已解析值插入模板。无数值的叙述性解释也必须附工具或知识库来源，但不应冒充数值结论。
