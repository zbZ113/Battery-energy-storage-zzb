# 受约束多智能体编排协议 v1

本协议定义“泉芯智寿”的四个专业 Agent 与监督器如何共同使用共享工具。它是确定性、可恢复的工作流内核；将来可由 LangGraph 或 OpenAI 兼容模型把自然语言转成候选步骤，但模型绝不能跳过本协议或直接产生工程数值。

## 角色与工具白名单

| 角色 | 允许调用的共享工具 |
| --- | --- |
| 数据质检 Agent | `validate_battery_data`、`audit_dataset_split`、`extract_early_cycle_features` |
| 寿命预测 Agent | `predict_cycle_life`、`predict_soh_trajectory`、`calibrate_prediction_interval`、`adapt_to_target_domain`、`update_cell_parameters` |
| 物理核验 Agent | `check_operating_condition` |
| 试验决策 Agent | `recommend_next_experiment` |
| 监督 Agent | `make_batch_decision`、`retrieve_battery_evidence`、`generate_audited_report` |

每一步都携带预定义 `AgentRole`、标准 `tool_name`、JSON 输入和可选人工审批标志。工作流先校验角色边界，再经 `ToolRegistry.execute_for_agent` 调用；该入口要求非空白名单、强类型输入、输入哈希和完整 `ToolResult`。因此 LLM 只能提议步骤、解释返回结果，不能选择未授权工具、修改工具数值或绕过来源链。

## 审批、恢复与降级

- 标记人工审批的步骤在执行前返回 `AWAITING_HUMAN_APPROVAL`，不会擅自继续。
- 恢复时必须使用同一 `request_id`、同一计划哈希和顺序前缀；既有 `ToolResult` 会重新验证名称和输入哈希。计划变化或审计字段不匹配会被拒绝。
- 工具输入不合法、工具不在白名单、工具结果审计不完整或工具失败时，工作流输出 `FAILED` 与明确的阻断代码，绝不补造数字。
- 当前版本不依赖 LangGraph、网络、数据库或 LLM；后续适配器只可保存/恢复该状态对象，不得复制领域工具逻辑。

该编排内核不是“多个模型聊天”。专业分工的价值在于工具最小权限、人工复核节点、可追溯的执行计划和可恢复状态，而所有 SOH、EOL80、区间、物理样本和试验建议仍只来自相应的数值工具。
