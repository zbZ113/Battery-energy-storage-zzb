# 共享强类型工具注册表协议 v1

## 目的与边界

`quanxin_life.tools` 是 FastAPI、MCP、Agent 和 Next.js 的唯一领域工具入口。注册表只负责输入契约校验、白名单授权、执行调度和审计契约核验；它不实现电芯寿命预测、SOH 轨迹、物理仿真或任何业务数值计算。

工具执行函数必须自行调用已经批准的数值模块，并自行产生 `ToolResult`。注册表不会补写、修正、四舍五入或重新解释工具返回的任何数值。LLM 仅可选择已注册工具、提供经契约约束的输入并解释经审计的结果。

## 标准工具名

`StandardToolName` 固定声明以下计划内工具名：

```text
validate_battery_data
audit_dataset_split
extract_early_cycle_features
predict_cycle_life
predict_soh_trajectory
calibrate_prediction_interval
update_cell_parameters
check_operating_condition
recommend_next_experiment
make_batch_decision
retrieve_battery_evidence
generate_audited_report
```

声明名称不代表注册表提供了对应业务实现。某项工具只能在其独立数值实现、输入契约、测试和版本信息齐备后注册；未注册调用必须失败，不得返回虚构降级结果。

## 注册与执行契约

每个 `ToolDefinition` 固定绑定：

- 一个 `StandardToolName`；
- 一个非空 `tool_version`；
- 一个继承 `quanxin_life.core.schemas.ContractModel` 的 Pydantic 输入模型；
- 一个接收已验证输入并返回 `quanxin_life.core.ToolResult` 的执行函数。

同一工具名只允许注册一个实现，禁止静默覆盖。注册时必须提供真正的 `StandardToolName`、非空版本、`ContractModel` 子类输入模型和可调用执行器。输入可以是映射或 `ContractModel`；即使调用方传入了目标输入模型的实例，注册表也必须先执行 `model_dump(mode="json")`，再由声明的输入模型重新 `model_validate`。该过程防止 `model_construct` 等绕过验证的伪实例进入数值工具。通过后的规范化输入再执行 `sha256_canonical`，成为调用的唯一输入哈希；执行器返回的 `ToolResult.input_hash` 必须完全相同。

`list_schemas()` 返回每个已注册工具的名称、版本和 Pydantic JSON Schema，供 MCP 和界面发现能力使用。列出 Schema 不执行工具，也不触发模型、网络、数据库或服务初始化。

## 授权与失败处理

调用方可传入 `allowed_tool_names`。当目标工具不在白名单中时，注册表在输入校验和执行前拒绝调用。未知标准名称、未注册名称、重复注册、输入校验失败、工具运行失败以及返回契约不一致均以明确异常结束；不会自动选择其他工具，也不会构造业务结果。

通用 `execute()` 保留给 API、界面和受控服务调用，默认不强制白名单。阶段 7 的 Agent 必须只调用 `execute_for_agent()`：该严格入口要求显式且非空的工具白名单，省略、传入 `None` 或空集合均会拒绝执行。这样 Agent 不能通过通用入口绕开职责白名单。

## 审计防火墙

每个正式返回必须是有效 `ToolResult`，并且满足：

1. `tool_name` 与注册名称一致；
2. `tool_version` 与注册版本一致；
3. `input_hash` 等于已验证输入的规范化 SHA-256；
4. `model_version`、`data_version`、`feature_version` 均为非空版本标识；
5. `provenance` 非空，且继续由 `ToolResult` 的公共 Pydantic 契约校验；
6. `result_id`、时间戳、数值、警告和来源均由具体工具生成，注册表不代写。

执行器返回后，注册表会对其执行 `model_dump(mode="json")`，再通过 `ToolResult.model_validate()` 重建公共契约对象。即使执行器恶意或错误地使用 `model_construct` 伪造 `ToolResult`，也必须在此处被拒绝；版本字符串在去除空白后仍须非空。

因此，Agent、报告器和 UI 必须只消费经 `ToolRegistry.execute()` 返回的 `ToolResult`。后续监督器可以根据 `result_id`、输入哈希、版本和来源建立全链路审计；没有这些字段的数字不得进入正式报告。

## 非目标

- 不在本层创建或加载模型权重；
- 不在本层访问外部 API、数据库、消息队列或网络；
- 不接受未核验的 pickle、joblib、`.pt` 或 `.pth` 制品；
- 不以工具名称、提示词或示例结果替代真实数值工具；
- 不将 PyBaMM 短期核验结果转换为长期寿命结论。
