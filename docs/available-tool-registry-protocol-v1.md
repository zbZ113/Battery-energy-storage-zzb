# 已实现工具注册表协议 v1

`create_available_tool_registry()` 是当前已实现领域工具的唯一装配点。FastAPI、MCP、
Agent、Next.js 和 Streamlit 应获取同一个注册表实例或使用同一装配函数，不能在各层复制
数值逻辑或自行拼接工具列表。

当前注册工具为：

- `validate_battery_data`
- `audit_dataset_split`
- `check_operating_condition`
- `make_batch_decision`

其余标准工具名在具有真实、可审计实现前不会注册。调用未实现工具会得到明确的
`UnknownToolError`，而非占位响应、模拟成功状态或编造数值。
