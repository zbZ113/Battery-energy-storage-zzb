# 已实现工具注册表协议 v1

`create_available_tool_registry()` 是当前已实现领域工具的唯一装配点。FastAPI、MCP、
Agent 和 Next.js 应获取同一个注册表实例或使用同一装配函数，不能在各层复制
数值逻辑或自行拼接工具列表。

当前无参注册工具为：

- `validate_battery_data`
- `audit_dataset_split`
- `check_operating_condition`

`make_batch_decision` 已迁移为上下文绑定工具：它需要审计账本和已批准策略解析器，不能在无参注册表中
暴露；详见 `batch-decision-tool-protocol-v2.md`。

需要已登记上游结果、已拟合内存模型、审计账本或受审核上下文解析器的工具（例如
`predict_soh_trajectory`、`update_cell_parameters`、
`recommend_next_experiment` 和 `retrieve_battery_evidence`）不进入默认注册表：调用方必须在模型制品
来源、哈希、训练上下文、上游证据链、试验候选目录或知识语料范围完成核验后，显式将其绑定到专用
注册表。主动试验推荐的公共输入只允许服务端上下文 ID；知识检索不接受调用方携带的文档或引用；
详见 `next-experiment-recommendation-tool-protocol-v1.md` 与
`battery-evidence-retrieval-tool-protocol-v1.md`。

其余标准工具名在具有真实、可审计实现前不会注册。调用未实现工具会得到明确的
`UnknownToolError`，而非占位响应、模拟成功状态或编造数值。
