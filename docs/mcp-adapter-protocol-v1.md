# 可选 MCP 工具适配层协议 v1

## 目标

`quanxin_life.tools.mcp_adapter` 为阶段 7 的 MCP 传输层提供一个可选、无副作用的适配边界。它不实现任何电芯数据、寿命、物理、试验或报告业务逻辑；所有发现和调用均委托给已经注册的 `ToolRegistry`。

模块导入时不会导入 MCP SDK、初始化网络、加载模型或连接服务。因此，即使环境未安装 MCP SDK，FastAPI、Next.js、数值工具和本地测试仍可正常使用。

## 调用模型

`McpToolCallRequest` 是 MCP JSON 请求的 Pydantic 契约：

```text
tool_name: StandardToolName
input: JSON mapping
```

输入映射须能通过项目的 canonical JSON 哈希化。适配器不会信任已构造的 Pydantic 实例：无论请求来自映射还是 `McpToolCallRequest`，均会先 `model_dump(mode="json")`，随后重新执行 `McpToolCallRequest.model_validate()`。

`discover_tools()` 直接返回 `ToolRegistry.list_schemas()` 的确定性结果。`call_tool()` 仅调用 `ToolRegistry.execute()`，用于 API、界面和其他受控服务；`call_tool_for_agent()` 仅调用 `ToolRegistry.execute_for_agent()`，因此 Agent 必须显式提供非空工具白名单，不能绕过注册表的授权边界。

适配器不得复制注册表的输入哈希、结果版本、数值审计、白名单执行或领域工具逻辑。返回值始终是注册表产生并验证过的 `ToolResult`。

## 可选 SDK 降级

`load_optional_mcp_sdk()` 才会延迟导入 `mcp` 包。缺少该可选依赖时，函数抛出 `McpSdkUnavailableError`，并清楚说明可以继续使用进程内 `McpToolAdapter` 或安装 MCP 额外依赖。该降级不产生伪造 MCP 服务、模拟工具结果或替代数值。

后续 stdio 或 Streamable HTTP MCP 主机必须：

1. 在进程启动路径显式调用 `load_optional_mcp_sdk()`；
2. 将 MCP 的发现请求映射到 `discover_tools()`；
3. 将通用调用映射到 `call_tool()`；
4. 将四个专业 Agent 的调用映射到 `call_tool_for_agent()`；
5. 继续将所有正式数字限制在 `ToolResult` 的审计链内。

在 MCP SDK、凭证或外部传输不可用时，系统只能明确报告降级状态，不得静默回退到未授权工具调用。
