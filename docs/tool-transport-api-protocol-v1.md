# 共享工具传输 API 协议 v1

## 目标与边界

`quanxin_life.api.ToolInvocationService` 是 HTTP、MCP、Streamlit、Next.js 和本地脚本调用领域工具的唯一服务边界。它只把经过 Pydantic 验证的 JSON 输入转交给 `ToolRegistry`；不包含 SOH、RUL、预测区间、退化率、物理仿真或批次决策的第二套实现。

数值工具必须先在 `ToolRegistry` 中登记。服务层返回的业务数值始终来自登记工具产生并通过审计契约复验的 `ToolResult`，不能由传输层、LLM、UI 或 HTTP 路由拼接。

## 传输中立调用契约

```json
{
  "tool_name": "validate_battery_data",
  "input_value": {
    "...": "该工具已登记的 JSON 输入"
  }
}
```

- `tool_name` 必须是 `StandardToolName` 枚举中的已登记名称；
- `input_value` 只能是 JSON 兼容映射；对象、可执行反序列化制品和未声明字段由后续工具输入契约拒绝；
- 外部调用走 `ToolInvocationService.invoke()`，它仅委托 `ToolRegistry.execute()`；
- Agent 调用走 `invoke_for_agent()`，必须提供非空工具白名单，并仅委托 `ToolRegistry.execute_for_agent()`；
- 每个成功响应是原始 `ToolResult` 的 JSON 表达，包含 `result_id`、工具与模型/数据/特征版本、输入哈希、来源链、警告及时间戳。

## 可选 FastAPI 工厂

`create_fastapi_app(service)` 延迟导入 FastAPI。未安装 `quanxin-life[api]` 时，调用工厂会抛出 `FastApiDependencyUnavailable`；核心算法、MCP 适配和本地测试不受影响，也不会返回伪造的 HTTP 服务。

已安装依赖后，工厂提供：

| 方法 | 路径 | 责任 |
| --- | --- | --- |
| `GET` | `/health` | 仅报告传输服务存活，不报告模型或工业系统状态。 |
| `GET` | `/v1/tools` | 从 `ToolRegistry.list_schemas()` 发现已登记工具。 |
| `POST` | `/v1/tools/{tool_name}` | 将请求体作为该已登记工具的 `input_value`，通过共享服务调用。 |

具体的 `/v1/predictions/*`、`/v1/physics/check`、`/v1/experiments/recommend`、`/v1/decisions/batch` 和 `/v1/reports/*` 只在相应数值工具具有完整输入契约、真实 `ToolResult` 及集成测试后，作为这一通用调用路径的受限别名加入。不得先创建返回示例数值的空路由。

## 错误与审计规则

1. FastAPI 可选依赖缺失：显式异常，不降级为静态接口。
2. 非法工具名、非 JSON 输入、未登记工具或工具契约失败：返回客户端可见的 422；不泄漏堆栈、密钥、原始敏感数据或完整用户提示。
3. 工具自身失败：保持 `ToolRegistryError` 边界；后续异步任务层可将它映射为可查询作业状态，但不能以成功响应掩盖失败。
4. HTTP 响应和日志都必须保留 `result_id`；报告生成仍须经过数值防火墙，不能把路由响应直接当作正式报告。
5. 该模块不代表真实 BMS/EMS、飞书或企业服务接入；这些适配器在无凭证时只能运行明确标注的本地沙箱。
