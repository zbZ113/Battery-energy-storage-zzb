# Aily MCP HTTPStreaming 接入

本文档描述新版“豆包工作伙伴 / Aily 工作助手”如何通过 MCP HTTPStreaming 调用泉芯智寿现有的受审计飞书业务链。旧版 `/v1/aily/*` Bearer API 保留，但新版工作助手应配置本文的 MCP URL。

## 安全边界

- MCP 默认关闭。只有同时加载飞书 override 和 MCP override 时才启用。
- MCP 使用独立高熵 URL 路径令牌，不复用旧 Aily Connector Bearer 密钥。
- MCP 令牌和启用配置只挂载到 API；migrate 与 worker 不接收秘密路径。
- 生产默认同时在 Nginx 与 API 使用精确出口 IP 白名单；受信网关模式默认关闭。
- Aily 自动发送的 `x-aily-user` 必须先映射到本地 `User.id`。
- `x-aily-email` 只做请求格式检查，不落库、不记日志，也不能单独授权。
- 每次创建、查询、读结果、读报告或建复检动作时，系统都会追溯到原始飞书上传，并重新验证活动项目、飞书绑定和本地 actor。
- 未登记且不带邮箱的 Aily 服务验证身份只允许 `initialize`、`notifications/initialized`、`ping` 和 `tools/list`；任何 `tools/call` 仍要求真实用户映射。
- LLM 只能编排和解释。SOH、RUL、EOL、寿命、区间、阈值结论及业务指标必须来自有效 `ToolResult`。

## MCP 工具

默认发布六个业务工具；配置复检表后再发布第七个：

| 工具 | 用途 |
| --- | --- |
| `quanxin_resolve_analysis_source` | 按 `电芯ID | cutoff-N` 解析当前用户已完成的飞书上传及内部任务引用 |
| `quanxin_create_scenario_context` | 保存工程师明确给出的温度、倍率、SOC、DoD 等工况假设 |
| `quanxin_create_analysis_task` | 创建异步寿命、SOH、参考工况或工程建议任务 |
| `quanxin_get_analysis_task` | 轮询持久化任务状态 |
| `quanxin_get_audited_result` | 读取 run-bound 且通过展示授权的中文标量结果 |
| `quanxin_get_audited_report` | 读取 SHA 校验通过的受审 Markdown 报告 |
| `quanxin_create_recheck_action` | 在复检能力已配置时创建幂等复检记录 |

通用 `ToolRegistry` MCP 不会暴露给 Aily。CyclePatch 与 BLAST-Lite 是两个独立工具，Aily 只能在同一业务任务中分别调用和综合解释，不能把两者数值互相换算。

## 运维文件

### 1. 端点令牌

在 `${QUANXIN_SECRETS_ROOT}/aily_mcp_endpoint_token` 写入一行 32 至 128 位的随机 URL-safe 字符。不要提交到 Git，不要出现在截图、日志或聊天中。

PowerShell 生成示例：

```powershell
$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$token = [Convert]::ToHexString($bytes).ToLowerInvariant()
Set-Content -LiteralPath "$env:QUANXIN_SECRETS_ROOT\aily_mcp_endpoint_token" `
    -Value $token -Encoding utf8NoBOM
Remove-Variable bytes, token
```

### 2. Aily 用户映射

在运行时配置目录创建 `aily-mcp-identity-bindings.json`：

```json
{
  "schema_version": "quanxin-aily-mcp-identity-bindings-v1",
  "bindings": [
    {
      "aily_user_id": "<Aily 请求头中的 x-aily-user>",
      "local_user_id": "<现有数据库 User.id>"
    }
  ]
}
```

`aily_user_id` 与 `local_user_id` 都必须一对一唯一。不要写入邮箱。Aily 文档把该值称为 `Feishu_user_id`，而仓库事件保存的是 `open_id`；在真实请求确认前不得假设两者相等。

### 3. 出口 IP

环境变量 `AILY_MCP_ALLOWED_SOURCE_IPS` 使用逗号分隔的精确地址，不接受 CIDR。Aily 官方文档当前列出的地址为：

```text
101.126.59.88,101.126.59.89,101.126.59.90,101.126.59.91,101.126.59.92,122.14.241.34,122.14.241.35,122.14.241.36,122.14.241.37,122.14.241.38
```

2026-08-15 的真实任务握手还观察到 `240e:b1:e401:3::a6`、`240e:83:200::33c` 和 `106.38.226.12`，说明任务执行出口会轮换且官方清单并不完整。严格生产部署可把已核验的地址逐个加入精确白名单，但不得据此猜测或放开宽泛 CIDR。

仅当 API 只在内部网络可达、本地网关绑定 `127.0.0.1:8080`、Cloudflare Tunnel 是唯一外部入口且端点令牌保持保密时，可在本地联调环境显式启用：

```text
AILY_MCP_TRUST_GATEWAY_SOURCE_IP=true
```

该模式仍要求网关提供格式合法的来源 IP，并继续执行秘密 URL、Host、请求大小、Aily 用户头、发现方法限制和业务授权校验；它只跳过不可靠的出口 IP 成员判断。生产默认保持 `false`。

## 启动

生产部署叠加三个 compose 文件：

```powershell
docker compose `
  -f deploy/competition.compose.yaml `
  -f deploy/competition.feishu-aily.override.yaml `
  -f deploy/competition.aily-mcp.override.yaml `
  up -d
```

本地完整运行时也可以叠加同一个 MCP override：

```powershell
docker compose `
  -f deploy/local.compose.yaml `
  -f deploy/competition.feishu-aily.override.yaml `
  -f deploy/competition.aily-mcp.override.yaml `
  up -d --build
```

生产 Nginx 使用直连来源 IP 并保留精确白名单。本地 Nginx 不自行判断 Aily 白名单，只把 Cloudflare 注入的 `CF-Connecting-IP` 传给 API；是否跳过成员判断由 `AILY_MCP_TRUST_GATEWAY_SOURCE_IP` 显式控制。不要把本地配置当作公网源站配置。

## Aily 中填写

服务类型选择 `MCP HTTPStreaming`。URL 必须带结尾斜杠：

```text
https://<当前外部HTTPS主机>/v1/aily/mcp/<端点令牌>/
```

Aily 会自动发送 `x-aily-user` 和 `x-aily-email`，无需在 UI 中另配 Header。若使用临时 `trycloudflare.com` 主机，主机名变化后必须同步更新 `QUANXIN_EXTERNAL_HTTPS_BASE_URL` 并重启 API；FastMCP 会拒绝旧 Host。

## 当前限制

- MCP 不会猜测电芯或自动枚举全库任务。用户提供 `电芯ID | cutoff-N` 后，Aily 先调用 `quanxin_resolve_analysis_source`；服务端只在当前用户获授权且已成功的飞书上传中解析 `source_run_id` 与 `data_batch_id`。
- 同一电芯和 cutoff 存在多个成功上传时，解析器按受审时间选择当前有效来源；Aily 不应要求用户手写内部 UUID。
- 报告单次返回受 Aily 约 2 万字上下文建议限制。大型轨迹数组不会直接返回，曲线应读取飞书附件和受审报告。
- 只有已完成、项目绑定有效、数据身份未变化的原始飞书上传可以作为 Aily 任务来源。

## 验证

```powershell
.\.venv\Scripts\python.exe -m ruff check src deploy scripts
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m compileall -q src deploy migrations
```

根目录测试代码仅在团队本地维护，不随 Git 分发。发布验收还必须在目标 Aily 中完成
MCP 安装、工具发现、真实用户授权、任务标签解析和受审工况查询。
