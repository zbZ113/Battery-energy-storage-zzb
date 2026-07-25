# 泉芯智寿科研工作台

该目录是 FastAPI 的薄 Streamlit 客户端。它不导入领域模型，不加载模型制品，也不计算
SOH、RUL、预测区间或批次决策。界面只提交自然语言目标、服务端审核过的标识，并原样展示
FastAPI 在当前 Agent Run 权限范围内签发的结果。

## 安全会话

- 每个 Streamlit 浏览器会话创建一个独立 `httpx.Client`，由它保存后端签发的 HttpOnly
  Session Cookie；客户端之间不共享 Cookie。
- 登录、改密、退出及其他 POST 请求固定发送 `QUANXIN_TRUSTED_ORIGIN`，页面不能修改 API
  地址或 Origin，避免把工作台变成可访问任意地址的代理。
- 密码只在当次 HTTPS/本机 HTTP 请求中使用，不写日志、文件、Git 或 `session_state`。
- 未登录只显示登录页；临时密码账号只显示改密页；改密完成后才显示科研工作台。
- 生产环境的 API 与可信 Origin 必须使用 HTTPS；开发环境只允许通过本机
  `localhost`、`127.0.0.1` 或 `::1` 的 HTTP 地址访问。

## 本地配置

PowerShell 中先设置显式环境变量：

```powershell
$env:QUANXIN_ENVIRONMENT = "development"
$env:QUANXIN_API_BASE_URL = "http://127.0.0.1:8000"
$env:QUANXIN_TRUSTED_ORIGIN = "http://127.0.0.1:8501"
```

然后在仓库根目录启动：

```powershell
.\.venv\Scripts\python.exe -m streamlit run workbench\streamlit_app.py `
  --server.address 127.0.0.1 --server.port 8501
```

若尚未安装可选依赖，在联网环境安装项目的 `streamlit` extra 后再启动。真实密码、API
密钥和飞书密钥不得写入 `.env.example` 或提交到仓库。

## 当前接口与限制

当前工作台调用后端认证接口和已有科研流程接口：

- `POST /v1/auth/login`
- `GET /v1/auth/me`
- `POST /v1/auth/change-password`
- `POST /v1/auth/logout`
- `GET /health`
- `GET /v1/tools`
- `POST /v1/datasets/{dataset_id}/batches/canonical-csv`
- `POST /v1/workflows/lifetime-decision`
- `POST /v1/agent/runs`
- `GET /v1/agent/runs/{run_id}`
- `POST /v1/agent/runs/{run_id}/approve`
- `POST /v1/agent/runs/{run_id}/reject`
- `POST /v1/agent/runs/{run_id}/cancel`
- `GET /v1/agent/runs/{run_id}/results/{result_id}`

“Agent 协同”页允许输入项目 ID、自然语言目标、数据集 ID 和期望输出，随后保存当前
`run_id`。页面可手动刷新状态、查看服务端计划、批准或拒绝审批、取消任务，并且只能通过
`run_id + result_id` 读取已完成步骤的 `ToolResult`。科研台不调用全局管理员结果或报告
接口，避免跨任务读取。

读取成功的 ToolResult 只缓存在当前 Streamlit 浏览器会话中，并按 `result_id` 保存。
“结果展示”和“审计报告”页面只展示这份已经经过 run 范围接口验证的缓存，不会把旧同步
寿命工作流返回的 result ID 拼接到当前 Agent Run。旧“工作流执行”页只用于兼容科研调试，
其响应 ID 不是 Agent 正式结果入口。

创建任务的 `Idempotency-Key` 与规范化后的项目、目标、数据集和输出指纹绑定。同一请求在
网络失败后重试会复用原键；用户修改请求内容后会生成新键；创建成功后立即清除临时键。

## 为什么科研台没有实时事件流

正式 Next.js 门户负责消费 `GET /v1/agent/runs/{run_id}/events` 的 SSE 实时事件流。Streamlit
科研台采用“手动刷新”模式，避免浏览器重跑机制与长连接互相阻塞。审批 ID 和已完成的
Result ID 应从 Next.js 任务时间线复制到科研台；科研台不会猜测审批对象或结果 ID，也不会
用轮询伪装 SSE。

当前后端还没有 run 范围内的 Markdown 报告导出接口，所以科研台只展示 run 范围内的
`ToolResult` 证据。在后端和 Next.js 补齐 run 范围报告导出前，科研台不会回退到全局
管理员报告接口。
