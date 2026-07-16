# 泉芯智寿科研工作台

该目录是 FastAPI 的薄 Streamlit 客户端。它不导入领域模型，不加载模型制品，也不计算
SOH、RUL、预测区间或批次决策。界面只提交服务端审核过的标识，并原样展示服务端签发的
结果和审计 Markdown。

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
- `POST /v1/batches/canonical-csv`
- `POST /v1/workflows/lifetime-decision`
- `GET /v1/reports/{result_id}`

旧科研流程仍通过全局 `GET /v1/results/{result_id}` 读取工具结果。正式产品的下一步是把
该工作台迁移到 Agent Run 范围内的结果接口，避免跨任务读取；在完成该迁移前，不把此页
作为评委的正式结果入口。
