# 泉芯智寿运行与部署指南

本仓库只保留两条受支持的运行路径：

1. Windows 本机完整竞赛栈：`deploy/local.compose.yaml`。
2. 服务器竞赛栈：`deploy/competition.compose.yaml`。

旧的 foundation-only Compose 和 Streamlit 研究工作台已移除。飞书机器人、Aily MCP、
CyclePatch、HybridPatch、BLAST-Lite、PostgreSQL、Redis、Celery、Next.js 与 Nginx
必须通过同一完整运行时装配，避免出现“接口能访问但模型、账本或任务队列未接通”的假成功。

## 运行边界

| 形态 | 入口 | 用途 |
| --- | --- | --- |
| 本机完整栈 | `deploy/local.compose.yaml` | 开发、飞书/Aily 联调和比赛演示 |
| 服务器完整栈 | `deploy/competition.compose.yaml` | 带 TLS、不可变镜像和受管目录的竞赛部署 |
| 独立回调 runner | `scripts/run_feishu_callback.py` | 仅验证飞书 URL verification、验签、解密与入队，不执行分析 |
| 独立 MCP Host | `scripts/run_mcp_host.py` | 本机通用工具协议检查，不代替 Aily 专用 MCP 入口 |

任何 SOH、RUL、区间、阈值比较或工况寿命数值都必须来自有效 `ToolResult`。
原始数据、模型权重、运行时配置和密钥不进入 Git。

## 环境要求

- Python 3.11。
- Docker Desktop 或兼容的 Docker Engine 与 Compose。
- 本地运行时目录、模型注册表、校准证据和 secrets。
- 启用飞书/Aily 时所需的 App、Bitable、端点令牌和身份映射。
- 临时公网联调时使用当前有效的 HTTPS 隧道地址；Quick Tunnel 重启后地址会变化。

开发依赖按需安装：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[agents,auth,data,dev,knowledge,llm,ml,api,infrastructure,persistence,reporting,scenarios,mcp]"
```

不需要为普通产品代码修改重新训练模型。正式 A100 结果与模型制品由受管注册表复用。

## Windows 本机完整栈

首次准备目录与基础秘密：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\local\prepare_runtime.ps1 -RuntimeRoot D:\QuanxinRuntime
```

安装已经审核的演示资产与模型注册表：

```powershell
.\.venv\Scripts\python.exe scripts\local\prepare_assets.py --target D:\QuanxinRuntime\runtime
```

复制 `deploy/local.env.example` 的字段到运行时私有 env 文件并填写本机值。
飞书/Aily secrets 只写入 `QUANXIN_SECRETS_ROOT` 指向的目录，不写进 env、代码或日志。

启动完整本地栈：

```powershell
docker compose --env-file D:\QuanxinRuntime\compose\local.env -f deploy/local.compose.yaml -f deploy/competition.feishu-aily.override.yaml -f deploy/competition.aily-mcp.override.yaml up -d --build
```

查看状态：

```powershell
docker compose --env-file D:\QuanxinRuntime\compose\local.env -f deploy/local.compose.yaml -f deploy/competition.feishu-aily.override.yaml -f deploy/competition.aily-mcp.override.yaml ps
Invoke-RestMethod http://127.0.0.1:8080/health
```

后端 Dockerfile 已把第三方依赖与项目源码拆成独立缓存层。只改源码时，
Docker 只执行快速的本项目安装与编译层；只有 `pyproject.toml` 发生变化才会重装
PyTorch 和其他依赖。

停止服务：

```powershell
docker compose --env-file D:\QuanxinRuntime\compose\local.env -f deploy/local.compose.yaml -f deploy/competition.feishu-aily.override.yaml -f deploy/competition.aily-mcp.override.yaml down
```

## 飞书事件与 Aily MCP

完整本地栈默认监听 `http://127.0.0.1:8080`。创建当前 HTTPS 隧道：

```powershell
cloudflared tunnel --protocol http2 --url http://127.0.0.1:8080
```

以 cloudflared 本次输出的主机名为准，并同步更新运行时的
`EXTERNAL_HTTPS_BASE_URL` / `QUANXIN_EXTERNAL_HTTPS_BASE_URL` 后重启 API。
不要沿用断电前的 trycloudflare 主机名。

飞书开放平台事件回调：

```text
https://<当前公网主机>/v1/integrations/feishu/events
```

域名和路径之间只能有一个斜杠。

Aily 自定义 MCP 服务：

```text
https://<当前公网主机>/v1/aily/mcp/<端点令牌>/
```

MCP URL 必须保留末尾斜杠。Aily 的出口 IP、测试身份发现权限和真实用户身份映射见
[Aily MCP HTTPStreaming 接入](integrations/aily-mcp-httpstreaming.md)。

## 服务器竞赛栈

服务器使用不可变镜像和受管挂载目录：

```powershell
docker compose --env-file <server-private-env> -f deploy/competition.compose.yaml -f deploy/competition.feishu-aily.override.yaml -f deploy/competition.aily-mcp.override.yaml up -d
```

部署前必须完成：

- 数据库迁移；
- deployment registry、模型制品和 calibration evidence 的 SHA-256 核验；
- 飞书事件密钥、Aily MCP 端点令牌和用户映射；
- active route 与候选 BLAST 展示授权；
- TLS、反向代理、备份与回滚路径。

详细步骤见 [竞赛 ECS 部署](deployment/competition-ecs.md)、
[运维 Runbook](deployment/operations-runbook.md) 和
[部署安全与 Secrets](deployment/security-and-secrets.md)。

## 轻量验证

普通代码和整理变更不运行训练。按影响范围执行：

```powershell
python -m compileall -q src deploy migrations
python -m ruff check src deploy scripts
python -m mypy
```

根目录测试代码作为本地维护材料，不随 Git 交付。飞书/Aily 发布前还需在目标租户执行
真实回调、MCP 工具发现、任务解析和工况查询冒烟。

前端：

```powershell
Set-Location frontend
pnpm install --frozen-lockfile
pnpm lint
pnpm typecheck
pnpm build
```

这些检查证明当前代码契约成立，不替代真实飞书租户、真实模型制品或长期运行验收。

## 数据与制品

保留的证据数据域为：

- MATR：CyclePatch RUL 与 HybridPatch 有限时域 SOH。
- Naumann cycle/calendar：BLAST-Lite 外部参考核验。
- LFP 280Ah DoD：大型方形 LFP 观测范围尺度核验。

原始数据不进入 Git。任何新 CSV 先完成字段、单位、周期完整性、来源 SHA-256 和支持域
检查；缺少必要字段时显式拒绝，字段充分时由服务端规范化并路由到已激活模型。
