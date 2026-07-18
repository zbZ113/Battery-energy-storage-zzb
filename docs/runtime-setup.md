# 泉芯智寿运行与部署指南

本指南区分两种运行形态：仓库可直接启动的“基础 API”，以及需要外部审核依赖才能装配的“完整竞赛应用”。前者适合检查传输、契约和工具发现；后者才包含 Canonical CSV 上传、寿命决策和报告链。

## Windows 10 本机运行

**结论：Windows 10 足够完成当前项目的开发、单机训练、API/Streamlit 演示和比赛录屏。** 基线模型与小型混合模型不要求服务器；有 NVIDIA GPU 时可加速部分 PyTorch 训练，但不是运行基础功能的前提。只有需要公网访问、多人长期使用、持续任务或更稳定的容器环境时，才建议租用 Linux 服务器。

### 环境准备

要求 Python `>=3.11,<3.12`。在仓库根目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[data,dev,ml,api,streamlit]"
```

若 PowerShell 禁止激活脚本，可直接使用虚拟环境解释器：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[data,dev,ml,api,streamlit]"
```

### 启动基础 API

```powershell
.\.venv\Scripts\python.exe -m uvicorn deploy.foundation_api:app --host 127.0.0.1 --port 8000
```

另开终端检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/v1/tools
```

基础 API 只装配不需要企业上下文的工具。`/health` 和 `/v1/tools` 可立即验证；不要把它描述成完整寿命决策服务。

### 启动 Streamlit

工作台需要可访问的 FastAPI 地址：

```powershell
.\.venv\Scripts\python.exe -m streamlit run workbench/streamlit_app.py
```

连接基础 API 时只支持健康检查与工具发现。上传、寿命决策和报告下载要求下文所述的完整竞赛应用。

### 本地验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m compileall -q src workbench deploy
```

## Linux 服务器运行

以下示例适用于安装了 Python 3.11 的 Linux 主机：

```bash
git clone <repository-url> Battery-energy-storage-zzb
cd Battery-energy-storage-zzb
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data,ml,api,streamlit]"
python -m uvicorn deploy.foundation_api:app --host 0.0.0.0 --port 8000
```

服务器应通过防火墙或反向代理限制访问范围。若使用 systemd、容器编排或反向代理，密钥和企业路径应由其秘密管理/环境注入能力提供，不要写入仓库或镜像。

健康检查：

```bash
curl --fail http://127.0.0.1:8000/health
```

Streamlit 可在另一个进程启动：

```bash
python -m streamlit run workbench/streamlit_app.py --server.address 0.0.0.0
```

只有在完整竞赛 API 已就绪时，才应向用户开放工作台的上传和决策操作。

## Docker Compose 基础 API

[`deploy/compose.yaml`](../deploy/compose.yaml) 只启动真实存在的基础 API 入口，不虚构数据库、任务队列、模型服务、Streamlit 或 MCP 进程：

```bash
docker compose -f deploy/compose.yaml up --build
docker compose -f deploy/compose.yaml ps
curl --fail http://127.0.0.1:8000/health
```

停止：

```bash
docker compose -f deploy/compose.yaml down
```

镜像只安装 `api` extra。它适合传输层冒烟检查，不包含科学计算、PyBaMM、MCP SDK 和完整竞赛依赖。

## 完整竞赛应用装配

完整 FastAPI 由 `create_competition_fastapi_app` 构造。仓库不会替调用方猜测企业数据、策略或模型：调用方必须创建并审核以下对象，再把它们注入工厂：

- `CompetitionToolDependencies`：共享 `AuditLedger` 或 `JsonlAuditLedger`、模型、校准/适配/策略/知识 Resolver 等。
- `FileSystemVerifiedEarlyCycleBatchStore`：推荐的重启可恢复 Canonical CSV 存储；单元测试可使用内存实现。
- 经清单与 SHA-256 校验的模型制品。
- 独立的校准队列和批次决策策略。

最小装配结构如下，省略号必须由使用方的审核实现替换，不能作为可运行占位代码：

```python
from quanxin_life.application import FileSystemVerifiedEarlyCycleBatchStore
from quanxin_life.application.http_application import create_competition_fastapi_app
from quanxin_life.audit import JsonlAuditLedger

ledger = JsonlAuditLedger("runtime/audit/tool-results.jsonl")
dependencies = build_reviewed_competition_dependencies(audit_ledger=ledger)
batch_store = FileSystemVerifiedEarlyCycleBatchStore("runtime/batches")
app = create_competition_fastapi_app(dependencies, batch_store=batch_store)
```

将这段装配代码放在使用方自己的运行模块后，可通过 Uvicorn 启动。完整 API 包括：

- `POST /v1/batches/canonical-csv`
- `POST /v1/workflows/lifetime-decision`
- `GET /v1/results/{result_id}`
- `GET /v1/reports/{result_id}`

文件系统批次存储会在每次读取时重新校验 CSV、元数据与来源哈希；JSONL 审计账本会在启动时逐条重建 `ToolResult` 并在损坏、重复或冲突时失败关闭。它们适合单进程比赛部署；多人并发或多副本服务仍应实现同契约的数据库/对象存储后端。

## MCP

MCP SDK 是惰性可选依赖。安装仓库已验证的 SDK 版本：

```bash
python -m pip install -e ".[mcp]"
```

stdio 方式适合由本机 MCP 客户端作为子进程启动：

```bash
python scripts/run_mcp_host.py --transport stdio
```

Streamable HTTP 方式默认只绑定本机回环地址：

```bash
python scripts/run_mcp_host.py --transport streamable-http --host 127.0.0.1 --port 8001
```

客户端地址为：

```text
http://127.0.0.1:8001/mcp
```

如需先检查解析后的非秘密配置而不启动服务：

```bash
python scripts/run_mcp_host.py \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8001 \
  --streamable-http-path /mcp \
  --print-config
```

`src/quanxin_life/tools/mcp_host.py` 提供：

- `create_mcp_host(service, config=...)`
- `run_mcp_host(service, config=...)`
- `stdio` 与 `streamable-http` 两种传输配置

独立入口只装配当前可安全独立运行的共享 `ToolInvocationService`，不会复制领域数值逻辑。当前会暴露数据质量、按电芯划分审计和短期物理核验工具；缺少已审核模型、策略或企业数据的工具不会以占位实现冒充可用。缺少 SDK 时 Host 会抛出显式不可用错误，导入模块不会自动启动服务。

不要直接把未鉴权的 MCP HTTP 端口暴露到公网。远程部署必须由调用方在受控网络中增加认证、TLS、访问控制和审计代理，并继续复用同一工具授权边界。

## 数据路径与制品

推荐把外部数据放在仓库外的受控目录，并在清单中记录绝对/挂载路径、来源 URI、SHA-256 和获取时间。`.env.example` 中的路径变量是运维模板，当前核心包不会自动读取它们；使用方装配模块应显式读取、校验后再传给 Resolver。

- MATR：按项目决定暂时保留接口，不要求下载或挂载原始大数据。
- HUST：官方 pickle 不能在主进程直接加载，必须先在隔离环境完成来源核验与安全转换。
- Naumann：Excel/MAT 需要使用审核后的布局配置转换。
- 企业数据：必须脱敏并通过受控存储提供，不进入 Git 历史、镜像或普通日志。
- 模型：优先 JSON/UBJ、ONNX、safetensors；必须验证清单和 SHA-256。

## 外部需提供项

完整运行前，由团队或企业提供并审核：

| 项目 | 是否必需 | 说明 |
|---|---|---|
| Python 3.11 与项目依赖 | 必需 | 版本必须与锁定环境一致 |
| 数据路径与数据清单 | 完整分析必需 | 包含来源、哈希、版本和字段/单位说明 |
| 模型制品与模型清单 | 寿命预测必需 | 不接受未核验可执行反序列化制品 |
| 校准队列 | 区间签发必需 | 必须与训练、验证、测试电芯互斥 |
| 企业决策策略 | 批次决策必需 | 包含策略 ID、适用范围和审核来源 |
| 企业数据/知识材料 | 可选 | 需脱敏、授权并建立来源链 |
| LLM Provider 与 API key | 可选 | 当前确定性工作流无需 LLM；接入解释/编排适配器后才需要，且不能产生工程数值 |
| MCP SDK | 可选 | 仅启用 MCP Host 时需要 |
| 飞书应用信息 | 可选 | 当前核心仓库尚无飞书运行入口 |
| PyBaMM | 可选 | 仅用于短时工况核验 |

所有 API key、Token、飞书密钥和数据库凭据都应保持为空模板，通过秘密管理器或本机环境提供。日志不得输出这些值。
