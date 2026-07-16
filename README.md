# 泉芯智寿

## 项目简介

泉芯智寿是面向新能源装备产业的储能电芯退化感知与研发决策原型。系统把数据质检、早期寿命预测、SOH 轨迹、预测区间、在线校正、短时物理核验、主动试验推荐和批次决策组织成一条可审计工具链。

项目的基本原则是：LLM 只负责理解任务、选择工具和解释结果；SOH、RUL、预测区间、阈值比较及其他工程数值必须由版本化工具返回，并能追溯到 `ToolResult`、输入哈希、模型/数据/特征版本与来源记录。

完整研发路线见[批准实施计划](docs/superpowers/plans/2026-07-12-quanxin-zhishou-full-implementation.md)，运行环境和部署说明见[运行指南](docs/runtime-setup.md)。

## 核心功能

- 统一电芯元数据、循环记录、数据来源和 SHA-256 清单。
- 按 `cell_id` 进行训练、验证、校准和测试划分，并扫描截断点后特征泄漏。
- 提供 Dummy、Variance、XGBoost、[独立 CPMLP](docs/cpmlp-implementation-note-v1.md) 与混合退化模型组件。
- 使用 Split/Normalized Conformal 签发可审计预测区间。
- 通过个体参数更新持续校正新到达观测，不在线改写全局模型。
- 使用 Naumann 工况数据、Gaussian Process 和成本约束策略推荐补充试验。
- 使用 PyBaMM 做短时间窗工况和边界核验，不生成长期退化标签。
- 依据数据质量、区间和审核策略输出入组、复检、降级或拒绝决策。
- 通过 FastAPI、Streamlit 和可选 MCP Host 复用同一 `ToolInvocationService`。

## 系统架构

```text
可信数据/审核配置
        │
        ▼
数据治理 ──→ 特征与模型 ──→ Conformal/在线校正
        │                         │
        └──→ 物理核验/主动试验 ──┤
                                  ▼
                         批次决策与审计报告
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
                 FastAPI      Streamlit       MCP Host
```

核心包采用 Python 3.11 和 `src/quanxin_life` 布局。`ToolRegistry` 负责强类型输入、白名单与结果契约；`AuditLedger`/`JsonlAuditLedger` 负责结果证据链；FastAPI、Streamlit 与 MCP 只提供传输和展示，不复制算法逻辑。

## 快速开始

### 1. 建立 Python 3.11 环境

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[data,dev,ml,api,streamlit]"
```

需要短时物理核验或 MCP Host 时，再按需安装：

```powershell
python -m pip install -e ".[physics]"
python -m pip install -e ".[mcp]"
```

Linux：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data,dev,ml,api,streamlit]"
```

### 2. 启动可独立验证的基础 API

仓库提供的 [`deploy/foundation_api.py`](deploy/foundation_api.py) 仅装配当前无外部企业依赖即可运行的三个工具，用于验证健康检查、工具发现和通用工具调用：

```powershell
python -m uvicorn deploy.foundation_api:app --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000/health`、`/v1/tools` 或 `/docs`。基础 API 不包含 Canonical CSV 上传和寿命决策工作流；完整接口必须由调用方提供审核后的模型、校准队列、策略和知识后端后，通过 `create_competition_fastapi_app` 装配。

也可以使用最小 Compose 配置：

```powershell
docker compose -f deploy/compose.yaml up --build
```

### 3. 执行质量门禁

```powershell
python -m pytest -q
python -m ruff check .
python -m mypy
python -m compileall -q src workbench deploy
```

安装 `dev` 依赖后可启用同一组本地提交门禁：

```powershell
python -m pre_commit install
python -m pre_commit run --all-files
```

### 4. 运行 Naumann 主动试验回放

先用已审核的 Naumann Excel/MAT 适配器生成条件观测，再把观测、工况边界、初始队列以及有来源的时间/设备成本写入严格 JSON 请求：

```powershell
python scripts/naumann_gp_pipeline.py --request reviewed-request.json --output-dir runtime/naumann-run
```

成功运行会生成可追溯的实验数据 JSON/CSV、GP 回放结果和运行清单；缺少审核成本时只生成明确的降级清单，不会补默认成本或伪造推荐结果。

## API、Streamlit 与 MCP

### FastAPI

基础入口为 `deploy/foundation_api.py`，公开：

- `GET /health`
- `GET /v1/tools`
- `POST /v1/tools/{tool_name}`

完整应用工厂位于 `src/quanxin_life/application/http_application.py`。当调用方注入完整 `CompetitionToolDependencies` 与可信批次存储后，还可提供批次上传、寿命工作流、结果查询和报告读取端点。

### Streamlit

[`workbench/streamlit_app.py`](workbench/streamlit_app.py) 是 HTTP-only 薄客户端：

```powershell
python -m streamlit run workbench/streamlit_app.py
```

工作台的完整上传和决策功能要求它连接到已装配的竞赛 API；连接基础 API 时只能使用健康检查与工具发现。

### MCP

[`src/quanxin_life/tools/mcp_host.py`](src/quanxin_life/tools/mcp_host.py) 提供惰性加载的 stdio/Streamable HTTP Host 工厂。MCP SDK 当前不是项目的默认依赖，宿主应用需安装经自身环境验证的兼容 SDK，再把已装配的 `ToolInvocationService` 传给 `create_mcp_host` 或 `run_mcp_host`。SDK 缺失时会显式报告不可用，不会返回占位结果。

## 数据与安全边界

- 原始文件必须登记来源、许可证/使用条件、SHA-256、下载时间和数据版本。
- 未核验来源时禁止加载 pickle、joblib、`.pt`、`.pth` 等可执行反序列化制品。
- 所有数据集必须按 `cell_id` 划分；标准化器、插值器和特征提取器不得读取测试集合或预测截断点后的信息。
- Canonical CSV 采用固定表头、逐行 Pydantic 校验、大小限制和来源哈希绑定。
- PyBaMM 仅用于短时滚动核验；默认参数集不能被描述为目标工业电芯标定结果。
- 缺测、域外、未校准、工具失败或数据质量阻断必须显式降级。
- 日志不得保存密钥、完整用户提示或原始敏感企业数据。

MATR 按需求暂时只保留接口，不下载大体量原始数据。HUST 官方 pickle 必须在隔离环境完成来源核验和安全转换后才能进入统一格式。企业数据不得放入源码仓库，建议由受控路径或对象存储提供。

## 已实现能力

- MATR 有界读取接口，以及 Naumann Excel/MAT 审核布局适配器。
- HUST ZIP 静态安全审计与 Canonical CSV 可信批次存储。
- 数据质检、标签、划分、泄漏扫描和早期循环特征。
- Dummy、Variance、XGBoost、CPMLP、混合退化、Conformal、CORAL/DANN 和在线个体校正组件。
- 受治理 XGBoost/Variance 制品注册与安全加载边界。
- `FileSystemVerifiedEarlyCycleBatchStore` 与 `JsonlAuditLedger` 提供重启可恢复、读取时复验的本地持久化。
- Naumann-GP 主动试验、PyBaMM 短时核验和批次决策组件。
- 十五个标准领域工具的显式装配、角色白名单、共享审计账本与 Markdown 报告；
  其中场景年份只能由已审计的循环寿命结果和明确运行策略换算，不能表述为长期实测寿命。
- Canonical CSV 到寿命决策报告的确定性工作流代码与 FastAPI 传输层。
- HTTP-only Streamlit 工作台及可选 MCP Host 工厂。
- pytest、Ruff、mypy、compileall 与 GitHub Actions 门禁配置。

## 验证边界

仓库当前提供的是学习和竞赛研发原型。MATR 原始数据按需求未落盘，HUST 安全转换和跨数据集真实评测仍待完成；仓库不发布未经真实数据复现的模型性能、覆盖率或业务收益。完整寿命工作流还要求使用方提供经过审核的模型制品、校准队列、策略及相关来源记录，现有结果不能用于现场控制、质保或安全承诺。
