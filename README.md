# 泉芯智寿

> 面向电芯研发的可信寿命预测、退化轨迹与受约束专业智能体工作流

泉芯智寿把真实电芯数据治理、早期寿命预测、有限时域 SOH 轨迹、
Split Conformal 区间和可审计交付组织成一条工程链。系统中的 SOH、RUL、
区间和决策数值只能由版本化工具产生，并以 `ToolResult` 进入审计账本；
LLM 只负责理解任务、编排已授权工具和解释已有证据。

当前仓库已经完成 MATR 三批数据上的正式 Advanced 训练、结果验收、
模型路由、项目级 API / Agent / 报告 / UI 代码链及测试环境纵向 E2E。
这不等于生产部署，也不构成跨数据集覆盖保证或工业寿命承诺。

## 当前状态

状态更新时间：2026-07-27。

| 层级 | 含义 | 当前结论 |
| --- | --- | --- |
| **Implemented** | 代码、契约和测试入口已存在 | 数据治理、模型训练、制品验证、项目工具、Agent、报告、Next.js UI、校准物化 |
| **Validated** | 已有正式实验、哈希验收或受控 E2E 证据 | MATR Advanced Final 80 次运行；指标闭环；Split Conformal；Advanced 单电芯纵向 E2E |
| **Planned** | 尚未完成或需要外部输入 | HUST 外部验证、删失感知区间、完整部署栈、真实 BMS/EMS、企业数据验收 |

完整成熟度矩阵见 [项目状态](docs/status.md)，已知限制见
[已知限制](docs/limitations.md)。

## 已验证结果

### MATR Advanced Final

正式矩阵：

```text
4 个 cutoff（20 / 50 / 100 / 150）
× 4 个模型
× 5 个随机种子（38 / 39 / 40 / 41 / 42）
= 80 次正式运行
```

结果包验证摘要：

```text
status=VERIFIED
mode=final
operations=80
files=2654
source_commit=232d9fc8957bb547ca2b80205802f377864e982b
output_sha256=d201224870a28c655f66a810bc94f90ad28133e06f2fb4a7285195274c303d82
```

代表性五种子结果：

| 任务 | 模型与 cutoff | MAE | RMSE | 其他指标 |
| --- | --- | ---: | ---: | --- |
| MATR 官方 cycle life | CyclePatch Direct，150 cycles | 98.19 cycles | 122.49 cycles | MAPE 12.62%，R² 0.8656，15%-Acc 64.44% |
| 有限时域 SOH | HybridPatch-v2，20 cycles | 1.281 SOH 百分点 | 2.700 SOH 百分点 | 单调违规率 0% |
| 有限时域 SOH | Current Hybrid，20 cycles | 1.610 SOH 百分点 | 2.578 SOH 百分点 | 单调违规率 0% |

这些结果说明：

- CyclePatch Direct 在 `cutoff=150` 获得当前最低 RUL 平均 MAE；
- Direct 与 CyclePatch-BatLiNet 的逐电芯配对区间跨过零，不能宣称统计显著胜出；
- HybridPatch-v2 的平均 SOH 误差较低，但 Current Hybrid 的 RMSE 和尾部风险更低；
- 当前 SOH 因而采用平均精度与尾部/效率双路由，不宣称单一全能冠军；
- 12 个 calibration 电芯属于小校准队列；90% 与 95% 在部分条件下使用相同有限样本分位数；
- RUL `cutoff=100` 的现有候选均未达到目标 90% PICP。

完整五种子表、PICP、MPIW 和晋级口径见
[Advanced Benchmark](docs/benchmark.md)。

## 核心技术路线

```text
真实电芯数据
  → 来源登记、SHA-256、Canonical Schema
  → 按 cell_id 隔离 train / validation / calibration / test
  → CyclePatch RUL / HybridPatch SOH
  → route-specific Split Conformal
  → ToolResult 与 Audit Ledger
  → exact AgentStep + fenced worker
  → API / 审计报告 / Next.js UI
```

### 1. 数据与标签

- MATR 三批 MATLAB v7.3/HDF5 数据采用来源清单、文件哈希和安全读取；
- 训练、验证、校准和测试严格按 `cell_id` 隔离；
- RUL 标量目标是 MATR 官方 `cycle-life`，不是统一 EOL80；
- SOH 监督和输出采用真实有限时域，当前正式轨迹最长到 cycle 500；
- HUST 官方 pickle 不在主进程直接加载，必须先在隔离环境完成来源核验与安全转换。

### 2. 正式模型与基线

| 类别 | 定位 |
| --- | --- |
| Dummy / Variance / XGBoost | 透明基线与经典特征基线 |
| CPMLP | legacy 曲线集合基线；不作为当前 Advanced 旗舰 |
| CyclePatch Direct | 带真实 cycle 位置、工况编码和跨循环 Transformer 的 RUL 模型 |
| CyclePatch-BatLiNet | 在 CyclePatch 表征上加入冻结参考库分支的 RUL 模型 |
| Current Hybrid | 带强单调先验的 SOH 尾部/效率基线 |
| HybridPatch-v2 | 学习式退化幅度与多尺度曲线编码的 SOH 平均精度模型 |

模型输入、路由、拒绝条件和安全制品要求见
[Advanced 模型卡](docs/model-card-advanced.md)。

### 3. 可信数值链

- 模型、数据、划分、特征、代码、路由和校准队列均带版本；
- safetensors、JSON、Parquet 等制品在消费前复验大小、清单和 SHA-256；
- 禁止未经核验加载 pickle、joblib、`.pt` 或 `.pth`；
- `AgentStep` 精确绑定工具、输入身份、依赖和审批证据；
- ledger 与 step 完成原子提交；
- `fenced claim` 阻止失效 Worker 覆盖新结果；
- Redis/Celery 校准任务只传 `materialization_id`，不传样本数组、路径或哈希；
- 报告和 UI 只消费已登记 `ToolResult`，不重新计算或编造业务数字。

## 成熟度边界

### 已经验证

- MATR 三批 Advanced Final：80 次正式运行；
- 1,080 行 RUL 测试预测和 500,000 行 SOH 逐循环预测对账；
- 80% / 90% / 95% Split 与 Normalized Conformal 离线评价；
- 15 条条件路由与具体 representative checkpoint 绑定；
- Advanced artifact v2、安全注册、人工激活/回退和 active-route resolver；
- target-aware MATR official RUL、finite-horizon SOH、route-specific Split Conformal；
- 项目 API、受约束 Agent、报告与单电芯 UI；
- calibration materialization 的 ADMIN API、identity-only 队列、Worker 和管理 UI；
- 使用真实认证、数据库、装配和证据解析的测试环境纵向 E2E。

### 尚未验证或尚未部署

- HUST 零样本外部泛化和跨域重校准；
- 右删失样本的 survival-aware 评价与 Conformal；
- 真实企业 BMS/EMS 数据、设备协议和安全联锁；
- 完整 PostgreSQL / Redis / Worker / 对象存储 / API / UI 一键部署；
- 生产监控、备份恢复、容量规划、密钥轮换和灾难恢复；
- 模型单样本推理延迟与服务吞吐基准；
- 工业环境中的长期稳定性、合规性和决策责任验收。

更完整的禁止性表述和适用范围见 [已知限制](docs/limitations.md)。

## 快速开始

要求 Python 3.11。

### 1. 安装核心与开发依赖

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[data,dev,ml,api]"
```

Linux：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data,dev,ml,api]"
```

按需安装高级能力：

```bash
python -m pip install -e ".[agents,auth,infrastructure,persistence,reporting,training]"
python -m pip install -e ".[mcp]"
```

### 2. 启动 foundation API

[`deploy/foundation_api.py`](deploy/foundation_api.py) 只提供健康检查、工具发现和
已装配基础工具，不是完整产品 API：

```bash
python -m uvicorn deploy.foundation_api:app --host 127.0.0.1 --port 8000
```

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/v1/tools
http://127.0.0.1:8000/docs
```

基础 Compose：

```bash
docker compose -f deploy/compose.yaml up --build
```

完整装配要求受审查的依赖与运行输入，见
[运行与装配指南](docs/runtime-setup.md)。文件系统原型可使用
`FileSystemVerifiedEarlyCycleBatchStore` 与 `JsonlAuditLedger`；并发环境应使用
数据库和对象存储实现同等契约。

### 3. Next.js 门户

前端版本与命令以 [`frontend/package.json`](frontend/package.json) 为准，
固定使用 `pnpm@10.28.1`：

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

### 4. Streamlit 科研工作台

```bash
streamlit run workbench/streamlit_app.py
```

工作台只通过 HTTP API 消费结果，不直接加载模型或访问服务端文件路径。

### 5. 三批 MATR 与 A100 入口

正式数据集标识为 `matr-three-batch`。训练入口保留用于复现和后续批准实验：

```bash
bash scripts/a100/preflight.sh
bash scripts/a100/train_dataset.sh matr-three-batch smoke
bash scripts/a100/train_dataset.sh matr-three-batch final
```

当前正式结果已经完成，不需要为了产品开发重复运行 A100。A100 离线服务器不使用
Git、SSH、SCP 或网盘；更新继续采用人工传输、离线归档和 SHA-256 校验。

复现层级和制品要求见 [可复现性](docs/reproducibility.md)。

## 扩展能力

以下能力存在代码、契约或沙箱，但不属于当前算法主结论：

- PyBaMM：短时工况边界核验，不制造长期退化标签；
- 主动试验：Gaussian Process、信息增益和成本约束候选排序；
- 知识库：审核材料、BM25、pgvector 与 reranker 的显式降级检索；
- MCP：[`src/quanxin_life/tools/mcp_host.py`](src/quanxin_life/tools/mcp_host.py)
  提供 stdio 和 Streamable HTTP；
- 飞书：`src/quanxin_life/integrations/feishu` 提供签名回调、幂等事件和引用卡片，
  HTTP 入口包含 `POST /v1/integrations/feishu/events`；
- 工业接口：REST / MQTT / Modbus / EMS 均为协议沙箱，不代表真实 BMS/EMS 已接入。

## 质量门禁

后端：

```bash
python -m pytest -q
python -m ruff check .
python -m mypy
python -m compileall -q src workbench deploy migrations
python -m pip check
```

前端：

```bash
cd frontend
pnpm test
pnpm lint
pnpm typecheck
pnpm build
```

GitHub Actions 使用 Python 3.11、Node.js 24 和锁定的 pnpm。通过测试只能证明当前
自动化契约成立，不能替代外部数据验证、生产环境验收或安全审计。

## 文档导航

### 当前证据

- [项目状态：Implemented / Validated / Planned](docs/status.md)
- [Advanced Benchmark](docs/benchmark.md)
- [Advanced 模型卡](docs/model-card-advanced.md)
- [已知限制](docs/limitations.md)
- [可复现性](docs/reproducibility.md)

### 工程与契约

- [架构说明](ARCHITECTURE.md)
- [运行与装配指南](docs/runtime-setup.md)
- [项目宪法](PROJECT_CHARTER.md)
- [数据契约](DATA_CONTRACT.md)
- [实验协议](EXPERIMENT_PROTOCOL.md)
- [完成定义](DEFINITION_OF_DONE.md)
- [技术路线图](docs/architecture/technical-roadmap.md)

### 数据与合规

- [HUST 数据卡](docs/data-cards/hust-public-data-v1.md)
- [Naumann 数据卡](docs/data-cards/naumann-public-data-v1.md)
- [许可证策略](LICENSE_POLICY.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)

`docs/superpowers/plans/` 保存历史实施计划和内部工程追踪，不作为当前公开状态的
单一事实源。部署链将在上述文档纠偏完成后单独设计和验收。
