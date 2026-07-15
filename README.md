# 泉芯智寿

## 项目定位

“泉芯智寿”是面向新能源装备产业的储能电芯退化感知与决策系统。当前仓库定位为**可信领域工具与算法原型**：以可复现的数据处理、寿命预测、不确定性校准、试验推荐和决策工具为数值来源，由受约束的工作流负责组织工具调用、证据链和报告。

项目正在按“华为杯”地方赛道目标向真实数据驱动、可端到端演示的作品推进，但目前**不应视为生产系统，也没有形成可对外宣称的真实模型性能**。

完整实施路线见 [批准计划](docs/superpowers/plans/2026-07-12-quanxin-zhishou-full-implementation.md)。

## 不可突破的可信边界

- LLM 只能编排工具和解释工具结果，不能计算、猜测、补齐或改写 SOH、RUL、预测区间、阈值比较及经营指标。
- 每个业务数值必须来自有效、版本化的 `ToolResult`，并可追溯到 `result_id`、工具版本、模型版本、数据版本、特征版本、输入哈希和来源记录。
- 训练、验证、校准与测试必须按 `cell_id` 划分；禁止同一电芯跨集合，也禁止特征读取预测截断点之后的数据。
- 未经来源核验，不加载 pickle、joblib、`.pt`、`.pth` 等可执行反序列化制品；模型制品需要清单与 SHA-256 校验。
- PyBaMM 只用于短时滚动物理核验和敏感性参考，不生成长期退化标签，也不替代真实测试数据。
- 缺测、域外、未校准和工具失败必须显式降级；不得静默填补或返回示例数值冒充结果。
- 所有外部接口复用 `quanxin_life.core` 的公共枚举和 Pydantic 契约。

## 当前已实现

### 数据与治理

- 统一的电芯、循环记录、来源清单和哈希契约；数据验证、存储、标签、泄漏审计与电芯级划分。
- MATR 有界读取适配器和 Naumann Excel/MAT 布局适配器。
- HUST 压缩包静态安全审计；主进程不会直接反序列化其 `.pkl` 文件。
- 规范化 CSV 的严格表头、逐行 Pydantic 校验、来源 SHA-256 绑定与内存批次注册器。
- 早期循环特征、曲线张量与 `ΔQ(V)` 方差特征。

### 模型与不确定性

- Dummy、Variance、XGBoost 和 CPMLP 寿命基线。
- 单调趋势与累计残差结合的混合退化轨迹模型。
- Split Conformal、Normalized Conformal，以及独立校准集约束。
- CORAL、DANN 目标域适应原型；在线个体参数校正。
- 受治理的模型制品注册表：仅接受 XGBoost JSON/UBJ 与透明 Variance JSON，注册和读取均复核大小、SHA-256、版本、数据集、截断周期与特征模式；显式拒绝 pickle、joblib、`.pt`、`.pth`。
- XGBoost 原生制品可从已核验注册表安全加载；只有这一路径的寿命预测结果才会标记为 `VERIFIED_ARTIFACT`，制品哈希同时进入来源链。

### 试验、物理与决策

- Naumann 工况数据到高斯过程训练输入的桥接。
- 高斯过程、最大方差和成本约束主动试验推荐与回放组件。
- PyBaMM 短时工况核验适配器，包含明确的不可用/失败降级。
- 基于预测区间、质量状态和策略来源的入组、复检、降级与拒绝决策。

### 工具、智能体与审计

- 统一 `ToolResult`、白名单 `ToolRegistry`、角色权限、输入哈希和追加式 `AuditLedger`。
- 十四个标准领域工具的显式应用装配：数据质检、划分审计、早期特征、新观测入库、寿命预测、SOH 轨迹预测、区间校准、目标域适应、个体更新、工况核验、试验推荐、批次决策、知识证据检索和审计报告。
- 下游工具只能消费已登记的上游结果；正式 Markdown 报告按结果 ID 和数值路径取值，阻止客户端或 LLM 注入工程数值。
- 确定性的寿命决策工作流已形成代码原型：它从服务端可信批次解析器取得完整记录，质量检查通过后依次执行特征提取、寿命预测、Conformal 校准与区间签发、批次决策和审计报告；质量阻断时不继续产生下游数值。正式模式拒绝未登记或未核验的模型制品。
- 通用 FastAPI 工具传输层；支持 Canonical CSV 注册、ID-only 寿命工作流、审计结果读取和 Markdown 报告读取。
- 薄 Streamlit 科研工作台通过 HTTP 调用 FastAPI，可上传 Canonical CSV 与审核后的注册 JSON、提交三个服务端 ID、原样显示和下载审计 Markdown，不复制数值逻辑。
- 传输中立 MCP 适配器及 stdio/Streamable HTTP Host；SDK 采用可选依赖与惰性加载，所有调用仍委托共享 `ToolInvocationService`。

## 当前限制

- MATR 仅保留读取与接入接口，原始数据未纳入仓库，也尚未形成 MATR 真实训练和评测结果。
- HUST 的安全隔离转换流程尚未实现；在此之前不能进行真实 MATR→HUST 外测、域适应或目标域重校准。
- Naumann 适配与 GP 组件已经存在，但仓库尚无可作为比赛结论的正式回放结果。
- 现有算法测试主要验证契约、边界和合成样例；仓库不提供真实性能承诺，不应引用虚构的 MAE、RUL、覆盖率或业务收益。
- PyBaMM 是可选依赖，且只承担短期核验；默认参数集不代表目标工业电芯已经标定。
- 当前已有单请求上传—工作流—报告的内存原型，但没有异步任务状态、跨进程持久化审计数据库、BMS/EMS 接口或生产部署配置；服务重启后内存批次与审计结果不会恢复。
- 本机尚未安装 MCP SDK，因此 Host 契约通过伪 SDK 测试，但不能声称真实 stdio/Streamable HTTP 传输已经运行。
- 知识检索目前以可信契约和后端接口为主，不等于已经建成生产级电池知识库。
- 本仓库是学习与竞赛研发原型，不具备生产安全、质保或现场控制用途。

## 目录结构

```text
Battery-energy-storage-zzb/
├── configs/                         # 数据源与运行配置
├── docs/                            # 协议、契约与批准实施计划
├── src/quanxin_life/
│   ├── core/                        # 公共枚举、Pydantic 契约与哈希
│   ├── data/                        # 数据治理、划分、存储与数据集适配器
│   ├── features/                    # 早期循环、曲线张量和方差特征
│   ├── models/                      # Dummy/Variance/XGBoost/CPMLP/Hybrid
│   ├── uncertainty/                 # Split/Normalized Conformal
│   ├── adaptation/                  # CORAL 与 DANN
│   ├── online/                      # 在线个体参数校正
│   ├── experiments/                 # Naumann 桥接、GP 与主动试验
│   ├── physics/                     # PyBaMM 短时核验
│   ├── decision/                    # 批次决策策略
│   ├── tools/                       # 十四个强类型领域工具及 MCP 适配器
│   ├── audit/                       # 数值防火墙与审计账本
│   ├── agents/                      # 角色受限的确定性工具编排
│   ├── application/                 # 完整工具装配与寿命决策工作流
│   ├── api/                         # 通用 FastAPI 工具调用层
│   └── reporting/                   # 可审计 Markdown 报告
├── workbench/                       # HTTP-only Streamlit 科研工作台
├── tests/
│   ├── unit/                        # 模块契约与边界测试
│   ├── leakage/                     # 截断点和数据泄漏测试
│   ├── integration/                 # 应用装配集成测试
│   └── e2e/                         # 确定性寿命决策工作流测试
├── AGENTS.md                        # 仓库协作与安全规则
└── pyproject.toml                   # Python 3.11 包与可选依赖
```

## Python 3.11 开发环境

项目要求 Python `>=3.11,<3.12`，采用 `src/quanxin_life` 布局。以下命令使用当前 `pyproject.toml` 声明的可选依赖：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[data,dev,ml,physics,api,streamlit,agents,mcp]"
```

若只开发核心契约和不依赖科学计算栈的模块，可使用：

```powershell
python -m pip install -e ".[dev]"
```

可选依赖不会在导入核心包时自动启动模型训练、网络访问或服务。

## 本地质量检查

执行以下命令可运行仓库当前配置的完整本地门禁；README 不预先声明其结果，实际状态以当前工作区命令输出为准。

```powershell
.\.venv\python.exe -m pytest -q
.\.venv\python.exe -m ruff check .
.\.venv\python.exe -m mypy
.\.venv\python.exe -m compileall -q src workbench
```

开发单个模块时可以先运行目标测试，完成集成前仍应重新执行上述完整门禁。

## 当前开发优先级

下一阶段重点不是继续增加协议或 Agent 角色，而是打通一条有真实证据的纵向链：

```text
可信数据接入
→ 固定电芯级划分
→ 基线与混合模型评测
→ Conformal 覆盖率
→ 批次决策
→ API/界面
→ 审计报告与复现包
```

在真实数据、实验指标、应用装配和演示链路完成前，任何模型效果、区间覆盖率和工业价值都只能作为待验证目标，不能作为项目既有成果。
