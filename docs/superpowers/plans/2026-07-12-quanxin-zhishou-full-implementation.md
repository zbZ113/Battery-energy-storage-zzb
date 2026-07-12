# 泉芯智寿最高配置完整实现计划

> **执行要求：** 使用 `subagent-driven-development` 分派短生命周期专业代理，主控代理在每阶段执行接口审查、测试审查与集成验收。

**目标：** 构建面向华为杯第八届中国研究生人工智能创新大赛济南地方赛题的、多智能体协同储能电芯寿命预测与退化感知决策系统，实现从公开 LFP 数据治理、寿命预测、不确定性量化、在线校正和主动试验，到物理核验、批次决策、知识解释、BMS/EMS 及飞书协同的完整闭环。

**架构：** 独立单仓库、模块化单体与异步任务架构。数值模型和 PyBaMM 产生全部工程数值；四个专业智能体与监督器仅负责规划、工具编排、解释和报告；FastAPI、MCP、Next.js、Streamlit 共享同一套领域服务与强类型工具。

**技术栈：** Python 3.11、PyTorch、XGBoost、scikit-learn、PyBaMM、FastAPI、Celery、PostgreSQL/pgvector、Redis、MinIO、MLflow、DVC、MCP Python SDK、LangGraph、Next.js/TypeScript、Streamlit、Plotly、MQTT、Modbus、Docker Compose、WSL2。

## 一、竞赛定位与不可变规则

- 项目名称固定为“泉芯智寿——面向新能源装备产业的多智能体协同储能电芯寿命预测与退化感知决策系统”。
- 主报开放赛题三“AI+X”，兼报济南地方赛题，突出新能源装备、储能电池、工业智能体和智能制造协同。
- 报名截止为 2026-08-25，作品提交截止为 2026-09-01；最终以赛事官方通知为准。
- 不加载未知来源的 `.pkl`、`.joblib`、`.pth`；模型权重仅加载本项目训练并登记哈希的产物。
- 不使用行级随机划分；训练、验证、Conformal 校准和测试必须按电芯完全隔离。
- LLM 不得生成或修改 SOH、RUL、预测区间、退化率或物理仿真值。
- 无 `result_id`、模型版本、数据版本、输入哈希和来源记录的数字不得进入正式报告。
- PyBaMM 仅承担短期工况响应和边界核验，不承担 15—25 年寿命预测。
- 公开实验室数据不能表述为企业真实工业电芯数据。
- 所有第三方代码须具有兼容许可证并登记至 `THIRD_PARTY_NOTICES.md`。

## 二、系统与公共接口

### 仓库结构

```text
src/quanxin_life/
├── core/             # 公共枚举、Schema、版本和哈希
├── data/             # Schema、适配器、质检、清洗、划分、泄漏审计
├── features/         # ΔQ(V)、容量、温度、内阻、协议和工况特征
├── models/           # Dummy、Variance、XGBoost、CPMLP、混合模型
├── uncertainty/      # Split/Normalized Conformal 与覆盖率评估
├── adaptation/       # CORAL、DANN、少样本微调、目标域重校准
├── online/           # 个体参数更新、历史版本与漂移监测
├── physics/          # PyBaMM SPMe 短期物理核验
├── experiments/      # Naumann GP、候选空间、主动试验推荐
├── decision/         # 入组、复检、降级及成本敏感策略
├── knowledge/        # 文献解析、混合检索、重排与引用
├── tools/            # Tool Registry、MCP 及强类型输入输出
├── agents/           # 四专业 Agent、监督器、状态图和记忆
├── audit/            # 数值防火墙、来源链、事件记录
├── reporting/        # JSON、Markdown、PDF 和 Word 报告
├── api/              # FastAPI、任务、模型与审计接口
└── integrations/     # MQTT、Modbus、BMS/EMS、飞书
frontend/             # Next.js 正式驾驶舱
workbench/            # Streamlit 算法与实验工作台
tests/                # unit、integration、leakage、api、agent、e2e
configs/              # 数据、模型、实验、服务和策略配置
scripts/              # 数据下载、处理、训练、评估和提交材料生成
docs/                 # 项目宪法、数据协议、实验协议和竞赛文档
```

### 核心数据定义

- `SOH = 当前诊断放电容量 / 初始参考容量`。
- 初始参考容量取形成阶段后首批有效诊断循环容量中位数。
- 同时保留数据集官方寿命标签和统一 `EOL80` 标签。
- `EOL80` 为 SOH 首次不高于 0.80，且后续两个有效诊断点持续不高于 0.80。
- 早期预测截断点固定为 20、50、100、150 循环。
- 固定随机种子 `20260712`。
- 同数据集按 60%/15%/10%/15% 划分训练、验证、校准、测试电芯，并按寿命分位数和协议分层。
- 额外设置协议完全留出、MATR→HUST 数据集留出和 Naumann 储能工况外部验证。
- 元数据保存为 JSON，采样点及循环数据保存为 Parquet；每个原始文件登记来源、许可证、论文、下载时间和 SHA-256。

### 核心工具契约

所有工具返回统一 `ToolResult`，至少包含：`result_id`、`tool_name`、`tool_version`、`model_version`、`data_version`、`feature_version`、`input_hash`、`values`、`uncertainty`、`warnings`、`provenance`、`created_at`。

完整工具集：

```text
validate_battery_data
audit_dataset_split
extract_early_cycle_features
predict_cycle_life
predict_soh_trajectory
calibrate_prediction_interval
adapt_to_target_domain
update_cell_parameters
check_operating_condition
recommend_next_experiment
make_batch_decision
retrieve_battery_evidence
generate_audited_report
```

FastAPI、MCP、Agent、Next.js 和 Streamlit 只能调用这些共享工具，不得复制业务逻辑。

### 系统服务

Docker Compose 包含 API、Celery worker、Next.js frontend、Streamlit workbench、PostgreSQL/pgvector、Redis、MinIO、MLflow 与 Mosquitto，并为每个服务提供健康检查与持久卷。

## 三、完整实施阶段

### 阶段 1：项目宪法与工程底座（7 月 12 日—7 月 15 日）

- 建立 `AGENTS.md`、项目章程、架构规范、数据契约、实验协议、许可证策略和完成定义。
- 初始化 Python 3.11、uv 锁文件、Next.js、Docker Compose、CI、Ruff、mypy、pytest 和 pre-commit。
- 配置 PostgreSQL、Redis、MinIO、MLflow、Mosquitto及健康检查。
- 建立模型、数据、工具、审计和报告的 Pydantic 公共类型。
- 验收：所有容器启动；API、Next.js、Streamlit 和数据库健康检查通过；CI 完成空项目质量门禁。

### 阶段 2：公开数据治理（7 月 16 日—7 月 22 日）

- 从官方来源获取 MATR、HUST、Naumann 循环及日历老化数据，保存许可证和哈希。
- 参考 BatteryML 数据结构思想，独立实现适配器；禁止复用其有问题的 Severson 特征和划分逻辑。
- 实现 JSON 元数据、Parquet 数组、DuckDB 查询和 DVC 版本链。
- 实现字段、单位、时间、循环完整性、异常跳变、重复数据和缺失数据检查。
- 实现电芯级划分、协议留出、数据集留出及特征泄漏扫描。
- 验收：三类数据均可转换；同一电芯不会跨分区；特征禁止访问截断点后数据；数据质量报告自动生成。

### 阶段 3：特征与算法基线（7 月 23 日—7 月 29 日）

- 实现容量趋势、ΔQ(V)、dQ/dV、内阻、充电时间、库仑效率、温度和协议特征。
- ΔQ(V) 采用固定电压网格、单循环 PCHIP 插值；重复电压点和非单调区间先清洗。
- 实现 Dummy、Variance、XGBoost 基线。
- 从 BatteryLife 仅薄改 CPMLP 结构，保留 MIT 声明，重写数据加载、训练、评价和权重保存。
- 使用 MLflow 记录配置、随机种子、数据版本、指标、图表和模型哈希。
- 验收：20/50/100 循环均产生 MAE、RMSE、MAPE、R²；所有模型使用完全相同的电芯划分。

### 阶段 4：核心混合退化模型（7 月 30 日—8 月 5 日）

```text
SOH(t) = 1 - Dtrend(t; θ, condition)
           - λ · cumulative_softplus(residual(x, t, condition))
```

- 趋势项包含平方根衰减、线性衰减和可学习拐点加速项。
- 参数通过正值变换确保退化方向正确。
- 残差采用累计非负退化增量，从结构上保证长期单调，而非仅依赖惩罚项。
- 加入平滑、边界、残差幅度及个体参数正则。
- RUL 由预测 SOH 首次跨越 EOL80 得到，不独立虚构。
- 完成无单调结构、无残差、无工况编码、无拐点项的消融实验。
- 验收：预测轨迹无非法长期恢复；固定种子可复现；与 XGBoost、CPMLP 和纯趋势模型形成完整对比。

### 阶段 5：可信区间、域适应和在线校正（8 月 6 日—8 月 11 日）

- 实现 Split Conformal 基线和 Normalized Conformal 主方法。
- 按预测跨度和模型难度缩放区间；主报告使用 90% 目标覆盖率，附 80% 和 95% 敏感性结果。
- 分别报告同分布、协议外、跨数据集和目标域重校准后的 PICP 与 MPIW。
- 实现 CORAL 特征对齐、CPMLP-DANN 域对抗和少样本目标域微调对照。
- MATR 训练、HUST 零样本测试，再用独立目标域校准电芯执行适配和 Conformal 重校准。
- 在线更新只优化电芯个体偏置、退化速率和拐点参数，全局模型保持冻结。
- 验收：20→50→100→150 循环更新后误差和区间变化均有轨迹；跨域覆盖率不得沿用源域声明。

### 阶段 6：主动试验与物理核验（8 月 12 日—8 月 16 日）

- Naumann 输入为温度、平均 SOC、DOD、充电倍率和放电倍率。
- GP 输出容量损失率、等效循环退化率、阻抗增长率和混合模型工况参数。
- 使用 Matérn 5/2 + WhiteKernel，输出预测均值和方差。
- 采集函数采用“预期整体方差下降 ÷ 标准化时间与设备成本”，并施加安全边界、重复试验惩罚和设备能力约束。
- 通过留一回放比较随机、均匀网格、最大方差和成本约束信息增益。
- 锁定发布版 PyBaMM，使用 SPMe + Prada2013 + Experiment，输出短期电压、电流、容量、SOC、终止原因和边界风险。
- PyBaMM 结果强制携带“参数未针对目标电芯标定，仅供短期趋势与边界核验”警告。
- 验收：主动试验策略具有可复现回放曲线；PyBaMM 接口不存在长期 RUL 字段。

### 阶段 7：多智能体、MCP 与知识库（8 月 17 日—8 月 22 日）

- 使用 LangGraph 实现带 PostgreSQL 检查点的受约束状态图。
- 数据质检、寿命预测、物理核验、试验决策 Agent 分别配置工具白名单。
- 监督器检查数值来源、预测区间、跨域状态、PyBaMM 边界、许可证和人工复核条件。
- 支持多轮任务规划、工具失败后的重新规划、人工确认节点和会话恢复。
- LLM 层使用可替换的 OpenAI 兼容 Provider；支持外部 API、Ollama 本地模型和无 LLM 模板模式。
- 建立 LFP 论文、数据卡、模型卡、PyBaMM 假设和工业接口规范知识库。
- 使用 pgvector + PostgreSQL 全文检索 + 中文 BM25 + reranker 混合检索，解释保留文档和页码。
- MCP 支持 stdio 和 Streamable HTTP，并暴露全部强类型工具。
- 验收：Agent 不能越权调用工具；无来源数字无法通过监督器；外部 LLM 不可用时数值流程和模板报告仍可运行。

### 阶段 8：后端、驾驶舱与工业集成（8 月 17 日—8 月 24 日）

FastAPI 提供数据接入、质检、寿命与轨迹预测、在线更新、物理核验、试验推荐、批次决策、Agent 运行、任务状态、报告和审计接口。

Next.js 驾驶舱包含：产业总览与批次健康地图；数据上传与 Schema 映射；Observed/Predicted/Newly Observed/Simulated 曲线；SOH/RUL/区间/拐点/模型对比；在线校正；入组/复检/降级/拒绝；Naumann 热图与试验推荐；PyBaMM 核验；Agent 计划、工具调用和人工审批；数值证据审计；JSON/Markdown/PDF/Word 导出。

Streamlit 保留同等数据与算法能力，定位为科研实验工作台，不重复后端逻辑。

工业集成包括 REST/MQTT/Modbus BMS 模拟、EMS 决策接口，以及飞书机器人告警、多维表格同步、报告文档、复检任务和审批卡片。无外部凭证时采用沙箱和协议模拟，不宣称生产部署。

### 阶段 9：系统实验与竞赛材料（8 月 25 日—8 月 31 日）

- 冻结数据、划分、特征、模型、Conformal、知识库和依赖版本。
- 运行 5 个随机种子的完整实验矩阵，报告均值、标准差及失败实验。
- 完成算法消融、域适应、区间校准、在线更新、主动试验、Agent 可靠性和端到端性能实验。
- Agent 实验比较无 Agent、单 Agent、受约束多 Agent，指标包括任务完成率、工具选择准确率、越权率、数值篡改率、证据覆盖率和时延。
- 编制 300 字作品简介、匿名项目文档、技术附录、系统架构图、实验表、模型卡、数据卡、许可证清单及符合官方要求的视频。
- 8 月 30 日在干净 WSL2/Docker 环境断网复现；8 月 31 日上传并回下载核验。

## 四、测试、验收与质量门禁

```text
pytest -q
pytest tests/leakage -q
pytest tests/integration -q
pytest tests/e2e -q
ruff check .
mypy src/quanxin_life
npm --prefix frontend run lint
npm --prefix frontend run typecheck
npm --prefix frontend run test
docker compose config
pip-audit
```

必须覆盖：电芯跨集合、截断后信息泄漏、测试集参与拟合/调参/校准、未知序列化产物、跨域未重校准、PyBaMM 长期字段、无来源数字、低质量数据决策、LLM 数值篡改/越权、外部依赖失败恢复，以及 CSV 到审计报告的浏览器端到端流程。

## 五、最终完成定义

- MATR、HUST、Naumann 主数据适配成功并可追溯。
- 数据划分和特征工程无已知泄漏。
- Dummy、Variance、XGBoost、CPMLP 与混合模型有真实实验结果。
- Conformal 报告实际覆盖率与区间宽度；跨数据集适配与目标域重校准完成。
- 在线个体参数校正有回放实验；主动试验推荐与随机基线诚实对比。
- PyBaMM 短期核验边界清晰；批次决策支持入组、复检、降级和拒绝。
- 四专业 Agent 与监督器可恢复、可审计；MCP、BMS/EMS、知识库和飞书适配完整可运行。
- Next.js、Streamlit、FastAPI 和异步任务端到端贯通。
- 每个正式数字均可追溯；Docker 干净环境可离线启动和复现核心 Demo。
- 初赛材料匿名、无夸大表述、许可证完整。

## 六、Codex 多代理执行规则

- 主控 Codex 担任首席研究工程师和集成负责人，独占公共 Schema、数据划分、`ToolResult`、数据库模型和 Docker Compose。
- 六类短生命周期子代理分别负责数据、算法、物理与试验、智能体可信工具、后端集成、前端可视化。
- 每项任务采用“失败测试 → 最小实现 → 目标测试 → 全量质量门禁 → 独立规格审查 → 独立质量审查 → 提交”的 TDD 流程。
- 使用独立 Git worktree，分支统一以 `codex/` 开头。
- 正式运行基线为 WSL2 + Docker；Windows 仅作为 Codex 宿主和浏览器演示环境。
- 正式产品端为 Next.js；Streamlit 为科研工作台；两者共享 FastAPI 和领域工具。
- 飞书真实租户、企业 BMS/EMS 和工业大容量电芯数据属于外部条件；无凭证或数据时以沙箱和协议模拟验证，不宣称真实生产部署。
- “最高配置”指所有模块均纳入实现和验收，不意味着虚构工业验证、保证模型优于所有基线或宣称精确预测 25 年。
