# 第三方声明登记表

本文件是发布审计入口，不替代第三方许可证原文。依赖的准确版本由锁文件和发布时生成的软件物料清单确定。

## Python 组件类别

| 类别 | 计划组件 | 用途 | 发布前动作 |
| --- | --- | --- | --- |
| 核心契约 | Pydantic | 数据校验与序列化 | 核对锁定版本及许可证文本 |
| 数据科学 | NumPy、pandas、SciPy、PyArrow | 数据处理与特征计算 | 生成依赖清单并保留声明 |
| 模型 | scikit-learn、PyTorch、XGBoost | 基线与学习模型 | 审查库与模型权重的独立许可 |
| 物理 | PyBaMM | 短时物理一致性参考 | 核对代码、参数集和引用要求 |
| 服务与 Agent | FastAPI、Uvicorn、LangGraph、LangChain Core | API 与编排 | 核对传递依赖及服务条款 |
| 分析界面 | Streamlit、Plotly | 研究工作台 | 核对静态资产与再分发声明 |

## 数据集登记

MATR、HUST、Naumann 数据只在对应数据卡记录的授权范围内使用。每个快照必须登记正式名称、发布方、原始链接、引用、条款版本、获取日期、是否允许再分发及文件哈希。仓库不直接包含这些原始数据。

- LFP 280Ah DoD 数据（Zenodo record `14576042`，CC BY 4.0）：仅用于 280Ah 方形 LFP 电芯在公开观测循环范围内的外部尺度参考核验。原始 ZIP 不进入 Git；来源 URI、许可证、文件大小、SHA-256、审核 layout 和被排除的缺温度 CSV 均由版本化 manifest 记录。该数据不构成 15～25 年自然寿命验证，也不代表海辰产品。

## 概念参考代码

- BLAST-Lite（NREL / Alliance for Energy Innovation, LLC，BSD-3-Clause）：仓库在 `src/quanxin_life/_vendor/blast_lite/` 受控保留上游提交 `b093495b47dc40dd96dba865d91f553619501e94` 的两个 LFP 模型及最小运行依赖，并保留 `LICENSE`、`NOTICE`、`UPSTREAM.json`、源文件 SHA-256 和本项目兼容性修改说明。其 `Lfp_Gr_250AhPrismatic` 仅作为大型方形 LFP 参考模型，不得描述为海辰 280Ah 专属模型或产品寿命承诺。
- BatteryML（Microsoft，MIT License）：MATR/HUST 原始字段语义和电芯—周期分层结构的概念参考。目标仓库的数据适配、校验、划分和安全持久化均为独立实现，不复制其 pickle 持久化、Severson 特征或划分逻辑。许可证原文位于本地审计工程 `BatteryML-main/BatteryML-main/LICENSE`，正式发布时随 SBOM 固化准确版本与版权声明。
- BatteryLife（Ruifeng Tan，MIT License）：仅作为 CPMLP 命名、逐循环曲线编码与跨循环聚合思想的对照参考。当前 `src/quanxin_life/models/cpmlp.py` 为 clean-room 独立实现，采用不同的强类型曲线契约、显式缺失掩码、掩码池化和 EOL80 下界约束，未复制 `BatteryLife/models/CPMLP.py` 源码或权重。许可证原文位于本地审计工程 `BatteryLife-main/BatteryLife-main/LICENSE`；若未来改为复用其源码，必须另行保留原版权声明与修改说明。

## 待发布清单

正式发行前，发布负责人必须用锁文件替换上述“计划组件”为准确版本，附上全部要求的许可证文本和版权声明，并登记前端 npm 包、容器基础镜像、字体/图标、模型权重、MCP 服务与知识库资料。任何未知来源项均阻断发布。
