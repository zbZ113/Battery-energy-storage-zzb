# Naumann 公开老化数据卡 v1

## 数据身份

| 字段 | 内容 |
| --- | --- |
| 数据卡版本 | `naumann-public-data-card-v1` |
| 核验日期 | 2026-07-16 |
| 电芯体系 | 公开研究中的 LFP/石墨电芯 |
| Cycle 来源 | Mendeley `6hgyr25h8d/1`，CC BY 4.0 |
| Calendar 来源 | Mendeley `kxh42bfgtj/1`，CC BY 4.0 |
| 本地文件总数 | 25 |
| 本地总大小 | 131,126 bytes |

本数据卡记录仓库本地已核验快照和允许用途。它不证明数据来自海辰储能或其他企业，也不授予超出原始许可证的权利。

## 本地组成

### NAUMANN_CALENDAR

- 4 个 `.xlsx`，共 67,487 bytes；
- 文件包括容量、直流内阻、EIS 和微分电压分析；
- 当前仅 `DischargeCapacity.xlsx` 有审核布局并真实适配成功。

容量文件核验记录：

| 字段 | 值 |
| --- | --- |
| 相对路径 | `data/NAUMANN_CALENDAR/DischargeCapacity.xlsx` |
| SHA-256 | `c684bdb2adc7314d7cf943b472068694e20425af8f84fbf249d5d7554dc7e3f0` |
| 布局 | `configs/data_layouts/naumann_calendar_capacity_v1.json` |
| 条件数 | 17 |
| 时间点数 | 35 |
| 输出观测数 | 595 |
| 输出类型 | `CalendarCapacityObservation` |

### NAUMANN_CYCLE

- 21 个 MATLAB v5 `.mat`，共 63,639 bytes；
- 包含容量、电阻、EIS、dV/dQ 和汇总条件矩阵；
- 当前仅 xDOD、40°C、50% SOC、1C/1C 的相对容量比 FEC 矩阵有审核布局并真实适配成功。

xDOD 文件核验记录：

| 字段 | 值 |
| --- | --- |
| 相对路径 | `data/NAUMANN_CYCLE/xDOD_1C1C_40°C_Capacity_CC_CV_FEC.mat` |
| SHA-256 | `2b3f640783d72dec17ec57d438d570e891057ecdf5690d444688dd1e1b2209da` |
| 布局 | `configs/data_layouts/naumann_cycle_xdod_capacity_fec_v1.json` |
| 条件数 | 7 |
| 每条件观测数 | 35 |
| 输出观测数 | 245 |
| 观测轴 | `equivalent_full_cycles` |
| 指标 | `relative_capacity_ratio`，无量纲 |
| 输出类型 | `CycleMatrixObservation` |

## 允许用途

- 公开实验室数据上的工况敏感性分析；
- GP 和主动试验方法回放；
- 不同审核策略的离线比较；
- 数据适配、来源追踪和审计机制验证；
- Calendar 静置容量变化参考。

## 禁止或尚未支持的用途

- 不作为海辰储能、济南企业或其他工业现场真实数据；
- 不作为企业质保、安全控制或电站调度依据；
- 不把相对容量比解释为 Ah、SOH、EOL80 或 RUL；
- 不用于训练 CPMLP 等要求电芯级早期循环轨迹和寿命标签的模型；
- 不从文件名、图例或缺失列猜测工况；
- 不用 PyBaMM 或 LLM 填补缺失测量；
- 不对无审核布局的 20 个 Cycle MAT 和 3 个 Calendar 工作簿声称已接入；
- 不把当前 6 个固定轴观测点的小样本回放描述为策略普遍优势或业务收益。

## 数据质量与已知限制

- `xCyC_80DOD_40°C_*` 含重复等效循环轴值，严格适配器会拒绝；
- `xDOD_1C1C_x°C_*` 含非有限尾部及重复轴值，严格适配器会拒绝；
- Loadcollectives 图例缺少完整 DOD/C-rate 证据；
- EIS MAT 使用结构体，当前条件矩阵适配器不支持；
- dV/dQ MAT 使用独立变量结构，当前条件矩阵适配器不支持；
- Calendar 观测没有循环 DOD 和 C-rate，进入五维工况 GP 前必须有科学用途映射；
- xDOD 各条件的原始观测轴不完全相同；横向比较必须使用版本化固定轴选择，并保存目标轴、容差及实际偏差；
- 原始数据不包含实验室预约时长和设备成本。

## 主动试验资源边界

没有审核资源成本时，仅允许比较：

- `random`；
- `uniform_grid`；
- `max_variance`。

此状态必须标记为 `completed_without_costs`，同时禁用 `cost_aware_eivr`。只有取得有证据的试验时长、设备成本和归一化配置后，才允许运行成本约束 EIVR。

## 来源与复现要求

- 每次适配前核对文件 SHA-256、来源 URI、许可证和数据版本；
- 原始文件不进入 Git；
- 审核布局、数据卡、manifest 和不可执行实验清单进入版本控制；
- 适配结果必须保留源文件、源哈希、布局和适配器版本；
- GP 回放必须冻结初始观测队列、候选空间、固定轴选择、预算和随机种子；
- 任何缺失、歧义、哈希不符或越界条件必须显式失败或降级。

## 当前完成度声明

当前已经验证的是：

1. Calendar 容量文件可按审核布局产生 595 条真实条件观测；
2. Cycle xDOD 容量比文件可按审核布局产生 245 条真实条件观测；
3. 数值来源、布局和适配器版本可以保留。
4. 版本化配置可在 10,000 FEC 附近、100 FEC 容差内选择 6 个纯 CC 工况的直接观测点；
5. 真实公开数据已完成 `random`、`uniform_grid` 和 `max_variance` 三策略有限池回放；
6. 资源字段保持 `null`，`cost_aware_eivr` 被明确列为不可用。

当前尚未据此宣称：

- 任一 GP 或主动试验策略已取得可泛化的性能提升；
- EIVR 优于三种无成本策略；
- 公开数据模型已适用于工业储能电芯；
- 已验证任何 15–25 年寿命、成本节省或经营收益。
