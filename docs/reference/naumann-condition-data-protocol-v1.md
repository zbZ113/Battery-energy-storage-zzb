# Naumann 工况级退化数据协议 v1

## 1. 目的与边界

本协议接入 Naumann 公开的 LFP/石墨电芯循环老化与日历老化数据，服务于：

- 工况—退化关系建模；
- Naumann 高斯过程（GP）与主动试验回放；
- 温度、平均 SOC、DOD、充放电倍率的工况敏感性分析；
- 储能静置工况的日历老化参考。

这些数据是**公开实验室研究数据**，不等价于海辰储能、济南企业或其他工业现场的储能电芯数据。它们也不是用于 `CycleRecord`、早期循环 `EOL80` 预测或 CPMLP 训练的电芯级寿命样本。任何适配、图表或报告都不得将条件级汇总测量描述为逐采样点电化学曲线、企业实测寿命或 15–25 年工业验证结果。

## 2. 已核验的本地数据快照

截至 2026-07-16，本地快照共 25 个文件、131,126 bytes：

| 数据集 | 本地文件 | 字节数 | 已核验接入状态 |
| --- | ---: | ---: | --- |
| `NAUMANN_CALENDAR` | 4 个 `.xlsx` | 67,487 | `DischargeCapacity.xlsx` 已按审核布局真实适配；其余 3 个工作簿未接入 |
| `NAUMANN_CYCLE` | 21 个 MATLAB v5 `.mat` | 63,639 | `xDOD_1C1C_40°C_Capacity_CC_CV_FEC.mat` 已按审核布局真实适配；其余 20 个文件未接入 |

来源登记位于 `configs/data_sources.json`。原始数据目录不进入 Git；正式运行仍必须为每个文件提供与来源 URI、许可证和 SHA-256 一致的 `RawFileManifest`。

### 2.1 Calendar 容量实测适配

已核验文件：

```text
data/NAUMANN_CALENDAR/DischargeCapacity.xlsx
```

审核布局：

```text
configs/data_layouts/naumann_calendar_capacity_v1.json
```

真实适配结果：

- 文件 SHA-256：`c684bdb2adc7314d7cf943b472068694e20425af8f84fbf249d5d7554dc7e3f0`；
- 17 个温度/SOC 条件；
- 35 个存储时间点；
- 595 条 `CalendarCapacityObservation`；
- 布局版本：`naumann-calendar-capacity-2021-v1`；
- 适配器版本：`naumann-calendar-condition-v1.0.0`。

上述数量只证明解析结果与审核布局一致，不是模型性能、主动试验收益或工业有效性指标。

### 2.2 Cycle xDOD 容量比实测适配

已核验文件：

```text
data/NAUMANN_CYCLE/xDOD_1C1C_40°C_Capacity_CC_CV_FEC.mat
```

审核布局：

```text
configs/data_layouts/naumann_cycle_xdod_capacity_fec_v1.json
```

真实适配结果：

- 文件 SHA-256：`2b3f640783d72dec17ec57d438d570e891057ecdf5690d444688dd1e1b2209da`；
- 7 个已审核 DOD 条件；
- 每个条件 35 个等效满循环观测点；
- 245 条 `CycleMatrixObservation`；
- 指标名称：`relative_capacity_ratio`；
- 布局版本：`naumann-cycle-xdod-capacity-fec-2021-v1`；
- 适配器版本：`naumann-cycle-mat-condition-v1.0.0`。

`relative_capacity_ratio` 是无量纲相对容量比，不是 Ah。不得把源矩阵的约 0–1 数值改名为绝对容量、SOH、EOL80 或 RUL。

## 3. Calendar 容量工作簿契约

`load_naumann_calendar_capacity` 只接受：

1. 已验证 SHA-256、来源 URI、许可证的 `RawFileManifest`；
2. 与该来源完全匹配的 `SourceCatalogEntry`；
3. 人工审阅并版本化的 `NaumannCalendarLayout`。

布局必须显式声明工作表、标题单元格、时间列、首个观测行，以及每个条件列的精确标题、条件 ID、温度和平均 SOC。适配器不会从类似 `TP_40°C,50%SOC` 的文本自动推断物理条件。

时间必须严格递增。任一条件值缺失、非有限、负容量、标题不符、来源不符或哈希不符都会阻断读取，不得插值、补零、重排行或静默删除。

Calendar 数据天然没有循环 DOD、充电倍率或放电倍率，这些字段保持显式 `null`。若下游要把 Calendar 观测放入五维循环工况 GP，必须逐条件提供有证据的科学用途映射；没有证据时不得猜测。

另外三个 Calendar 工作簿当前边界如下：

- `DifferentialVoltageAnalysis.xlsx`：无审核布局，不接入；
- `ElectrochemicalImpedanceSpectroscopy.xlsx`：无审核布局，不接入；
- `ResistanceR_DC10s.xlsx`：无审核布局，不接入。

## 4. Cycle 条件矩阵契约

`load_naumann_cycle_matrix` 只读取布局中明确声明的：

- `X_Axis_Data_Mat`；
- `Y_Axis_Data_Mat`；
- `Legend_Vec`；
- 观测轴类型、指标类型和每列工况。

布局列必须覆盖矩阵的每一列，图例必须逐字符匹配；观测轴必须严格递增；数值必须有限且非负。适配器不从文件名或图例自动推断温度、SOC、DOD 或倍率，也不自动裁剪 NaN 尾部或合并重复轴值。

当前只有 xDOD 容量比文件具有已提交、已真实验证的 Cycle 布局。其余文件边界如下：

- 4 个 EIS MAT 文件包含结构体，当前矩阵适配器不读取；
- 5 个 dV/dQ MAT 文件使用 `Q1`、`Q2`、`Qactual`，当前矩阵适配器不读取；
- 其余容量/电阻汇总矩阵尚无审核布局，不进入正式流水线；
- `xCyC_80DOD_40°C_*` 存在重复等效循环轴值，现有严格递增校验会阻断；
- `xDOD_1C1C_x°C_*` 存在非有限尾部和重复轴值，现有校验会阻断；
- Loadcollectives 图例未完整给出 DOD 和倍率，缺少证据时不得补造；
- 归一化容量或归一化电阻必须使用准确的无量纲指标契约，不得伪装为 Ah 或 Ω。

## 5. 固定观测轴选择

条件间比较若要求“同一等效满循环位置”，必须使用版本化的 `ReviewedAxisSelection`，显式声明：

- `selection_version`；
- `target_axis_value`；
- `max_absolute_deviation`；
- 允许参与比较的 `condition_ids`。

选择器只从每个条件的直接源观测中选择距离目标轴最近且处于容差内的一点，并记录绝对偏差。它不插值、不外推、不平均、不填补缺失值。若最近点并列、条件缺失、超出容差或混入不同来源/指标上下文，必须拒绝。

因此，“固定轴”不能由界面、LLM 或实验脚本临时挑选；目标轴、容差和条件集合必须先审核、版本化，再用于回放。

## 6. 无成本主动试验回放

Naumann 原始文件不提供实验预约时长和设备成本。缺少经过审核的资源信息时，系统不得填入默认时长、默认成本或占位数值。

当目标变换和工况映射已经审核，但资源字段为 `null` 时，流水线允许运行三种**不含成本**的有限候选池回放：

1. `random`：随机基线；
2. `uniform_grid`：均匀网格基线；
3. `max_variance`：最大预测方差策略。

此时必须：

- 将状态标记为 `completed_without_costs`；
- 写入警告 `COST_AWARE_EIVR_DISABLED_MISSING_REVIEWED_RESOURCES`；
- 把 `cost_aware_eivr` 列入不可用策略；
- 在数据制品中保持 `duration_hours` 和 `equipment_cost` 为 `null`。

只有同时提供有证据的 `duration_hours`、`equipment_cost` 以及对应归一化配置时，才允许启用成本约束 EIVR。不能用无成本回放推断“成本更低”或“设备利用率提升”。

若连目标变换或必要工况映射都未审核，即 `mapping` 整体缺失，流水线只能输出 `degraded_missing_reviewed_mapping` 清单，不生成 GP 回放结果。

## 7. 可追溯与对外表述

- 原始目录 `data/NAUMANN_CYCLE/` 和 `data/NAUMANN_CALENDAR/` 不提交 Git；
- 数据卡、来源登记、审核布局、SHA-256、转换配置和不可执行结果清单进入版本控制；
- `.mat` 与 `.xlsx` 只在来源和 SHA-256 核验成功后读取；
- 每条下游观测保留源文件、源 SHA-256、布局版本、适配器版本、桥接版本和映射版本；
- 对外只可表述为“公开实验室 LFP/石墨研究数据上的方法验证”；
- 不得声称为海辰储能真实数据、企业现场验证、质保依据或安全控制依据；
- 文件数、观测条数和适配成功不等于模型准确率或业务收益；
- 未完成真实回放前，不报告 GP 优于基线、误差下降或成本收益。

## 8. 当前可复现入口

原始适配入口：

```python
from quanxin_life.data.adapters.naumann_calendar import (
    load_naumann_calendar_capacity,
    load_naumann_calendar_layout,
)
from quanxin_life.data.adapters.naumann_cycle_mat import (
    load_naumann_cycle_layout,
    load_naumann_cycle_matrix,
    select_reviewed_axis_observations,
)
```

下游 GP 回放入口：

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\naumann_gp_pipeline.py `
  --request <审核后的请求.json> `
  --output-dir <空输出目录>
```

该 CLI 消费已经适配并带来源的严格 JSON 请求，不直接猜测原始 MAT/XLSX 的结构。实际实验还需要显式的初始观测队列、候选池边界、查询预算、目标变换和固定轴选择。

真实 xDOD 固定轴回放入口：

```powershell
.\.venv\Scripts\python.exe scripts\run_reviewed_naumann_replay.py `
  --repository-root . `
  --config configs\experiments\naumann_xdod_10000fec_baselines_v1.json `
  --output-dir artifacts\naumann_xdod_10000fec_run_v1
```

该配置从 7 个已审核工况中排除与纯 CC 工况具有相同五维向量、但充电协议不同的 `CC+CV` 条件，保留 6 个纯 CC 工况；在 10,000 FEC 目标附近、100 FEC 容差内各选一个直接观测点，不进行插值。当前真实运行状态为 `completed_without_costs`，仅包含三种无成本策略，不能据此宣称任何策略普遍更优。

本次受审查运行的精简证据记录保存于 `reports/experiments/naumann_xdod_10000fec_baselines_v1/evidence_record.json`。记录同时保存原始 `run_manifest.json` 的 SHA-256、源文件哈希、配置哈希、运行 ID、产物哈希与降级警告；完整本地中间产物仍位于 `artifacts/`，提交该记录不代表完成工业验证。

## 9. 后续接入顺序

已完成：25 文件哈希清单、xDOD 布局、10,000 FEC 固定轴选择配置，以及 6 个直接观测点的三策略真实回放。

后续顺序：

1. 对固定轴选择和仅 6 个工况的小样本限制进行科研复核；
2. 增加更多具有完整工况证据且不重复的条件矩阵；
3. 取得可核验资源成本后，再单独启用成本约束 EIVR；
4. 逐文件审核其他容量、电阻、EIS 和 dV/dQ 布局，不批量猜测。
