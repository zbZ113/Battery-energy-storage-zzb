# 早期循环特征协议 v1

`early-cycle-v1` 是“泉芯智寿”阶段 3 的基础特征协议。它只处理已经由数据适配器
标准化为 `CycleRecord` 的单电芯记录；所有字段单位、来源与语义以数据契约和适配器
登记结果为准。

## 使用边界

- 支持的预测截断点固定为第 20、50、100、150 个 canonical cycle index。
- 提取器收到任何 `cycle_index > cutoff_cycle` 的输入都会直接失败；不会静默丢弃，
  以便调用方发现未来数据泄漏。
- 输入必须只来自一个 `dataset_id` 与一个 `cell_id`。结构性数据错误、重复样本与
  非单调时间会阻断提取；`valid=false` 记录会被排除并记入警告。
- 本协议不按电流正负号猜测充放电阶段。容量相关计算仅使用适配器已明确标注语义的
  `discharge_capacity_ah` 与 `charge_capacity_ah` 字段。

## 特征组

| 组别 | 主要字段 | 不可用时的策略 |
| --- | --- | --- |
| 容量 | 首末放电容量、变化量、相对变化、每周期斜率 | 保持 `null`，记录 `DISCHARGE_CAPACITY_UNAVAILABLE` |
| 库仑效率 | 可配对充/放电容量的均值、末值、标准差 | 保持 `null`，记录 `COULOMBIC_EFFICIENCY_UNAVAILABLE` |
| 时间 | 完整循环首末持续时间及变化量 | 保持 `null`，记录 `CYCLE_DURATION_UNAVAILABLE` |
| 温度 | 样本均值、最大值、标准差、缺失比例 | 温度统计保持 `null`，记录 `TEMPERATURE_UNAVAILABLE` |
| 内阻 | 首末内阻、变化量、每周期斜率 | 保持 `null`，记录 `INTERNAL_RESISTANCE_UNAVAILABLE` |
| ΔQ(V) | 均值、方差、最小/最大值、L2 幅度 | 保持 `null`，记录 `DELTA_Q_UNAVAILABLE` |

## ΔQ(V) 计算

1. 对每个候选循环使用 `(voltage_v, discharge_capacity_ah)`；按电压升序排列，并把
   重复电压的容量以确定性均值合并。
2. 优先使用至少两条诊断循环；不足时才使用非诊断循环，并在结果中记入
   `DELTA_Q_NON_DIAGNOSTIC_FALLBACK`。
3. 两条曲线只在共同电压区间内，以固定 0.01 V 网格作 PCHIP 插值。
   网格不会跨出共同边界，也不允许外推。
4. 任何曲线、交集或插值值不足都返回不可用，而不是填入 0 或未来周期数据。

输出 `EarlyCycleFeatureSet` 始终保留 `feature_version`、完整有效 source cycle 集与
实际用于 ΔQ 的两个 cycle。该结果不是模型预测，不应直接作为正式报告中的业务数值；
后续模型工具会将其版本和输入哈希纳入 `ToolResult` 来源链。

## 独立方差基线特征

`delta-q-variance-v1` 由 `features.variance` 提供，是后续 Variance 回归基线的独立
输入原语。调用方必须显式传入 `anchor_cycle` 和 `comparison_cycle`，两者都不得超过
截断点；模块不会依据数组下标或文件顺序猜测比较周期。输出保留两个 cycle、固定网格
有效点数及不可用原因，供模型层在训练前决定排除样本或执行已登记的缺失策略。
