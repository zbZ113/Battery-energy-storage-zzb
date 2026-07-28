# Advanced Benchmark

本页报告 MATR 三批 Advanced Final 的正式、冻结结果。所有指标来自已完成
SHA-256 验收的结果包、逐样本预测导出、metrics closure、Conformal 和模型晋级制品。
结果只适用于声明的数据、划分、目标和评估协议。

## 实验口径

| 项目 | 口径 |
| --- | --- |
| 数据 | MATR 2017-05-12、2017-06-30、2018-04-12 三批 |
| 总电芯 | 140 |
| RUL 标签 | 138 个 MATR 官方 cycle-life |
| 划分 | train 82 / validation 19 / calibration 12 / test 27 |
| SOH test | 25 个满足有限轨迹条件的测试电芯 |
| cutoff | 20 / 50 / 100 / 150 cycles |
| 随机种子 | 38 / 39 / 40 / 41 / 42 |
| 正式运行 | 4 模型 × 4 cutoff × 5 seeds = 80 |
| RUL 目标 | MATR 官方 cycle life，不是统一 EOL80 |
| SOH 目标 | 真实轨迹，最大监督/输出边界 cycle 500 |

表中 RUL 是五种子逐电芯聚合指标；SOH 是五种子逐轨迹点聚合指标。

## RUL 五种子结果

单位：MAE、RMSE 为 cycles；MAPE、15%-Acc 为百分比；R² 无量纲。

| 模型 | cutoff | MAE | RMSE | MAPE | R² | 15%-Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CyclePatch-BatLiNet | 20 | 122.58 | 157.94 | 15.98% | 0.7781 | 55.56% |
| CyclePatch-BatLiNet | 50 | 114.26 | 141.36 | 15.56% | 0.8224 | 57.78% |
| CyclePatch-BatLiNet | 100 | 103.00 | 129.56 | 14.07% | 0.8513 | 58.52% |
| CyclePatch-BatLiNet | 150 | 101.92 | 128.92 | 12.76% | 0.8507 | 67.41% |
| CyclePatch Direct | 20 | 116.05 | 143.36 | 16.54% | 0.8167 | 58.52% |
| CyclePatch Direct | 50 | 111.44 | 137.54 | 15.51% | 0.8308 | 60.00% |
| CyclePatch Direct | 100 | 104.52 | 130.38 | 13.86% | 0.8490 | 58.52% |
| CyclePatch Direct | 150 | **98.19** | **122.49** | **12.62%** | **0.8656** | 64.44% |

`CyclePatch Direct + cutoff 150` 获得最低平均 MAE、RMSE 和 MAPE，但：

- BatLiNet-150 的 15%-Acc 为 67.41%，高于 Direct-150；
- Direct 与 BatLiNet 的逐电芯 MAE 配对 Bootstrap 区间跨过零；
- 因此当前证据不支持“Direct 统计显著全面优于 BatLiNet”。

## SOH 五种子结果

MAE、RMSE 使用 SOH 比例表示；括号内换算为 SOH 百分点。

| 模型 | cutoff | MAE | RMSE | 单调违规率 |
| --- | ---: | ---: | ---: | ---: |
| Current Hybrid | 20 | 0.01610（1.610） | 0.02578（2.578） | 0% |
| Current Hybrid | 50 | 0.01618（1.618） | 0.02622（2.622） | 0% |
| Current Hybrid | 100 | 0.01661（1.661） | 0.02710（2.710） | 0% |
| Current Hybrid | 150 | 0.01703（1.703） | 0.02789（2.789） | 0% |
| HybridPatch-v2 | 20 | **0.01281（1.281）** | 0.02700（2.700） | 0% |
| HybridPatch-v2 | 50 | 0.01287（1.287） | 0.02753（2.753） | 0% |
| HybridPatch-v2 | 100 | 0.01338（1.338） | 0.02826（2.826） | 0% |
| HybridPatch-v2 | 150 | 0.01380（1.380） | 0.02889（2.889） | 0% |

解释：

- HybridPatch-v2 在四个 cutoff 的平均 MAE 均较低；
- Current Hybrid 在四个 cutoff 的 RMSE 和尾部电芯误差均较低；
- 两者单调违规率均为 0；
- 远期 201–350 cycles 误差增大，较高误差集中在 2017-06-30 批次；
- 当前采用双路由，不将 HybridPatch-v2 宣称为唯一 SOH 冠军。

## 90% Split Conformal

校准集 12 个电芯，测试集 27 个电芯。PICP 是经验覆盖率，MPIW 单位为 cycles。

| 模型 | cutoff | 90% PICP | MPIW |
| --- | ---: | ---: | ---: |
| CyclePatch-BatLiNet | 20 | 85.19% | 404.52 |
| CyclePatch-BatLiNet | 50 | 92.59% | 429.29 |
| CyclePatch-BatLiNet | 100 | 88.89% | 400.74 |
| CyclePatch-BatLiNet | 150 | 96.30% | 357.93 |
| CyclePatch Direct | 20 | 92.59% | 540.89 |
| CyclePatch Direct | 50 | 81.48% | 356.07 |
| CyclePatch Direct | 100 | 88.89% | 397.12 |
| CyclePatch Direct | 150 | 85.19% | 341.41 |

边界：

- 所有结果带 `SMALL_CALIBRATION_COHORT`；
- 12 个 calibration 电芯使 90% 与 95% 在部分条件下使用同一最大秩；
- `cutoff=100` 的两个候选都未达到目标 90% PICP；
- Normalized Conformal 未稳定改善覆盖—宽度权衡，不能自动替代 Split 基线；
- 覆盖结果只适用于声明的 MATR calibration/test 划分。

## 条件路由

所有路由均为 `CONDITIONAL`，需要人工审批，不等于在线激活。

| cutoff | RUL 点精度 | RUL coverage | SOH 平均精度 | SOH 尾部/效率 |
| ---: | --- | --- | --- | --- |
| 20 | Direct | Direct | HybridPatch-v2 | Current Hybrid |
| 50 | Direct | BatLiNet | HybridPatch-v2 | Current Hybrid |
| 100 | BatLiNet | Direct（未达 90% 目标） | HybridPatch-v2 | Current Hybrid |
| 150 | Direct | BatLiNet | HybridPatch-v2 | Current Hybrid |

## 结果完整性

```text
Advanced Final operations: 80
indexed files: 2654
safetensors files: 897
RUL test predictions: 1080 rows
SOH trajectory predictions: 500000 rows
RUL calibration predictions: 480 rows
source commit: 232d9fc8957bb547ca2b80205802f377864e982b
output SHA-256: d201224870a28c655f66a810bc94f90ad28133e06f2fb4a7285195274c303d82
```

本地 CPU 使用 safetensors 复算后，与 A100 聚合指标的最大差异为：

- RUL：0.015913 cycles；
- SOH：`7.05 × 10^-8`。

该差异属于 CPU/CUDA 浮点执行差异。完整复现和来源差异见
[可复现性](reproducibility.md)，研究边界见 [已知限制](limitations.md)。

## 公开图件与 Source Data

README 使用的三张轻量 PNG 和四份聚合 source data 是从已验证 Nature 图包逐字节
复制的公开子集，没有重新计算或手工改写指标：

- [RUL cutoff 性能](assets/benchmark/advanced-final-20260723/Figure_2_rul_cutoff_performance.png)；
- [SOH 代表轨迹](assets/benchmark/advanced-final-20260723/Figure_4_soh_trajectory_examples.png)；
- [模型综合权衡](assets/benchmark/advanced-final-20260723/Figure_6_model_tradeoff.png)；
- [公开证据 manifest](assets/benchmark/advanced-final-20260723/manifest.json)；
- [聚合 source data](source-data/advanced-final-20260723/)。

Figure 4 的五种子 min–max 阴影不是 Conformal prediction interval；Figure 6 的耗时和
显存来自训练过程，不是在线单样本推理基准。算法解释见
[算法原理](algorithms/README.md)。
