# CPMLP-DANN 跨域适应协议 v1

## 目的与边界

本协议定义 `quanxin_life.adaptation.dann` 的 CPMLP-DANN 对照实验。它用于比较：

1. 仅使用 MATR 源域早期循环曲线与 `EOL80` 标签训练回归器；
2. 在不读取 HUST 目标域寿命标签的前提下，利用独立目标域曲线进行域对抗表征学习；
3. 在完全隔离的 HUST 测试电芯上进行后续推理与离线评估。

该组件是研究模型，不生成 `ToolResult`，不由 LLM 产生任何寿命数值，也不加载外部权重、pickle、joblib、`.pt` 或 `.pth` 制品。

## 电芯级分区约束

`SourceDomainBatch` 必须与源域 `SplitManifest.train` **完全一致**，并携带观测到的源域 `EOL80` 标签。

`TargetDomainBatch` 没有标签字段；其构造函数不接受 `eol80_labels`。适配时仅允许下列两个目标域队列：

| `TargetDomainCohort` | 标准 `SplitManifest` 对应字段 | 可用目的 |
| --- | --- | --- |
| `ADAPTATION` | `train` | 无标签域对抗适配 |
| `CALIBRATION` | `calibration` | 独立无标签适配／重校准前的表征适配 |

项目当前公共 `SplitManifest` 没有单列 `adaptation`，因此 `ADAPTATION` 明确映射为**目标域**的 `train` 分区；该分区只作无标签适配，绝不能被当作源域回归标签。`validation` 与 `test` 电芯不得进入 `fit`、归一化统计或域损失。目标域 `test` 仅可在模型训练结束后通过 `predict` 推理。

本组件采用更保守的全局身份规则：即使 `dataset_id` 不同，源／目标 `SplitManifest` 的任意分区也不能重用同一个裸 `cell_id`。这避免来源不清的同名电芯被静默混同。若同一物理电芯确实跨数据集出现，必须在上游完成可审计的身份重命名与来源登记，而不能直接进入 DANN。

## 曲线特征契约

`SourceDomainBatch` 与 `TargetDomainBatch` 都必须携带不可变的 `CurveFeatureContract`：

```text
feature_version
cutoff_cycle
cycle_indices
voltage_grid_v
```

契约要求：

- `cycle_indices` 为严格递增的整数，最后一个索引等于 `cutoff_cycle`；
- `voltage_grid_v` 全部有限且严格递增；
- 曲线张量的 cycle 和 voltage 轴长度分别与两套坐标完全匹配；
- 源、目标契约逐字段完全相同，且两者的 `feature_version` 必须精确匹配 `DANNConfig.feature_version`。

因此，两个张量即使尺寸相同，只要电压网格、循环网格、截断点或特征版本不同，也会在训练前被拒绝。

所有输入同时验证：

- 每个批次中 `cell_id` 唯一；
- 源／目标适配电芯互不重叠；
- 曲线张量为 `[cell, cycle, voltage]`；
- 掩码为同一 `[cell, cycle]` 轴的 `torch.bool`；
- 已观测曲线值有限，未观测整行保持 `NaN`；
- 源、目标域的 cycle／voltage 轴完全一致。

## 模型与优化

共享编码器采用轻量 CPMLP 风格结构：逐循环电压曲线 MLP 编码、掩码加权池化、观测覆盖率拼接和表征 MLP。表征分两路输出：

- 回归头：仅对源域样本计算标准化 `EOL80` 均方误差；
- 域分类头：对源、目标无标签样本计算二分类 BCE，并经 Gradient Reversal Layer（GRL）把反向梯度传回共享编码器。

总损失为：

```text
L = MSE(source regression)
    + domain_loss_weight × [BCE(source domain) + BCE(target domain)] / 2
```

GRL 前向恒等，反向梯度乘以 `-gradient_reversal_coefficient`。该实现使用 PyTorch autograd，不是只在日志中声明的伪域对抗。

## 归一化与可重复性

特征均值、特征标准差、标签均值和标签标准差只由源域训练电芯计算。目标适配及测试曲线只应用这些固定统计量，不能重新拟合。训练固定 `DANNConfig.random_seed`（默认 `20260712`），并启用 PyTorch 确定性算法；输出同时携带模型版本、源域、目标域和目标适配队列状态。

每个 `DANNPrediction` 和内部拟合状态还携带：

- `feature_version` 与完整 `CurveFeatureContract`；
- `source_split_version`、`target_split_version`；
- `random_seed`；
- 归一化方法版本 `source-train-zscore-v1`；
- `normalization_input_hash`：仅针对参与归一化拟合的源域训练曲线、掩码、标签、坐标契约和 cell ID 计算的 SHA-256。

该哈希不包含目标适配、目标校准或目标测试输入，因此可用于审计“源域统计只拟合一次”的边界。它不是 `ToolResult`；正式业务报告仍需在后续工具与审计层生成 `result_id`、模型版本和数据版本链。

## 解释限制

CPMLP-DANN 的输出为早期曲线条件下的模型数值，仅供离线研究对照。它不构成工业储能电芯寿命承诺、质保结论、长期物理仿真或现场 BMS/EMS 决策。跨域预测的覆盖率和批次决策必须由后续独立 Conformal 重校准、工具层与审计层处理，不能由本组件隐含声明。
