# 周期寿命预测工具协议 v1

`predict_cycle_life` 是 EOL80 周期寿命的受审计数值工具。它只将已登记的早期循环特征证据交给已拟合的数值预测器；LLM、前端、Agent 和调用方均不能提交或改写特征、EOL80 标签、RUL、区间或模型版本。

## 输入边界

公开输入只有 `upstream_result_id`，且必须是由
`extract_early_cycle_features` 产生并登记在 `AuditLedger` 中的 UUID。工具会重新校验：

- 上游工具名称、工具版本和特征模型版本；
- `quanxin_life.early_cycle_trajectory_evidence.v1` 工件类型；
- 外层与工件内的数据、特征和划分版本；
- 数据集与电芯标识，以及 `OBSERVED` 来源记录；
- 预测器要求的完整、有限、非填补特征列。

旧的裸 `values.features`，未登记 ID，调用方直接给出的标签、RUL、区间、版本或来源字段都必须拒绝。

## 数值模型边界

工具只调用已经在受控进程中拟合的 `CycleLifePredictor.predict()`。预测器必须声明与上游证据一致的 `feature_version`、`split_version`、`data_version`、`cutoff_cycle` 和精确的特征列名。缺失特征或非有限特征会失败，绝不以零值、均值或虚构代理变量填补。

工具不读取 pickle、joblib、`.pt`、`.pth` 或未知权重。当前内存预测器尚未接入已登记的模型制品清单，因此输出固定携带：

```text
model_artifact_status = UNREGISTERED_IN_MEMORY
warning               = MODEL_ARTIFACT_UNREGISTERED_IN_MEMORY
```

这表示结果可用于研发流程演示和后续 Conformal 校准输入，但不能作为已登记模型的正式性能声明或生产部署依据。

## 标准输出

```text
tool_name     = predict_cycle_life
tool_version  = cycle-life-prediction-tool-v1
artifact_type = quanxin_life.predicted_cycle_life.v1
```

`values.artifact` 包含记录批次、数据集/电芯、截断循环、严格校验后的 `LifePrediction`、由点预测减截断循环得到的派生 RUL、源清单哈希、上游结果 ID、划分版本、实际使用特征列以及模型制品状态。`ToolResult` 继承上游 provenance，输入哈希仅基于公开 ID，且 `uncertainty` 固定为 `null`。

本工具不会生成经 Conformal 校准的预测区间；任何区间结论必须由独立的校准工具在电芯隔离的校准集合上产生。

## 适用范围

该工具只处理可追溯的公开实验室早期循环数据。它不把公开 MATR 或 Naumann 数据表述为济南企业或工业储能现场真实数据，也不支持精确外推 15—25 年寿命。跨数据集或目标域使用时，必须经过独立的域适应与目标域重校准流程，并在结果中显式标识其状态。
