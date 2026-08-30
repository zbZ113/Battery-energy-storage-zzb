# SOH 轨迹预测工具协议 v1

`predict_soh_trajectory` 是受审计账本约束的有限窗口轨迹预测工具。它只将
已登记的 `extract_early_cycle_features` 结果交给已拟合的内存
`HybridDegradationPredictor`，不读取 pickle、joblib、`.pt`、`.pth` 或任何
外部模型文件，也不会自行创建、补全或改写 SOH、EOL80、RUL 数值。

## 最小输入与上游绑定

调用者只能提交 UUID 字段 `upstream_result_id`。任何调用方提供的电芯身份、
周期、SOH、工况、版本或 provenance 都会被 `ContractModel(extra="forbid")`
拒绝。

工具通过 `AuditLedger.resolve_registered_result()` 解析 ID，并且要求上游结果：

- `tool_name` 为 `extract_early_cycle_features`；
- `tool_version` 和 `model_version` 分别等于当前早期特征工具的已登记版本；
- `values.artifact_type` 严格等于
  `quanxin_life.early_cycle_trajectory_evidence.v1`；
- `values.artifact` 能通过严格的 `TrajectoryPredictionEvidence` 契约；
- 外层与包络内的 `data_version`、`feature_version` 完全一致；
- provenance 至少含有一条 `OBSERVED` 来源。

旧版 `values["trajectory_input"]` 裸负载不兼容且必须拒绝。这样可避免 UI、
Agent 或调用者把未经来源校验的数值注入模型。

## 早期循环证据边界

标准上游包络包含：可信 `record_batch_id`、电芯身份、截断点、截断点以前的
诊断 SOH、参考容量及其方法、完整特征值、有限可用的工况特征、源循环、
源清单哈希和数据/特征/划分版本。

工具在调用模型前再次保证：

- 观测周期非负、严格递增、不晚于截断点，最后一点等于截断点；
- 观测 SOH 位于 `(0, 1.5]`，且截断点尚未达到 EOL80；
- 工况及非空特征值均为有限数，缺失特征必须保留为 `null`，不能静默填零；
- 源循环位于截断窗口内，源清单哈希符合 SHA-256 格式；
- 上游数据、特征、划分版本与已拟合预测器上下文完全一致。

早期特征工具会保留全部有限工况特征。轨迹工具仅从中选择预测器训练时声明的
`condition_feature_names` 子集；缺少任何模型必需工况即失败，额外来源特征不会
被伪造成模型输入。输出包络同时记录实际使用的工况字段名称。

## 输出与适用边界

输出 `ToolResult.values` 使用：

```json
{
  "artifact_type": "quanxin_life.predicted_soh_trajectory.v1",
  "artifact": {
    "record_batch_id": "...",
    "prediction_cycles": ["..."],
    "predicted_soh": ["..."],
    "eol80_crossing": {"...": "..."},
    "derived_rul_cycle": "...",
    "model_version": "...",
    "feature_version": "...",
    "split_version": "...",
    "data_version": "...",
    "upstream_result_id": "...",
    "model_condition_feature_names": ["..."]
  }
}
```

`derived_rul_cycle` 只能由预测轨迹第一次满足 `SOH <= 0.80` 的交叉点推导；
若有限预测窗口内不存在交叉点，二者均为 `null` 并产生
`NO_EOL80_CROSSING_IN_FINITE_HORIZON` 警告。这不构成窗口外寿命、质保或
长期工业寿命承诺。

本工具不输出 Conformal 区间；`uncertainty` 固定声明
`finite_horizon_only=true` 与 `conformal_interval_included=false`。正式批次
决策仍须消费独立、按电芯校准的区间工具结果。

当前上下文工具只接收内存中的已拟合预测器，不加载任何模型制品。因此输出会强制
携带 `MODEL_ARTIFACT_UNREGISTERED_IN_MEMORY` 警告和
`model_artifact_status=UNREGISTERED_IN_MEMORY`。该状态只能用于开发、回放和
集成验证；服务工厂接入已校验 SHA-256 的模型清单前，监督器不得把它作为正式竞赛
性能结论或生产决策的模型来源。

Agent、API、MCP 与 Next.js 必须经同一注册路径调用此工具；禁止由
LLM 或界面直接拼接或改写数值。
