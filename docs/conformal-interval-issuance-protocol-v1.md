# 目标电芯 Normalized Conformal 区间签发协议 v1

本协议约束“点预测 → 校准 → 目标区间 → 批次决策”的数值证据链。它是
[`conformal-calibration-tool-protocol-v1.md`](conformal-calibration-tool-protocol-v1.md)
中目标区间签发模式的实施补充，不替代数据集级覆盖率评估。

## 输入边界

公开调用方仅可提交：

```json
{
  "prediction_result_id": "UUID",
  "calibration_result_id": "UUID"
}
```

调用方不能提交 `observed_eol_cycle`、`difficulty_scale_cycle`、`alpha`、残差、区间边界、目标域校准状态、模型版本或来源信息。`extra="forbid"` 是强制边界，违规字段必须在 Pydantic 解析阶段失败。

## 受信任依赖

签发器仅从以下受控来源读取数值：

1. `AuditLedger` 中的 `predict_cycle_life` 标准制品；
2. `AuditLedger` 中的 `normalized_conformal_calibration.v1` 标准制品；
3. `VerifiedPredictionDifficultyScaleResolver` 解析的、与点预测结果 ID 一一绑定的模型难度尺度。

点预测、难度尺度与校准证据的 `dataset_id`、`cell_id`、`cutoff_cycle`、`feature_version`、`split_version`、`model_version`、`data_version` 和 `scale_version` 必须保持相容；难度尺度的 `source_manifest_hash` 还必须与点预测标准制品中的来源清单哈希相同。目标 `cell_id` 不得出现在校准电芯列表中。任一不一致、未登记结果、非 UUID、缺失预测来源或暴露目标 EOL 标签的点预测都必须拒绝。

## 数值生成与领域状态

区间只通过确定性函数 `make_normalized_prediction_interval()` 生成。RUL 不单独建模；它继续由点预测寿命与截断点导出。LLM、UI、MCP 和 Agent 只能传递结果 ID、调用工具和解释结果，不能计算或改写区间。

`target_domain_calibrated` 由服务端计算：

```text
target_domain_calibrated = (target_domain_id == calibration_domain_id)
```

若校准队列还声明 `target_domain_recalibration`，结果会附加
`TARGET_DOMAIN_RECALIBRATION_APPLIED`。若两个领域不一致，结果必须附加
`TARGET_DOMAIN_UNCALIBRATED`，后续批次决策只能复检，不得将源域 90% 覆盖率外推为目标域保证。

## 输出制品和审计

区间结果必须使用：

```text
artifact_type: quanxin_life.normalized_prediction_interval.v1
tool_name: calibrate_prediction_interval
tool_version: conformal-calibration-tool-v1
```

制品包含目标区间、两条上游 `result_id`、领域状态和难度尺度来源清单哈希；`ToolResult.uncertainty` 包含覆盖率目标、难度尺度与上下界。来源链合并并去重点预测、校准和难度尺度的来源记录。

批次决策工具必须再次从 `AuditLedger` 解码该制品，并核对其中的 `calibration_result_id` 与显式输入一致。没有 `result_id`、版本、输入哈希和来源链的数字不得进入正式报告。

## 验收

```powershell
.\.venv\python.exe -m pytest tests\unit\tools\test_conformal_interval_issuance_tool.py tests\unit\tools\test_batch_decision_tool.py -q
.\.venv\Scripts\ruff.exe check src\quanxin_life\tools\conformal_calibration.py src\quanxin_life\tools\batch_decision.py tests\unit\tools
.\.venv\Scripts\mypy.exe src\quanxin_life\tools\conformal_calibration.py src\quanxin_life\tools\batch_decision.py
```
