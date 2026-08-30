# 可信 Normalized Conformal 校准与区间签发工具协议 v1

## 目的与边界

`calibrate_prediction_interval` 是具有两种互斥安全操作模式的受约束工具：

1. **校准证据模式**：从已经过来源核验、按 `cell_id` 完全隔离的校准队列中计算 Normalized Conformal 残差分位数；
2. **目标区间签发模式**：从已登记点预测、已登记校准证据，以及服务端受信任的同版本预测难度尺度中，签发一颗目标电芯的寿命区间。

工具不接受调用方、LLM、HTTP、MCP 或前端传入的标签、点预测、残差、难度尺度、`alpha`、区间上下界、模型版本、数据版本或来源哈希。所有工程数值只能来自受信任解析器、`AuditLedger` 和确定性 Conformal 算法。

## 公共输入

### 模式 A：校准证据

```json
{
  "calibration_cohort_id": "trusted-calibration-cohort-20260715"
}
```

`calibration_cohort_id` 是服务端登记标识，不要求调用方上传文件、预测值或标签。

### 模式 B：目标电芯区间签发

```json
{
  "prediction_result_id": "UUID",
  "calibration_result_id": "UUID"
}
```

两个 ID 必须引用当前 `AuditLedger` 中已登记的结果：

- `prediction_result_id` 必须对应 `predict_cycle_life` 的标准点预测制品；
- `calibration_result_id` 必须对应模式 A 输出的标准校准制品。

服务端再通过 `VerifiedPredictionDifficultyScaleResolver` 解析与点预测 ID 一一绑定的模型难度尺度。尺度的电芯、数据集、截断点、特征、划分、模型、数据和 `scale_version` 必须逐项匹配已登记点预测及校准证据。

输入模型采用 `extra="forbid"`；调用方不能混用两种模式，也不能额外传入数值字段。

## 受信任校准队列

模式 A 的 `VerifiedNormalizedCalibrationCohort` 必须满足：

- 预测只属于 `SplitManifest.calibration` 的电芯级校准集合；
- 预测共享同一数据集、`cutoff_cycle`、目标、特征、划分、模型、数据和尺度版本；
- 每条预测均有真实、非右删失的 `EOL80` 标签；
- 队列来源至少包含一条 `OBSERVED` 记录，并登记来源清单哈希；
- 解析后的合约会再次经 Pydantic 校验。

因此，调用方无法通过伪造普通 DTO、跨切分电芯、不同版本、删除标签或裸 CSV 来生成校准证据。

## 输出制品

### 模式 A：校准证据

```text
tool_name: calibrate_prediction_interval
tool_version: conformal-calibration-tool-v1
artifact_type: quanxin_life.normalized_conformal_calibration.v1
```

`values.artifact` 含校准分位数、样本数、尺度版本、校准域、范围、校准电芯 ID、切分清单哈希和来源清单哈希，但不含逐电芯真实寿命标签或残差。数值来自 `calibrate_normalized_conformal()`。

### 模式 B：目标电芯区间

```text
tool_name: calibrate_prediction_interval
tool_version: conformal-calibration-tool-v1
artifact_type: quanxin_life.normalized_prediction_interval.v1
```

`values.artifact` 至少包含：

- `prediction_interval`：由 `make_normalized_prediction_interval()` 生成的 `NormalizedPredictionInterval`；
- `prediction_result_id` 与 `calibration_result_id`：完整上游结果链；
- `calibration_domain_id`、`target_domain_id` 与 `target_domain_calibrated`；
- `difficulty_scale_source_manifest_hash`：受信任难度尺度的来源清单哈希。

结果的 `uncertainty` 仅反映已计算的覆盖率目标、难度尺度与区间上下界；所有值由登记点预测、校准制品和受信任尺度导出。其 `provenance` 合并点预测、校准和难度尺度来源，并去重保留来源链。

## 覆盖率与跨域声明

每份结果始终带有：

```text
COVERAGE_VALID_ONLY_FOR_DECLARED_CALIBRATION_COHORT
```

当校准队列声明 `target_domain_recalibration`，且目标域与校准域匹配时，额外带有：

```text
TARGET_DOMAIN_RECALIBRATION_APPLIED
```

当目标域与校准域不匹配时，模式 B 必须带有：

```text
TARGET_DOMAIN_UNCALIBRATED
```

此时 `target_domain_calibrated=false`，后续批次决策工具必须降级为 `RECHECK`，不得将源域 90% 覆盖率外推为目标域保证。跨域 PICP/MPIW 必须基于独立目标域评测电芯报告。

## 注册与运行时依赖

工具依赖可信解析器，故不得被无参的 `create_available_tool_registry()` 自动注册。装配层必须显式注入校准队列解析器；模式 B 还必须显式注入 `AuditLedger` 与难度尺度解析器。不得为了方便调用而读取裸 CSV、加载未登记模型制品或构造占位标签。

模式 A 的 `uncertainty` 为 `null`，因为它不是单电芯结论；模式 B 的 `uncertainty` 必须绑定已登记上游数值。

## 验收

```powershell
.\.venv\Scripts\ruff.exe check src\quanxin_life\tools\conformal_calibration.py
.\.venv\Scripts\mypy.exe src\quanxin_life\tools\conformal_calibration.py
```

数值契约回归用例由团队在本地维护，不随 Git 分发。
