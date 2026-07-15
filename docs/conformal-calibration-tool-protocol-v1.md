# 可信 Normalized Conformal 校准工具协议 v1

## 目的与边界

`calibrate_prediction_interval` 负责从已经过来源核验、按 `cell_id` 完全隔离的校准队列中计算
Normalized Conformal 的残差分位数。它输出的是**校准证据**，不是某颗电芯的寿命区间；后续
区间应用工具必须再绑定一条已登记的点预测、同版本难度尺度以及本工具产生的校准证据。

该工具不接受、也不会从 LLM、HTTP 请求或前端传入下列值：

- 单电芯 `observed_eol_cycle` 标签；
- 点预测、残差、难度尺度或预测区间；
- `alpha` 覆盖率参数；
- 模型版本、数据版本、切分版本或来源哈希。

这些信息仅能由 `VerifiedNormalizedCalibrationCohortResolver` 从受信任的评测制品和数据清单中
解析。当前版本使用平台固定的 `alpha=0.10`，即 90% 目标覆盖率；80% 和 95% 灵敏度分析应由
独立、版本化实验任务生成，不可在调用期由智能体临时设定。

## 公共输入

```json
{
  "calibration_cohort_id": "trusted-calibration-cohort-20260715"
}
```

`calibration_cohort_id` 是服务端登记的标识，不要求调用方上传文件、预测值或标签。输入模型采用
`extra="forbid"`，故任何附加数值字段都会被拒绝。

## 受信任队列契约

解析器返回的 `VerifiedNormalizedCalibrationCohort` 必须包含：

- 仅属于 `SplitManifest.calibration` 的电芯级预测；
- 与该切分清单相同的数据集、`cutoff_cycle`、目标、特征、切分、模型和数据版本；
- 真实、非右删失的 `EOL80` 标签；
- 同一 `scale_version` 和正的模型难度尺度；
- 原始数据清单 SHA-256 与至少一条 `OBSERVED` 来源记录；
- 已声明的 `calibration_domain_id` 与 `calibration_scope`。

解析器返回的数据会再次经 Pydantic 合约验证。即使调用方绕过解析器正常构造过程，缺少观测来源、
跨切分电芯、不同版本或删失标签都会被拒绝。

## 输出证据

输出 `ToolResult` 使用以下固定身份：

```text
tool_name: calibrate_prediction_interval
tool_version: conformal-calibration-tool-v1
artifact_type: quanxin_life.normalized_conformal_calibration.v1
```

`values.artifact` 含校准分位数、样本数、尺度版本、校准域、校准范围、校准电芯 ID、切分清单哈希
和来源清单哈希。它**不**包含逐电芯真实寿命标签或残差。

所有输出数值来自 `calibrate_normalized_conformal()`；LLM 只能解释已登记 `ToolResult`，不得生成、
修改或重新计算区间数值。

## 覆盖率声明

每份结果始终带有：

```text
COVERAGE_VALID_ONLY_FOR_DECLARED_CALIBRATION_COHORT
```

当解析器声明 `target_domain_recalibration` 时，额外带有：

```text
TARGET_DOMAIN_RECALIBRATION_APPLIED
```

这些警告表示该结果只能用于已声明的校准队列；它不构成对 HUST、Naumann、工业储能电芯或未重校准
目标域的覆盖率承诺。跨域 PICP/MPIW 必须以独立目标域评测电芯报告。

## 注册与运行时依赖

工具依赖可信解析器，故不得被无参的 `create_available_tool_registry()` 自动注册。装配层必须显式注入
来源核验器；不得为了方便调用而读取裸 CSV、加载未登记模型制品或构造占位标签。

当前工具也不直接应用预测区间，因此 `uncertainty` 字段为 `null`。这避免将校准统计量误表述为某个
工程对象的寿命结论。

## 验收测试

```powershell
.\.venv\python.exe -m pytest tests\unit\tools\test_conformal_calibration_tool.py tests\unit\uncertainty\test_normalized_conformal.py -q
.\.venv\Scripts\ruff.exe check src\quanxin_life\tools\conformal_calibration.py tests\unit\tools\test_conformal_calibration_tool.py
.\.venv\Scripts\mypy.exe src\quanxin_life\tools\conformal_calibration.py tests\unit\tools\test_conformal_calibration_tool.py
```
