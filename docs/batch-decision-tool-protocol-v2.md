# 批次入组／复检／降级决策工具协议 v2

## 迁移原因

旧版 `make_batch_decision` 包装器允许调用方同时提交预测区间、质量报告、策略阈值、领域校准状态、
来源和决策时间。即使携带 UUID，也无法证明这些业务数值来自已登记结果，不能作为 API、MCP、Agent
或界面层的正式入口。

v2 保留 `quanxin_life.decision.make_batch_decision()` 作为研究与单元测试使用的纯确定性函数；但对外的
工具只允许引用经 `AuditLedger` 重验证的证据与服务端已批准策略。

## 公共输入

```json
{
  "prediction_interval_result_id": "UUID",
  "calibration_result_id": "UUID",
  "quality_result_id": "UUID",
  "policy_id": "approved-policy-synthetic-v1"
}
```

三个结果 ID 必须不同并存在于当前请求上下文的 `AuditLedger`。调用方不能提交：

- SOH、RUL、寿命点预测、上下界、覆盖率或质量分数；
- 数据质量问题列表、领域是否已校准、阈值或策略版本；
- 来源记录、执行时间或任意业务结果；
- 由 UI、Agent 或 LLM 构造的中间 DTO。

## 上游证据要求

### 目标电芯区间

区间结果必须来自 `calibrate_prediction_interval` 的兼容版本，且使用
`quanxin_life.normalized_prediction_interval.v1` 制品。制品包含：

- `NormalizedPredictionInterval`；
- 其上游 `prediction_result_id` 与难度尺度来源清单哈希；
- 对应的 `calibration_result_id`；
- 校准域、目标域和目标域校准状态。

当 `target_domain_calibrated=true` 时，目标域必须与校准域一致。不能把源域校准结果表述为未经重校准
的 HUST、Naumann 或工业场景目标域结论。

### Conformal 校准

校准结果必须来自同版本 `calibrate_prediction_interval` 的
`quanxin_life.normalized_conformal_calibration.v1` 制品。工具逐字段核对其模型、数据、特征版本和
`NormalizedConformalCalibration` 是否与目标区间完全一致。

### 数据质量

质量结果必须来自 `validate_battery_data` 的兼容规则引擎。工具重新解析问题列表，并重新计算阻断状态、
问题数与确定性质量分，拒绝任何被篡改的汇总字段。

### 决策策略

`policy_id` 由 `VerifiedBatchDecisionPolicyResolver` 解析为经人工批准、版本化的策略。策略域必须与目标
区间数据集匹配；阈值不允许从调用请求传入。策略来源清单和来源记录会并入派生结果。

## 输出与降级

v2 输出 `batch-decision-tool-v2`。所有决策数值由底层确定性策略函数在已解析的证据上计算，来源链由：

```text
目标区间 ToolResult + 校准 ToolResult + 质量 ToolResult + 已批准策略来源
```

组成。质量阻断时输出 `REJECT`；目标域未校准时输出 `RECHECK`；只有整个区间严格跨过策略要求时才允许
`ADMIT` 或 `DOWNGRADE`。

无参 `create_available_tool_registry()` 不再注册该工具。运行时必须显式注入 `AuditLedger` 与可信策略解析器；
否则工具保持不可用，而非接受自由数值或构造演示结果。
