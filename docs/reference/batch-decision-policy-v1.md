# 批次入组、复检、降级与拒绝决策协议 v1

`batch-policy-v1` 把已校准的 EOL80 预测区间转化为研发/质检分流建议。它不是寿命模型：不计算 SOH、RUL、退化率或置信区间，也不由 LLM 生成阈值。`required_eol_cycle` 必须由获授权的测试或质量策略输入，并与 `policy_version` 一起登记。

## 输入边界

- 输入只能是 `PredictionInterval` 或 `NormalizedPredictionInterval`；其数据、模型、特征和校准版本已由上游模型登记。
- `DataQualityReport.dataset_id` 必须与区间一致。任何 `BLOCKING` 数据质量问题会阻断数值决策。
- `target_domain_calibrated` 是必填布尔状态。它为 `false` 表示目标域没有独立校准证据；系统不会假定源域覆盖率可直接迁移。
- `batch-decision-tool-v1` 已将结果包装为 `make_batch_decision` 的 `ToolResult`。工具输入必须提供至少两个互异的上游 `result_id`（预测区间和数据质量报告）、来源链、模型/数据/特征版本以及输入哈希；包装层只保留确定性领域决策结果，不生成新的寿命或区间数值。

## 规则与优先级

1. 数据质量含 `BLOCKING`：`REJECT`，原因 `DATA_QUALITY_BLOCKING`；需先修复或重新采集数据。
2. 目标域未校准：`RECHECK`，原因 `TARGET_DOMAIN_UNCALIBRATED`；不得因点预测优秀而入组。
3. 区间下界严格高于策略要求：`ADMIT`，原因 `LOWER_BOUND_PASSES`。
4. 区间上界严格低于策略要求：`DOWNGRADE`，原因 `UPPER_BOUND_BELOW_REQUIREMENT`。
5. 其余情况（包括区间接触或跨越要求）：`RECHECK`，原因 `INTERVAL_CROSSES_REQUIREMENT`。

采用严格不等式是为了避免把阈值边界上的不确定性伪装成确定合格或确定不合格。公开实验室数据的输出只能作为研究原型分流依据，不得被表述为济南企业储能电芯的生产放行结论。
