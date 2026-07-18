# MATR Hybrid 早期特征时间容差热修复设计

## 故障现象

A100 三批 MATR Smoke 已完成截止循环 20 和 50 的模型套件。在进入截止循环
100 时，Hybrid 轨迹任务通过 `extract_early_cycle_features` 再次执行严格时间校验，
并因原始 MAT 浮点时间戳中约 `10^-11 s` 的数值反转触发
`NON_MONOTONIC_TIME`。

前一轮热修复仅将已审核的 `1e-9 s` MATR 数值容差传入
`CurveTensorConfig`，因此标量寿命模型可运行；Hybrid 使用的
`EarlyCycleFeatureConfig` 未携带同一容差，形成两条特征路径规则不一致。

## 设计决策

1. `EarlyCycleFeatureConfig` 新增非负、有限的
   `time_monotonic_tolerance_s`，默认值为 `0.0`。
2. 默认值保持所有通用调用和其他数据集的严格时间校验，不扩大容忍范围。
3. 仅 MATR 真实训练装载器显式传入已审核的 `1e-9 s`。
4. 容差内的微小反转必须在特征结果中保留
   `TIME_WITHIN_NUMERIC_TOLERANCE` 告警，不静默吞掉数据质量事实。
5. 超过配置容差的反转仍以 `NON_MONOTONIC_TIME` 阻断。
6. 修复不改变 SOH、寿命标签、数据划分、训练配置或任何模型数值。

## A100 运行一致性

修复会改变训练代码提交，因此不得将旧代码完成的截止循环 20/50 结果与新代码的
100/150 结果合并成同一 Smoke。服务器应保留旧运行目录作为审计备份，安装新代码
后从头运行完整 Smoke；预处理缓存和三批原始数据无需重新上传。

## 验收

- 显式配置 `1e-9 s` 时，容差内反转可继续且告警可见；
- 默认配置仍拒绝相同反转；
- 超过 `1e-9 s` 的反转仍被拒绝；
- MATR Hybrid 装载器确实将 `1e-9 s` 传给早期特征配置；
- 相关单元测试、Ruff、mypy 与项目回归通过；
- 生成内容明确、带 SHA-256 的第二个 A100 热修复包。
