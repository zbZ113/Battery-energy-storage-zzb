# 在线个体参数校正工具协议 v1

`update_cell_parameters` 将 `IndividualTrajectoryCalibrator` 封装为一个
受审计的在线个体校正工具。它只在已登记、冻结且有限的全局 SOH 轨迹内，利用
已登记的新增诊断 SOH 观测来校正单个电芯的偏置、退化速率和拐点偏移；不会训练、
微调或替换全局模型，也不会生成独立的 RUL、EOL 或置信区间。

## 最小输入与账本绑定

公开输入只有：

- `trajectory_result_id`：必须解析为已登记的 `predict_soh_trajectory` 结果；
- `observation_result_id`：必须解析为已登记的 `ingest_newly_observed_soh` 结果；
- `update_version`：必须精确等于当前 `CalibrationConfig` 的版本与规范 JSON
  SHA-256 组合。

两个结果 ID 必须不同。`global_trajectory`、`observations`、provenance、模型版本
以及任何直接数值均会被 `ContractModel(extra="forbid")` 拒绝。

所有上游 ID 由 `AuditLedger.resolve_registered_result()` 重新构造。旧版裸字段：

```text
values.prediction_cycles
values.predicted_soh
values.newly_observed_soh
```

不兼容且必须拒绝；调用者不能将有效 ID 与替换后的数值拼接使用。

## 上游证据契约

### 轨迹证据

上游必须是工具版本匹配的 `predict_soh_trajectory`，并且：

```text
values.artifact_type = quanxin_life.predicted_soh_trajectory.v1
```

其 `values.artifact` 必须包含可验证的批次 ID、电芯身份、截断点、预测周期、预测
SOH、模型/数据/特征/划分版本、上游早期特征结果 ID、训练期工况字段名称和模型
制品状态。外层和包络内的模型、数据、特征版本必须完全一致，再重构为
`FrozenGlobalTrajectory`。

### 新增观测证据

上游必须是工具和规则引擎版本都匹配的 `ingest_newly_observed_soh`，并且：

```text
values.artifact_type = quanxin_life.newly_observed_soh_evidence.v1
```

其包络须包含同一测量批次内严格递增的唯一 UUID 测量记录、真实测量时间、
源记录哈希、`NEWLY_OBSERVED` 类型、参考容量及其方法、观测数量和完整版本。
外层与包络内的数据/特征版本、观测与轨迹的身份/数据/特征/划分版本均需一致。
上游 provenance 至少有一条 `NEWLY_OBSERVED` 记录。

任何来源不明、版本不一致、跨电芯、超出冻结轨迹窗口、重复测量或结构不完整的
结果都会失败，而不是静默填补、插值或推测。

## 校正范围与降级

数值内核只调用：

```text
IndividualTrajectoryCalibrator.calibrate(
  global_trajectory=<frozen>,
  observations=<newly observed>,
  update_version=<config-bound>
)
```

校正器只能改变电芯个体参数；全局网络、特征工程、数据划分和 Conformal 校准器均
不在可变范围内。观测不足、优化失败或拟合质量不足时，工具保留冻结全局轨迹，并
返回 `RECHECK` 与明确原因码。

若上游轨迹仍标记为 `UNREGISTERED_IN_MEMORY`，输出会保留该状态并增加
`UPSTREAM_MODEL_ARTIFACT_UNREGISTERED` 警告。该降级状态可用于开发和回放，
但在服务工厂接入已核验 SHA-256 的模型制品清单前，监督器不得将其作为正式性能
结论或生产决策模型来源。

## 标准输出

```text
tool_name     = update_cell_parameters
tool_version  = online-update-tool-v1
artifact_type = quanxin_life.online_individual_calibration.v1
```

`values.artifact` 包含：

- `status` 与 `reason_code`；
- 校正后的有限轨迹；
- 内核返回的完整审计记录和校正配置哈希；
- 内核警告、两个上游结果 ID、划分版本和上游模型制品状态。

ToolResult 的模型、数据与特征版本继承自冻结全局轨迹；输入哈希只基于三项公开
输入；provenance 从两个上游结果按规范 JSON SHA-256 去重合并；`created_at` 仅使用
执行时的 UTC 时钟。

Agent、FastAPI、MCP 与 Next.js 必须经同一工具注册路径调用本工具，不得
复制校正逻辑或由 LLM 改写数值。
