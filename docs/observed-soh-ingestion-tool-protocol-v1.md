# 新增观测 SOH 接入工具协议 v1

`ingest_newly_observed_soh` 是在线个体校正链路中唯一允许产生
`NEWLY_OBSERVED` SOH 证据的受控源工具。它不接受客户端上传的容量、SOH、
版本或来源记录；公有请求只引用受信存储中已经核验的测量批次。工具在服务端
解析该批次，按“诊断放电容量 / 参考容量”确定性计算 SOH，并生成标准证据包。

## 不可变边界

1. API、MCP、Agent、Next.js 和 Streamlit 的公开输入只能是非空
   `measurement_batch_id`。空白 ID、SOH、容量、参考容量、预测轨迹、模型结果、
   数据/特征/划分版本和 provenance 都必须被拒绝。
2. 原始测量值只能由服务端注入的 `VerifiedMeasurementResolver` 返回。该解析器
   代表已验证的 BMS、实验室文件或设备接入记录，不能由客户端、LLM 或 UI 构造。
3. resolver 的返回值会序列化后重新经 `VerifiedObservationBatch` 校验；因此即使
   一个不可信适配器使用 `model_construct` 绕过自身构造校验，时间、身份、容量、
   来源与哈希约束仍会在数值计算前阻断。
4. 该工具不加载任何模型制品，不预测 RUL/EOL/区间，也不允许 LLM 生成或修改
   数值。
5. 未注入可信 resolver 时，工具不得加入默认 bootstrap 注册表；它只能由拥有
   受信存储依赖的 API 服务工厂注册。

## 受信批次契约

`VerifiedObservationBatch` 必须满足：

- 所有测量属于同一 `dataset_id + cell_id`；
- `measurement_id` 为 UUID 且批次内唯一；
- 循环号严格递增；测量 UTC 时间不能倒退，也不得晚于实际执行时间；
- 放电容量与参考容量均为有限正数，且一个批次使用同一参考容量；
- 每条测量有 SHA-256 格式的 `source_record_hash`；
- 批次携带非空数据、特征与划分版本；
- provenance 至少含一条 `NEWLY_OBSERVED` 记录。

少于三条可信观测允许登记。观测不足是否应复检由后续在线校正数值内核明确
返回；接入工具不得补点、插值或伪造来源。

## 标准输出

工具返回：

```text
tool_name     = ingest_newly_observed_soh
artifact_type = quanxin_life.newly_observed_soh_evidence.v1
```

`values.artifact` 至少包含：

- `measurement_batch_id`、`dataset_id`、`cell_id`；
- `reference_capacity_ah` 与 `reference_capacity_method`；
- 每条由工具计算的观测：`measurement_id`、cycle、SOH、`NEWLY_OBSERVED`、
  UTC `measured_at` 和 `source_record_hash`；
- `observation_count`；
- `data_version`、`feature_version`、`split_version`。

外层 `ToolResult` 的数据/特征版本和 provenance 必须直接继承受信批次；
`input_hash` 只基于公开的 `measurement_batch_id` 请求。`created_at` 表示工具实际
执行时间，不替代测量时间。

## 下游消费

`update_cell_parameters` 必须通过 `AuditLedger.resolve_registered_result()` 解析
本工具结果，并同时验证工具名、标准 `artifact_type`、身份、内外版本一致性和
`NEWLY_OBSERVED` 来源。旧的裸 `newly_observed_soh` 列表格式不得兼容。

任何 resolver 找不到批次、返回批次 ID 不匹配、数据质量不满足本协议或外部服务
不可用的情况，都必须显式失败或由上层返回降级状态；不得生成部分 `ToolResult`。
