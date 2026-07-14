# 早期循环特征与轨迹证据工具协议 v1

`extract_early_cycle_features` 是生成早期 SOH 轨迹与特征证据的受控源工具。它只
接受服务端受信存储签发的 `record_batch_id`，再由注入的
`VerifiedEarlyCycleBatchResolver` 解析 canonical Parquet 记录、元数据、特征配置、
版本和 `OBSERVED` 来源链。API、MCP、Agent、Next.js 和 Streamlit 不能直接提交
原始记录、元数据、特征配置、SOH、版本或 provenance。

## 输入与信任边界

公有请求仅含一个去首尾空白后的非空 `record_batch_id`。下列字段出现在公有请求
中时必须按 `extra="forbid"` 拒绝：

- `records`、`metadata`、`feature_config`；
- `data_version`、`split_version`、provenance；
- `observed_soh`、`condition_features`、`prediction_cycles` 或任何模型输出。

resolver 返回 `VerifiedEarlyCycleBatch`，并且返回对象在工具内部 `model_dump` 后重新
`model_validate`。因此适配器即使错误地用 `model_construct` 绕过自身的构造校验，
也不能让无效数据进入数值层。

受信批次必须满足：

1. 记录与元数据只有一个完全一致的 `dataset_id + cell_id`；
2. 全部记录不晚于已登记的特征截断循环；
3. 截断循环本身有有效的诊断放电容量，避免将不完整窗口伪装成 20/50/100/150
   循环证据；
4. provenance 至少含一条 `OBSERVED` 来源记录，源清单哈希为 SHA-256；
5. 元数据额定容量和可选参考容量都是有限值；
6. 数据、划分和特征版本均由受信批次提供。

该上下文绑定工具不得加入 `create_available_tool_registry()` 默认注册表；只有拥有
受信数据解析器的服务工厂可以注册它。

## 数值派生

工具先调用已有的 `extract_early_cycle_features`，复用其截断检查、单电芯检查和
ERROR/BLOCKING 数据质量拒绝逻辑。随后仅从 valid diagnostic record 的每周期最大
诊断放电容量生成观测序列；缺失周期不插补。

参考容量优先采用受信元数据中的 `reference_capacity_ah`。若其缺失，只允许在截断
窗口内对有效诊断容量取确定性中位数，并标记为
`first_valid_diagnostic_median`；严禁回退为 `nominal_capacity_ah`。SOH 只由
`diagnostic_discharge_capacity / reference_capacity` 计算，且必须有限、位于 `(0, 1.5]`。

## 标准输出

工具返回：

```text
tool_name     = extract_early_cycle_features
artifact_type = quanxin_life.early_cycle_trajectory_evidence.v1
```

`values.artifact` 包含：

- `record_batch_id`、电芯身份与 `cutoff_cycle`；
- 实际诊断 `observed_cycles` 和由工具派生的 `observed_soh`；
- 参考容量、参考方法、特征/工况、源循环和特征警告；
- `source_manifest_hash`；
- data / feature / split 版本。

外层 `ToolResult` 的 data 与 feature 版本和 artifact 必须一致，provenance 直接继承
受信批次，`input_hash` 只来自公有 `record_batch_id`，`created_at` 只来自实际 UTC
执行时钟。旧的 `values.trajectory_input` 结构不得继续作为正式下游接口。

其中 `feature_values` 保留完整特征字典，缺失信号显式为 `null`；
`condition_features` 仅包含确定性可得的有限数值，供预测器消费。缺失特征不会被
替换为零、均值或任何隐式代理。

## 下游限制

`predict_soh_trajectory` 必须通过 `AuditLedger` 按 result ID 重新解析本证据，并验证
工具名、`artifact_type`、内外版本、身份、截断安全性和 `OBSERVED` 来源。解析失败、
受信批次不存在或数据不足时，系统必须显式失败或降级，不得补齐数值或伪造轨迹。
