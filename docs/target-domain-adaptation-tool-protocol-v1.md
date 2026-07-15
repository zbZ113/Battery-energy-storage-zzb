# 跨数据集 CORAL 目标域适配工具协议 v1

`adapt_to_target_domain` 实现阶段 5 的第一条可信跨数据集适配链：以 MATR 源域训练电芯的早期特征为输入，经由 HUST 目标域**无标签训练电芯**的二阶统计量进行 CORAL 对齐。该工具不是寿命预测器、不是目标域校准器，也不输出 SOH、RUL、置信区间或业务决策。

## 调用边界

公开输入只有一个服务端登记标识：

```json
{
  "adaptation_cohort_id": "matr-to-hust-train-v1"
}
```

调用方、LLM、UI、Agent 与 MCP 均不能提交原始特征、目标域标签、结果 ID 列表、目标测试电芯、适配器超参数或协方差矩阵。输入采用 `extra="forbid"`，额外字段会在契约解析阶段失败。

## 受信任适配上下文

`VerifiedAdaptationCohortResolver` 由服务端解析 `adaptation_cohort_id`，返回：

- 源域和目标域各自已登记的 `extract_early_cycle_features` 结果 ID；
- 两份 `SplitManifest`；
- `adapter_version`、特征名和协方差正则化配置。

工具重新从 `AuditLedger` 获取每一条结果，逐条验证标准早期特征制品、版本字段与 `OBSERVED` 来源。源域结果的 `cell_id` 必须精确等于源域 `split_manifest.train`；目标域结果的 `cell_id` 必须精确等于目标域 `split_manifest.train`。验证、校准、测试电芯、跨域同名电芯、未登记结果、错误工具、缺失来源或不一致的特征/截断点/版本均被拒绝。

源域与目标域可以具有不同 `data_version` 和 `split_version`，但必须共享相同的 `feature_version`、`cutoff_cycle` 和已配置特征模式。目标域数据只用于无标签统计对齐；EOL80、SOH 真值和任何后验标签均不会进入 CORAL。

## 数值与输出

工具固定使用 `CORALFeatureAdapter`，以训练集合的协方差矩阵计算源域白化与目标域着色。输出标准制品：

```text
tool_name: adapt_to_target_domain
tool_version: target-domain-adaptation-tool-v1
artifact_type: quanxin_life.target_domain_adaptation.v1
```

制品包括源/目标数据集、两组上游结果 ID、两份切分清单哈希、特征名、统一截断点、域内数据与切分版本、适配后的源域训练特征及其方法和适配器版本。所有数值均来自已登记早期特征证据和确定性 CORAL 计算；LLM 不得生成或修改任何对齐特征。

输出的 `uncertainty` 固定为 `null`，并强制带有：

```text
TARGET_DOMAIN_ADAPTATION_DOES_NOT_ESTABLISH_CONFORMAL_COVERAGE
```

这表示 CORAL 只减少特征分布差异，不能替代 HUST 目标域 Conformal 重校准，也不能成为目标域 90% 覆盖率的声明。

## 注册边界与后续使用

该工具需要显式注入 `AuditLedger` 和可信适配上下文解析器，因此不进入无参 `create_available_tool_registry()`。后续训练、预测、难度尺度和 Conformal 校准必须显式绑定同一适配结果 ID 与目标域上下文；在这些下游契约接入前，适配制品只作为可审计科研训练输入，不直接改变现有预测结果。

## 验收

```powershell
.\.venv\python.exe -m pytest tests\unit\adaptation tests\unit\tools\test_target_domain_adaptation_tool.py -q
.\.venv\Scripts\ruff.exe check src\quanxin_life\adaptation src\quanxin_life\tools\target_domain_adaptation.py tests\unit\adaptation tests\unit\tools\test_target_domain_adaptation_tool.py
.\.venv\Scripts\mypy.exe src\quanxin_life\adaptation src\quanxin_life\tools\target_domain_adaptation.py
```
