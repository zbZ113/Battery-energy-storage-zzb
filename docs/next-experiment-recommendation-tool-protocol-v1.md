# 主动试验推荐工具协议 v1

## 1. 定位

`recommend_next_experiment` 是对 `quanxin_life.experiments.gp_active` 的受治理封装。它面向 Naumann 等公开实验室 LFP/石墨工况级数据，基于已审查的真实观测拟合固定核高斯过程，并在已审核的候选空间中输出下一批试验的优先级。

它不产生工业安全批准、企业生产部署结论或长期储能寿命结论。候选“安全已审批”仅表示服务端目录中存在明确审批状态；每一项推荐仍必须经过人工安全和设备排程确认。

## 2. 公共输入边界

公共 API、MCP、Agent 与前端的输入只有：

```json
{
  "recommendation_context_id": "naumann-cycle-capacity-loss-v1"
}
```

客户端、LLM 和 Agent 不得提交或修改：

- 观测退化率、容量损失率、阻抗增长率及其标签；
- 温度、SOC、DOD、充放电倍率等候选工况；
- 安全审批、设备可用性、试验时长或设备成本；
- GP 边界、采集策略、参考空间、批量大小或核参数。

这些值只能由 `VerifiedExperimentRecommendationContextResolver` 从服务端批准的上下文中返回。该约束防止 LLM 伪造数值、绕过设备限制或通过修改成本改变试验排序。

## 3. 可信上下文

服务端上下文至少包含：

- `dataset_id`、`data_version`、`feature_version`、来源清单 SHA-256；
- Naumann 桥接映射版本和候选目录版本；
- 实际观测的条件、目标值、资源消耗、源观测 ID 与记录哈希；
- 固定的五维运行边界、采集配置、参考条件空间、策略与批量大小；
- 候选条件及明确的 `safety_approved`、`equipment_available` 状态；
- 至少一条 `OBSERVED` 来源记录。

上下文在使用前重新通过 Pydantic 契约验证。观测 ID、源观测 ID、候选 ID 必须唯一；已观测条件和参考条件必须落在边界内；参考条件不得重复。候选越界或未批准不会被裁剪或默认放行，而由数值模块以明确拒绝码记录。

## 4. 数值执行与制品

工具构造 `GaussianProcessExperimentRecommender`，仅以可信上下文中的实际观测调用：

```text
fit(observed Naumann condition measurements)
→ recommend_batch(reviewed candidates, reviewed reference space)
→ ToolResult
```

GP 模型固定为 `Matérn(ν=2.5) + WhiteKernel`，随机种子为 `20260712`。主策略是成本约束 EIVR；`MAX_VARIANCE` 只能由服务端上下文明确选为基线。工具不读取或保存 pickle、joblib、`.pt`、`.pth` 等可执行反序列化制品。

输出 `ToolResult` 的固定字段为：

```text
tool_name       recommend_next_experiment
tool_version    next-experiment-recommendation-tool-v1
artifact_type   quanxin_life.experiment_recommendation.v1
model_version   naumann-condition-gp-v2
```

制品记录上下文 ID、数据/映射/候选目录版本、来源清单哈希、完整策略哈希、已使用源观测 ID 和记录哈希，以及每个被选中或拒绝候选的预测均值、后验标准差、成本、重复惩罚、采集分数和拒绝原因。数值仅存在于该 `ToolResult` 内；报告必须通过 `result_id` 和审计账本引用。

## 5. 安全、来源与降级

所有结果强制携带：

- `RECOMMENDATION_REQUIRES_HUMAN_SAFETY_APPROVAL`；
- `PUBLIC_LABORATORY_DATA_NOT_INDUSTRIAL_DEPLOYMENT_EVIDENCE`。

上下文不存在、来源未含 `OBSERVED`、边界/参考空间不合规、观测不足或 GP 返回非有限值时，工具直接失败并返回可诊断错误；不得补零、插值、替代标签或返回伪成功结果。候选未通过审批、设备不可用、越界或与已观测条件重复时则作为明确的拒绝候选保留在同一审计制品中。

## 6. 注册与服务集成

该工具依赖服务端的上下文解析器，因此不进入无参 `create_available_tool_registry()`。FastAPI、MCP、Agent 与 Next.js 必须通过同一 `register_recommend_next_experiment_tool()` 绑定同一个已审核上下文解析器；禁止在不同接入层复制 GP、排序或候选过滤逻辑。
