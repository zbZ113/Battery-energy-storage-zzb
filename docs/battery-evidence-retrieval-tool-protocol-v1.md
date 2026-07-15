# 电池知识证据检索工具协议 v1

## 1. 目的与定位

`retrieve_battery_evidence` 为监督 Agent、报告生成器与产品界面提供带来源的解释性证据。它检索 LFP 退化机理论文、公开数据卡、模型卡、PyBaMM 假设和工业接口规范；它**不**预测 SOH、RUL、区间、退化率、工况成本或企业经营指标。

检索命中属于 `DOMAIN_KNOWLEDGE` 或 `PHYSICS_REFERENCE` 等叙述性证据。它们可用于解释模型适用范围和提示人工复核，但不能代替已登记数值工具的结果，也不能直接成为正式数值报告的 `NumericEvidence`。

## 2. 公共调用契约

公共调用只允许：

```json
{
  "knowledge_scope_id": "lfp-public-evidence-v1",
  "query": "磷酸铁锂温度退化机理",
  "top_k": 5
}
```

`query` 是非数值检索意图；`top_k` 限定为 1—20。调用方不得提交文档正文、URL、哈希、嵌入向量、页码、BM25 分数、向量分数、重排分数、引用条目、证据等级或 provenance。Pydantic 契约使用 `extra="forbid"`，以阻断客户端/LLM 附加伪造来源。

## 3. 服务端知识范围

`VerifiedKnowledgeScopeResolver` 只返回已审核的知识范围，其中包含：

- `corpus_version`、`index_version`、`retrieval_policy_version`；
- 每份批准文档的 ID、标题、URI、SHA-256、许可证和证据等级；
- 至少一条 `OBSERVED` 来源记录。

返回值会在工具内重新验证。文档 ID 与 source ID 必须唯一。检索后端产生的每个命中都必须回链到该范围内的批准文档，且命中的证据等级必须与批准文档匹配；未知文档、重复 chunk、超过 `top_k` 的响应或不可序列化分数均会使调用失败。

## 4. 最高配置检索后端

生产后端实现 `HybridBatteryEvidenceBackend`，目标模式固定为：

```text
pgvector 语义召回 + PostgreSQL 全文/BM25 关键词召回 + 中文/中英重排器
```

后端返回 `HybridEvidenceSearchResponse`，每个命中包含 `document_id`、`chunk_id`、页码或章节、原文摘录、BM25 分数、向量分数、重排分数和最终分数。工具不自行重算这些分数，也不让 LLM 修改它们。未来的 pgvector 存储、embedding 服务和 reranker 仅能通过该后端协议接入，不得在 FastAPI、MCP、Agent、Next.js 或 Streamlit 中另写检索逻辑。

若后端明确返回非 `pgvector_bm25_reranker` 模式，工具保留结果但强制附加 `KNOWLEDGE_RETRIEVAL_DEGRADED_FROM_PGVECTOR_HYBRID`；该降级标记必须被前端、Agent 与报告展示，不得静默隐藏。

## 5. ToolResult 与审计边界

工具输出：

```text
tool_name       retrieve_battery_evidence
tool_version    battery-evidence-retrieval-tool-v1
artifact_type   quanxin_life.battery_evidence_retrieval.v1
model_version   后端版本
data_version    语料库版本
feature_version 检索策略版本
```

制品记录检索范围、索引与策略版本、检索模式、查询、`top_k` 和完整引用。每条引用都含来源 URI、SHA-256、许可证、页码/章节、摘录、证据等级及三类检索分数。输出始终携带 `RETRIEVAL_OUTPUT_IS_NOT_A_NUMERICAL_PREDICTION`。

正文中的任意工程数值仍只能通过 `AuditLedger` 引用其他数值工具的 `ToolResult.values.*`。知识检索的分数仅是信息检索元数据，不能被解释为电芯性能、模型精度或工业收益。

## 6. 注册与多智能体使用

该工具需要服务端知识范围解析器与混合检索后端，因此不进入无参 `create_available_tool_registry()`。应用启动时以 `register_retrieve_battery_evidence_tool()` 显式绑定。监督 Agent 只可通过其白名单调用该注册工具；没有已审核语料库时工作流必须显式失败或显示降级，而不是使用 LLM 记忆补写引用。
