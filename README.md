# 泉芯智寿

“泉芯智寿”是面向新能源装备产业的多智能体协同储能电芯寿命预测与退化感知决策系统。系统把可复现的数据与模型工具作为唯一数值来源，由多智能体负责调用工具、组织证据、解释不确定性并生成可追溯建议。

当前提交建立项目宪法、数据与实验契约以及 `quanxin_life.core` 公共类型，不包含业务模型、数据适配器、Docker Compose 或前后端实现。

## 核心原则

- LLM 不计算、猜测或补齐业务数值；所有数值必须来自版本化工具的 `ToolResult`。
- 训练、验证和测试按电芯划分，禁止同一电芯跨集合泄漏。
- 未知来源的 pickle、joblib、PyTorch checkpoint 不得反序列化。
- PyBaMM 只用于短期物理一致性核验，不作为长期寿命真值生成器。
- 每项结论都必须带证据级别、来源哈希、版本、不确定性与警告。

## 开发环境

项目锁定 Python 3.11。核心包使用 `src` 布局；前端规划为 Next.js，分析工作台规划为 Streamlit，服务编排目标环境为 WSL2 + Docker。

```powershell
python -m pytest tests/unit/core -q
python -m compileall src
```

完整实施路线见 [批准计划](docs/superpowers/plans/2026-07-12-quanxin-zhishou-full-implementation.md)。
