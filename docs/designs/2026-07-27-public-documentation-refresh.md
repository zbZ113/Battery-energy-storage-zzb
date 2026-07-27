# 公开文档纠偏设计

## 目标

让首次访问仓库的读者在 README 第一屏和文档导航中准确区分：

- 已实现（Implemented）：代码和契约已经存在；
- 已验证（Validated）：已有测试、正式实验或受控 E2E 证据；
- 计划中（Planned）：尚未完成或尚未获得外部输入；
- 非目标：不得把测试环境闭环、协议沙箱或 MATR 内部结果表述为生产接入、跨域保证或工业寿命承诺。

本次只整理公开文档，不修改模型、业务契约、部署拓扑或运行数据。

## 单一事实源

公开文档按职责分层，避免在多个页面复制不断变化的长篇状态：

| 文档 | 唯一职责 |
| --- | --- |
| `README.md` | 项目入口、核心价值、代表结果、快速开始和文档导航 |
| `docs/status.md` | 当前成熟度、已验证闭环和未完成事项 |
| `docs/benchmark.md` | 正式实验口径、五种子结果、Conformal 和晋级结论 |
| `docs/model-card-advanced.md` | Advanced 模型结构、输入输出、路由、适用范围和拒绝条件 |
| `docs/limitations.md` | 科研、数据、校准、部署和产品边界 |
| `docs/reproducibility.md` | 数据版本、划分、配置、制品、SHA-256 和复现层级 |
| `ARCHITECTURE.md` | 组件、信任边界、数值流、恢复语义和部署边界 |
| `docs/runtime-setup.md` | 可实际执行的入口，以及 foundation 与完整应用的差异 |

内部实施计划保留为历史和工程追踪材料，但不再作为公开状态的主入口。

## 叙事结构

核心技术主线固定为：

```text
真实电芯数据
→ cell_id 隔离与版本化数据制品
→ CyclePatch RUL / HybridPatch SOH
→ route-specific Split Conformal
→ ToolResult 与审计账本
→ 受约束专业智能体工作流
→ API / 报告 / UI
```

PyBaMM、主动试验、知识库、MCP、飞书和工业协议沙箱列为扩展能力，不与核心寿命预测证据并列。

## 数值发布规则

- 所有正式指标只来自已验收的 Advanced Final、metrics closure、Conformal 和 promotion 制品。
- README 只展示代表结果；完整表格放在 `docs/benchmark.md`。
- 明确区分 cycles、SOH 比例、SOH 百分点、百分比和无量纲 R²。
- 不把 MATR 官方 cycle life 表述为统一 EOL80。
- 不把训练耗时或峰值显存表述为推理延迟。
- 不把 `CONDITIONAL` 路由表述为已自动激活模型。
- 不发布本机绝对路径、原始敏感数据、密钥或未经 Git 管理的大结果文件。

## 验收标准

1. README 不再声称 A100 Smoke、五种子训练或正式 Conformal 仍在进行。
2. README 首屏包含真实结果和明确的成熟度边界。
3. CPMLP 与 Current Hybrid 被标记为基线，Advanced 正式模型成为算法主线。
4. 所有新增文档互相链接，术语和数值一致。
5. runtime 文档明确 foundation API 不是完整应用，完整产品栈尚未一键部署。
6. 现有文档契约测试更新并通过；Ruff、mypy 和 compileall 不因本次改动退化。
