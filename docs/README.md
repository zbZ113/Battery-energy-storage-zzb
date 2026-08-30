# Hiro 文档

本目录只保留当前代码、模型证据、运行部署和外部集成需要的文档。比赛提交稿、阶段计划、一次性审计快照和被替代的发布记录不随产品仓库维护。

## 从这里开始

| 主题 | 文档 |
| --- | --- |
| 当前成熟度 | [项目状态](status.md) |
| 本地与服务器运行 | [运行与部署](runtime-setup.md) |
| 已知限制 | [限制与拒绝条件](limitations.md) |
| 复现实验 | [可复现性](reproducibility.md) |

## 架构与代码

- [仓库架构](../ARCHITECTURE.md)
- [系统总体设计](architecture/system-overview.md)
- [模块边界](architecture/module-design.md)
- [代码走读](development/code-walkthrough.md)
- [HTTP API](api/README.md)
- [Architecture Decision Records](adr/README.md)

## 算法与证据

- [算法说明](algorithms/README.md)
- [Advanced Benchmark](benchmark.md)
- [Advanced 模型卡](model-card-advanced.md)
- [Naumann 数据卡](data-cards/naumann-public-data-v1.md)
- [公开图件与来源数据](assets/benchmark/advanced-final-20260723/manifest.json)

## 飞书与 Aily

- [飞书与 Aily 模型工具联动](integrations/feishu-aily-model-tools.md)
- [Aily MCP HTTPStreaming](integrations/aily-mcp-httpstreaming.md)
- [Aily MCP 系统提示词](integrations/aily-mcp-system-prompt.md)
- [Aily Connector OpenAPI](integrations/aily-connector-openapi.yaml)

## 部署与运维

- [竞赛 ECS 部署](deployment/competition-ecs.md)
- [运维 Runbook](deployment/operations-runbook.md)
- [安全与 Secrets](deployment/security-and-secrets.md)
- [当前不可变发布记录](deployment/releases/2026.08.02-1.md)

## 训练与开发参考

- [A100 Advanced 训练指南](training/a100-advanced-model-training-guide.md)
- [MATR 三批 A100 操作手册](training/a100-three-batch-training-runbook.md)
- [版本化工具与数值协议](reference/README.md)
