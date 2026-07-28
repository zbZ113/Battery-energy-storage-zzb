# Architecture Decision Records

Architecture Decision Record（ADR）记录会长期影响数据语义、安全边界、运行时和部署的
架构决策。ADR 解释“为什么这样做”，具体接口和操作步骤仍以代码、API 文档和部署手册为准。

## 状态

- `Proposed`：讨论中，尚未成为约束；
- `Accepted`：当前有效决策；
- `Superseded`：已被后续 ADR 替代；
- `Deprecated`：仍可能存在兼容代码，但不再推荐。

ADR 一旦接受不原地改写结论。需要改变决策时，新建 ADR，并在双方文档中建立
`Supersedes / Superseded by` 链接。

## 决策索引

| ADR | 决策 | 状态 |
| --- | --- | --- |
| [ADR-0001](0001-numeric-audit-firewall.md) | 业务数值必须通过 ToolResult 数值防火墙 | Accepted |
| [ADR-0002](0002-cell-id-splitting.md) | 数据集必须按 `cell_id` 隔离 | Accepted |
| [ADR-0003](0003-safe-model-artifacts.md) | 模型只使用可验证的非执行制品 | Accepted |
| [ADR-0004](0004-active-route-and-calibration-materialization.md) | 使用 active route 与 READY calibration materialization | Accepted |
| [ADR-0005](0005-postgresql-as-system-of-record.md) | PostgreSQL 是正式系统记录源 | Accepted |
| [ADR-0006](0006-acr-and-ecs-competition-deployment.md) | 比赛采用私有 ACR 和单机 ECS Compose | Accepted |

## 新增 ADR 的最低内容

每份 ADR 至少包含：

1. 状态和日期；
2. 背景与问题；
3. 决策；
4. 备选方案；
5. 后果与风险；
6. 代码或测试证据；
7. 当前部署成熟度。

状态标签的含义见[系统总体设计](../architecture/system-overview.md#7-成熟度)。
