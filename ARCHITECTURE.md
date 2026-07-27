# 架构说明

## 架构目标

泉芯智寿的首要约束不是“调用更多模型”，而是保证每个业务数字都能回答：

1. 来自哪个数据、划分、特征、模型和校准队列；
2. 由哪个已授权工具计算；
3. 使用了哪个 active route 和具体制品；
4. 是否经过完整性验证和审计登记；
5. 在失败、重试、回退和并发 Worker 下是否仍保持同一语义。

LLM 不能生成 SOH、RUL、区间、阈值比较或经营数字。它只能编排工具和解释已经登记的
`ToolResult`。

## 核心数值流

```text
受控数据与清单
  │  来源 / schema / cell_id split / SHA-256
  ▼
Record Batch 与 Advanced artifact v2
  │  服务端身份，不接受客户端路径和哈希
  ▼
Active Route Resolver
  │  task + cutoff + role + approved artifact
  ├── RUL POINT_ACCURACY
  ├── RUL COVERAGE
  ├── SOH MEAN_ACCURACY
  └── SOH TAIL_EFFICIENCY
  ▼
READY Calibration Materialization
  │  calibration ToolResult + manifest SHA-256
  ▼
项目级数值工具
  │  target-aware RUL / finite-horizon SOH /
  │  route-specific Split Conformal
  ▼
ToolResult + Audit Ledger
  │  结果身份、值、单位、来源和告警
  ▼
Agent exact-step / 报告 / API / UI
```

任何环节缺少或冲突都必须失败关闭。

## 逻辑分层

### 1. 核心契约层

`quanxin_life.core` 定义公共枚举、Pydantic 输入输出、`ToolResult` 和错误语义。
核心模块不在导入时连接数据库、网络或加载模型。

职责：

- 统一 RUL、SOH、模型任务、路由角色和状态语义；
- 验证有限数值、单位和结构；
- 禁止不同 API 或 Agent 另造等价 DTO；
- 为基础设施适配器提供稳定边界。

### 2. 数据与制品层

职责：

- MATR/HUST/Naumann 来源登记；
- Canonical schema 和版本化 Parquet；
- 以 `cell_id` 为单位的 train/validation/calibration/test 划分；
- Advanced Final 输出索引和 artifact v2；
- safetensors、JSON、Parquet、文件大小和 SHA-256 复验；
- 拒绝 pickle、joblib、`.pt`、`.pth`、符号链接和路径逃逸。

数据文件、模型权重和大型结果不写入数据库，也不通过客户端路径定位。
数据库保存身份和受管 URI，字节留在受控文件系统或对象存储。

### 3. Application 服务层

服务层协调领域契约和外部适配器，不依赖 FastAPI 请求对象。

关键服务包括：

- Advanced suite 安全导入与注册；
- deployment bundle 与 managed artifact catalog；
- 模型路由激活、回退和 active-route resolver；
- 项目 record batch 与 ToolResult binding；
- calibration evidence 验证、物化、claim/recovery；
- Agent 执行上下文解析；
- 审计报告导出。

服务端只接收业务身份，例如 `project_id`、`record_batch_id`、
`materialization_id`。本地路径、哈希、样本数组和模型指标不由浏览器提交。

### 4. 数值工具层

工具层是唯一可以产生业务数值的边界。

正式 Advanced 工具：

- target-aware MATR official RUL；
- finite-horizon SOH trajectory；
- route-specific Split Conformal；
- Advanced 单电芯审计报告。

工具输入由服务端冻结，输出必须：

- 通过 Pydantic 校验；
- 使用明确单位；
- 包含模型、数据、特征、划分、路由和校准身份；
- 包含适用范围、告警和拒绝理由；
- 作为 `ToolResult` 原子登记。

### 5. 受约束专业智能体工作流

当前 Agent 是 role-based、policy-constrained workflow，不宣称自主协商或智能涌现。

流程：

```text
持久 AgentRun intent
→ 服务端解析 target batch
→ 持久 AgentStep 按 ordinal 排序
→ exact AgentStep 绑定工具和输入
→ Supervisor 校验角色、依赖和审批
→ Worker 执行
→ ToolResult 与 step 完成原子提交
```

`AgentStep` 不能在 Worker 运行时被客户端改写。服务端冻结：

- canonical input hash；
- 工具名和工具版本；
- 上游依赖；
- active route；
- calibration materialization；
- 项目授权；
- 人工审批证据。

### 6. API、报告与 UI

FastAPI 适配器只负责：

- 认证、授权和项目可见性；
- Origin、幂等键和请求契约；
- 映射应用服务错误；
- 返回安全摘要，不暴露绝对路径或秘密。

报告只消费审计账本中的 ToolResult。PDF/DOCX 是同一审计 Markdown 的格式视图，
不能重新计算业务值。

Next.js UI 只消费 API：

- 项目、record batch、Agent run 和结果；
- 单电芯 RUL/SOH/Conformal；
- calibration materialization 状态；
- 报告与告警。

UI 不加载模型，不计算指标，也不接收“示例数值”代替后端结果。

## Active Route 与 Advanced artifact v2

路由分离聚合研究结论和具体可执行制品：

```text
Promotion Report（聚合、CONDITIONAL）
→ representative seed（只看 validation）
→ artifact v2（具体 checkpoint）
→ managed candidate registry
→ 人工 activation ledger
→ active route stream head
```

路由身份至少包含：

- task；
- cutoff；
- role；
- candidate；
- representative seed；
- artifact；
- activation decision；
- route stream revision。

回退追加新决策，不覆盖历史记录。Resolver 每次使用数据库中的当前 stream head，
不信任客户端缓存。

## Calibration Materialization

Conformal 运行前，需要把项目当前路由对应的 calibration 证据物化为 READY 记录：

```text
ADMIN identity-only POST
→ PENDING record
→ Redis payload: materialization_id
→ Worker claim
→ 服务端重建项目、路由和数据上下文
→ 读取受控 manifest / split / Parquet
→ 产生 calibration ToolResult
→ ProjectMaterializationCommit
→ READY
```

状态：

- `PENDING`：等待 claim；
- `RUNNING`：有有效 lease 的 Worker 正在执行；
- `READY`：证据已完整提交，可供 Agent resolver 使用；
- `FAILED`：当前有效 fence 内的执行失败；
- `STALE`：active route 或绑定上下文已变化，旧证据不可继续使用。

客户端不能提交 calibration 样本 ID、数组、路径或哈希。Agent resolver 必须找到一个
与 task、cutoff、route 和项目上下文精确匹配的 READY materialization。

## 并发、原子性与崩溃恢复

### Fenced claim

Worker claim 产生递增 fence token 和 lease。所有完成或失败写入都必须同时匹配：

- materialization identity；
- claim identity；
- fence token；
- 当前状态。

lease 过期后，新 Worker 可以 recovery claim。旧 Worker 即使稍后返回，也不能以旧
fenced claim 覆盖新结果或写入 FAILED。

### 原子提交

成功执行必须在同一事务中：

1. 登记 `ToolResult`；
2. 建立项目 ToolResult binding；
3. 写入 materialization/sample binding；
4. 标记 AgentStep 或 materialization 完成；
5. 提交 ledger 关联身份。

若任一写入失败，整个事务回滚。系统不能出现“step 已完成但结果不存在”或
“结果存在但不属于项目”的半提交状态。

### 幂等

- API 使用显式 `Idempotency-Key`；
- 相同身份和上下文返回原记录；
- 相同键但不同请求拒绝；
- 重试只重新入队 materialization identity；
- Worker 从数据库重建上下文，不复用不可信消息体。

## 信任边界

| 边界 | 主要风险 | 控制 |
| --- | --- | --- |
| 数据文件 | 篡改、格式欺骗、泄漏 | manifest、文件头、大小、SHA-256、`cell_id` split |
| 模型制品 | 可执行反序列化、错配 | safetensors/JSON、artifact v2、禁用格式 |
| 浏览器/API | 越权、CSRF、路径注入 | session、project ACL、trusted Origin、identity-only DTO |
| Redis/Celery | 消息篡改、旧任务回写 | 只传 ID、服务端重建、lease、fenced claim |
| LLM/Agent | 编造数字、越权工具 | exact AgentStep、角色白名单、ToolResult-only |
| 报告/UI | 二次计算、断章取义 | ledger-bound render、单位、告警、来源身份 |
| MCP/飞书/工业协议 | 外部不可信输入 | 版本契约、签名、幂等、沙箱、显式降级 |

日志不得记录密钥、原始敏感数据、完整用户提示或本地绝对模型路径。

## 组合根与可选依赖

`src/quanxin_life/application/assembly.py` 显式组装工具、项目预测和 calibration
组件。高阶依赖在函数内部延迟导入，核心包导入不触发 Torch、PyArrow、数据库、
网络或 Worker 初始化。

主要组合入口：

- `create_competition_tool_registry`；
- `create_competition_tool_invocation_service`；
- `create_project_prediction_tool_registry`；
- `create_project_prediction_tool_invocation_service`；
- `create_advanced_calibration_components`；
- `create_competition_fastapi_app`。

若某项依赖未配置，其 API 不应出现或必须显式返回不可用；不得用随机数、内存假实现或
示例数值冒充完整能力。

## 部署边界

当前仓库包含两种运行形态：

### Foundation API

`deploy/foundation_api.py` 和 `deploy/compose.yaml` 可直接启动健康检查和基础工具入口。
它不包含完整项目认证、数据库路由、Worker、对象存储或 Advanced 数值链。

### 完整应用装配

代码和测试装配已存在，但目标环境仍需提供：

- PostgreSQL 与 Alembic migrations；
- Redis 和 Celery Worker；
- 受管制品目录或对象存储；
- 认证、项目、record batch 和审批数据；
- 完整 FastAPI 组合根；
- Next.js 环境配置；
- 监控、TLS、备份、恢复和密钥管理。

因此“纵向 E2E 已验证”不等于“生产部署已完成”。部署链将在独立设计中决定，
不在架构文档中假设外部基础设施已经存在。

## 扩展能力

以下能力通过同一 ToolResult 和权限边界接入，但不是当前核心数值主线：

- PyBaMM 短时边界核验；
- 主动试验；
- 审核知识库；
- MCP；
- 飞书协同；
- REST/MQTT/Modbus/EMS 协议沙箱。

扩展组件不可绕过模型、数据、路由、校准和审计约束。

## 相关文档

- [项目状态](docs/status.md)
- [Advanced Benchmark](docs/benchmark.md)
- [Advanced 模型卡](docs/model-card-advanced.md)
- [已知限制](docs/limitations.md)
- [可复现性](docs/reproducibility.md)
- [运行与装配指南](docs/runtime-setup.md)
- [数据契约](DATA_CONTRACT.md)
- [实验协议](EXPERIMENT_PROTOCOL.md)
