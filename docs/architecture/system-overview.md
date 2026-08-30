# 系统总体设计

更新时间：2026-07-28。

本文说明 Hiro 解决什么问题、端到端链路如何工作，以及当前能力处于什么成熟度。
代码级依赖关系见[模块设计](module-design.md)，完整软件边界见
[架构说明](../../ARCHITECTURE.md)。

## 1. 产品定位

Hiro 不是让大语言模型直接猜测电池寿命，而是把真实数据、正式模型、区间校准、
审计账本和受约束 Agent 组成一条可追溯的数值链。

系统必须让每个 RUL、SOH、区间和决策数字回答：

1. 输入来自哪个数据版本、`cell_id` 划分和 record batch；
2. 使用哪个 active route、模型 checkpoint 和 calibration evidence；
3. 由哪个版本化工具计算；
4. 结果是否作为 `ToolResult` 原子登记；
5. 哪个用户、项目、Agent run 或报告使用过它。

LLM 只能规划工具调用和解释已经登记的结果，不能生成业务数值。

## 2. 当前目标与非目标

### 当前目标

当前交付目标是一个可复现、可审计的单机比赛演示：

```text
HTTPS 登录
→ 项目与 record batch
→ 人工激活正式模型路由
→ READY calibration materialization
→ RUL / SOH / Split Conformal
→ ToolResult 与 Agent run
→ 审计报告与 Next.js UI
```

该演示需要证明真实代码、数据库、消息队列、Worker、模型制品和浏览器界面能够贯通，
而不是只展示离线截图。

### 当前非目标

以下能力不属于个人比赛演示的完成条件：

- 多可用区、高可用和自动扩缩容；
- 企业 KMS、SIEM、灾难恢复和正式 SLA；
- 连接真实 BMS/EMS 并直接控制设备；
- 未经授权的外部或企业数据验证；
- 使用 PyBaMM 生成长期退化标签；
- 将受约束工作流宣称为自主多智能体协商。

## 3. 端到端技术路线

```mermaid
flowchart LR
    A["真实电芯数据"] --> B["来源、Schema、单位与 SHA-256"]
    B --> C["cell_id 隔离与冻结版本"]
    C --> D["Advanced artifact v2"]
    D --> E["人工激活 Active Route"]
    E --> F["Route-specific Calibration"]
    F --> G["RUL / SOH / Split Conformal"]
    G --> H["ToolResult 与 Audit Ledger"]
    H --> I["Exact AgentStep 与 Fenced Worker"]
    I --> J["API / 报告 / Next.js UI"]
    J --> K["HTTPS 比赛演示"]
    K -.后续研究.-> L["跨域、删失感知与企业集成"]
```

### 各阶段作用

| 阶段 | 核心作用 | 失败时的行为 |
| --- | --- | --- |
| 数据治理 | 冻结来源、语义、单位和 `cell_id` 划分 | 拒绝未登记或哈希不一致的输入 |
| Advanced artifact v2 | 将训练结论绑定到具体安全 checkpoint | 拒绝缺少清单、上下文或 SHA-256 的制品 |
| Active Route | 由人工决定当前项目使用哪个候选和目标角色 | 不从验证排名自动激活 |
| Calibration materialization | 为当前路由生成独立校准证据 | 路由变化后旧证据变为 `STALE` |
| 数值工具 | 计算 target-aware RUL、有限时域 SOH 和区间 | 不确定或域外输入显式降级 |
| ToolResult / Ledger | 保存值、单位、版本、输入哈希和来源 | 原子提交失败则全部回滚 |
| Agent / Worker | 按持久步骤编排工具并支持崩溃恢复 | 旧 fence 不能覆盖新 claim |
| API / UI / 报告 | 安全展示已登记结果 | 不在客户端二次计算业务值 |

## 4. 主要业务流程

### 4.1 项目预测

```text
用户选择项目与 record batch
→ 服务端验证会话、成员关系和项目状态
→ 从数据库解析当前 active route
→ 从受管目录验证模型清单与 checkpoint
→ 按 cell_id 构造目标输入
→ 运行 RUL 或 SOH 工具
→ 在项目账本登记 ToolResult
→ API、Agent 与报告只引用 result_id
```

客户端只提交身份，不提交本地路径、模型哈希或自行构造的预测数组。

### 4.2 Calibration materialization

```text
ADMIN identity-only POST
→ PENDING
→ Redis 只传 materialization_id
→ Worker claim 与递增 fence
→ 服务端重建项目、路由、数据和模型上下文
→ 生成 calibration ToolResult
→ 原子提交 sample bindings 与 READY
```

状态为 `PENDING / RUNNING / READY / FAILED / STALE`。RUNNING lease 过期后可以恢复；
旧 Worker 即使稍后返回，也不能写入结果或失败状态。

### 4.3 Agent 执行

```text
持久 AgentRun intent
→ 按 ordinal 读取持久 AgentStep
→ 服务端冻结工具、输入哈希、依赖、路由和校准证据
→ Worker 执行 exact step
→ ToolResult 与 step 完成原子提交
→ 生成审计报告
```

Agent 是受角色和白名单约束的工作流，不依靠多个聊天机器人自由讨论。

## 5. 数据与算法主线

### RUL

- 目标明确保留 MATR official cycle-life 语义；
- cutoff 为 20、50、100、150；
- 正式模型包括 CyclePatch Direct 与 CyclePatch-BatLiNet；
- 剩余循环由目标周期与 cutoff 的关系派生，不另造独立标签。

### SOH

- Current Hybrid 与 HybridPatch-v2 预测有限时域轨迹；
- 正式监督轴截至真实 cycle 500；
- 轨迹指标使用 SOH 比例或 percentage points；
- 单调结构用于降低非物理回升，但不能替代跨域验证。

### 不确定性

- 校准集与测试集按 `cell_id` 隔离；
- Split Conformal 证据绑定 route、cutoff、模型、数据和项目；
- 当前正式证据不等于删失感知或跨域覆盖保证。

正式实验结果见 [Advanced Benchmark](../benchmark.md)，适用边界见
[Advanced 模型卡](../model-card-advanced.md)和[已知限制](../limitations.md)。

## 6. 运行拓扑

### 运行入口

`deploy/local.compose.yaml` 和 `deploy/competition.compose.yaml` 分别装配本机与服务器
完整运行时；飞书/Aily 与 MCP 通过显式 override 启用。

### Competition

`deploy/competition.compose.yaml` 定义：

- PostgreSQL；
- Redis；
- 一次性 migration；
- FastAPI API；
- 单并发 Celery Worker；
- Next.js frontend；
- Nginx HTTPS gateway。

PostgreSQL 与 Redis 只位于内部 `backend` 网络，只有 gateway 映射 80/443。
后端服务以只读根文件系统、`no-new-privileges`、secret file 和显式资源上限运行。

## 7. 成熟度

状态定义：

- `Implemented`：代码、契约或配置已存在；
- `Validated`：已有自动化测试、正式实验或受控 E2E 证据；
- `Published`：镜像已发布并获得不可变 digest；
- `Deployed`：目标 ECS 上真实运行；
- `Demonstrated`：公网浏览器完整流程通过；
- `Planned`：尚未实现或仍需外部输入。

| 能力 | 当前状态 | 证据或剩余门禁 |
| --- | --- | --- |
| MATR 数据与固定划分 | Validated | 三批 140 个电芯，按 `cell_id` 隔离 |
| 80 次 Advanced Final | Validated | 4 模型 × 4 cutoff × 5 seeds |
| artifact v2、路由与回退 | Validated | 服务、数据库和哈希测试 |
| RUL / SOH / Conformal 工具链 | Validated | API、Agent、报告和受控 E2E |
| Calibration API/Worker/UI | Validated | claim、recovery、fence 和纵向测试 |
| Competition Compose | Implemented | 尚需在目标 ECS 解析、拉取并启动 |
| ACR 镜像 | Implemented | workflow 已存在；成功发布和 digest 待确认 |
| 公网 HTTPS 产品 | Planned | ECS 基础设施已准备，应用栈未验收 |
| 外部域零样本与删失感知 | Planned | 需要合规数据和新实验 |
| 企业 BMS/EMS 与 HA | Planned | 不属于当前比赛目标 |

最新事实以[项目状态](../status.md)为准。

## 8. 比赛演示完成标准

只有同时满足以下条件，才可以标记 `Demonstrated`：

1. 五个私有镜像有不可变 digest；
2. ECS migration 成功，所有服务健康；
3. ADMIN、项目和 record batch 由真实数据库保存；
4. 正式候选注册并由 ADMIN 激活；
5. calibration materialization 到达 `READY`；
6. 浏览器完成 RUL、SOH、Conformal、Agent 和报告流程；
7. 页面上的每个业务数字可追溯到 `result_id`；
8. 重启后数据仍在，错误输入与旧 route 失败关闭；
9. 未公开 PostgreSQL、Redis、秘密文件或模型绝对路径。

## 9. 路线图

```text
P0  ACR 发布与 digest 冻结
P1  ECS Compose、migration 与健康检查
P2  模型候选注册、人工激活与 calibration READY
P3  公网浏览器纵向 E2E、重启与回退演练
P4  README、API、算法、ADR、复现和演示材料
P5  外部域、删失感知、消融与尾部误差研究
P6  企业数据、BMS/EMS、监控、HA 与合规
```

P0–P4 构成个人比赛闭环；P5 是研究增强；P6 是企业化路线，不能混写成当前能力。
