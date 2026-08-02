# 项目状态

更新时间：2026-08-02。

本页是仓库公开成熟度的单一事实源。`Implemented` 表示代码和契约存在，
`Validated` 表示已有自动化测试、正式实验、哈希验收或受控 E2E 证据，
`Published` 表示不可变镜像或发布包已经进入目标 registry，`Deployed` 表示已经在
目标环境启动并通过服务验收，`Demonstrated` 表示公网浏览器真实业务路径已贯通，
`Planned` 表示尚未完成。这些状态不能互相替代。

## 状态摘要

| 能力 | Implemented | Validated | Deployed | Planned / 未完成 |
| --- | :---: | :---: | :---: | --- |
| MATR 来源登记、HDF5 安全读取、Canonical 数据 | 是 | 三批正式数据 | 不适用 | 外部数据持续版本化 |
| `cell_id` 级 train/validation/calibration/test 隔离 | 是 | 140 个电芯的固定划分 | 不适用 | 新数据集重新冻结 |
| Advanced RUL / SOH 正式训练 | 是 | 80 次 A100 Final | 不适用 | 不重复使用冻结 test 调参 |
| 指标闭环与论文图件 | 是 | RUL/SOH 逐样本对账、12 张图 | 不适用 | 投稿材料独立审查 |
| Split / Normalized Conformal | 是 | MATR 12 calibration + 27 test | 否 | 扩大 calibration、删失感知、跨域重校准 |
| Advanced artifact v2 与制品注册 | 是 | 哈希、清单、代表 checkpoint | 否 | 目标 ECS 激活 |
| 人工激活、回退、active route | 是 | 数据库和服务测试 | 否 | 目标 ECS 人工审批验收 |
| target-aware RUL / finite-horizon SOH | 是 | 工具、API、Agent、报告测试 | 否 | 公网真实模型验收 |
| exact `AgentStep` 与原子 ledger | 是 | claim/recovery 测试 | 否 | 多副本压力测试 |
| calibration materialization | 是 | ADMIN API、Worker、UI、纵向 E2E | 否 | 真实 Worker 与证据部署 |
| Next.js 项目门户 | 是 | 单元测试、类型检查、构建、E2E | 否 | 公网浏览器验收 |
| 竞赛单机服务栈 | Compose、API、Worker、迁移、网关已实现 | 完整 CI、真实 PostgreSQL、不可变 ACR 构建 | 否（公网 edge 可达，ECS 内部未对账） | RepoDigest、Compose、备份与演练 |
| HUST 外部验证 | 接入与安全转换组件存在 | 尚无正式零样本结果 | 否 | 零样本、重校准、域适配 |
| 工业 BMS/EMS | 协议沙箱存在 | 沙箱测试 | 否 | 企业凭证、网络、设备和安全联锁 |

## 已验证的科学证据

### 数据与划分

- 三批 MATR 共 140 个电芯；
- 138 个电芯具有可用官方 cycle-life 标签；
- 固定划分为 train 82、validation 19、calibration 12、test 27；
- SOH 正式测试使用 25 个满足轨迹条件的测试电芯；
- 所有划分以 `cell_id` 为单位，不按行或周期随机拆分。

### 正式训练

```text
cutoff：20 / 50 / 100 / 150
模型：CyclePatch Direct / CyclePatch-BatLiNet /
      Current Hybrid / HybridPatch-v2
随机种子：38 / 39 / 40 / 41 / 42
正式运行：80
```

- 21 个运行跑满预设 epoch；
- 59 个运行按冻结规则提前停止；
- Final 目录含 2,654 个被输出索引覆盖的文件；
- 结果包含 897 个 safetensors 文件；
- 训练/验证日志、测试指标和安全制品均有 SHA-256 证据。

正式指标见 [Advanced Benchmark](benchmark.md)。

### 产品代码链

当前测试环境已经验证：

```text
身份认证
→ 项目和 record batch
→ active-route-aware runtime resolver
→ READY calibration materialization
→ target-aware RUL / SOH
→ route-specific Split Conformal
→ ToolResult
→ exact AgentStep
→ 审计报告
→ Next.js UI
```

纵向 E2E 使用真实应用装配、认证 API、数据库记录、manifest/split/Parquet/SHA
解析和报告账本。模型前向边界使用显式测试适配器，因此该 E2E 证明工程闭环，
不构成真实部署环境的性能或可用性证明。

## 当前正式路由状态

- 路由结论是 `CONDITIONAL`，需要人工审批；
- 聚合晋级结论不会自动激活具体 checkpoint；
- RUL 分为 `POINT_ACCURACY` 与 `COVERAGE` 目标；
- SOH 分为 `MEAN_ACCURACY` 与 `TAIL_EFFICIENCY` 目标；
- 路由改变会使旧 calibration materialization 变为 `STALE`；
- 服务端从当前 active route 解析模型和校准证据，客户端不能提交路径、哈希或样本数组。

## 当前竞赛发布状态

私有 ACR release `2026.08.02-1` 已由 GitHub Actions run `30739080847` 成功发布并
记录五个不可变镜像 digest，来源提交为
`daca47a7d0a2f8e82a7c549b6b97d0f25b5596a7`，前端公开 Origin 为
`https://47.99.69.138`。发布前同一提交的 Python、真实 PostgreSQL 和前端 CI 全部通过。

这使竞赛镜像状态达到 `Published`，但尚未达到 `Deployed`：公网严格 TLS edge 可达，
SSH 22 在认证前连接超时，目标 ECS 的 RepoDigest、Compose、migration、Worker、正式
模型/calibration、九步 ToolResult 和浏览器 E2E 仍未对账。完整发布记录见
[ACR release `2026.08.02-1`](deployment/releases/2026.08.02-1.md)。历史 release
`2026.07.28-1` 保留原 source commit、旧 IP 和 digest，不再代表当前源码。

## 未完成事项

### 研究

1. HUST 零样本外部验证和目标域重校准；
2. 右删失数据的 survival baseline、IPCW 或删失感知 Conformal；
3. 更大的 calibration cohort；
4. CyclePatch 位置、工况、Transformer 和 BatLiNet 分支消融；
5. HybridPatch-v2 尾部误差、clamp 饱和和批次偏移诊断；
6. 参数量、单样本推理延迟、吞吐和并发基准。

### 部署

1. 恢复受控 SSH/ECS 管理通道，并对账 release `2026.08.02-1` 的五个 RepoDigest；
2. 在目标 ECS 拉取镜像、执行 Alembic `0001`–`0015` 并启动竞赛 Compose；
3. 初始化首个 ADMIN、项目、15 个正式候选和人工 active route；
4. 部署真实 calibration evidence 并由 Worker 生成 `READY` 物化证据；
5. 完成浏览器到 Worker、ToolResult、Agent、报告的公网纵向 E2E；
6. 完成重启、route 回退、失败恢复、数据库/Redis 备份与恢复演练；
7. 补充监控、告警、审计留存和密钥轮换。企业级 HA 与灾难恢复不阻塞个人比赛演示。

### 外部输入

真实部署或工业验证需要使用方提供：

- 目标环境、域名、TLS 和网络边界；
- PostgreSQL、Redis、对象存储和备份策略；
- 密钥管理与身份提供方；
- 经授权、脱敏且带来源清单的企业数据；
- BMS/EMS 协议、设备映射和安全联锁；
- 决策策略、拒绝条件和人工审批责任人；
- HUST 或其他外部数据的合规获取和使用确认。

具体限制见 [已知限制](limitations.md)，复现要求见
[可复现性](reproducibility.md)。系统全景和比赛完成标准见
[系统总体设计](architecture/system-overview.md)，目标环境步骤见
[竞赛 ECS 部署](deployment/competition-ecs.md)。
