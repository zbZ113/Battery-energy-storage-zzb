# 项目状态

更新时间：2026-07-27。

本页是仓库公开成熟度的单一事实源。`Implemented` 表示代码和契约存在，
`Validated` 表示已有自动化测试、正式实验或受控 E2E 证据，`Planned` 表示尚未完成。
三者不能互相替代。

## 状态摘要

| 能力 | Implemented | Validated | Planned / 未完成 |
| --- | :---: | :---: | --- |
| MATR 来源登记、HDF5 安全读取、Canonical 数据 | 是 | 三批正式数据 | 外部数据持续版本化 |
| `cell_id` 级 train/validation/calibration/test 隔离 | 是 | 140 个电芯的固定划分 | 新数据集重新冻结 |
| Advanced RUL / SOH 正式训练 | 是 | 80 次 A100 Final | 不重复使用冻结 test 调参 |
| 指标闭环与论文图件 | 是 | RUL/SOH 逐样本对账、12 张图 | 投稿材料独立审查 |
| Split / Normalized Conformal | 是 | MATR 12 calibration + 27 test | 扩大 calibration、删失感知、跨域重校准 |
| Advanced artifact v2 与制品注册 | 是 | 哈希、清单、代表 checkpoint | 生产对象存储激活 |
| 人工激活、回退、active route | 是 | 数据库和服务测试 | 生产审批策略 |
| target-aware RUL / finite-horizon SOH | 是 | 工具、API、Agent、报告测试 | 外部域验收 |
| exact `AgentStep` 与原子 ledger | 是 | claim/recovery 测试 | 多副本压力测试 |
| calibration materialization | 是 | ADMIN API、Worker、UI、纵向 E2E | 真实 Worker 基础设施部署 |
| Next.js 项目门户 | 是 | 单元测试、类型检查、构建、E2E | 真实环境配置与运营验收 |
| 完整服务栈 | 组件存在 | 测试环境装配 | 一键 Compose、监控、备份、密钥管理 |
| HUST 外部验证 | 接入与安全转换组件存在 | 尚无正式零样本结果 | 零样本、重校准、域适配 |
| 工业 BMS/EMS | 协议沙箱存在 | 沙箱测试 | 企业凭证、网络、设备和安全联锁 |

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

## 未完成事项

### 研究

1. HUST 零样本外部验证和目标域重校准；
2. 右删失数据的 survival baseline、IPCW 或删失感知 Conformal；
3. 更大的 calibration cohort；
4. CyclePatch 位置、工况、Transformer 和 BatLiNet 分支消融；
5. HybridPatch-v2 尾部误差、clamp 饱和和批次偏移诊断；
6. 参数量、单样本推理延迟、吞吐和并发基准。

### 部署

1. 完整 PostgreSQL / Redis / Celery Worker / 对象存储 / API / UI Compose；
2. 生产配置、密钥轮换、TLS、监控、告警和审计留存；
3. 备份恢复、灾难恢复和容量规划；
4. 真实模型制品与真实 calibration 证据在目标环境中的激活验收；
5. 浏览器到 Worker 的真实环境冒烟和故障演练。

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
[可复现性](reproducibility.md)。
