# 已知限制

本页集中记录当前证据不能支持的结论。任何 README、演示、论文、报告、API 或 UI
表述都不得绕过这些边界。

## 数据与标签限制

### MATR 内部证据

当前正式指标来自 MATR 三批固定划分。三批数据存在协议和寿命分布差异，
2017-06-30 批次在部分 SOH 远期预测中误差较高。总体平均值不能代表每个批次、
协议或寿命区间都具有相同表现。

### RUL 目标语义

当前标量目标是 MATR 官方 cycle life：

- 不能称为统一 EOL80；
- 不能直接换算为真实 15–25 年工业寿命；
- 不能与其他数据集的 EOL 定义混用；
- 不能把回归 R²称为分类准确率。

### SOH 有限时域

SOH 轨迹使用真实监督并结束于 cycle 500：

- 不能宣称可靠预测完整 1000+ cycles 轨迹；
- 未在有限时域跨越阈值不等于已证明长期不会跨越；
- PyBaMM 只做短时工况核验，不能制造长期退化标签。

## 右删失与 Conformal 限制

当前 RUL 校准和覆盖评价要求非右删失的官方 cycle-life：

- 右删失电芯不能进入当前 RUL calibration；
- 当前 PICP/MPIW 只描述 event-observed cohort；
- 不能将该覆盖率外推到删失人群；
- survival-aware baseline、IPCW 和删失感知 Conformal 尚未实现。

校准集只有 12 个电芯：

- 所有区间结果必须带 `SMALL_CALIBRATION_COHORT`；
- 90% 和 95% 在部分条件下使用相同最大有限样本秩；
- 经验 PICP 波动较大；
- cutoff 100 的现有 RUL 候选均未达到目标 90% PICP；
- Normalized Conformal 未稳定改善覆盖—宽度权衡。

## 模型限制

### CyclePatch

- 正式优化采用 small-data full-batch 训练；
- Direct 与 BatLiNet 的逐电芯配对区间跨过零；
- 长寿命、少数批次和最差电芯仍存在较大误差；
- 训练耗时与峰值显存不是推理延迟或服务吞吐。

### CPMLP

CPMLP 是 legacy 曲线集合基线。其掩码聚合不显式表达跨循环顺序，
不能代表当前 CyclePatch Advanced 主线。

### Current Hybrid

- 固定幅度缩放具有数据集依赖性；
- clamp 可能把超范围输出压回合法区间；
- 强单调结构不表达测量回弹或容量恢复。

### HybridPatch-v2

- 平均 MAE 较低，但 RMSE 和尾部风险没有同步改善；
- 远期误差增加；
- `[0, 1.5]` 裁剪仍需饱和诊断；
- 不能仅凭平均 MAE 宣称为唯一 SOH 冠军。

## 外部泛化限制

当前仅有 MATR 的正式深度模型证据；Naumann/280Ah 结果属于独立物理参考情景：

- 没有正式零样本指标；
- 没有目标域 calibration 或覆盖保证；
- 当前仓库未保留目标域适配实现，跨域能力必须重新建立独立证据；
- Naumann 数据和企业数据也没有当前 Advanced 路由的正式结果。

任何跨域使用都应先：

1. 校验来源、许可、字段、单位和协议；
2. 按 `cell_id` 独立划分；
3. 运行域偏移检查；
4. 使用独立目标域 calibration；
5. 重新报告点误差、PICP、MPIW 和拒绝率；
6. 经人工审批后建立新 active route。

## 工程与产品限制

### 测试闭环不等于生产部署

当前 Advanced API / Agent / 报告 / UI 和 calibration materialization 已通过受控
纵向 E2E，但这不等于生产部署：

- `deploy/local.compose.yaml` 与 `deploy/competition.compose.yaml` 已实现 PostgreSQL、Redis、migrations、
  API、Worker、Next.js 和 Nginx HTTPS 拓扑，并通过部署契约测试；
- 目标 ECS、Docker、UFW、ACR 和正式 IP TLS 前置设施已经建立，但五个应用镜像、
  Compose、数据库迁移、模型路由和 calibration evidence 尚未完成公网纵向验收；
- 当前竞赛拓扑使用宿主机受管目录和 Docker secrets file，不等于企业级 KMS、
  对象存储或多节点密钥管理已经验收；
- 没有容量、并发、故障注入、灾难恢复或长期运行结果；
- E2E 的模型前向边界使用显式测试适配器。

### 工业接口

REST、MQTT、Modbus 和 EMS 是协议沙箱：

- 不代表生产 BMS/EMS 已接入；
- 不包含企业凭证、现场网络和设备安全联锁；
- 不能下发安全关键控制；
- 真实接入必须由企业提供协议、责任边界和验收环境。

### Agent 能力

当前系统是受约束专业智能体工作流：

- 使用精确 `AgentStep`、角色白名单和冻结依赖；
- 不宣称多智能体自主协商、涌现或无限制动态规划；
- LLM 不生成 SOH、RUL、区间或决策数值；
- 缺少已登记 `ToolResult` 时必须拒绝，而不是补写结果。

## 制品与溯源限制

- 正式离线结果包未包含 `mlruns`，不能声称 MLflow 存储随结果包交付；
- A100 训练时 `input_bundle_sha256` 与本地重建清单哈希不同；
- 逐样本复算已与 A100 指标对账，但该来源差异仍必须保留；
- 大型结果包和原始数据不进入 Git；
- 公开仓库不能独立重建私有或受控数据字节。

## 禁止性表述

当前证据不支持：

- “工业生产系统已经上线”；
- “达到 SOTA”；
- “对任意电池都有 90%/95% 覆盖保证”；
- “支持右删失寿命区间”；
- “HybridPatch-v2 全面优于 Current Hybrid”；
- “CyclePatch Direct 统计显著优于 BatLiNet”；
- “A100 本身构成算法创新”；
- “真实 BMS/EMS 已接入”；
- “训练显存等于部署显存，训练耗时等于推理延迟”。

当前正式结果见 [Advanced Benchmark](benchmark.md)，模型拒绝策略见
[Advanced 模型卡](model-card-advanced.md)。
