# Advanced 模型卡

版本：Advanced artifact v2，2026-07-27。

本模型卡覆盖当前正式 Advanced 路由使用的四个模型。它描述模型角色、输入输出、
制品要求和拒绝条件，不构成模型自动激活批准。

## 共同任务定义

| 项目 | RUL | SOH |
| --- | --- | --- |
| 目标 | MATR 官方 cycle life | 真实有限时域 SOH 轨迹 |
| cutoff | 20 / 50 / 100 / 150 cycles | 20 / 50 / 100 / 150 cycles |
| 输出 | cycle-life 点预测 | 从 cutoff 后到 cycle 500 的轨迹 |
| 区间 | 独立 calibration 上的 route-specific Split Conformal | 当前不发布同口径轨迹覆盖保证 |
| 数据隔离 | 以 `cell_id` 隔离 | 以 `cell_id` 隔离 |

所有模型输入必须来自服务端冻结的项目 record batch、active route 和已验证制品。
客户端不能提交模型路径、SHA-256、calibration 样本数组或服务端文件位置。

## CyclePatch Direct

### 结构

- 充电和放电曲线使用多尺度 PhasePatch 编码；
- 加入真实 cycle 位置和工况条件；
- 使用可学习位置编码与 masked Transformer 聚合跨循环演化；
- 直接回归 MATR 官方 cycle life。

### 角色

- cutoff 20：`DEFAULT`；
- cutoff 50：`POINT_ACCURACY`；
- cutoff 100：`COVERAGE` 候选，但 90% PICP 未达目标；
- cutoff 150：`POINT_ACCURACY`。

### 主要风险

- 正式训练采用小样本 full-batch 优化；
- 长寿命和跨批次样本仍有较大误差；
- Direct 与 BatLiNet 的配对区间跨过零；
- 不能把最低平均 MAE解释为统计显著全面胜出。

## CyclePatch-BatLiNet

### 结构

- 复用 CyclePatch 的时序感知编码；
- 加入冻结、哈希绑定的参考库分支；
- 使用检索到的参考寿命证据辅助回归；
- 参考库不能来自测试标签或运行时未登记路径。

### 角色

- cutoff 50：`COVERAGE`；
- cutoff 100：`POINT_ACCURACY`；
- cutoff 150：`COVERAGE`；
- cutoff 20 未独立承担默认路由。

### 主要风险

- 参考库质量、覆盖和域偏移会影响预测；
- cutoff 增加时显存需求增加；
- 不能把参考检索表述为外部知识泛化；
- cutoff 100 的 90% Split PICP 为 88.89%，未达目标。

## Current Hybrid

### 结构

- 使用平方根、线性、knee 加速和累计非负残差构造退化；
- 结构保证预测轨迹单调不增；
- 采用固定幅度缩放和输出范围裁剪。

### 角色

- 四个 cutoff 均为 `TAIL_EFFICIENCY`；
- 当系统只能保留单条 SOH 路由时，当前策略保留 Current Hybrid。

### 主要风险

- 固定缩放常数具有数据集依赖性；
- `[0, 1.5]` 裁剪可能掩盖未校准输出；
- 不允许测量回弹或容量恢复；
- 平均 MAE 高于 HybridPatch-v2。

## HybridPatch-v2

### 结构

- 多尺度早期曲线编码；
- 学习式正退化幅度、knee 位置和正残差率；
- 累计退化保持轨迹单调；
- 输出仍限制在 `[0, 1.5]`。

### 角色

- 四个 cutoff 均为 `MEAN_ACCURACY`。

### 主要风险

- MAE 较低，但 RMSE 和尾部误差高于 Current Hybrid；
- 远期轨迹误差增加；
- 2017-06-30 批次表现较弱；
- clamp 饱和和跨域稳定性尚未完成专项验证。

## 制品与加载安全

正式服务只接受：

- safetensors 权重；
- JSON 架构、特征和 manifest；
- 经验证的参考库文件；
- 文件大小、相对 URI 和 SHA-256 全部匹配的 artifact v2。

拒绝：

- pickle、joblib、`.pt`、`.pth`；
- 符号链接和路径逃逸；
- 未登记的绝对路径；
- 与 active route 不匹配的 checkpoint；
- manifest、特征、架构或权重哈希变化；
- 被新路由淘汰的 `STALE` calibration materialization。

## 运行时拒绝与降级

以下情况必须拒绝或显式降级：

- 项目、session、record batch 或 active route 不可见；
- 输入 cutoff、目标任务或数据版本不匹配；
- 目标电芯出现在 calibration cohort；
- calibration materialization 不是唯一、精确匹配的 `READY` 记录；
- RUL calibration 含右删失或无官方 cycle-life 的电芯；
- 请求超出 SOH cycle 500 有限边界；
- Conformal route 未达到目标覆盖；
- 输入处于未验证的外部域；
- 业务策略要求人工复核。

## 预期用途

- MATR 同协议研究和比赛展示；
- 可追溯的单电芯辅助分析；
- 模型、区间和失败边界研究；
- 受控研发流程中的候选排序和复检提示。

## 非预期用途

- 直接承诺真实储能系统剩余年限；
- 未经重校准应用到 HUST、Naumann 或企业数据；
- 安全关键控制、质保赔付或无人审批的批次放行；
- 将预测区间解释为参数置信区间；
- 由 LLM 修改、补全或生成模型数值。

正式结果见 [Advanced Benchmark](benchmark.md)，完整限制见
[已知限制](limitations.md)。
