# 泉芯智寿 A100 真实训练与评价流水线实施计划

## 目标与锁定决策

建立从原始数据、特征生成、模型训练、验证早停、断点恢复、最终测试、实验汇总到安全制品导出的真实流水线。

- 仅使用物理 GPU 1；Shell 设置 `CUDA_VISIBLE_DEVICES=1`，PyTorch 内部使用 `cuda:0`。
- MATR 首轮预测目标为独立的 `MATR_OFFICIAL_CYCLE_LIFE`，不得冒充统一 EOL80。
- CPMLP、Hybrid、DANN 最大训练轮数分别为 300、500、300；每 5 epoch 验证，连续 10 次验证无改善早停。
- XGBoost 最大 2000 boosting rounds，100 rounds 无改善早停；Dummy/Variance 无 epoch。
- Hybrid 只使用实测 `QDischarge`，统一预测到循环 500，不制造外推标签。
- 采用 Smoke 单种子检查和四截断点、五种子的正式实验。
- 终端、JSONL、CSV 与本地文件型 MLflow 同时记录。
- Conda Python 3.11 独立环境，通过 Shell/tmux 启动。
- A100 训练包包含来源已登记、哈希已核验的原始 MATR `.mat`。
- 每个数据集一条命令运行语义适用的模型套件。

## 数据与公共契约

### 目标语义

公共目标类型至少区分：

```text
UNIFIED_EOL80_CYCLE
MATR_OFFICIAL_CYCLE_LIFE
```

通用预测契约包含 `target`、`predicted_cycle`、`observed_cycle`、`right_censored`、电芯身份、截断点及数据/特征/划分/模型版本。现有 `LifePrediction` 保留为 EOL80 兼容接口，正式报告必须显示目标名称。

本批 MATR 中，训练、验证、校准、测试分别有 26、5、4、9 个可用官方 cycle-life 标签；两个右删失电芯不进入标量回归损失，但保留在数据审计和轨迹任务中。

### 特征和监督隔离

从原始 HDF5 生成两类内容寻址制品：

- 早期输入只含截止循环 20、50、100、150 以内的数据；
- 监督制品保存官方 cycle-life 和循环 500 以内的实测 SOH 轨迹。

特征提取器不能读取监督目录；未来容量只允许进入损失和评价。原始 `.mat` 只有在 HDF5 文件头、批准路径、大小与 SHA-256 全部匹配时才允许进入训练包。

## 统一训练系统

版本化配置包含数据集、目标、截断点、模型、种子、设备、精度、批量大小、最大轮数、验证间隔、耐心、优化器、调度器、版本上下文、检查点和日志策略。

统一 Trainer 负责设备、训练循环、验证、早停、调度、日志、检查点、恢复和最终测试；CPMLP、Hybrid、DANN、XGBoost 与经典基线适配器只负责数据批次、前向、损失和任务指标。现有 CPU 入口继续兼容，正式训练不再硬编码 CPU。

### 模型策略

| 模型 | 上限 | 主验证指标 | 早停 |
|---|---:|---|---:|
| CPMLP | 300 epoch | 官方 cycle-life MAE | 10 次验证 |
| Hybrid | 500 epoch | 循环 500 内轨迹 MAE | 10 次验证 |
| DANN | 300 epoch | 源域验证 MAE | 10 次验证 |
| XGBoost | 2000 rounds | 官方 cycle-life MAE | 100 rounds |
| Dummy/Variance | 一次拟合 | 官方 cycle-life MAE | 不适用 |

深度模型使用 AdamW 与 `ReduceLROnPlateau`，连续 3 次验证无改善时学习率减半，最低 `1e-6`。校准集只用于 Conformal；测试集只在模型和超参数冻结后评价一次。DANN 不能使用目标域标签进行训练或早停。

正式结果默认使用 FP32 确定性训练；BF16 仅作可选效率实验。

## 安全断点恢复

每个深度模型每个 epoch 原子保存：

```text
checkpoint/
├── model.safetensors
├── optimizer.safetensors
├── optimizer_state.json
├── scheduler_state.json
├── rng_state.safetensors
├── progress.json
└── manifest.json
```

检查点绑定 epoch、global step、最佳指标、早停计数、模型/优化器/调度器/随机数状态，以及数据、划分、特征、配置、代码和输入包哈希。保留 `last`、`best` 和最近三个周期检查点；损坏或上下文不一致时拒绝恢复。SIGTERM/SIGINT 在安全边界保存后退出，强制终止最多损失一个 epoch。

XGBoost 使用 UBJ 和 JSON 状态；Dummy/Variance 只登记最终安全制品。已完成任务重跑时跳过，中断任务从最后有效检查点恢复。

## A100 与一键命令

Shell 固定：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

启动前验证物理 GPU 1 为 A100、总显存不少于 75 GiB、占用不高于 1024 MiB、PyTorch 只看到一张 GPU 且内部设备为 `cuda:0`。训练任务不得访问 GPU 0。

统一命令：

```bash
bash scripts/a100/train_dataset.sh matr smoke
bash scripts/a100/train_dataset.sh matr final
bash scripts/a100/train_dataset.sh hust final
bash scripts/a100/train_dataset.sh naumann-cycle final
bash scripts/a100/train_dataset.sh naumann-calendar final
bash scripts/a100/train_all_ready.sh final
```

MATR 正式运行四个截断点、五类模型和五个固定种子：`20260712` 至 `20260716`。任务顺序执行；单项失败返回非零，再次运行时跳过已完成项并恢复失败项。

HUST 在安全 Canonical Parquet 和目标语义对齐前返回 `DATASET_NOT_READY`。Naumann Cycle 运行退化率基线、GP 和主动试验回放；Naumann Calendar 运行日历老化与温度/SOC模型，不强行运行 CPMLP 或 DANN。

## 指标、日志和结果

终端与结构化日志记录数据集、模型、截断点、种子、epoch、训练/验证损失、MAE、RMSE、MAPE、R²、学习率、耗时、GPU显存、最佳轮次和早停状态。

每个运行目录至少包含：

```text
run_manifest.json
config_resolved.json
training_log.jsonl
metrics_epoch.csv
metrics_validation.csv
metrics_test.json
metrics_test.csv
aggregate_metrics.json
environment.json
model_card.md
plots/
checkpoints/
artifacts/
mlruns/
```

正式汇总报告五种子均值、标准差和失败实验；Hybrid 轨迹误差与单调性；Conformal PICP/MPIW和小校准集警告；训练时间、峰值显存和最佳轮次；官方 cycle-life 与统一 EOL80 的差异。

## 服务器环境和打包

服务器采用 Conda Python 3.11.13。PyTorch 使用 2.12 稳定版默认 CUDA 13.0 wheel，不安装系统 CUDA Toolkit，不采用实验性 CUDA 13.2 nightly。其余依赖通过 Linux/Python 3.11 哈希锁文件安装，包含数据、机器学习、MLflow、绘图和环境记录依赖。

跨平台打包工具从干净 Git 工作树生成代码、配置、来源已登记的原始 MATR、处理制品、划分、训练配置、依赖锁、全包清单和源提交记录；排除 `.git`、环境变量、密钥、HUST pickle、缓存、旧模型及未登记可执行反序列化文件。

服务器顺序执行：

```bash
bash scripts/a100/preflight.sh
bash scripts/a100/train_dataset.sh matr smoke
bash scripts/a100/train_dataset.sh matr final
```

## 测试与验收

- GPU 1 绑定与占用门禁；无 CUDA 或错误设备时阻断。
- 官方 cycle-life 与统一 EOL80 不能混用。
- 截断点后数据不得进入特征。
- 四类电芯分区完全隔离。
- 中断恢复与连续训练在容差内一致。
- 损坏、缺失或上下文不一致检查点被拒绝。
- 每五轮产生验证指标，早停保存最佳模型。
- 测试集不参与调参，校准集不参与训练。
- 终端、JSONL、CSV和MLflow记录一致。
- 禁止 pickle、joblib、`.pt`、`.pth`。
- Smoke 从原始 MATR 贯通预处理、训练、验证、恢复和制品导出。
- 正式命令完成四截断点、五种子与完整汇总。

## 实施顺序

1. 新增官方 cycle-life 目标契约和兼容适配。
2. 构建 MATR 循环 500 监督轨迹及防泄漏边界。
3. 实现统一 Trainer、设备抽象、验证和安全断点。
4. 接入 CPMLP、Hybrid、DANN、XGBoost 与经典基线。
5. 建立数据集模型矩阵、配置和一键 Shell。
6. 接入 JSONL、CSV、本地 MLflow与汇总报告。
7. 建立 A100 依赖锁、预检与安全打包。
8. 完成本地合成数据/CPU门禁，由用户在 GPU 1 运行 Smoke。
9. Smoke 验收后运行 MATR 正式五种子实验。
10. HUST 安全转换完成后启用 DANN 与 MATR→HUST 评价。

