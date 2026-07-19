# 泉芯智寿先进电池寿命双旗舰模型升级与 A100 科学验证计划

## 一、目标与锁定决策

在现有三批 MATR、A100 训练流水线和安全制品体系上实现两条先进模型主线：

1. **CyclePatch-BatLiNet**：预测 `MATR_OFFICIAL_CYCLE_LIFE`。
2. **HybridPatch-v2**：预测循环 500 以内的真实、结构性单调 SOH 轨迹，RUL 只能由轨迹跨越阈值推导。

固定随机种子为 `38、39、40、41、42`：Smoke 使用 38；Select 初筛使用 38；Select 稳健性复核使用 38、39、40；Final 使用全部五个种子。所有入口同步设置 Python、NumPy、PyTorch、CUDA 和 `PYTHONHASHSEED`，并开启项目现有确定性训练策略。

首轮数据范围固定为三批 MATR：140 颗电芯，标量寿命标签可用 138 颗，Hybrid 轨迹样本可用 120 颗；固定划分为训练 82、验证 19、校准 12、测试 27；截断点为 20、50、100、150 循环。模型晋级只依据训练集和验证集，校准集只用于 Conformal，测试集在配置冻结后评价一次。

## 二、模型设计

### 1. 多通道早期循环输入

新增 `EarlyCycleSequence`：

```text
dataset_id / cell_id / cutoff_cycle
cycle_indices
values[cycle, phase, sample, variable]
cycle_mask / sample_mask
condition_features / condition_mask
data_version / feature_version / input_hash
```

固定处理规则：

- 只读取 `cycle_index <= cutoff_cycle`；
- 充电和放电阶段各重采样 150 点；
- 输入变量至少包括电压、电流和阶段容量；
- 缺失循环和采样点通过 Mask 表达，不使用零值冒充缺测；
- 标准化统计量只能由训练电芯产生；
- 寿命标签和未来 SOH 制品不能传入特征提取器；
- 电压重复点、非单调插值坐标、时间倒退必须清洗或拒绝。

### 2. CyclePatch 编码器

每个循环执行充电/放电独立 Patch Embedding、卷积核 3/5/7 的多尺度循环内编码、Masked Attention Pooling，并拼接循环编号、温度、协议等结构化条件，形成循环级 Token。循环 Token 进入带位置编码的 Transformer，同时输出 CLS 和 Masked Attention Pooling 表示，经门控融合形成电芯表示 `z_i`。

候选范围：

```text
d_model: 128 / 256
inter_cycle_layers: 2 / 4
attention_heads: 4 / 8
dropout: 0.05 / 0.10
```

直接分支为 `y_direct = DirectHead(z_i)`，训练目标使用仅由训练电芯计算的标准化官方 cycle-life，损失使用 Huber。

### 3. BatLiNet 电芯间分支

只允许训练电芯进入参考库：

```text
pair_ij = concat(z_i - z_j, abs(z_i - z_j), z_i * z_j)
delta_y_ij ~= y_i - y_j
y_pair_ij = y_j + delta_y_ij
```

训练损失：

```text
L_scalar =
    Huber(y_direct, y_i)
  + lambda_pair * Huber(delta_y_ij, y_i - y_j)
  + lambda_rank * PairwiseRankingLoss
```

候选参数：

```text
lambda_pair: 0.25 / 0.50 / 1.00
lambda_rank: 0.00 / 0.10
reference_count: 16 / 32 / 64
fusion_alpha: 0.25 / 0.50 / 0.75
```

推理使用 `alpha * y_direct + (1-alpha) * median(y_pair_ij)`。参考电芯按训练寿命分位数分层选择，选择由种子控制并登记哈希；验证、校准和测试电芯不得进入参考库。

### 4. HybridPatch-v2

使用相同的多通道早期循环编码器，增加工况特征 AdaLN 调制、循环间注意力、0/8/16 个可选退化模式查询 Token、历史 SOH 重建辅助头，并连接当前 Hybrid 的结构性单调解码器：

```text
SOH(t) = SOH(cutoff)
       - D_sqrt(t)
       - D_linear(t)
       - D_knee(t)
       - cumulative_softplus(D_residual(t))
```

所有退化参数通过正值变换；支持真实、非均匀循环坐标；监督终点不超过循环 500；不生成循环 500 后的训练标签；不设置独立 RUL 头；单调性来自网络结构。

损失：

```text
L_trajectory =
    MaskedHuber(predicted_soh, real_soh)
  + lambda_history * HistoryReconstructionLoss
  + lambda_smooth * SecondDifferenceLoss
  + lambda_order * DegradationOrderLoss
  + lambda_residual * ResidualMagnitudeLoss
```

候选参数：

```text
lambda_history: 0.00 / 0.10
lambda_smooth: 0.00 / 0.01
lambda_order: 0.00 / 0.05
lambda_residual: 0.001 / 0.01
```

## 三、训练选择协议

### Smoke

```bash
bash scripts/a100/train_advanced_models.sh matr-three-batch smoke
```

使用种子 38、截断点 50、最多 10 epoch，验证数据加载、GPU 反向传播、日志、检查点、恢复、安全制品和汇总，不形成性能结论。

### Select

```bash
bash scripts/a100/train_advanced_models.sh matr-three-batch select
```

Select 只构造训练和验证 DataLoader：截断点 100、种子 38 的全部候选先运行 30 epoch，按验证 MAE 淘汰后 50%，剩余候选继续到 90 epoch；每条主线保留前两名，再用四个截断点和种子 38、39、40 做稳健性复核；最终输出冻结配置、选择证据和 SHA-256，冻结后禁止命令行覆盖超参数。

### Final

```bash
bash scripts/a100/train_advanced_models.sh matr-three-batch final
```

使用种子 38、39、40、41、42，运行四截断点乘以 CyclePatch direct-only、CyclePatch-BatLiNet、current Hybrid 和 HybridPatch-v2。已有基线只有在数据、划分、特征和目标版本完全一致时才能导入。

## 四、晋级与评价规则

CyclePatch-BatLiNet 必须在三个选择种子、四个截断点的平均验证 MAE 上优于 XGBoost 至少 2%，且任一截断点不恶化超过 10%；参考库只包含训练电芯，运行和制品可复验。

HybridPatch-v2 必须在平均验证轨迹 MAE 上优于当前 Hybrid 至少 2%，单调违规率为 0，轨迹 RMSE 不恶化，不访问循环 500 后监督，RUL 只能从轨迹阈值跨越推导。

未晋级候选保留为消融，不在 README 和竞赛材料中称为旗舰。正式报告包含 MAE、RMSE、MAPE、R2、轨迹误差、PICP、MPIW、五种子均值与标准差、失败实验、10,000 次逐电芯配对 Bootstrap 区间、训练时间、峰值显存、最佳 epoch 和消融结果。

## 五、A100 训练配置

物理设备固定：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

PyTorch 必须只看到一张设备，并将物理 GPU 1 映射为 `cuda:0`。CyclePatch 系列使用 AdamW、300 epoch、每 5 epoch 验证、10 次验证早停、ReduceLROnPlateau、梯度裁剪 1.0 和 FP32。HybridPatch-v2 使用相同策略但上限 500 epoch。BF16 只作为效率实验。总预算允许 48 小时以上，但采用阶段淘汰避免无效训练。

## 六、工程改造

新增 `EarlyCycleSequence`、`CycleLifePairBatch`、`CyclePatchEncoder`、`CyclePatchLifePredictor`、`CyclePatchBatLiNetPredictor`、`HybridPatchV2Predictor`、对应训练任务、`AdvancedModelSelectionManifest` 和 safetensors 架构白名单。继续复用 `PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE`、公共预测契约、`TrainingEngine`、安全检查点、`ToolResult`、Conformal 和 A100 预检。

新增配置目录 `configs/training/advanced/`，包含 smoke、selection、final 以及三类搜索配置；新增 `scripts/a100/train_advanced_models.sh`、`scripts/run_advanced_model_suite.py` 和 `scripts/a100/verify_advanced_run.sh`。参考仓库继续位于被忽略的 `research/upstream`，正式模型不得运行时导入参考项目。

## 七、恢复、日志和制品

每个 epoch 原子保存 safetensors 模型/优化器/RNG、JSON 优化器/调度器/进度和清单。检查点上下文增加 seed、参考库哈希、架构哈希、选择清单哈希和标准化哈希。完成任务安全跳过，中断任务自动恢复，上下文变化时拒绝恢复。

终端、JSONL、CSV 和本地文件型 MLflow 统一记录模型、种子、截断点、epoch、训练/验证损失、MAE、RMSE、MAPE、R2、轨迹误差、单调违规率、学习率、显存、最佳 epoch、早停计数和耗时。所有正式权重使用 safetensors，继续禁止 pickle、joblib、`.pt` 和 `.pth`。

## 八、测试与实施顺序

测试必须覆盖未来信息泄漏、电芯划分、训练集标准化、Mask 语义、循环顺序敏感性、参考库隔离、Select 测试集隔离、冻结配置、单调轨迹、派生 RUL、tiny cohort 过拟合、断点恢复、种子确定性、日志一致性、GPU 绑定和安全制品。

实施顺序：计划入库；多通道张量；CyclePatch；BatLiNet；HybridPatch-v2；辅助损失；安全制品；Smoke/Select/Final；A100 脚本；CPU 合成验证；训练包；用户 A100 Smoke、Select、Final；结果导入与竞赛报告。

正式工程代码、测试、配置和文档进入当前 Git 仓库；参考项目、论文、原始缓存和训练产物不提交，由用户通过 GitHub Desktop 推送。

## 九、A100 执行

```bash
cd /data/abd/z/AI-B
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quanxin-a100
bash scripts/a100/preflight.sh
tmux new -s quanxin-advanced
bash scripts/a100/train_advanced_models.sh matr-three-batch smoke 2>&1 | tee logs/a100/advanced-smoke.log
bash scripts/a100/train_advanced_models.sh matr-three-batch select 2>&1 | tee logs/a100/advanced-select.log
bash scripts/a100/train_advanced_models.sh matr-three-batch final 2>&1 | tee logs/a100/advanced-final.log
```

Codex 负责本地实现、验证、提交、打包和运行说明；用户负责上传并在 A100 执行，日志回传后继续诊断和验收。

## 十、完成定义

- 五个正式种子统一为 38、39、40、41、42；
- 两条旗舰由本项目自主实现，参考代码不成为运行时依赖；
- Select 完全隔离测试集，Final 配置冻结且可复验；
- 轨迹结构性单调；
- 一条命令可分别运行 Smoke、Select 和 Final；
- A100 训练具备验证、早停、恢复、日志和安全制品；
- 五种子、四截断点、消融和统计区间完整；
- 所有正式代码和文档进入项目 Git；
- 未经实验支持不得宣称跨数据集或工业级 SOTA。
