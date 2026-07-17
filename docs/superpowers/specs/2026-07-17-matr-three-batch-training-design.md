# MATR 三批次联合训练设计

## 1. 目标

在不改变已经验证的 2018-04-12 单批次训练语义和产物路径的前提下，将 MATR 的 2017-05-12、2017-06-30 与 2018-04-12 三批数据接入同一套 A100 真实训练与评价流水线。

新增联合入口固定为：

```bash
bash scripts/a100/train_dataset.sh matr-three-batch smoke
bash scripts/a100/train_dataset.sh matr-three-batch final
```

原有 `matr` 入口继续表示已经验证的 2018-04-12 单批次，不允许在同名配置下静默更换数据。

## 2. 已核验原始数据

| 批次 | batch index | 电芯数 | 文件大小 | SHA-256 |
|---|---:|---:|---:|---|
| 2017-05-12 | 1 | 46 | 3,025,320,241 | `9d928ab978f0e3c70b31cb833a749fedd35094d01af76475d69b40aa3497f5ba` |
| 2017-06-30 | 2 | 48 | 2,007,331,155 | `63ab200d09ecb237fee5ef3a5c5db76e3212e3206a0bd92f769e1427fed338b8` |
| 2018-04-12 | 3 | 46 | 3,236,690,412 | `62c30e413b63e6144720e016deed3661fac8468641794a5807b123fe84717998` |

三个文件均具有 MATLAB 7.3 文件头和偏移 512 的 HDF5 签名，包含 `cycles`、`summary`、`policy_readable` 与 `cycle_life` 等必需引用。原始文件只进入来源清单、A100 训练包和本地忽略目录，不提交到 Git。

## 3. 目标与样本资格

### 3.1 官方 cycle-life 标量任务

- 三批共 140 颗电芯。
- 2017 两批共 94 颗电芯均有有效官方 cycle-life。
- 2018 批次有 44 个有效官方标签与 2 个右删失电芯。
- Dummy、Variance、XGBoost、CPMLP 使用 138 个事件标签；两颗右删失电芯保留在数据审计与划分中，但不得进入标量回归损失、Conformal 残差或测试指标。
- 目标名称保持 `MATR_OFFICIAL_CYCLE_LIFE`，不得表述为统一 EOL80。

### 3.2 Hybrid 循环 500 轨迹任务

- 2017-05-12 的 46 颗电芯均具有超过 500 个真实容量点。
- 2017-06-30 有 28 颗满足条件，另外 20 颗只有 170 至 500 个容量点。
- 2018-04-12 的 46 颗均满足条件。
- Hybrid 使用 120 颗具备完整真实循环 1 至 500 `QDischarge` 的电芯。
- 20 颗短轨迹电芯只记录 `INSUFFICIENT_REAL_TRAJECTORY_500` 资格原因，不填充、不外推、不生成训练标签。
- Hybrid 预测与评价仍固定到循环 500。

## 4. 数据制品与合并边界

三批分别生成并复验：

```text
data/processed/MATR/2017-05-12-cutoff150/
data/processed/MATR/2017-06-30-cutoff150/
data/processed/MATR/2018-04-12-cutoff150/

data/processed/MATR/2017-05-12-supervision500/
data/processed/MATR/2017-06-30-supervision500-eligible/
data/processed/MATR/2018-04-12-supervision500/
```

每批拥有独立原始清单、转换报告、监督报告和 SHA-256。联合数据版本不复制或改写批次内制品，而是使用一个版本化联合清单引用三个已验证组件。联合清单记录：

- 批次日期、batch index、原始 SHA-256；
- 处理根目录与转换报告哈希；
- 标量标签数量、右删失数量；
- Hybrid 循环 500 合格与不合格电芯清单；
- 联合数据版本和清单 SHA-256。

特征加载器只能读取三个 `cutoff150` 根目录；未来监督加载器只能读取独立监督目录。

## 5. 三批联合划分

每批先按现有协议与批内寿命分位数生成确定性电芯级 60%/15%/10%/15% 划分，再按同名分区合并：

```text
combined.train       = batch1.train + batch2.train + batch3.train
combined.validation  = batch1.validation + batch2.validation + batch3.validation
combined.calibration = batch1.calibration + batch2.calibration + batch3.calibration
combined.test        = batch1.test + batch2.test + batch3.test
```

这样每个分区均包含不同批次，同时保持 `cell_id` 完全隔离。联合审计必须验证：

- 140 个 `cell_id` 恰好出现一次；
- `MATR_b1*`、`MATR_b2*`、`MATR_b3*` 均在所有分区中有代表；
- 标量任务只过滤右删失标签，不重分区；
- Hybrid 只按循环 500 资格过滤，不重分区；
- 校准集只用于 Conformal，测试集只在冻结后评价一次。

## 6. 联合训练配置

新增：

```text
configs/training/matr_three_batch_smoke.json
configs/training/matr_three_batch_final.json
```

模型、截断点、种子、验证频率和早停规则与已经批准的 A100 计划一致：

- 截断点：20、50、100、150；
- 模型：Dummy、Variance、XGBoost、CPMLP、Hybrid；
- Smoke：单种子和短训练；
- Final：五种子；
- 物理 GPU 1，内部设备 `cuda:0`；
- 正式结果使用 FP32。

每个运行清单必须携带联合数据清单哈希、三批原始 SHA-256、联合划分版本、目标、模型、截断点、种子、代码提交和输入包哈希。

## 7. A100 三批训练包

新增三批训练包，不覆盖现有 2018 单批包：

```text
dist/quanxin-matr-three-batch-a100.zip
dist/quanxin-matr-three-batch-a100.sha256.json
```

包内包含：

- 干净 Git 提交的代码、配置、脚本和依赖锁；
- 三个来源已登记、大小与 SHA-256 已核验的原始 `.mat`；
- 三批早期特征制品；
- 三批监督制品和 Hybrid 资格审计；
- 三批独立划分与联合划分；
- 联合数据清单和 `source_revision.json`。

包内不得包含 `.git`、`.env*`、密钥、HUST pickle、缓存、旧模型、`.pt`、`.pth`、pickle 或 joblib。整包使用 ZIP64，外部索引在解压前校验完整 ZIP 字节，内部清单在解压后再次校验全部文件。

## 8. 错误处理与恢复

- 任一原始文件头、大小、SHA-256、批次日期或 batch index 不匹配时阻断。
- 已存在处理制品必须逐文件复验，损坏时拒绝复用，不静默覆盖。
- 2017-06-30 的短轨迹电芯明确降级为 Hybrid 不合格，不影响标量模型。
- 三批联合配置与单批配置使用不同运行根目录和上下文哈希，禁止交叉恢复检查点。
- 再次执行联合命令时，已完成任务跳过，中断任务从相同联合上下文的最后有效检查点恢复。

## 9. 验收

本地仅新增并验证两批数据相关能力，不重复已经通过的 2018 单批算法测试。完成条件：

- 两个 2017 原始清单完成并被 Git 跟踪；
- 两批截止 150 转换及逐电芯复验完成；
- batch1 的 46 颗和 batch2 的 28 颗生成真实循环 500 监督轨迹；
- batch2 的 20 颗短轨迹具有显式资格审计；
- 三批联合划分覆盖 140 颗电芯且无泄漏；
- 联合 Smoke 计划展开为 20 个串行任务，Final 展开为 100 个串行任务；
- 三批 A100 ZIP64 包及外部索引完成独立复验；
- 用户在服务器物理 GPU 1 上运行联合 Smoke 后，回传预检、汇总和训练日志进行最终 GPU 验收。
