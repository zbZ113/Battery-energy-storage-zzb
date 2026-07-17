# MATR 2018-04-12 批次处理协议 v1

## 目标与边界

本协议将用户从 MATR 官方项目下载的 MATLAB v7.3/HDF5 批次转换为逐电芯、内容寻址的 Parquet 与 JSON 制品，用于 20/50/100/150 循环早期寿命实验。原始 `.mat` 保持只读且不进入 Git；公开实验室电芯不得表述为企业工业电芯。

原始文件由 [matr_2018_04_12_batch_v1.json](../configs/data_manifests/matr_2018_04_12_batch_v1.json) 固定：

- 文件：`data/2018-04-12_batchdata_updated_struct_errorcorrect.mat`
- 大小：3,236,690,412 bytes
- SHA-256：`62c30e413b63e6144720e016deed3661fac8468641794a5807b123fe84717998`
- 批次日期：`2018-04-12`
- 电芯数：46
- 来源许可证状态仍为 `must_verify_before_download`；在许可证条款完成独立核验前，不把该字段写成已确认的开放许可证。

## 版本化语义

1. 批次编号为 3，与本地 BatteryML 参考工程的四批次顺序一致；不复制其 pickle 输出和特征实现。
2. 原始 `t` 明确解释为分钟并乘以 60 转为秒。抽样检查中，`dQc/dt × 60` 与记录电流的量纲关系吻合，而按秒解释会产生约 60 倍偏差。
3. cycle 0 作为形成/初始化边界保留在规范化记录中，但标记为非诊断；cycle 1 起的完整固定放电循环标记为诊断循环。
4. `reference_capacity_ah` 固定取 cycles 1–5 的 summary `QDischarge` 中位数。窗口不随 20/50/100/150 截断点改变，也不使用额定 1.1 Ah 代替实测参考容量。
5. 官方 `cycle_life` 只保存在 `official_life_label`，不得与项目统一 EOL80 标签混用。
6. 首版处理上限为 cycle 150（含），足以覆盖计划中的所有早期截断点；不为首轮实验重复存储后续数百至上千循环的采样点。

## 转换命令

```powershell
.\.venv\Scripts\python.exe scripts\convert_matr_batch.py `
  --raw data\2018-04-12_batchdata_updated_struct_errorcorrect.mat `
  --manifest configs\data_manifests\matr_2018_04_12_batch_v1.json `
  --output-root data\processed\MATR\2018-04-12-cutoff150 `
  --report reports\data_quality\matr_2018_04_12_cutoff150.json `
  --batch-index 3 `
  --batch-date 2018-04-12 `
  --time-unit minutes `
  --max-cycle-index 150
```

适配器只验证原始文件一次，并在同一 HDF5 句柄中逐颗电芯读取。每颗电芯发布后立即回读验证 Parquet Schema、元数据、清单路径、文件大小和 SHA-256。`data/processed/` 为可再生成制品，不进入 Git；质量报告、协议、配置与代码进入 Git。

## 电芯级分区

转换成功后生成固定分区：

```powershell
.\.venv\Scripts\python.exe scripts\build_matr_split.py `
  --conversion-report reports\data_quality\matr_2018_04_12_cutoff150.json `
  --split-output configs\data_splits\matr_2018_04_12_cell_split_v1.json `
  --evidence-output reports\data_quality\matr_2018_04_12_split_v1.json `
  --split-version matr-2018-04-12-cell-split-v1 `
  --quantile-count 4
```

分区严格按 `cell_id` 进行，目标比例为 60%/15%/10%/15%，随机种子为 `20260712`。分层键由充电协议与官方事件寿命四分位组合；官方寿命为 NaN 的电芯进入独立 `right-censored` 层，绝不作为已观测寿命事件排序。

## 官方寿命与统一 EOL80 审计

官方 `cycle_life` 与项目统一 EOL80 必须分别保存。运行：

```powershell
.\.venv\Scripts\python.exe scripts\audit_matr_labels.py `
  --raw data\2018-04-12_batchdata_updated_struct_errorcorrect.mat `
  --manifest configs\data_manifests\matr_2018_04_12_batch_v1.json `
  --conversion-report reports\data_quality\matr_2018_04_12_cutoff150.json `
  --output reports\data_quality\matr_2018_04_12_eol80_audit_v1.json
```

统一 EOL80 使用实测参考容量和连续三个诊断点确认。若原始实验在达到该阈值前结束，统一标签必须保持右删失；不得因官方标签存在而改写为统一 EOL80 事件。训练与报告必须声明所用目标是 `MATR_cycle_life` 还是项目统一 EOL80。
