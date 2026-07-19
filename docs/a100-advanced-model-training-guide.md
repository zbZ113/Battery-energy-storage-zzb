# 泉芯智寿高级模型 A100 训练指南

本指南对应 CyclePatch-BatLiNet 与 HybridPatch-v2 的受控训练入口。训练只使用批准的三批 MATR 数据、按电芯隔离的划分和安全模型制品。默认绑定物理 GPU 1，PyTorch 内部设备为 `cuda:0`。

## 服务器准备

```bash
cd /data/abd/z/AI-B
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quanxin-a100
python -m pip check
bash scripts/a100/preflight.sh
```

预检必须报告 NVIDIA A100 80GB、物理设备 1、显存至少 75 GiB，且 PyTorch 只看到一张卡。GPU 1 被占用时会阻断，不会误用 GPU 0。

## 上传前检查

将工程代码、配置、三批 MATR 原始文件和已生成的 Parquet 制品复制到 `/data/abd/z/AI-B`。不要上传 `.git`、密钥、pickle/joblib、`.pt/.pth` 或旧训练缓存。上传后先运行：

```bash
python scripts/run_advanced_model_suite.py matr-three-batch smoke --plan-only
```

## Smoke、Select、Final

建议使用 tmux。各阶段均可重复执行；检查点会使已完成任务跳过，未完成任务从最后有效状态恢复。

```bash
tmux new -s quanxin-advanced

# 10 epoch、种子38，仅验证数据/GPU/日志/检查点链路
bash scripts/a100/train_advanced_models.sh matr-three-batch smoke \
  2>&1 | tee logs/a100/advanced-smoke-console.log

# 候选筛选：阶段1/2及38、39、40稳健性复核
bash scripts/a100/train_advanced_models.sh matr-three-batch select \
  2>&1 | tee logs/a100/advanced-select-console.log

# 选择清单冻结后，四截断点、五种子、四条模型线
bash scripts/a100/train_advanced_models.sh matr-three-batch final \
  2>&1 | tee logs/a100/advanced-final-console.log
```

Shell 入口会设置 GPU1 可见性、逐种子 `PYTHONHASHSEED`、CUBLAS 确定性配置和无显示后端。每个种子使用独立 Python 进程，模型任务顺序执行，不在 GPU 1 上并行抢占。

## 结果验收

完成阶段后，使用外部输出索引复验结果目录：

```bash
bash scripts/a100/verify_advanced_run.sh \
  runs/a100/matr-three-batch/advanced/final \
  runs/a100/matr-three-batch/advanced/final/output_index.json
```

验收会重新检查运行清单、文件清单、SHA-256、目录边界、秘密内容和危险序列化格式。缺文件、字节变化或未登记模型制品都会失败。

## 日志查看

```bash
tail -f logs/a100/advanced-smoke-*.log
tail -f logs/a100/advanced-select-*.log
tail -f logs/a100/advanced-final-*.log
nvidia-smi -i 1
```

正式结果至少包含运行配置、逐 epoch JSONL/CSV、验证与测试指标、最佳 epoch、显存峰值、检查点、safetensors 模型、模型卡和结果清单。Smoke 只验证流程，不产生正式性能结论；Select 只使用训练/验证电芯；Final 配置必须绑定选择清单。

## 常见恢复

- 命令被中断：重新执行同一阶段命令，从最后有效检查点继续。
- 选择清单未冻结：Final 返回 `FINAL_CONFIG_NOT_BOUND`，先完成 Select 并写入实际清单哈希。
- 数据集未准备：入口返回 `DATASET_NOT_READY`，不会伪造训练样本。
- MLflow 服务不可用：使用本地文件型追踪，JSONL/CSV 和核心训练仍继续。
