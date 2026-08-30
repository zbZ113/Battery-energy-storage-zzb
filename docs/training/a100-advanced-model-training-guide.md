# Hiro 高级模型 A100 训练操作手册

本手册用于现有服务器 `/data/abd/z/AI-B` 和 Conda 环境 `quanxin-a100`。三批 MATR 原始数据、处理后的 Parquet、监督制品和经典模型结果继续原位复用，不重新上传约 9 GB 数据，不重建环境，也不重跑旧 Dummy、Variance、CPMLP 或经典 Hybrid 实验。

高级训练固定使用物理 GPU 1；设置 `CUDA_VISIBLE_DEVICES=1` 后，PyTorch 内部设备显示为 `cuda:0` 是正确行为。

## 一、当前高级训练内容

- Smoke：cutoff 50、seed 38、4 个模型、每个最多 10 epoch，共 4 个任务。
- Select：13 个候选先跑 30 epoch；7 个生存者续跑到 90 epoch；7 个 finalist 在 4 个 cutoff、3 个 seed 上复核，共记录 104 次训练操作。
- Final：4 个 cutoff × 4 条模型线 × 5 个 seed，共 80 个任务。
- CyclePatch 两条标量寿命线最多 300 epoch；current Hybrid 和 HybridPatch-v2 最多 500 epoch。
- 每 5 epoch 验证一次，连续 10 次验证无改进时早停。
- 每个 epoch 保存安全断点；正式权重使用 safetensors，不使用 `.pt`、`.pth`、pickle 或 joblib。
- Final 每个任务在恢复最佳检查点后评价一次测试集，并写入 `metrics_test.json`。

## 二、服务器代码备份与更新

先登录服务器并备份当前来源记录、未提交补丁和已有高级运行。此操作不移动经典结果目录 `runs/a100/matr-three-batch/final`。

```bash
cd /data/abd/z/AI-B
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p backups/$STAMP

test -f source_revision.json && cp -a source_revision.json backups/$STAMP/
test -d runs/a100/matr-three-batch/advanced && \
  tar -czf backups/$STAMP/advanced-runs-before-update.tar.gz \
  runs/a100/matr-three-batch/advanced
test -d logs/a100 && \
  find logs/a100 -maxdepth 1 -type f -name 'advanced-*.log' -print0 | \
  tar --null -czf backups/$STAMP/advanced-logs-before-update.tar.gz --files-from=-
```

如果服务器目录包含 `.git`，使用 Git 更新：

```bash
cd /data/abd/z/AI-B
git status --short --branch | tee backups/$STAMP/git-status-before.txt
git diff --binary > backups/$STAMP/worktree-before.patch
git diff --cached --binary > backups/$STAMP/index-before.patch

# 工作区必须干净；若输出非空，先保留上面的补丁并停止，不要 reset --hard。
test -z "$(git status --porcelain)"

git fetch origin codex/quanxin-full
git switch codex/quanxin-full
git pull --ff-only origin codex/quanxin-full
git rev-parse HEAD
```

如果服务器目录没有 `.git`，在 Windows 项目根目录生成代码包，仅上传代码，不含数据、缓存和运行结果：

```powershell
cd 'D:\guet_learning\26 AI acting\Battery-energy-storage-zzb'
git archive --format=zip --output advanced-code.zip HEAD
Get-FileHash .\advanced-code.zip -Algorithm SHA256
scp .\advanced-code.zip dell@<A100服务器IP>:/data/abd/z/
```

然后在服务器核验并覆盖代码文件。数据和运行目录不在 Git archive 中，不会被覆盖：

```bash
cd /data/abd/z
sha256sum advanced-code.zip
unzip -o advanced-code.zip -d /data/abd/z/AI-B
cd /data/abd/z/AI-B
```

无 `.git` 部署还必须写入上传代码对应的提交号：

```bash
export QUANXIN_SOURCE_COMMIT='<Windows 上 git rev-parse HEAD 的 40 位提交号>'
python - <<'PY'
import json, os
from datetime import datetime, timezone
payload = {
    "schema_version": "source-revision-v1",
    "git_commit": os.environ["QUANXIN_SOURCE_COMMIT"],
    "git_dirty": False,
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}
open("source_revision.json", "w", encoding="utf-8").write(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
)
PY
```

## 三、复用现有环境并做预检

```bash
cd /data/abd/z/AI-B
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quanxin-a100

python --version
python -m pip install -e . --no-deps
python -m pip check

find data/cache/advanced_sequences -name sequence.safetensors 2>/dev/null | wc -l
du -sh data/cache/advanced_sequences 2>/dev/null || true

bash scripts/a100/preflight.sh
python - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "visible_gpu_count": torch.cuda.device_count(),
    "pytorch_device_0": torch.cuda.get_device_name(0),
})
PY
```

预检必须确认物理 GPU 1 是 A100 80GB、空闲显存满足门禁、PyTorch 只看到一张卡。GPU 1 已被其他任务占用时先停止，不要改脚本去使用 GPU 0。

## 四、只检查计划，不启动训练

```bash
bash scripts/a100/train_advanced_models.sh matr-three-batch smoke --plan-only
bash scripts/a100/train_advanced_models.sh matr-three-batch select --plan-only

# Select 完成前，Final 应明确返回 FINAL_CONFIG_NOT_BOUND，这是正常门禁。
bash scripts/a100/train_advanced_models.sh matr-three-batch final --plan-only || true
```

## 五、运行 Advanced Smoke

```bash
cd /data/abd/z/AI-B
tmux new -s qx-advanced-smoke
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quanxin-a100
bash scripts/a100/train_advanced_models.sh matr-three-batch smoke
```

按 `Ctrl+B`，再按 `D` 退出 tmux；训练继续运行。重新进入：

```bash
tmux attach -t qx-advanced-smoke
```

另开一个 SSH 窗口监控：

```bash
cd /data/abd/z/AI-B
LOG=$(ls -1t logs/a100/advanced-smoke-*.log | head -n 1)
tail -F "$LOG"
```

GPU、进度、磁盘和进程监控：

```bash
watch -n 2 'nvidia-smi -i 1'

watch -n 5 '
echo "run_status:" $(find runs/a100/matr-three-batch/advanced/smoke -name run_status.json | wc -l)
echo "validation:" $(find runs/a100/matr-three-batch/advanced/smoke -name metrics_validation.csv | wc -l)
echo "checkpoints:" $(find runs/a100/matr-three-batch/advanced/smoke -name model.safetensors | wc -l)
'

watch -n 10 'du -sh data/cache/advanced_sequences runs/a100/matr-three-batch/advanced/smoke 2>/dev/null; df -h /data'
pgrep -af 'train_advanced_models|run_advanced_model_suite'
```

Smoke 结束后检查错误并验证结果索引：

```bash
LOG=$(ls -1t logs/a100/advanced-smoke-*.log | head -n 1)
grep -Ein 'traceback|exception|error|out of memory|non-finite|nan' "$LOG" | tail -n 50 || true

bash scripts/a100/verify_advanced_run.sh \
  runs/a100/matr-three-batch/advanced/smoke
```

验证应报告 `mode=smoke`、`operations=4`。Smoke 只验证数据、GPU、训练、验证、断点和输出链，不作为正式性能结论。

## 六、运行 Advanced Select

Smoke 通过后运行：

```bash
tmux new -s qx-advanced-select
cd /data/abd/z/AI-B
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quanxin-a100
bash scripts/a100/train_advanced_models.sh matr-three-batch select
```

监控：

```bash
cd /data/abd/z/AI-B
LOG=$(ls -1t logs/a100/advanced-select-*.log | head -n 1)
tail -F "$LOG"

watch -n 2 'nvidia-smi -i 1'
watch -n 10 '
echo "status files:" $(find runs/a100/matr-three-batch/advanced/selection -name run_status.json | wc -l)
echo "validation files:" $(find runs/a100/matr-three-batch/advanced/selection -name metrics_validation.csv | wc -l)
echo "selection manifest:" $(test -s runs/a100/matr-three-batch/advanced/selection/selection_manifest.json && echo READY || echo WAITING)
'
```

Select 结束后验收冻结证据：

```bash
cd /data/abd/z/AI-B
test -s runs/a100/matr-three-batch/advanced/selection/selection_manifest.json
test -s runs/a100/matr-three-batch/advanced/selection/selection_trace.json
test -s runs/a100/matr-three-batch/advanced/selection/final_config_resolved.json

sha256sum runs/a100/matr-three-batch/advanced/selection/selection_manifest.json
bash scripts/a100/verify_advanced_run.sh \
  runs/a100/matr-three-batch/advanced/selection

bash scripts/a100/train_advanced_models.sh matr-three-batch final --plan-only
```

验证应报告 `mode=select`、`operations=104`；Final plan 应报告 80 个任务。不要手工修改 `final.json` 或选择清单哈希。

## 七、运行 Advanced Final

```bash
tmux new -s qx-advanced-final
cd /data/abd/z/AI-B
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quanxin-a100
bash scripts/a100/train_advanced_models.sh matr-three-batch final
```

监控：

```bash
cd /data/abd/z/AI-B
LOG=$(ls -1t logs/a100/advanced-final-*.log | head -n 1)
tail -F "$LOG"

watch -n 2 'nvidia-smi -i 1'
watch -n 10 '
echo "completed runs:" $(find runs/a100/matr-three-batch/advanced/final -name run_status.json | wc -l) / 80
echo "test metrics:" $(find runs/a100/matr-three-batch/advanced/final -name metrics_test.json | wc -l) / 80
echo "validation logs:" $(find runs/a100/matr-three-batch/advanced/final -name metrics_validation.csv | wc -l) / 80
'
```

Final 结束后验收：

```bash
cd /data/abd/z/AI-B
LOG=$(ls -1t logs/a100/advanced-final-*.log | head -n 1)
grep -Ein 'traceback|exception|error|out of memory|non-finite|nan' "$LOG" | tail -n 100 || true

bash scripts/a100/verify_advanced_run.sh \
  runs/a100/matr-three-batch/advanced/final

python - <<'PY'
import json
from pathlib import Path
root = Path("runs/a100/matr-three-batch/advanced/final")
aggregate = json.loads((root / "aggregate_metrics.json").read_text())
print({
    "aggregate_run_count": aggregate["run_count"],
    "run_status_files": len(list(root.rglob("run_status.json"))),
    "test_metric_files": len(list(root.rglob("metrics_test.json"))),
    "output_index": (root / "output_index.json").is_file(),
})
PY
```

验证应报告 `mode=final`、`operations=80`，并且 80 个任务都有测试指标和安全 safetensors 检查点。

## 八、中断与断点恢复

优先使用 `Ctrl+C` 或 `SIGTERM`，训练会在安全边界保存。不要先用 `kill -9`。

```bash
pgrep -af 'run_advanced_model_suite|train_advanced_models'
kill -TERM <Python训练进程PID>
```

恢复时必须保持同一代码提交、同一配置、同一数据和同一运行目录，直接重新执行原命令：

```bash
bash scripts/a100/train_advanced_models.sh matr-three-batch smoke
bash scripts/a100/train_advanced_models.sh matr-three-batch select
bash scripts/a100/train_advanced_models.sh matr-three-batch final
```

已完成任务会跳过；未完成任务从最后有效检查点继续。不要编辑 `checkpoints/`、不要复制别的 seed 检查点、不要修改冻结配置来绕过上下文哈希。若更新了代码提交，应先备份旧高级运行目录，再启动新一轮，不要混用不同来源版本的断点。

## 九、结果打包与下载

完整审计包包含高级运行目录、日志、预检和来源清单：

```bash
cd /data/abd/z/AI-B
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p exports
tar -czf exports/quanxin-advanced-final-$STAMP.tar.gz \
  runs/a100/matr-three-batch/advanced/selection \
  runs/a100/matr-three-batch/advanced/final \
  logs/a100/advanced-select-*.log \
  logs/a100/advanced-final-*.log \
  a100-preflight.json \
  source_revision.json
sha256sum exports/quanxin-advanced-final-$STAMP.tar.gz | \
  tee exports/quanxin-advanced-final-$STAMP.tar.gz.sha256
```

Windows 下载并复验：

```powershell
scp dell@<A100服务器IP>:/data/abd/z/AI-B/exports/quanxin-advanced-final-<时间戳>.tar.gz `
  'D:\guet_learning\26 AI acting\Battery-energy-storage-zzb\downloads\'
Get-FileHash `
  'D:\guet_learning\26 AI acting\Battery-energy-storage-zzb\downloads\quanxin-advanced-final-<时间戳>.tar.gz' `
  -Algorithm SHA256
```

下载后的 SHA-256 必须与服务器 `.sha256` 文件一致，再进入本地结果导入与论文/竞赛分析流程。
