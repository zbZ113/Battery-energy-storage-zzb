# MATR 三批数据 A100 真实训练操作手册

本手册适用于以下三批已登记 MATR MATLAB v7.3/HDF5 数据：

- `2017-05-12`：46 颗电芯；
- `2017-06-30`：48 颗电芯；
- `2018-04-12`：46 颗电芯。

组合数据共 140 颗电芯。官方 cycle-life 标量任务使用 138 个已观察标签；Hybrid 循环 500 轨迹任务只使用 120 颗具有真实完整轨迹的电芯，另外 20 颗以 `INSUFFICIENT_REAL_TRAJECTORY_500` 明确排除，不插补、不外推。

## 一、本地 Windows：提交后生成训练包

训练包只能从干净 Git 工作树生成。先在 GitHub Desktop 确认没有未提交文件，或在 PowerShell 检查：

```powershell
cd "D:\guet_learning\26 AI acting\Battery-energy-storage-zzb"
git status --short
```

输出必须为空。然后生成三批 ZIP64 包及外部 SHA-256 索引：

```powershell
.\.venv\Scripts\python.exe scripts\build_matr_three_batch_a100_package.py `
  dist\quanxin-matr-three-batch-a100.zip `
  --output-index dist\quanxin-matr-three-batch-a100.sha256.json
```

程序会：

1. 检查 Git 工作树干净且记录当前 commit；
2. 逐一校验三份原始 MAT 的登记 SHA-256；
3. 检查三份 MAT 均为 MATLAB v7.3/HDF5；
4. 加入三批早期输入和真实监督 Parquet/JSON；
5. 拒绝 `.pkl/.pickle/.joblib/.pt/.pth`、密钥、环境文件和符号链接；
6. 生成 ZIP 内部清单；
7. 重新读取整个 ZIP，逐文件复验大小和 SHA-256；
8. 生成 ZIP 外部索引。

再次本地验证：

```powershell
.\.venv\Scripts\python.exe scripts\verify_matr_three_batch_a100_package.py `
  dist\quanxin-matr-three-batch-a100.zip `
  dist\quanxin-matr-three-batch-a100.sha256.json
```

只上传这两个文件：

```text
quanxin-matr-three-batch-a100.zip
quanxin-matr-three-batch-a100.sha256.json
```

## 二、上传到 A100 服务器

以下路径可按实验室实际目录调整：

```bash
ssh dell@<服务器地址>
mkdir -p ~/quanxin-transfer ~/quanxin-runs
exit
```

在本地 PowerShell 上传：

```powershell
scp dist\quanxin-matr-three-batch-a100.zip `
  dist\quanxin-matr-three-batch-a100.sha256.json `
  dell@<服务器地址>:~/quanxin-transfer/
```

若使用移动硬盘、SFTP 或实验室网页上传，也必须同时传 ZIP 和 JSON 索引，不能只传 ZIP。

## 三、服务器：解压前校验外部 ZIP 哈希

登录服务器：

```bash
ssh dell@<服务器地址>
cd ~/quanxin-transfer
python - <<'PY'
import hashlib
import json
from pathlib import Path

archive = Path("quanxin-matr-three-batch-a100.zip")
index = json.loads(
    Path("quanxin-matr-three-batch-a100.sha256.json").read_text(encoding="utf-8")
)
digest = hashlib.sha256()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
actual = digest.hexdigest()
assert archive.stat().st_size == index["size_bytes"], "ZIP size mismatch"
assert actual == index["archive_sha256"], "ZIP SHA-256 mismatch"
print("external archive verified:", actual)
PY
```

任何断言失败都应停止，重新上传，不要继续解压。

建立新的空目录并解压：

```bash
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
PROJECT_DIR="$HOME/quanxin-runs/$RUN_TAG"
mkdir -p "$PROJECT_DIR"
python -m zipfile -e \
  "$HOME/quanxin-transfer/quanxin-matr-three-batch-a100.zip" \
  "$PROJECT_DIR"
cd "$PROJECT_DIR"
```

## 四、服务器：创建 Conda Python 3.11 环境

不安装独立 CUDA Toolkit，使用 PyTorch wheel 自带 CUDA runtime，由服务器 595.71.05 驱动承载。

推荐直接运行工程脚本：

```bash
cd "$PROJECT_DIR"
bash scripts/a100/create_env.sh
conda activate quanxin-a100
```

若服务器的 Conda 未初始化：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quanxin-a100
```

脚本等价于：

```bash
conda create -n quanxin-a100 python=3.11.13 pip -y
conda activate quanxin-a100
python -m pip install --upgrade "pip>=26.1.2,<27" "setuptools>=83,<84" wheel
python -m pip install torch==2.12.0
python -m pip install --require-hashes -r requirements/a100-linux-py311.lock
python -m pip install -e . --no-deps
python -m pip check
```

环境安装完成后，使用包内受控验证器再次检查 ZIP 内部清单。验证器依赖项目的 Python 环境，因此必须放在 Conda 环境创建之后：

```bash
cd "$PROJECT_DIR"
conda activate quanxin-a100
python scripts/verify_matr_three_batch_a100_package.py \
  "$HOME/quanxin-transfer/quanxin-matr-three-batch-a100.zip" \
  "$HOME/quanxin-transfer/quanxin-matr-three-batch-a100.sha256.json"
```

预期状态：

```text
MATR_THREE_BATCH_A100_PACKAGE_VERIFIED
```

检查关键版本：

```bash
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("visible gpu count:", torch.cuda.device_count())
PY
```

此时尚未设置 `CUDA_VISIBLE_DEVICES`，看到服务器全部 GPU 属于正常现象；正式训练脚本会只暴露物理 GPU 1。

## 五、确认物理 GPU 1 空闲并执行预检

查看物理 GPU：

```bash
nvidia-smi
nvidia-smi -i 1
```

要求：

- 物理 GPU 1 为 `NVIDIA A100 80GB`；
- 显存总量不少于 75 GiB；
- 启动前 GPU 1 已用显存不超过 1024 MiB；
- 不要终止 GPU 0 上其他成员的任务。

执行项目预检：

```bash
cd "$PROJECT_DIR"
conda activate quanxin-a100
bash scripts/a100/preflight.sh
```

训练脚本固定：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
```

因此 PyTorch 内部只看到一张卡，并把物理 GPU 1 映射为 `cuda:0`。这是正确行为，不代表使用了物理 GPU 0。

## 六、先运行三批 Smoke

创建 tmux 会话：

```bash
tmux new -s quanxin-smoke
```

在 tmux 中执行：

```bash
cd "$PROJECT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quanxin-a100
bash -o pipefail -c \
  'bash scripts/a100/train_dataset.sh matr-three-batch smoke 2>&1 | tee smoke-console.log'
```

Smoke 自动完成：

```text
三批原始哈希复验
→ 已处理制品复验/缺失时重建
→ GPU 1 预检
→ 4 个截断点
→ Dummy / Variance / XGBoost / CPMLP / Hybrid
→ 单种子短训练
→ 验证、日志、测试指标与安全制品
```

离开 tmux 但保持训练运行：

```text
Ctrl-b 然后按 d
```

重新进入：

```bash
tmux attach -t quanxin-smoke
```

另开 SSH 监控物理 GPU 1：

```bash
watch -n 2 nvidia-smi -i 1
```

Smoke 输出目录：

```text
runs/a100/matr-three-batch/smoke/
```

检查汇总：

```bash
cat runs/a100/matr-three-batch/smoke/aggregate_metrics.json
find runs/a100/matr-three-batch/smoke -name metrics_test.json | sort
```

Smoke 只验证流水线，不作为正式性能结论。

## 七、中断与断点恢复

正常中断优先在 tmux 中按：

```text
Ctrl-C
```

Trainer 会在安全边界保存检查点。重新执行完全相同命令：

```bash
bash scripts/a100/train_dataset.sh matr-three-batch smoke
```

系统会：

- 跳过已经完成且上下文哈希一致的任务；
- 从最后一个有效检查点恢复中断的深度模型；
- 拒绝损坏或配置、数据、划分、特征、代码 commit 不一致的检查点；
- 强制终止时最多损失当前一个 epoch。

不要手工复制 `last` 为 `best`，不要编辑 `progress.json`，不要删除清单后强行加载权重。

## 八、运行正式五种子实验

只有 Smoke 完成且日志、指标、GPU 绑定均正常后再运行正式实验：

```bash
tmux new -s quanxin-final
```

在 tmux 中：

```bash
cd "$PROJECT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quanxin-a100
bash -o pipefail -c \
  'bash scripts/a100/train_dataset.sh matr-three-batch final 2>&1 | tee final-console.log'
```

正式命令顺序执行：

```text
4 个截断点 × 5 个模型 × 5 个种子 = 100 个任务
```

不会在 GPU 1 上并行抢占。训练上限与验证策略：

| 模型 | 上限 | 验证 | 早停 |
|---|---:|---:|---:|
| CPMLP | 300 epoch | 每 5 epoch | 10 次验证无改善 |
| Hybrid | 500 epoch | 每 5 epoch | 10 次验证无改善 |
| XGBoost | 2000 rounds | 每轮 | 100 rounds 无改善 |
| Dummy/Variance | 无 epoch | 拟合后验证 | 不适用 |

正式输出目录：

```text
runs/a100/matr-three-batch/final/
```

中断后只需重新运行同一条 `final` 命令。不要创建第二份配置来“接着跑”，否则上下文哈希会变化并被拒绝恢复。

## 九、训练结束后的验收与回传

确认顶层文件存在：

```bash
test -f runs/a100/matr-three-batch/final/aggregate_metrics.json
test -f runs/a100/matr-three-batch/final/metrics_test.csv
find runs/a100/matr-three-batch/final -name metrics_test.json | wc -l
```

查看失败或未完成任务：

```bash
grep -R '"training_status"' runs/a100/matr-three-batch/final \
  --include='metrics_test.json' | sort
find runs/a100/matr-three-batch/final -name progress.json -print
```

打包结果，不包含原始 MAT：

```bash
cd "$PROJECT_DIR"
tar -czf "$HOME/quanxin-transfer/quanxin-matr-three-batch-results.tgz" \
  runs/a100/matr-three-batch/final \
  final-console.log
sha256sum "$HOME/quanxin-transfer/quanxin-matr-three-batch-results.tgz" \
  > "$HOME/quanxin-transfer/quanxin-matr-three-batch-results.tgz.sha256"
```

下载回本地：

```powershell
scp dell@<服务器地址>:~/quanxin-transfer/quanxin-matr-three-batch-results.tgz `
  dell@<服务器地址>:~/quanxin-transfer/quanxin-matr-three-batch-results.tgz.sha256 `
  .\server-results\
```

回传给 Codex 联合检查时，至少提供：

- `smoke-console.log` 或 `final-console.log`；
- `aggregate_metrics.json`；
- `metrics_test.csv`；
- 任一 CPMLP、Hybrid 和 XGBoost 任务的 `metrics_test.json`；
- 发生恢复时对应的 `progress.json` 和最后 100 行终端日志；
- `nvidia-smi -i 1` 截图或文本。

## 十、禁止事项

- 不要把物理 GPU 0 改成训练卡；
- 不要删除 `CUDA_VISIBLE_DEVICES=1`；
- 不要将官方 cycle-life 写成 EOL80；
- 不要让 20 颗短轨迹电芯进入 Hybrid 循环 500 损失；
- 不要加载 `.pkl/.joblib/.pt/.pth`；
- 不要手改 split、数据版本或模型制品哈希；
- 不要将 Smoke 指标写入正式比赛结论。
