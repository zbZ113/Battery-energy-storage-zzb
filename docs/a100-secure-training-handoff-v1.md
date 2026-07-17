# A100 安全训练交接协议 v1

本协议只负责“训练前检查”和“安全文件清单”，不自动上传文件，也不保存实验室账号或密钥。

## 服务器环境从零安装

工程固定使用 Conda Python 3.11.13、PyTorch 2.12.0 和 Linux/Python 3.11 哈希锁。服务器驱动 595.71.05 可以承载 PyTorch wheel 使用的 CUDA 13.0 运行时，不安装独立 CUDA Toolkit，也不使用 CUDA 13.2 nightly。

```bash
bash scripts/a100/create_env.sh
conda activate quanxin-a100
bash scripts/a100/preflight.sh
```

环境脚本会先安装 `torch==2.12.0`，再以 `--require-hashes` 安装 `requirements/a100-linux-py311.lock`，最后执行 `pip check`。若同名环境已经存在，脚本会拒绝覆盖。

正式训练建议在 `tmux` 中执行：

```bash
tmux new -s quanxin-matr
bash scripts/a100/train_dataset.sh matr smoke
# Smoke 验收后：
bash scripts/a100/train_dataset.sh matr final
```

## 你要准备什么

在 HUST 隔离转换和人工字段审核完成后，单独建立一个训练暂存目录。目录只允许包含：

- 安全转换后的 Parquet；
- 已冻结的数据、划分、特征和训练 JSON/YAML 配置；
- 必要的 Python 训练代码；
- 依赖锁文件、数据卡和实验说明。

不得放入原始 HUST pickle、`.env`、API Key、飞书凭证、SSH 私钥、聊天记录、`pickle/joblib/.pt/.pth` 模型或未审核文件。

## 第一步：在目标机器运行环境预检

```powershell
python scripts/a100_preflight.py --output preflight.json
```

结果含义：

- `READY`：Python 3.11、CUDA、GPU 和必需训练包均可见；
- `DEGRADED`：可生成报告，但当前机器不具备完整 GPU 训练条件；
- `BLOCKED`：Python 版本不符或必需包缺失，不能开始正式训练。

报告不会读取或输出环境变量、用户名、主机名和密钥。

正式的一键脚本会在 GPU 预检前执行 `scripts/prepare_matr_training_data.py`。该步骤先核验原始 MATR 文件的来源清单与 SHA-256；缓存存在时逐电芯复验内容寻址 Parquet/JSON 和循环 500 监督制品，缓存缺失时才从已批准的 `.mat` 重建。特征制品上限固定为循环 150，未来监督轨迹仍保存在独立目录。

## 第二步：在上传前生成训练包清单

```powershell
python scripts/build_a100_training_manifest.py `
  D:\safe-training-staging `
  D:\safe-training-manifest.json `
  --data-version hust-safe-v1 `
  --split-version hust-cell-split-v1
```

清单必须放在暂存目录外，否则清单本身会改变待校验文件集合。程序逐文件检查路径、类型、敏感内容、大小和 SHA-256；任何危险文件都会阻断整个清单生成。

## 第三步：人工上传

仅上传暂存目录和清单。上传后在 A100 机器再次运行相同清单校验，确保传输前后字节一致。当前仓库尚未取得你的实验室任务提交方式，因此不自动执行上传、排队或训练。

## 当前明确限制

- HUST inventory 已获人工批准，但原始 ZIP 尚未完成批准后复审，且字段尚未安全转换，因此不能创建正式训练包；
- HUST 字段仍未映射成 Canonical Parquet；
- CPMLP 与 Hybrid 已有 safetensors + 严格架构/特征 JSON 往返通道，但尚未用真实 HUST 训练制品验证；
- 现在的预检和清单工具不能证明模型性能，也不代表海辰真实工业数据验证。

## 深度模型输出格式

CPMLP 与 Hybrid 的正式推理制品只能包含：

```text
<artifact_id>/
├── model.safetensors
├── architecture.json
├── feature_config.json
└── manifest.json
```

加载器会重新检查四个文件的目录边界、SHA-256、大小、架构白名单、特征上下文、权重键、形状、float32 类型和有限值。目录里多出任何未登记文件也会拒绝加载。优化器状态不进入正式推理制品。

## 训练结果下载验收

A100 输出必须包含 `run_manifest.json`、解析后的配置、JSON/CSV 指标、JSONL 日志、模型卡、环境清单、至少一张图和至少一个安全模型制品。`run_manifest.json` 必须记录输入训练包与预检哈希、干净代码提交、数据/划分/特征版本、截断点、模型、随机种子和 UTC 时间。

输出索引应放在结果目录外。下载后执行：

```powershell
python scripts/verify_a100_training_output.py `
  D:\downloaded-a100-run `
  D:\downloaded-a100-run-index.json
```

缺文件、多文件、哈希变化、符号链接、秘密内容或 `.pkl/.pt/.pth` 等危险格式都会阻断验收。

## 生成 MATR A100 训练包

在本机确认 Git 工作树干净后运行：

```powershell
python scripts/build_matr_a100_package.py dist/quanxin-matr-a100.zip
```

命令只会收集 Git 已跟踪的源码与配置，并显式加入来源清单已登记的
`2018-04-12_batchdata_updated_struct_errorcorrect.mat`、截止循环 150 的早期输入制品和循环 500
监督制品。原始 MAT 必须同时通过路径、文件名、SHA-256、MATLAB 7.3 头和偏移 512 的 HDF5
签名校验。任意其他 MAT、pickle、joblib、`.pt`、`.pth`、符号链接或未登记处理格式都会阻断。
当前来源清单中的许可证字段仍为 `must_verify_before_download`；这不影响团队内部受控训练传输，
但在对外发布数据或竞赛复现包前必须补齐 MATR 官方许可依据，不能把该字段表述为已获公开再分发授权。

程序生成两个文件：

```text
dist/quanxin-matr-a100.zip
dist/quanxin-matr-a100.sha256.json
```

ZIP 内含 `source_revision.json` 和逐文件哈希清单；外部索引绑定整个 ZIP 字节流。二者必须一起
上传。由于已批准的原始 MATR 文件约 3.24 GB，打包与首次复验会顺序读取该文件多次，这是完整
字节校验的预期行为。

服务器收到文件后，在解压前运行：

```bash
python scripts/verify_matr_a100_package.py \
  /path/to/quanxin-matr-a100.zip \
  /path/to/quanxin-matr-a100.sha256.json
```

只有输出 `MATR_A100_PACKAGE_VERIFIED` 后才能解压。验证器不会调用 `extractall`，会先核验整包
SHA-256，再核验包内清单、精确文件库存、每个文件的大小与 SHA-256，以及原始 MAT 的 HDF5
签名。
