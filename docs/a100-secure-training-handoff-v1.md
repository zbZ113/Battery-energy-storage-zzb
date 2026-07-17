# A100 安全训练交接协议 v1

本协议只负责“训练前检查”和“安全文件清单”，不自动上传文件，也不保存实验室账号或密钥。

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
