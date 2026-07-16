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

- HUST 候选清单尚未获人工批准，不能创建正式训练包；
- HUST 字段仍未映射成 Canonical Parquet；
- CPMLP 与 Hybrid 还没有 safetensors 安全制品通道；
- 现在的预检和清单工具不能证明模型性能，也不代表海辰真实工业数据验证。
