# 可复现性

本项目把复现分为代码门禁、受控证据复核、正式训练和完整应用四个层级。
不同层级需要不同数据与基础设施，不能把“代码测试通过”表述为“正式训练已复现”。

## 版本与基本要求

- Python 3.11；
- 包布局：`src/quanxin_life`；
- 前端：Node.js 24、`pnpm@10.28.1`；
- 正式 Advanced source commit：
  `232d9fc8957bb547ca2b80205802f377864e982b`；
- Advanced Final output SHA-256：
  `d201224870a28c655f66a810bc94f90ad28133e06f2fb4a7285195274c303d82`。

依赖范围以 [`pyproject.toml`](../pyproject.toml) 和前端锁文件为准。

## 层级 0：代码与契约

不需要原始 MATR 或模型权重：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[agents,auth,data,dev,knowledge,llm,ml,api,infrastructure,persistence,reporting]"
python -m pytest -q
python -m ruff check .
python -m mypy
python -m compileall -q src workbench deploy migrations
python -m pip check
```

前端：

```powershell
cd frontend
pnpm install --frozen-lockfile
pnpm test
pnpm lint
pnpm typecheck
pnpm build
```

该层证明实现满足自动化契约，不证明正式模型性能。

## 层级 1：foundation API

```powershell
python -m uvicorn deploy.foundation_api:app --host 127.0.0.1 --port 8000
```

可检查：

```text
GET /health
GET /v1/tools
GET /docs
```

foundation API 不包含完整项目认证、Advanced route、Worker 和 UI 数值闭环。

## 层级 2：正式结果包复核

需要由受控人工通道取得：

```text
quanxin-advanced-final-results-20260723T015211Z.tgz
quanxin-advanced-final-results-20260723T015211Z.tgz.sha256
```

先验证传输哈希，再解压到独立只读接收目录。不得直接把下载目录登记为可服务模型。

正式结果包应满足：

```text
mode=final
operations=80
files=2654
source_commit=232d9fc8957bb547ca2b80205802f377864e982b
output_sha256=d201224870a28c655f66a810bc94f90ad28133e06f2fb4a7285195274c303d82
```

安全导入入口：

```powershell
python scripts/import_a100_suite_run.py `
  <解压后的-final-目录> `
  configs/training/matr_three_batch_final.json `
  <受管注册目录> `
  --expected-source-commit 232d9fc8957bb547ca2b80205802f377864e982b `
  --transfer-archive <结果归档> `
  --transfer-sha256 <归档SHA-256>
```

`A100SuiteRunImporter` 在不加载权重的情况下检查完整矩阵、输出索引、文件大小、
SHA-256、符号链接、路径逃逸和禁止格式。

## 层级 3：正式训练

### 原始数据

需要三批 MATR MATLAB v7.3/HDF5 文件：

```text
2017-05-12_batchdata_updated_struct_errorcorrect.mat
2017-06-30_batchdata_updated_struct_errorcorrect.mat
2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

对应清单：

- `configs/data_manifests/matr_2017_05_12_batch_v1.json`
- `configs/data_manifests/matr_2017_06_30_batch_v1.json`
- `configs/data_manifests/matr_2018_04_12_batch_v1.json`

每个文件必须复验类型、大小和 SHA-256。原始数据不进入 Git。

### 划分

固定按 `cell_id`：

```text
train=82
validation=19
calibration=12
test=27
```

禁止按行、周期或采样点随机划分。训练和模型选择不得读取 test 标签；
当前冻结 test 已用于一次性正式晋级，不应继续用于调参。

### A100 运行

```bash
bash scripts/a100/create_env.sh
conda activate quanxin-a100
bash scripts/a100/preflight.sh
bash scripts/a100/train_dataset.sh matr-three-batch smoke
bash scripts/a100/train_dataset.sh matr-three-batch final
```

正式 Advanced 还使用：

```bash
bash scripts/a100/train_advanced_models.sh matr-three-batch select
bash scripts/a100/train_advanced_models.sh matr-three-batch final
```

A100 服务器采用离线人工文件传输，不要求 Git、SSH、SCP 或网盘。
代码热修复和结果归档必须附独立 SHA-256 清单。

## 层级 4：完整 Advanced 应用

需要：

- PostgreSQL 并升级 Alembic 到 head（当前包含 0009–0015 的 Advanced 路由、
  exact-step 和 calibration migration）；
- Redis 与 Celery Worker；
- 受管模型、结果和 calibration 证据目录或对象存储；
- 认证、项目、record batch 和人工审批数据；
- active route 和 representative checkpoint；
- READY calibration materialization；
- FastAPI 完整装配与 Next.js UI。

纵向验收入口：

```powershell
python -m pytest tests/e2e/test_project_advanced_calibration_workflow.py -q
```

该测试使用真实应用装配和证据解析，但模型前向边界为显式测试适配器。
目标环境仍需单独完成真实 Worker、真实权重和浏览器冒烟。

## 指标复核

正式结果包括：

- `training_log.jsonl`；
- `metrics_epoch.csv`；
- `metrics_validation.csv`；
- `metrics_test.json`；
- `run_status.json`；
- safetensors 和逐文件 SHA-256；
- RUL/SOH 逐样本 CSV 与 Parquet；
- metrics closure、Conformal、promotion 和 figure manifest。

本地 CPU 复算与 A100 聚合指标最大差异：

```text
RUL: 0.015913 cycles
SOH: 7.05e-8
```

这是 CPU/CUDA 浮点执行差异。若出现更大或结构性差异，应停止发布并检查样本顺序、
特征版本、checkpoint、数据清单和目标定义。

## 已知来源差异

```text
A100 input_bundle_sha256:
1f031ca773407fc08c65ca4753e97258391b37114d9135c541d2f325ff43a025

本地重建 input_bundle_sha256:
6a055cd4eea20c58ad1ff263313c5b3791074bde58dffc44288f664c8d13a8fd
```

逐样本预测已经与 A100 指标对账，但两个清单字节哈希不相同，必须保留这一溯源差异。
正式结果包也未包含 `mlruns`，因此不能声称 MLflow 存储已经随归档交付。

## 不可复现时的停止条件

遇到以下任一情况应停止，而不是补写数值：

- 原始数据、结果归档或模型制品 SHA-256 不匹配；
- 数据不是按 `cell_id` 隔离；
- source commit、配置、特征或 split 版本不一致；
- 使用 pickle、joblib、`.pt` 或 `.pth`；
- 缺少 calibration/test 隔离证据；
- 结果未被 `ToolResult` 和审计账本登记；
- 将 MATR 官方 cycle life 与统一 EOL80 混用。

正式结果见 [Advanced Benchmark](benchmark.md)，部署边界见
[已知限制](limitations.md)。
