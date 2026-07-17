# HUST 隔离转换协议 v1

## 1. 用大白话说明

HUST 原始文件是 pickle。pickle 不只是“数据文件”，打开时可能执行 Python 指令。因此主程序、云服务器和 A100 训练机都不能直接打开它。

安全流程分为三道门：

1. **盘点但不打开**：只看 ZIP 目录和文件字节哈希；
2. **人工确认盘点单**：确认这就是准备处理的固定版本；
3. **一次性隔离转换**：在没有密钥、没有网络、输入只读的临时环境中逐个转换。

目前已完成第 2 道门：候选清单已经人工批准，77 个已解压成员也已作为不透明字节流逐一匹配清单并移至工作区外。由于原始 ZIP 当前不可用，批准清单约束下的二次 ZIP 审计尚未执行，代码仍会拒绝将它标记为可转换。

## 2. 静态审计命令

```powershell
.\.venv\Scripts\python.exe scripts\audit_hust_archive.py `
  --archive <工作区外的our_data.zip绝对路径> `
  --manifest data\interim\hust\raw-manifest.json `
  --catalog configs\data_sources.json `
  --output data\interim\hust\archive-audit-unfrozen-v2.json
```

该命令不会反序列化 pickle。未冻结时输出必须是：

```text
ready_for_conversion = false
inventory_version = null
```

## 3. 生成候选 inventory

```powershell
.\.venv\Scripts\python.exe scripts\freeze_hust_inventory.py `
  --audit data\interim\hust\archive-audit-unfrozen-v2.json `
  --inventory-version hust-mendeley-v2-local-bytes-v1 `
  --output data\interim\hust\inventory-candidate-v1.json
```

候选清单的 `review_status` 必须是 `CANDIDATE`。候选只便于审核，不能授权转换。

## 4. 人工批准

项目负责人确认来源、外层哈希、文件数量和成员清单后，才运行：

```powershell
.\.venv\Scripts\python.exe scripts\approve_hust_inventory.py `
  --candidate data\interim\hust\inventory-candidate-v1.json `
  --approved-by project-owner `
  --confirm-inventory-sha256 <候选清单中显示的inventory_sha256> `
  --output configs\data_manifests\hust_mendeley_v2_inventory_v1.json
```

`--confirm-inventory-sha256` 要求批准人把审阅页看到的候选摘要原样填入，避免只写一个批准人名称就误操作放行。这一步的结果是：项目明确把这一个固定本地快照作为后续隔离转换的输入基准。它不证明上游网站字节一致，也不证明 pickle 内部安全。

## 5. 按批准清单二次审计

```powershell
.\.venv\Scripts\python.exe scripts\audit_hust_archive.py `
  --archive <工作区外的our_data.zip绝对路径> `
  --manifest data\interim\hust\raw-manifest.json `
  --catalog configs\data_sources.json `
  --inventory configs\data_manifests\hust_mendeley_v2_inventory_v1.json `
  --output data\interim\hust\archive-audit-approved-v1.json
```

只有全部字节重新匹配时，输出才允许出现：

```text
ready_for_conversion = true
review_status = APPROVED（记录在 inventory 中）
```

## 6. 隔离环境的强制条件

正式转换环境必须同时满足：

- 无项目、LLM、飞书、云服务器或数据库密钥；
- 无不必要网络；
- 非 root 用户；
- 根文件系统只读；
- 原始 ZIP/成员目录只读；
- 输出目录独立可写；
- `cap_drop: ALL` 与 `no-new-privileges`；
- CPU、内存、进程数和输出文件大小有限制；
- 每次只处理一个成员；
- 先输出字段发现 JSON，字段和单位经人工审核后才运行正式转换；
- 输出只允许 Parquet、JSON、JSONL 和哈希清单；
- 输出不得包含 `.pkl`、`.pickle`、`.joblib`、`.pt` 或 `.pth`。

当前 Windows 环境没有 Docker，因此本协议不允许在 `CD` 主环境中临时反序列化。必须等待可用的一次性虚拟机或隔离容器。

## 7. 当前本地状态与风险

经用户明确授权，原位于 `data/HUST/` 的 77 个 pickle 已逐文件校验大小和 SHA-256，并移动至工作区外 `D:\guet_learning\26 AI acting\hust-quarantine-raw`。移动前后文件数均为 77，总字节数均为 `4,245,730,747`；未删除或反序列化任何成员。

原始 `our_data.zip` 当前不可用，因此不能生成 `ready_for_conversion=true` 的批准后审计结果。主应用、Docker 构建和 A100 上传必须继续排除外部原始目录；在重新取得并核验原 ZIP、完成隔离字段发现和人工字段审核前，不得把这些 pickle 用作训练输入。
