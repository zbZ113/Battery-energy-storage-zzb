# 泉芯智寿产品补全、可信知识库、云端部署与 A100 模型接入计划

> 当前执行版本：2026-07-24  
> 项目：泉芯智寿——面向新能源装备产业的多智能体储能电芯退化感知与决策系统  
> 状态：A100 高级模型训练已完成，进入指标收口、模型晋级、产品接入与竞赛交付阶段  
> 本计划承接 `2026-07-12-quanxin-zhishou-full-implementation.md`、`2026-07-17-a100-real-training-evaluation-pipeline.md` 和 `2026-07-19-cyclepatch-batlinet-hybridpatch-a100-model-upgrade.md`。

## 一、当前事实基线

### 1. 已完成并通过验证的 A100 训练

- 高级 Select 已完成，`selection_manifest.json` 和 `final_config_resolved.json` 已冻结。
- Advanced Final 完成 4 个 cutoff、4 个模型、5 个随机种子，共 80 个正式运行。
- A100 输出验证结果：

```text
status=VERIFIED
mode=final
operations=80
files=2654
source_commit=232d9fc8957bb547ca2b80205802f377864e982b
output_sha256=d201224870a28c655f66a810bc94f90ad28133e06f2fb4a7285195274c303d82
```

- 80 个运行均包含 `training_log.jsonl`、`metrics_epoch.csv`、`metrics_validation.csv` 和 `metrics_test.json`。
- 21 个运行正常跑满，59 个运行按既定规则提前停止。
- 结果包包含 897 个 safetensors 文件；模型与结果文件均有 SHA-256 证据。
- A100 环境不使用 Git、SSH、SCP 或网盘。后续若需更新运行代码，继续采用人工文件传输通道、离线 ZIP、清单和 SHA-256 校验。

### 2. 已完成的本地逐样本预测取证

本地结果根目录：

```text
server-results/advanced-final-20260723T015211Z/
```

该目录由 `/server-results/` 规则排除，不进入 Git。

已生成并验证：

- 40 个 RUL 运行、1,080 行逐电芯预测；
- 40 个 SOH 运行、500,000 行逐电芯逐循环轨迹；
- 20、50、100、150 四个 cutoff；
- 随机种子 38、39、40、41、42；
- CSV 与 Parquet 双格式；
- 预测导出清单与对账报告。

本地 CPU 复算与 A100 聚合指标对账结果：

```text
RUL 最大绝对差异：0.015913 cycles
RUL 平均绝对差异：0.002222
SOH 最大绝对差异：7.05e-8
SOH 平均绝对差异：4.77e-9
```

差异属于 CPU/CUDA 浮点执行差异，不是样本、模型或配置错位。

### 3. 当前正式实验结果

#### RUL 当前最佳候选

`CyclePatch Direct + cutoff 150`：

| 指标 | 五种子均值 |
|---|---:|
| MAE | 98.19 cycles |
| RMSE | 122.49 cycles |
| MAPE | 12.62% |
| R² | 0.8656 |

该结果可用于科研、比赛展示和离线辅助分析，但不得称为分类准确率、SOTA 或工业寿命承诺。

#### SOH 当前候选

| 模型 | 最低 MAE | 对应 RMSE | 单调违规率 |
|---|---:|---:|---:|
| HybridPatch-v2 | 1.281 SOH 百分点 | 2.700 SOH 百分点 | 0% |
| Current Hybrid | 1.610 SOH 百分点 | 2.578 SOH 百分点 | 0% |

HybridPatch-v2 平均误差更低，Current Hybrid 尾部大误差更小。SOH 正式模型尚未完成晋级判断。

### 4. 已完成的论文图件

已生成 6 张主图与 6 张补充图，采用 Python/Matplotlib 和真实实验数据：

- 数据集与任务全景；
- RUL 指标随 cutoff 变化；
- RUL 真实值与预测值；
- SOH 代表电芯轨迹；
- SOH 远期误差；
- 模型精度、稳定性、耗时与显存权衡；
- 训练曲线、随机种子稳定性、批次性能、寿命分组误差、效率和证据完整性。

每张图均包含 SVG、PDF、PNG 和 600 DPI TIFF，并附 source data、元数据和 SHA-256。

### 5. 必须保留的证据边界

- 本次离线结果包未包含 `mlruns`，不得声称 MLflow 存储已随结果包交付。
- A100 训练时 `input_bundle_sha256` 与当前本地重建值不同；该差异必须写入数据溯源说明。
- 当前标量目标是 MATR 官方 cycle life，不得表述为统一 EOL80 或真实 15–25 年工业寿命。
- 当前 SOH 轨迹验证到真实 cycle 500，不得宣称已可靠预测完整 1000+ 循环轨迹。
- HUST 外部验证、跨数据集 Conformal 重校准和真实工业数据验证尚未完成。

## 二、后续执行总顺序

1. 指标收口与模型晋级；
2. A100 正式套件登记与 ToolResult 接入；
3. 高完成度 Next.js 门户；
4. 真实数据上传、质检、自动冻结、分析和下载；
5. 可信知识库和个人研究工作空间；
6. 多 LLM、Agent 记忆、邮件和飞书协同；
7. 本地完整 Docker Compose；
8. 阿里云、域名备案、HTTPS 和 OSS 备份；
9. HUST 外部验证、HybridPatch-v2.1 等后续研究；
10. 华为杯材料、演示视频和匿名复现包。

模型结果已足够支持工程继续推进，不再以重复 A100 训练作为产品开发前置条件。

## 三、阶段 A：指标收口与正式模型晋级

### 1. 补齐易理解和论文级指标

基于现有逐样本预测直接计算，不重新训练：

- ±5%、±10%、±15%、±20% 相对误差命中率；
- BatteryLife 同口径 `15%-Acc`；
- 逐电芯绝对误差和相对误差；
- 短寿命、中寿命、长寿命分组误差；
- 三个 MATR 批次分别的指标；
- 误差分位数 P50/P75/P90/P95/max；
- 10,000 次逐电芯 Bootstrap 95% 置信区间；
- 模型间逐电芯配对差异与置信区间；
- SOH 按预测距离、cycle 区间、电芯和批次分组的 MAE/RMSE；
- SOH 尾部高误差电芯清单；
- 单调性、非法范围和异常恢复检查。

所有新增结果写入 `server-results/.../analysis/metrics-closure/`，不进入 Git；可复用分析代码进入 `scripts/` 或 `src/quanxin_life/` 并纳入 Git。

### 2. Conformal 与可信区间

- 使用独立 calibration 电芯，测试电芯不得参与校准；
- Split Conformal 作为基线；
- Normalized Conformal 作为主方法；
- 报告 80%、90%、95% 目标覆盖率；
- 输出 PICP、MPIW、分批次覆盖率、寿命分组覆盖率；
- 跨域未重校准时不得宣称覆盖保证；
- 区间跨过决策阈值时输出 `RECHECK`，不能给出强制通过结论。

### 3. 模型晋级规则

RUL：

- 以固定 test split、五种子聚合结果为主；
- 综合 MAE、RMSE、MAPE、R²、15%-Acc、P90 误差、Conformal 覆盖和推理成本；
- 不因单一 MAE 最低就自动晋级；
- 当前优先评估 `CyclePatch Direct + cutoff 150`。

SOH：

- 同时检查 MAE、RMSE、尾部误差、远期误差、单调性和推理成本；
- 比较 Current Hybrid 与 HybridPatch-v2；
- 若 HybridPatch-v2 的尾部风险不能接受，则不得仅凭平均 MAE 晋级；
- 可按应用角色保留“平均精度模型”和“保守尾部模型”，但报告必须明确路由规则。

### 4. 阶段交付

- 正式模型晋级报告；
- 模型卡；
- 数据卡；
- 指标总表；
- 分组指标；
- Bootstrap 与 Conformal 报告；
- 失败案例清单；
- 适用范围和拒绝条件；
- 每个正式制品的 SHA-256。

## 四、阶段 B：A100 结果登记与正式推理接入

### 1. 正式套件导入

- 导入完整 Advanced Final 套件，而不是单独复制权重；
- 验证运行清单、数据版本、split、配置、日志、指标、模型卡和文件哈希；
- 不完整套件只可登记为候选，不得激活；
- 保存 A100 来源提交、输出索引哈希和已知输入 bundle 差异；
- 禁止加载 pickle、joblib、`.pt`、`.pth`。

### 2. 模型注册

- 20、50、100、150 cutoff 分别登记候选与冠军；
- 历史模型不可覆盖；
- 支持人工激活和一键回退；
- 每个结果绑定实际模型版本、数据版本、feature 版本和输入哈希；
- CPU smoke 与正式 A100 模型严格分离。

### 3. ToolResult 接入

- RUL、SOH、Conformal、在线校正和批次决策全部返回统一 ToolResult；
- LLM 不得修改 ToolResult 数值；
- 报告和 UI 只读取经过审计的 ToolResult；
- 正式数字缺少 `result_id`、模型版本或数据版本时必须失败关闭。

## 五、阶段 C：高完成度前端和真实数据链

### 1. 页面顺序

1. Owner 登录、首次初始化和邮箱找回；
2. 项目驾驶舱；
3. 普通数据导入与专家导入；
4. 数据质量和版本冻结；
5. 单电芯诊断；
6. 批次风险与复检；
7. 科研证据工作台；
8. Agent 执行中心与页面侧边助手；
9. 知识库和 Agent 记忆；
10. 技术报告与导出中心；
11. LLM、模型、通知、备份和集成管理后台。

产品界面仅呈现单 Owner 和演示沙箱，不继续开发评委账号。

### 2. 上传与自动分析

- 普通上传采用固定模板 CSV/Parquet；
- 支持多文件同批上传、分片、暂停和重试；
- 电脑端单文件 500 MB、单数据集 2 GB；
- 手机端超过 50 MB 提示使用电脑；
- 浏览器使用短期签名凭证写入 MinIO 临时区；
- 单位必须确认；
- 坏文件进入隔离区，不自动跳过；
- 质检全绿后自动创建不可变数据版本并启动分析；
- 有警告时停止自动冻结；
- 冻结版本不能原地修改，只能创建新版本。

### 3. 专业展示

- 强制标注预测目标定义；
- Observed、Predicted、Newly Observed、Simulated 四类曲线；
- Conformal 区间；
- 在线更新前后对比；
- 全屏图表、缩放、框选、同步十字线和数据下载；
- 批次风险矩阵和决策原因；
- 模型对比、消融、稳定性、效率、Naumann 和 PyBaMM 页面；
- 现有 12 张论文图作为科研证据入口，但页面优先使用机器可读数据动态绘图。

### 4. 下载

- 支持技术报告、图表、CSV、ToolResult、数据卡、模型卡和配置逐项下载；
- 多选后异步生成 ZIP；
- ZIP 附清单、版本、生成时间和 SHA-256；
- 产品只生成技术报告，比赛材料单独维护。

## 六、阶段 D：可信知识库与研究工作空间

### 1. 语料范围

- 公共电池基础库：论文、数据集说明、PyBaMM、Conformal、主动试验和工业规范；
- 项目私有库：实验协议、数据卡、模型卡、失败分析和项目笔记；
- 默认搜索公共库和当前项目私有库；
- 禁止跨项目检索私有文档。

### 2. 文档流程

支持 PDF、Markdown、TXT、DOCX：

```text
上传
→ 安全预检
→ 许可证检查 / OCR
→ Owner 二次审核
→ 语义切块
→ Embedding 与 BM25 索引
→ 发布不可变 corpus_version
```

- 未知许可证可暂存，但不能检索和引用；
- 扫描 PDF 按需本地 OCR，并人工确认；
- 文档新版本不覆盖旧版本；
- 报告绑定具体文档、语料和索引版本；
- 文档中的提示注入只作为资料内容，不能成为 Agent 指令。

### 3. 检索与降级

```text
权限过滤
→ 中文 BM25
→ 远程 Embedding
→ 候选合并
→ 本地轻量 reranker
→ 带页码证据
```

降级顺序：远程 Embedding → 本地 Embedding → BM25。reranker 失败时保留召回结果并显示警告。

### 4. 研究能力

- PDF 页码跳转、高亮、批注和收藏；
- 跨文献对比；
- DOI、作者、年份和期刊元数据；
- BibTeX、RIS、GB/T 7714；
- 中英文术语表；
- 带页码的证据关系图；
- 新论文关键词订阅和候选审核；
- Zotero Web API 手动同步与每日同步；
- Zotero 只自动同步元数据，附件需人工审核；
- 同步冲突必须人工解决。

### 5. 回答边界

- Agent 回答每个实质结论必须带引用；
- 引用可打开原文对应页；
- 证据不足时明确拒绝补写；
- 知识证据不得覆盖模型 ToolResult；
- 外部搜索必须由 Owner 主动触发，候选资料审核后才可正式引用。

## 七、阶段 E：LLM、Agent 记忆和外部协同

### 1. 多 LLM

- 支持多套 OpenAI 兼容服务；
- 每套配置主模型、经济模型和 Embedding 模型；
- API Key 加密保存，主密钥只存在服务器 Secret；
- 服务故障可按优先级切换；
- 月预算达到硬上限后所有供应商停止调用；
- LLM 断线时固定数值工作流继续运行。

### 2. Agent 记忆

- 原始聊天保存 30 天；
- 只有 Owner 确认的结构化信息进入长期记忆；
- 提供查看、修正、停用、导出和删除；
- 记忆、论文证据和 ToolResult 必须分开显示。

### 3. 通知与飞书

- 站内、邮件、飞书三路通知；
- 飞书支持创建任务、状态卡、审批、多维表格和报告链接；
- 外部通知只发送状态、少量审计摘要和安全链接；
- 验签、去重、幂等、重试和死信必须完整；
- 未配置真实凭证时明确使用沙箱。

## 八、阶段 F：本地与阿里云部署

### 1. 本地 Compose

```text
caddy
frontend
api
worker
scheduler
streamlit
postgres
redis
minio
mailpit
ocr-worker
```

Prometheus 和 Grafana 不常驻；管理页提供服务、磁盘、数据库、队列、对象仓、知识索引和备份的轻量健康状态。

### 2. 阿里云

- 当前 ECS 先执行实际报价门；
- 月预算不超过 300 元；
- 4 核 16 GB 超预算时降为 4 核 8 GB；
- Worker 并发为 1；
- 增加约 8 GB swap 和约 100 GB 数据盘；
- 保留 Ubuntu 22.04；
- 绑定 EIP；
- 购买域名、实名认证、备案并启用 HTTPS；
- 数据库、Redis 和 MinIO 控制台不得暴露公网；
- 原始 MATR/HUST 全集不进入比赛服务器。

### 3. 备份

- 每日数据库加密备份；
- 每日对象清单；
- 每周关键对象完整备份；
- OSS 异地备份；
- 每月恢复演练；
- 备份密钥和服务器登录密钥分离。

## 九、阶段 G：后续研究与比赛交付

### 1. 后续研究

- HUST 安全转换和外部测试；
- MATR 两批训练、第三批完全留出；
- CORAL、DANN 和目标域重校准；
- HybridPatch-v2 尾部误差定位；
- 仅在有明确研究假设时运行 V2.1，不以盲目调参阻塞产品主线；
- 更长 SOH 轨迹实验必须使用真实达到目标周期的电芯。

### 2. 华为杯证据主线

1. 早期循环数据治理；
2. 可信寿命与 SOH 预测；
3. Conformal 风险区间；
4. 在线个体更新；
5. PyBaMM 短时物理核验；
6. 主动试验推荐；
7. 受约束多智能体编排；
8. ToolResult 数值防火墙；
9. 单电芯主场景和批次第二场景；
10. 真实实验、失败案例和边界说明。

### 3. 材料

- 300 字项目简介；
- 匿名完整项目文档；
- 实验协议；
- 数据卡、模型卡和 Agent 卡；
- 正式实验表和 12 张图；
- 消融、Conformal、在线更新和 Agent 可靠性结果；
- 演示视频；
- 一键部署与复现说明；
- 第三方许可证和数据来源清单；
- 已知限制和失败案例。

## 十、公共接口增量

需要新增或扩展的公共契约包括：

- `UploadSession`
- `DatasetImportStatus`
- `DatasetFileStatus`
- `UnitMapping`
- `DatasetQualitySummary`
- `ExportSelection`
- `ModelPromotionDecision`
- `LlmProviderProfile`
- `LlmBudgetState`
- `AgentMemoryRecord`
- `KnowledgeScope`
- `KnowledgeDocumentVersion`
- `KnowledgeCorpusSnapshot`
- `KnowledgeEmbeddingIndex`
- `KnowledgeCitation`
- `ZoteroSyncRun`
- `DemoSession`

不得新建与 `quanxin_life.core` 现有类型语义重复的 DTO。

## 十一、质量门禁

每个纵向切片执行：

```text
失败测试
→ 最小实现
→ 目标测试
→ 相关集成测试
→ 全量门禁
→ 差异审查
→ 本地 Git 提交
```

后端：

```text
pytest -q
pytest tests/leakage -q
pytest tests/integration -q
pytest tests/e2e -q
ruff check .
mypy src/quanxin_life
python -m compileall -q src workbench deploy
pip-audit
```

前端：

```text
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm playwright test
```

部署：

```text
docker compose config
docker compose up -d
docker compose ps
健康检查
数据库迁移检查
对象存储校验
OSS 备份与恢复演练
```

没有最新命令输出，不得声称完成或通过。

## 十二、当前立即执行的任务

### Task 1：指标收口工具

状态：**已完成（2026-07-24）**。

已生成：

```text
server-results/advanced-final-20260723T015211Z/analysis/metrics-closure/rul-v1/
server-results/advanced-final-20260723T015211Z/analysis/metrics-closure/soh-v1/
```

完成内容：

- RUL 10,000 次逐电芯 Bootstrap；
- RUL ±5%/±10%/±15%/±20% 命中率；
- RUL 批次、寿命分组、模型配对比较和失败电芯清单；
- SOH 10,000 次逐电芯 Bootstrap；
- SOH ±1/±2/±5 个百分点命中率；
- SOH 批次、预测距离、模型配对比较和尾部失败电芯清单；
- 输入预测 Parquet、原预测清单和全部输出 SHA-256 绑定。

当前新增判断：

- Direct-150 的 15%-Acc 为 64.44%，20%-Acc 为 81.48%；
- Direct 与 BatLiNet 的逐电芯 MAE 配对区间跨过零，不能声称 Direct 在当前测试集上统计显著优于 BatLiNet；
- HybridPatch-v2 在四个 cutoff 的逐电芯轨迹 MAE 配对区间均优于 Current Hybrid；
- Current Hybrid 的总体 RMSE 仍更低，HybridPatch-v2 不能仅凭 MAE 直接成为唯一 SOH 冠军；
- SOH 远期 201–350 cycles 区间误差明显放大，高误差电芯主要集中在 2017-06-30 批次。

读取现有逐样本 RUL/SOH Parquet，生成：

- accuracy-at-tolerance；
- Bootstrap 置信区间；
- 批次、寿命分组和预测距离指标；
- 失败电芯清单；
- 机器可读 JSON/CSV 和 SHA-256 清单。

#### Conformal 与可信区间

状态：**已完成（2026-07-24）**。

已生成：

```text
server-results/advanced-final-20260723T015211Z/analysis/calibration_predictions/
server-results/advanced-final-20260723T015211Z/analysis/conformal/rul-v1/
```

完成内容：

- 从 40 个 SHA-256 已验证的 RUL safetensors 检查点导出 480 行 calibration 预测；
- calibration 固定为 12 个独立电芯，test 固定为 27 个电芯，两者无交集；
- 保留完整 80-run 训练输入哈希闭包核验和本地重建 bundle 哈希差异；
- Split Conformal 基线与 Normalized Conformal 均报告 80%、90%、95% 目标覆盖；
- Normalized 难度尺度仅使用同模型五个随机种子预测的样本标准差，不读取标签；
- 输出 48 条模型/方法/cutoff/覆盖率汇总、1,296 个测试区间和 336 条分组覆盖；
- 输出 PICP、MPIW、分批次覆盖、寿命四分位覆盖、逐电芯区间和 SHA-256 清单。

当前新增判断：

- 12 个 calibration 电芯属于小校准队列，所有结果必须带 `SMALL_CALIBRATION_COHORT`；
- 有限样本秩在该队列上使 90% 与 95% 使用相同的最大校准分位数，不能解释为两种不同强度的经验保证；
- 五种子分歧尺度的 Normalized Conformal 未稳定改善覆盖-宽度权衡，不能自动作为正式主方法；
- 当前结果只对已声明的 MATR calibration/test 划分有效，不形成 HUST、Naumann 或工业域覆盖保证。

### Task 2：正式模型晋级报告

状态：**已完成（2026-07-24）**。

已生成：

```text
server-results/advanced-final-20260723T015211Z/analysis/model-promotion/v1/
```

完成内容：

- 评估 Direct、BatLiNet、Current Hybrid 和 HybridPatch-v2；
- RUL 按点预测精度与 Split coverage 两个可观测目标冻结条件路由，不用未经批准的权重强行合成单一冠军；
- SOH 冻结平均精度与尾部/效率双模型路由；
- representative seed 只按 `best_validation_metric` 选择，不读取 test 指标；
- 生成 15 条 `CONDITIONAL` 推荐、4 份模型卡、机器可读拒绝策略和 9 个文件的 SHA-256 清单；
- 所有推荐保持 `NOT_ACTIVATED`，不把聚合晋级报告冒充具体模型制品激活；
- 明确训练耗时与峰值显存不是推理延迟，当前没有推理延迟基准；
- 明确冻结 test split 已被本次一次性晋级使用，后续不得继续用其调参或重选模型。

当前冻结路由：

- RUL cutoff 20：CyclePatch Direct 同时承担点精度与 coverage 默认角色；
- RUL cutoff 50：Direct 为 `POINT_ACCURACY`，BatLiNet 为 `COVERAGE`；
- RUL cutoff 100：BatLiNet 为 `POINT_ACCURACY`，Direct 为当前可用的 `COVERAGE` 候选，但两者 90% PICP 均未达到目标；
- RUL cutoff 150：Direct 为 `POINT_ACCURACY`，BatLiNet 为 `COVERAGE`；
- SOH 四个 cutoff：HybridPatch-v2 为 `MEAN_ACCURACY`，Current Hybrid 为 `TAIL_EFFICIENCY`；
- 若 SOH 只能单路由，保留 Current Hybrid，不能仅凭平均 MAE 让 HybridPatch-v2 成为唯一冠军。

Task 2 不新增公共 `ModelPromotionDecision`。聚合统计推荐与具体可激活制品必须分层；公共激活契约、追加式决策账本、人工审批和回退在 Task 3 实施。

### Task 3：正式套件注册与 ToolResult

状态：**实施中（2026-07-25 完成部署制品、managed candidate registry、人工激活/回退账本、active-route resolver、可信 project-scoped invocation context、record batch binding、持久 project ToolResult binding 与持久 AgentRun per-step grant；2026-07-26 完成 `0013` exact-step binding、服务端冻结 canonical input/依赖/审批证据、PROJECT Worker 原子提交与 fenced-claim 恢复，并完成 target-aware RUL、finite-horizon SOH、route-specific Split Conformal、项目 API、Agent、报告和单电芯 UI 代码链；2026-07-27 完成来源验证的 Advanced calibration materialization、identity-only ADMIN API/队列、Agent READY resolver、校准管理 UI 与测试环境纵向 E2E）**。

已生成：

```text
server-results/advanced-final-20260723T015211Z/analysis/deployment-bundles/v1/
server-results/advanced-final-20260723T015211Z/analysis/deployment-registry/v1/
server-results/advanced-final-20260723T015211Z/analysis/advanced-final-registry/v1/
```

已完成内容：

- 将 15 条聚合晋级路由逐条绑定到 representative seed 的具体 best checkpoint；
- representative seed 继续只来自验证指标，不重新读取 test 指标选模型；
- 对完整 Advanced Final 输出索引、promotion 清单、来源证据、checkpoint 上下文和六文件 SHA-256 闭包重新验收；
- 推理恢复只反序列化 `model.safetensors`，optimizer、scheduler 和 RNG 仅做字节校验；
- 为 Advanced Current Hybrid 新增独立制品语义，不复用输入契约不同的 legacy Hybrid；
- 导出 15 个独立 `DeepModelArtifact`，每个包含 safetensors、架构、特征上下文和清单，BatLiNet 另含 reference library；
- 四个模型族均完成原 checkpoint 与导出后安全 loader 的 CPU 前向逐张量对账；
- 总索引显式绑定 data/split/feature、candidate、normalization、target scaler/reference library、selection、promotion 和 A100 来源哈希；
- 独立钉死 promotion manifest `sha256=7fd7e56322a7266b23422b4cd31839f9dc4f729bbab1d6538de970c60bd9f085`，并从 80-run 验证指标重新核对 representative seed；
- 每个导出 artifact 的 weights SHA-256 必须与其源 checkpoint `model.safetensors` 完全一致，resume 不接受仅包内自洽的替换权重；
- 部署索引 `manifest_sha256=9657e34d77122d79e81b5f9d75bbdc0b783e57b1a002ab05a51e3319fbe5d1eb`；
- `input_bundle_hashes_match=false` 原样保留，未把本地重建输入冒充 A100 原始输入；
- 所有 bundle 与路由仍为 `NOT_ACTIVATED`；managed registry 只登记候选，不执行审批、激活、回退或 ToolResult 接入；
- 新增 Advanced deployment managed registry：外部钉死 deployment manifest SHA，按封闭目录复验精确 15 条路由和全部 Deep artifact 字节后原子复制、幂等登记；
- 新增 Advanced Deep artifact Catalog trusted source：每次解析重新验证 managed bundle，向现有产品 Catalog 只暴露 `REGISTERED_CANDIDATE / NOT_ACTIVATED` 候选；
- Catalog metadata 保存 Final、promotion、selection、A100/local input hash 差异、candidate、seed、best epoch 和 checkpoint SHA provenance，不保存 MAE/RMSE 等业务指标；
- 登记过程不调用 safetensors loader、不构造模型、不产生 active route、审批、回退或 ToolResult。
- 真实 managed record `record_sha256=bacb8be304319e1815f1c44eabdff848f5ae58cd55cc86e8b18b05576ee1dc62`，连续登记保持同一时间与摘要；
- Windows managed path 使用完整 SHA 校验身份、16 字符前缀落盘，避免长路径失效；`bundles/records/artifacts` junction、复制期 junction swap、naive UTC 时间和记录篡改均 fail closed。
- 新增 Advanced Final 80-run 专用 importer，不复用只支持单一 RUL target 和三层目录的 legacy importer；
- importer 以外部 transfer SHA、`output_index.json` 文件 SHA 和 Final output SHA 为信任锚，重新验证 2,654 个索引文件与 80 个 best checkpoint，但不复制 15 GB 结果树；
- 80 个任务按 `4 family × 4 cutoff × 5 seed` 精确登记为 40 个 RUL 与 40 个 SOH，SOH 输出目标显式记录为 `soh_trajectory`，不伪装成 MATR 官方 cycle life；
- 登记记录保留 21 个 `COMPLETED`、59 个 `EARLY_STOPPED`、A100/local input bundle 哈希差异、selection/config/source/data/split/feature 和 checkpoint SHA provenance；
- importer 只解析 JSON、路径和 SHA，不读取测试指标选模、不调用 safetensors loader；外部字节篡改、矩阵替换、嵌套 junction、records junction swap 和并发首次登记均 fail closed；
- 真实 Advanced Final 登记 `import_id=d201224870a28c655f66a810bc94f90ad28133e06f2fb4a7285195274c303d82`，`record_sha256=bb68dbe0dd2a80a6d40b0a5de059e186b61fb32331eec1f0d30ede680e5a8c7a`。
- Advanced trusted Catalog source 新增一次复验、固定排序的 `resolve_all()`，完整 15 候选只触发一次 managed bundle 字节验证；
- 产品 `ModelArtifactCatalogService` 新增 15 候选单事务批注册：先做全量 identity/digest/project/metadata 冲突检查，再一次 flush/commit；任一冲突、唯一约束或来源异常均整批回滚；
- 带 Advanced provenance 的制品禁止再走 legacy 单 artifact 注册接口，防止管理员通过 1–14 次单项写入绕过原子批注册；
- 相同并发批请求若发生唯一约束竞态，输家会在新事务中重新读取并核对全部 15 条；完全一致时幂等返回，任一缺失或上下文不同仍按冲突失败；
- 批注册响应固定为 `REGISTERED_CANDIDATE / NOT_ACTIVATED`，不创建 active route、审批、回退或 ToolResult，也不复制 MAE/RMSE 等业务指标；
- 新增 trusted-origin 管理 API `POST /v1/admin/model-artifacts/advanced-candidates`，请求只接受 `project_id`，不接受调用方注入 artifact IDs、路径、哈希或状态；
- 产品 Catalog 本阶段不新增数据库 migration；现有 `model_artifacts/model_manifests` 已能保存 15 个候选及严格 provenance；
- 新增 `model_route_activation_events` 追加式 SQL 账本，按 `project + task + cutoff + role` 维护严格递增序号和 SHA-256 哈希链；历史激活与回退事件不可覆盖；
- 人工激活与回退只允许 ADMIN 在 ACTIVE、可见项目内执行，必须提供 trusted origin、人工原因、`Idempotency-Key` 和预期账本头；请求摘要绑定决策 ADMIN 身份，陈旧账本头、跨管理员复用、重复键异义请求、路由不匹配与历史篡改均失败关闭；
- 激活前重新验证 managed Advanced 候选、Catalog 持久记录及完整 route provenance；并发唯一约束恢复也会重新验证 ACTIVE 项目、完整账本和可信候选，不凭数据库事件单独返回成功；
- 回退只接受更早的有效 `ACTIVATE` 事件 ID，目标 artifact、哈希和 provenance 全部由服务端历史派生，调用方不能注入回退制品；回退通过追加 `ROLLBACK` 事件实现，不修改旧记录；
- 新增管理 API `POST /v1/admin/model-routes/activations`、`POST /v1/admin/model-routes/rollbacks` 和只读历史 API `GET /v1/model-routes/activation-events`；写接口不接受调用方注入路径、状态或 provenance 哈希，competition HTTP 正式装配强制注入该 adapter；
- 新增 `model_route_activation_stream_heads` 完整性锚，只保存路由坐标、head event ID/sequence/hash，不保存 active artifact；`0010` 会为既有 `0009` 事件流按最高序号回填 head；
- 激活/回退事件与 stream head 在同一事务提交；resolver 对项目内全部哈希链和 head 一致性做 fail-closed 校验，可阻断有 event 无 head、有 head 无 history、head 篡改及 SQLite 尾删除后的静默回退；
- 新增冻结的 `VerifiedActiveModelRoute` 契约与内部 resolver：只解析精确 `project + task + cutoff + role`，不自动 fallback；每次解析重新验证 ACTIVE/可见项目、trusted source、Catalog、事件快照和 route provenance；
- Catalog 候选继续保持 `REGISTERED_CANDIDATE / NOT_ACTIVATED`，当前生效状态只由已验证决策账本派生；不新增保存 active artifact 的 current projection；
- resolver 不开放新的 HTTP endpoint、不加载 safetensors、不构造模型、不读取指标、不生成 ToolResult；HTTP、推理加载和 ToolResult 留给下一子切片；
- 新增服务端签发的冻结 `VerifiedProjectInvocationContext`：绑定 project、actor、session、role 和调用来源；签发与每次使用均检查 ACTIVE 用户、密码轮换状态、有效未撤销 session、ACTIVE/可见项目和当前成员关系；伪造签名、项目归档、成员撤销、用户禁用和 session 撤销均 fail closed；
- Tool registry 新增显式 `GLOBAL / PROJECT` 执行域；PROJECT 工具不会出现在通用 HTTP/MCP discovery，也不能经通用 service、MCP 或普通 Agent 入口执行；project executor 只能接收经同一 context service 现场复验的上下文；
- 新增 project-scoped `ProjectAuditLedger`，将 ToolResult 与 exact project、actor、session、tool 和 input hash 绑定；跨项目解析失败，未配置 project ledger 时在 executor 启动前失败；
- 新增 `POST /v1/projects/{project_id}/tools/{tool_name}` 可信项目入口：principal 只来自认证 Cookie dependency，project 只来自 path，经 trusted Origin 和同一 context service 校验后进入 PROJECT registry 与 project audit ledger；请求 payload 不能注入 project、actor 或 route；
- project HTTP 集成测试使用真实 `ToolInvocationService + PROJECT ToolDefinition + ProjectAuditLedger`，不以 fake service 或 GLOBAL tool 冒充项目链路；
- 新增 `0011_record_batch_bindings` 迁移与追加式绑定模型：公开 `record_batch_id` 为服务端 UUID，同一 canonical CSV 内容可在不同项目复用内部 content ID，但每个 dataset 获得独立 opaque binding；公开响应不暴露内部 `content_batch_id`；
- 新增唯一正式上传入口 `POST /v1/datasets/{dataset_id}/batches/canonical-csv`：只允许 ADMIN/MEMBER 向可见 ACTIVE 项目的 DRAFT dataset 上传，project 只从数据库中的 `Dataset.project_id` 派生，请求体不能注入 project、dataset、content ID 或服务端哈希；
- record batch resolver 只接受经现场复验的 `VerifiedProjectInvocationContext`，只在同一 ACTIVE 项目及 FROZEN dataset 内解析；跨项目和不存在资源统一隐藏，dataset、binding、CSV、registration/provenance 或版本快照漂移均 fail closed；
- 相同内容在共享内容存储中遇到不同 registration/provenance 时会在持久化 binding 前拒绝；旧 `/v1/batches/canonical-csv` 无作用域入口已从公开 FastAPI 工厂、Workbench 与运行文档移除，不能再返回内部 content ID；
- 新增 `0012_project_tool_result_bindings` 迁移与 `SqlProjectAuditLedger`：将原样 `ToolResult`、完整 provenance 和 project/actor/session/source/tool/input 绑定在同一数据库事务中追加写入，并保存 ToolResult SHA-256 与 binding SHA-256；
- SQL project ledger 每次解析都重新复验可信 context，只允许同一 ACTIVE 项目读取；服务重启后可恢复，跨项目访问、session 撤销、ToolResult 内容篡改、provenance 篡改和 binding 篡改均 fail closed；
- project HTTP 集成链已改用 SQL ledger，不再以进程内字典冒充正式 project result persistence；内存 `ProjectAuditLedger` 仅保留为轻量边界与单元测试实现；
- 新增从持久 `AgentRun + dispatch + plan + AgentStep + claim + lease` 派生的服务端签名 `VerifiedAgentRunInvocationGrant`：冻结 exact step、plan hash 和单一 PROJECT tool allowlist，不接受 Agent payload 自报 project、actor、run 或 allowlist；
- grant 每次使用前均重新复验 ACTIVE run/session/project/member、计划与持久 step 一致性、DISPATCHED 状态和当前 claim/lease；普通 project service 明确拒绝 AGENT context，防止绕过 per-step grant；
- PROJECT executor 返回后、写 ledger 前再次复验同一 grant；若执行期间 claim 被替换、回收或过期，结果不得写入 `tool_results` 或 `project_tool_result_bindings`；
- 新增 `0013_exact_agent_step_bindings`：`AgentStep` 持久冻结 canonical input、execution snapshot、dependency evidence 与 claim digest；Agent project binding v2 精确绑定 `agent_step_id/step_id/plan_hash`、claim attempt/lease、审批与依赖摘要；无法推导 exact evidence 的 legacy AGENT v1 行在任何 schema 变更前 fail closed，有 v2 审计行时禁止破坏性 downgrade；
- 新增 exact-step 原子提交：同一数据库事务内锁定 run/dispatch/step，重新验证 claim、lease、冻结输入、依赖和审批，写入 ToolResult/provenance/project binding，清除 claim、完成 step 并追加 `STEP_COMPLETED`；任一写入失败全部回滚；
- Worker 先从 plan/reference 编译输入，再通过对应 Pydantic schema canonicalize 并持久化输入 JSON/hash；grant、ToolResult 与最终事务必须绑定同一 input hash，调用方不能替换输入；
- dependency evidence 只接受 ordinal 更早且已完成的显式依赖，每个依赖必须存在唯一持久 ToolResult，并绑定 result、provenance 与可用 project binding 摘要；审批请求必须绑定 exact `agent_step_id + execution_snapshot_sha256 + plan_hash` 和唯一、未过期的 APPROVED action；
- fenced-claim 恢复协议已覆盖 active lease 拒绝抢占、expired lease 重新 claim、旧 claim 禁止提交或清除新 claim、原子写入失败回滚、Worker 重启续跑和重复队列投递不重跑；原始 claim token 不进入 binding 或事件；
- Worker 已按 `GLOBAL / PROJECT` execution scope 分流；PROJECT step 只接受由当前持久 claim 派生的 per-step grant，并通过 `invoke_for_project_agent` 在执行前后复验；普通 project service 以及内存/SQL ledger 的 legacy `register_result` 均拒绝 AGENT context，不能绕过 exact-step commit；
- RUL、SOH、Conformal 的显式 PROJECT wrapper 已从 assembly 骨架贯通到正式 Advanced 数值链：全部重新解析同项目 Advanced raw-input attestation、复验 active-route-aware managed v2 runtime，RUL 固定使用 MATR official cycle life，SOH 固定为 cycle 500 内 finite horizon，Split Conformal 按 route 独立校准和签发；legacy normalized EOL80 链不再承担本纵向切片的业务数值；
- Advanced Deep artifact 已升级为 v2 自包含推理制品：四类 Advanced 模型均绑定完整 train-only normalizer statistics 与显式 output target；BatLiNet 额外绑定 safetensors reference batch、reference library 和逐文件 SHA-256；加载时重新核对 feature/inference context、normalizer、reference metadata 与制品字节，legacy v1 仍可解析但不能冒充自包含 v2 runtime；
- Advanced deployment bundle/index 已支持 v2，并将 output target 精确传播到 route、artifact 与 Catalog schema；真实 rebuild 固定生成 v2，BatLiNet round-trip 使用包内 reference batch；既有 v1 bundle、registry 与 Catalog 测试夹具保持可解析兼容；
- 新增 active-route-aware Advanced runtime resolver：每次调用重新验证 project context、精确 active route、managed bundle、deployment route、Catalog provenance 和 v2 artifact 全部字节；模型准备前后双重解析并在推理后支持复验，route/artifact 中途切换、v1 bundle、文件篡改或 provenance 漂移均 fail closed；四类真实 v2 safetensors loader 均已覆盖，缓存只保留私有冻结模板，每次返回隔离的无梯度推理实例，runtime 契约不暴露文件系统路径；
- PROJECT registry 已新增 Advanced raw-input attestation：复用 `extract_early_cycle_features` 标准工具名但采用独立 `advanced-input-tool-v1` 契约，只接受同项目服务端 `record_batch_id`；现场复验 FROZEN dataset/canonical records 后构建 label-free multichannel sequence，ToolResult 只记录 source、transform config 与 raw sequence SHA-256、固定轴和版本，不包含标签、RUL、SOH、区间、模型预测张量或 active route；后续数值工具必须重新解析 batch、核对 raw sequence hash，再用现场 active runtime 的 train-only normalizer materialize；
- PROJECT registry 已接入 Advanced MATR 官方 cycle life 与 finite-horizon SOH 工具：RUL 目标固定为 `matr_official_cycle_life`，SOH 只输出 cutoff 后且不超过 cycle 500 的单调有限轨迹，不从 SOH 推导 RUL；
- Split Conformal 已按冻结路由接入：cutoff 20 使用 `DEFAULT`，cutoff 50/100/150 的区间只使用 `COVERAGE`；点精度结果与 coverage 区间中心分开保存和展示；无法由 calibration cell 数量支持的目标覆盖率会失败关闭，不再截断秩后虚标覆盖保证；
- 新增项目安全结果读取 API、固定 Agent 单电芯链、四 ToolResult 审计报告与 Next.js 单电芯页面；报告和 UI 只显示持久 ToolResult 数值，并展示 route、模型、数据、feature、split、制品 SHA-256、SOH simultaneous finite band 与服务端警告；
- 新增 `0014`/`0015`：持久化 Advanced calibration materialization、样本绑定、创建时的 ADMIN session、claim token 摘要、attempt 与 lease；Worker 使用 fenced claim、续租心跳和崩溃恢复，过期或被替换的 claim 不能提交 READY/FAILED；
- 新增注册式三批 MATR calibration evidence resolver：RUL 观测只来自非删失的官方 cycle life，SOH 观测只来自逐文件 SHA-256 验证的 supervision Parquet 且 horizon 不超过 cycle 500；split、cell、观测、预测、runtime/source identity 与 sample manifest 全部由服务端冻结；
- 新增 identity-only ADMIN materialization API 与 Celery 队列：调用方只能选择合法 task/cutoff/route，Redis 只携带 `materialization_id`；样本 ToolResult、provenance、project binding、sample binding、manifest hash 和 READY 状态在一个事务中提交；
- 新增 Agent READY-materialization resolver：按 record batch 推导目标 route，精确匹配当前 runtime，按 ordinal 解析并复验全部样本 ToolResult/binding/manifest，拒绝目标 cell 进入 calibration cohort，并在返回前再次复验 active route；
- 新增 Next.js calibration 管理页：ADMIN 只能触发服务端列出的合法 route，MEMBER 只读；页面展示 readiness、冻结 identity、时间和 SHA 证据，不展示 calibration cell、sample result IDs、观测值或预测数组；1440×900 与 390×844 布局已完成无溢出浏览器 QA；
- 新增测试环境纵向 E2E：通过真实认证 ADMIN HTTP、生产 assembly、identity-only queue、assembled worker、注册式三批 manifest/split/Parquet/SHA resolver、Agent context、Split Conformal、report ledger 与 project result API 验证代码链；仅模型 tensor/inference 边界使用显式 test-only adapter，因此该测试不等同于真实产品数据库或真实激活模型验收；
- 实际产品数据库写入需在明确的数据库环境、ACTIVE 项目和已完成初始化的 ADMIN 身份下运行，当前仓库未伪造项目或管理员记录，也未修改任何真实产品数据库。

可复现源码入口：

```text
scripts/export_advanced_deployment_bundles.py
scripts/register_advanced_deployment_bundles.py
scripts/register_advanced_final_suite.py
scripts/register_advanced_candidates_to_catalog.py
src/quanxin_life/application/model_route_activation.py
src/quanxin_life/application/advanced_runtime.py
src/quanxin_life/tools/advanced_input.py
src/quanxin_life/application/advanced_prediction.py
src/quanxin_life/application/advanced_split_conformal.py
src/quanxin_life/tools/advanced_cycle_life_prediction.py
src/quanxin_life/tools/advanced_soh_prediction.py
src/quanxin_life/tools/advanced_conformal.py
src/quanxin_life/tools/advanced_project_report.py
src/quanxin_life/application/invocation_context.py
src/quanxin_life/application/agent_run_execution.py
src/quanxin_life/application/agent_run_invocation.py
src/quanxin_life/application/assembly.py
src/quanxin_life/audit/project_ledger.py
src/quanxin_life/tools/cycle_life_prediction.py
src/quanxin_life/tools/trajectory_prediction.py
src/quanxin_life/tools/conformal_calibration.py
src/quanxin_life/tools/registry.py
src/quanxin_life/api/service.py
src/quanxin_life/api/app.py
src/quanxin_life/api/model_routes.py
src/quanxin_life/application/record_batch_bindings.py
src/quanxin_life/api/record_batches.py
src/quanxin_life/audit/sql_project_ledger.py
migrations/versions/0009_model_route_activation_ledger.py
migrations/versions/0010_model_route_stream_heads.py
migrations/versions/0011_record_batch_bindings.py
migrations/versions/0012_project_tool_result_bindings.py
migrations/versions/0013_exact_agent_step_bindings.py
migrations/versions/0014_advanced_calibration_materializations.py
migrations/versions/0015_advanced_calibration_claims.py
src/quanxin_life/application/advanced_calibration_evidence.py
src/quanxin_life/application/advanced_calibration_materialization.py
src/quanxin_life/application/advanced_calibration_jobs.py
src/quanxin_life/application/advanced_agent_execution_context.py
src/quanxin_life/infrastructure/calibration_queue.py
src/quanxin_life/api/advanced_calibration.py
src/quanxin_life/tasks/advanced_calibration.py
frontend/app/projects/[projectId]/cells/[recordBatchId]/page.tsx
frontend/components/single-cell-analysis-contract.ts
frontend/components/single-cell-analysis.tsx
frontend/components/soh-trajectory-chart.tsx
frontend/app/projects/[projectId]/calibration/page.tsx
frontend/components/calibration-materialization-contract.ts
frontend/components/calibration-materialization-panel.tsx
```

Task 3 后续顺序：在明确的目标产品数据库执行 `0009`–`0015` migrations → 批注册 15 个候选 → 由已初始化 ADMIN 人工激活批准 route → 注册真实 MATR evidence root 并通过 ADMIN API materialize calibration ToolResult → 使用真实模型、真实持久 result IDs 完成 API / Agent / 报告 / UI 浏览器端到端验收。

- [x] A100 Final importer、deployment bundle、managed candidate registry 与 activation ledger 源码和离线制品；
- [ ] 在目标产品数据库执行 `0009`–`0015` migrations、15 候选批注册和人工 route activation；该步骤需要明确数据库环境、ACTIVE project 与已初始化 ADMIN；
- [x] exact AgentStep、原子 ledger、fenced-claim 恢复、冻结输入/依赖/审批与 PROJECT Worker 可信执行基础设施；
- [x] Advanced artifact v2 自包含 inference context、BatLiNet reference batch、显式 output target 与 deployment bundle/index v2；
- [x] active-route-aware runtime resolver、managed v2 字节复验、四类正式 loader 与隔离推理实例；
- [x] project Advanced raw-input ToolResult、同项目 record batch 复验、label-free sequence/config SHA-256 与 PROJECT assembly；
- [x] Advanced RUL/SOH 真实数值链，包括正式 safetensors runtime、target-aware ToolResult 与 finite-horizon SOH；
- [x] route-specific Split Conformal 校准/签发、项目 API、Agent、报告与 Next.js 单电芯纵向代码链；
- [x] 实现可信 calibration-sample producer/materializer、来源验证、fenced Worker、identity-only ADMIN API/队列、Agent READY resolver 与校准管理 UI；calibration split、cell、观测标签、预测结果和来源哈希均由服务端冻结，调用方不能自报；
- [ ] 在真实产品环境执行 migrations、15 候选批注册、人工 route activation、真实 evidence registration 和 calibration materialization 后，完成带真实模型与真实 result IDs 的浏览器端到端验收。

Task 3 的仓库内代码链已经贯通；完成上述真实产品环境验收后，进入 Next.js 门户其余页面和真实数据上传链。

## 十三、Git 与制品规则

- `server-results/`、原始数据、缓存、模型结果和生成图不提交 Git；
- 可复用源码、测试、配置、设计和计划进入 Git；
- 不使用 `git add .` 暂存结果目录；
- 每个完成切片形成小提交；
- Codex 不执行 `git push`，由用户通过 GitHub Desktop 审核后推送；
- 不移动或删除用户已有结果和未提交文件。

## 十四、比赛版完成定义

只有同时满足以下条件才可称比赛版完成：

- A100 正式模型完成晋级、注册、回退和 ToolResult 接入；
- 15%-Acc、Bootstrap、Conformal 和分组误差完成；
- 单电芯真实纵向链贯通；
- 批次风险第二场景贯通；
- Next.js、Streamlit、FastAPI、Worker、PostgreSQL、Redis 和 MinIO 贯通；
- 知识库返回带页码、版本和许可证的证据；
- Agent 受白名单、审批、预算和数值防火墙约束；
- 邮件和飞书完成真实联调，或明确展示沙箱状态；
- Docker 干净环境可启动；
- 云端 HTTPS 可稳定演示；
- OSS 备份恢复成功；
- 所有正式数字可追溯到 ToolResult；
- 竞赛文档、图表、视频和匿名复现包完成；
- 不夸大为 SOTA、工业质保或真实 BMS/EMS 生产部署。
