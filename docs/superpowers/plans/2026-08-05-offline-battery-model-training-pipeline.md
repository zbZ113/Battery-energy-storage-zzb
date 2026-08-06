# 泉芯智寿多源电池离线模型训练流水线实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development`（推荐）或 `executing-plans` 按任务逐项实施。所有步骤采用复选框跟踪；任何任务不得跳过红测、门禁或证据产物。

**Goal:** 在当前仓库内建立从八类公开电池数据的一次性安全处理、模型专属视图、统一训练适配、A100断点训练、验证选择、独立校准、一次性测试到可供智能体离线加载的晋级模型包的完整流水线。

**Architecture:** `data/raw` 只作为不可变来源，先生成具有真实物理语义的 Canonical Bundle，再按任务生成只拟合训练集归一化参数的 Model View。所有模型通过统一 Adapter 接入现有 `TrainingEngine`；`plan → smoke → select → final` 四阶段严格隔离测试集。最终只有完成五种子评价、逐电芯证据、校准、OOD边界、安全制品和人工批准的 `promoted_models` 才允许被产品智能体加载。

**Tech Stack:** Python 3.11、Pydantic 2、PyTorch 2.12、safetensors、NumPy/pandas/PyArrow/SciPy、scikit-learn/XGBoost、GPyTorch（BattGP）、MLflow、pytest、ruff、mypy、Bash/tmux、NVIDIA A100 80GB（物理 GPU 1）。

---

## 0. 范围、事实与不可变决策

### 0.1 当前真实能力

- 已可运行的基线链仅覆盖 MATR/MATR-three-batch：Dummy、Variance、XGBoost、CPMLP、Current Hybrid。
- 已可运行的高级链仅覆盖 MATR-three-batch：CyclePatch Direct、CyclePatch-BatLiNet、Current Hybrid、HybridPatch-v2。
- `scripts/a100/train_all_ready.sh` 当前只运行 MATR-three-batch 基线，不是全模型入口。
- `scripts/run_training_suite.py` 对 HUST/Naumann 返回 `DATASET_NOT_READY`（退出码 42）。
- `configs/training/advanced/final.json` 必须在 Select 产生并绑定 `selection_manifest` SHA 后才能运行 Final；不得取消该门禁。
- 现有 `TrainingEngine` 已具备原子检查点、best/last、SIGINT/SIGTERM恢复、上下文哈希和完成任务跳过，计划复用并扩展，不重新实现。
- 现有 CPMLP、Hybrid 和高级任务实际为 full-batch；配置中的 `batch_size=64` 尚未真正进入 DataLoader。完成微批和梯度累积前，不得宣称复现上游有效 batch。
- 当前 `data/processed` 只有 MATR；其他数据域尚未形成统一、安全、冻结的训练制品。
- 当前指标链会遗漏模型专属训练 loss，best 指标默认取 `mae`，逐电芯/轨迹测试结果并非在首次测试时原子落盘，尚无通用 OOD 实现。

### 0.2 锁定数据角色

| 数据域 | 真实层级 | 允许承担的任务 | 禁止事项 |
|---|---|---|---|
| MATR | 多电芯循环实验 | 早期寿命、SOH轨迹、同域基线 | 不得改写官方 cycle life 为 EOL80 |
| HUST | 多电芯循环实验，原始ZIP含不安全序列化 | 跨协议早期寿命、目标域适配 | 主进程/A100不得反序列化pickle |
| Naumann Cycle | condition-level循环老化观测 | 工况参数拟合、情景证据 | 不伪造cell_id或电芯寿命标签 |
| Naumann Calendar | condition-level日历老化观测 | 温度/SOC日历参数拟合 | 不强塞进电芯级随机划分 |
| 280Ah DoD | 大容量方形LFP、多DoD | 尺度校准、外部验证、少样本适配 | 不单独训练大型深网后声称泛化 |
| 280Ah TEMPEST | 基本为单只280Ah电芯 | 温度敏感性和案例证据 | 不把五个温度当作五只独立电芯 |
| 180Ah Forklift | 多只大容量电芯/多轮RPT与老化 | 外部轨迹验证、少样本适配 | RPT与Ageing不得混同 |
| 160Ah Field | 28套系统级现场时序 | BattGP、在线趋势、异常 | 不从时间/cycle伪造SOH或RUL |

### 0.3 锁定模型—任务矩阵

| 优先级 | 任务 | 模型 | 主要数据 | 主要输出 |
|---|---|---|---|---|
| P0-A | 同域早期寿命 | CPMLP、CyclePatch Direct、CyclePatch-BatLiNet | MATR | cycle life、RUL、排序 |
| P0-A | 同域SOH轨迹 | Current Hybrid、HybridPatch-v2 | MATR真实监督至cycle 500 | SOH轨迹、单调性、偏差 |
| P0-B | 多域早期寿命 | PBT、DITING/CPTransformer | MATR/HUST兼容视图 | Seen/Unseen、Zero/5/10-shot |
| P0-C | 多域SOH轨迹 | BatteryMFormer | BatteryLife兼容视图、MATR/HUST/大容量域 | 长跨度轨迹、跨域表现 |
| P1 | 工况域偏移轨迹 | MAGNet | 官方LFP/NCA/NCM兼容视图 | Qd/Ed、未见工况泛化 |
| P1 | 现场在线监测 | BattGP | 160Ah Field | GP状态、方差、异常风险 |
| P1 | 循环/日历情景 | BLAST-Lite/参数化基线 | Naumann、280Ah | 情景参数、残差、边界 |
| P2 | 超短片段特征 | Smart Feature Python等价实现 | 有可核验充电片段的数据 | SOH特征、稳定性、误差 |

### 0.4 不可违反的科学和安全规则

1. 训练、验证、校准、测试按完整 `cell_id` 隔离；现场数据还必须按 `system_id` 整组隔离。
2. 有官方电芯级划分时冻结官方划分；没有校准集时只能从训练侧预留校准电芯，不得从测试集抽取。
3. 训练/Select不得打开测试集；Final只能使用冻结配置，测试集只评价一次并生成不可变标记。
4. 归一化、特征选择、PCA、OOD阈值只使用训练集或预先声明的验证集，禁止全数据拟合。
5. 不加载未知或未核验的 pickle、joblib、`.pt`、`.pth`、`.ckpt`；正式权重使用 safetensors，结构与统计使用 JSON/Parquet。
6. Qwen不是部署依赖。若PBT/BatteryMFormer确需条件嵌入，只允许在可信环境一次性生成、审核并冻结为 safetensors；训练和推理不部署千问。
7. PyBaMM/BLAST只承担物理参考与情景敏感性，不制造15–25年真实标签。
8. 五种子模型分歧不是置信区间；只有独立校准集材料化的Conformal结果才能称预测区间。
9. 未知单位、缺少许可、域外输入、右删失和缺测必须显式标记；不得静默填补或制造标签。
10. LLM/Agent不参与本计划中的任何数值生成。

## 1. 目标文件结构与依赖顺序

### 1.1 计划新增或扩展的核心文件

```text
src/quanxin_life/data/
├── canonical.py
├── dataset_bundle.py
├── processing.py
├── adapters/
│   ├── naumann_bundle.py
│   ├── hust.py
│   ├── lfp_280ah_dod.py
│   ├── lfp_280ah_tempest.py
│   ├── lfp_180ah_forklift.py
│   └── lfp_field_160ah.py
└── model_views/
    ├── schemas.py
    ├── registry.py
    ├── builder.py
    ├── early_life.py
    ├── trajectory.py
    ├── field_monitoring.py
    ├── degradation_conditions.py
    └── partial_charge.py

src/quanxin_life/training/
├── matrix.py
├── batching.py
├── metric_logging.py
├── resource_monitor.py
├── promotion_bundle.py
└── adapters/
    ├── base.py
    ├── registry.py
    ├── native.py
    ├── pbt.py
    ├── diting.py
    ├── batterymformer.py
    ├── magnet.py
    ├── battgp.py
    ├── blast.py
    └── smart_feature.py

src/quanxin_life/evaluation/
├── metric_contracts.py
├── task_metrics.py
├── prediction_records.py
├── multi_seed.py
├── calibration.py
├── ood.py
├── figure_sources.py
└── model_card.py

scripts/data/
├── prepare_all_datasets.py
├── verify_processed_datasets.py
├── build_model_views.py
├── verify_model_views.py
└── audit_training_inputs.py

scripts/a100/
└── launch_all_training.sh

scripts/
├── run_training_matrix.py
├── export_promoted_model_bundle.py
└── build_offline_training_package.py

runs/a100/
├── task_runs/                 # 全部原始任务，含失败任务
├── aggregate/                 # 五种子与跨域汇总
├── calibration/               # Conformal材料
├── ood/                       # 支持域和OOD评价
├── figure_source_data/        # 画图唯一数值来源
├── promoted_models/           # 经门禁但尚未激活的离线模型
├── manifests/                 # 数据、配置、源码和环境清单
└── exports/                   # A100回传包
```

### 1.2 阶段依赖

```text
Gate D0 来源与Canonical契约
  → Gate D1 数据集Adapter与一次性处理
  → Gate V1 冻结Model View
  → Gate M1 统一模型Adapter/有效batch
  → Gate R1 plan/smoke/select/final编排
  → Gate E1 指标、逐电芯、校准、OOD
  → Gate P1 模型晋级与离线Bundle
  → Gate A1 A100离线包和正式训练
```

任何上游Gate未通过，下游任务必须标记 `BLOCKED`，不得静默跳过后汇报“全部完成”。

---

## Task 1：冻结训练任务矩阵与统一配置契约

**Files:**
- Create: `src/quanxin_life/training/matrix.py`
- Create: `configs/training/task_matrix_v1.json`
- Create: `tests/unit/training/test_matrix.py`
- Modify: `src/quanxin_life/training/config.py`
- Modify: `src/quanxin_life/training/suite.py`

- [ ] **Step 1：先写任务身份和非法组合红测**

```python
def test_task_identity_binds_all_reproducibility_fields() -> None:
    task = TrainingMatrixEntry.model_validate(VALID_ENTRY)
    assert task.task_id
    assert task.data_sha256
    assert task.model_view_sha256
    assert task.split_sha256
    assert task.config_sha256

def test_naumann_cannot_be_assigned_to_cycle_life_supervision() -> None:
    payload = {**VALID_ENTRY, "dataset_id": "NAUMANN_CYCLE", "task_type": "cycle_life"}
    with pytest.raises(ValueError, match="not compatible"):
        TrainingMatrixEntry.model_validate(payload)

def test_select_task_cannot_reference_test_split() -> None:
    payload = {**VALID_ENTRY, "mode": "select", "readable_splits": ["train", "validation", "test"]}
    with pytest.raises(ValueError, match="test"):
        TrainingMatrixEntry.model_validate(payload)
```

- [ ] **Step 2：运行红测**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/training/test_matrix.py -q`

Expected: FAIL，提示 `quanxin_life.training.matrix` 不存在。

- [ ] **Step 3：实现强类型矩阵契约**

`TrainingMatrixEntry` 至少包含：

```text
task_id, mode, task_type, target_semantics,
model_family, model_version, candidate_id,
dataset_id, dataset_version, model_view_version, split_version,
cutoff_cycle, prediction_horizon, seed, fold,
optimizer, loss_names, micro_batch_size,
gradient_accumulation_steps, effective_batch_size,
precision, max_epochs, validation_interval,
early_stopping_patience, selection_metric_name,
selection_metric_direction, required_artifacts,
data_sha256, model_view_sha256, split_sha256,
config_sha256, source_commit
```

固定 `validation_interval=5`、`early_stopping_patience=10`；Smoke可缩短epoch，但不能删除真实验证、检查点和恢复。

- [ ] **Step 4：登记P0/P1/P2任务并提供plan解析**

`task_matrix_v1.json` 先完整登记所有任务；尚未具备数据视图或许可证的任务使用显式阻断字段：

```json
{
  "enabled": false,
  "blocked_reason": "BLOCKED_LICENSE" 
}
```

允许的阻断码固定为 `BLOCKED_DATA_VIEW`、`BLOCKED_LICENSE`、`BLOCKED_DEPENDENCY`、`BLOCKED_SELECTION`。

- [ ] **Step 5：运行契约与配置测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/training/test_matrix.py tests/unit/training/test_config.py tests/unit/training/test_suite.py -q`

Expected: PASS。

- [ ] **Step 6：静态检查并提交**

Run: `./.venv/Scripts/python.exe -m ruff check src/quanxin_life/training/matrix.py tests/unit/training/test_matrix.py`

Run: `./.venv/Scripts/python.exe -m mypy src/quanxin_life/training/matrix.py`

Commit: `git commit -m "feat(training): freeze multi-model task matrix"`

---

## Task 2：修复数据来源登记和历史路径，不改变既有标签语义

**Files:**
- Modify: `configs/data_sources.json`
- Create: `configs/data_manifests/naumann_public_files_v2.json`
- Modify: `scripts/prepare_matr_three_batch_data.py`
- Modify: `src/quanxin_life/data/source_catalog.py`
- Test: `tests/unit/data/test_source_catalog.py`
- Test: `tests/unit/training/test_prepare_advanced_matr_data.py`
- Test: `tests/unit/training/test_a100_package.py`

- [ ] **Step 1：写实际raw路径和四个新来源红测**

断言MATR三批路径全部位于 `data/raw/MATR/v1/`；Naumann v2清单分别位于 `data/raw/NAUMANN_CYCLE/v1/` 与 `data/raw/NAUMANN_CALENDAR/v1/`；四个大容量/现场数据源具备URL、许可状态、下载UTC、SHA和ingestion mode。

- [ ] **Step 2：运行红测确认旧路径失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/data/test_source_catalog.py tests/unit/training/test_prepare_advanced_matr_data.py -q`

Expected: FAIL，指出 `data/<MATR文件>.mat` 或旧Naumann路径不匹配。

- [ ] **Step 3：版本化修复路径**

只修改新消费者指向raw新路径；保留历史清单v1和既有processed目录，不覆盖旧证据。MATR目标继续为 `MATR_OFFICIAL_CYCLE_LIFE`，不得借路径迁移重算为EOL80。

- [ ] **Step 4：重验三批MATR证据一致性**

Run: `./.venv/Scripts/python.exe scripts/prepare_matr_three_batch_data.py smoke`

Expected: 三个raw SHA全部通过；140 cells、eligibility和现有split一致；有效processed制品被验证并复用，而不是静默覆盖。

- [ ] **Step 5：运行相关回归门禁**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/data tests/unit/training/test_prepare_advanced_matr_data.py tests/unit/training/test_a100_package.py -q`

Expected: PASS。

- [ ] **Step 6：提交**

Commit: `git commit -m "fix(data): bind manifests to reviewed raw layout"`

---

## Task 3：建立多类型Canonical Bundle和幂等处理编排

**Files:**
- Create: `src/quanxin_life/data/canonical.py`
- Create: `src/quanxin_life/data/dataset_bundle.py`
- Create: `src/quanxin_life/data/processing.py`
- Create: `scripts/data/prepare_all_datasets.py`
- Create: `scripts/data/verify_processed_datasets.py`
- Create: `tests/unit/data/test_canonical_bundle.py`
- Create: `tests/unit/data/test_dataset_bundle.py`
- Create: `tests/unit/data/test_processing.py`
- Create: `tests/integration/data/test_prepare_all_datasets_cli.py`

- [ ] **Step 1：写Bundle物理层级和闭世界清单红测**

Canonical Bundle必须支持且区分：

```text
cell_cycle_telemetry
trajectory_observations
condition_observations
field_system_telemetry
metadata
targets
quality_report
artifact_manifest
```

测试要求每个数值能关联 `source_file`、`source_sha256`、`adapter_version`、单位和质量状态；未知表类型和未知单位必须拒绝。

- [ ] **Step 2：写幂等与原子发布红测**

```python
def test_second_identical_build_is_skipped(tmp_path: Path) -> None:
    first = processor.build(spec)
    second = processor.build(spec)
    assert first.output_sha256 == second.output_sha256
    assert second.status == "SKIPPED_VALID"

def test_changed_raw_sha_requires_new_dataset_version(tmp_path: Path) -> None:
    processor.build(spec)
    tamper_raw_file(spec.raw_files[0])
    with pytest.raises(ValueError, match="new dataset version"):
        processor.build(spec)
```

- [ ] **Step 3：运行红测**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/data/test_canonical_bundle.py tests/unit/data/test_dataset_bundle.py tests/unit/data/test_processing.py -q`

Expected: FAIL，模块不存在。

- [ ] **Step 4：实现staging→校验→原子rename**

正式输出固定为：

```text
data/processed/<DATASET>/<dataset-version>/canonical-v1/
├── metadata/
├── observations/
├── targets/
├── dataset_manifest.json
├── quality_report.json
├── artifact_manifest.json
└── COMMITTED
```

失败只保留隔离staging，不暴露半成品；正式目录存在且所有SHA一致时返回 `SKIPPED_VALID`。

- [ ] **Step 5：实现plan/build/verify CLI**

Run: `./.venv/Scripts/python.exe scripts/data/prepare_all_datasets.py plan`

Expected: 逐数据集输出 `READY`、`SKIPPED_VALID` 或明确阻断码，不写processed。

Run: `./.venv/Scripts/python.exe scripts/data/prepare_all_datasets.py build --dataset MATR`

Expected: 验证或生成MATR bundle。

- [ ] **Step 6：运行数据核心门禁**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/data/test_canonical_bundle.py tests/unit/data/test_dataset_bundle.py tests/unit/data/test_processing.py tests/integration/data/test_prepare_all_datasets_cli.py -q`

Expected: PASS。

- [ ] **Step 7：静态检查并提交**

Run: `./.venv/Scripts/python.exe -m ruff check src/quanxin_life/data scripts/data tests/unit/data tests/integration/data`

Run: `./.venv/Scripts/python.exe -m mypy src/quanxin_life/data`

Commit: `git commit -m "feat(data): add immutable canonical dataset bundles"`

---

## Task 4：规范化Naumann循环/日历工况证据

**Files:**
- Create: `src/quanxin_life/data/adapters/naumann_bundle.py`
- Modify: `configs/data_layouts/naumann_cycle_xdod_capacity_fec_v1.json`
- Modify: `configs/data_layouts/naumann_calendar_capacity_v1.json`
- Test: `tests/unit/data/adapters/test_naumann_bundle.py`
- Test: `tests/unit/data/adapters/test_naumann.py`
- Test: `tests/unit/data/adapters/test_naumann_cycle_mat.py`
- Test: `tests/unit/data/adapters/test_naumann_calendar.py`

- [ ] **Step 1：写condition-level语义红测**

测试输出包含 `condition_id`、FEC或自然时间、温度、SOC、DoD、倍率、relative capacity/阻抗及真实单位，同时明确 `identity_level="condition"` 和 `cell_level_split_supported=false`。

- [ ] **Step 2：写禁止假cell_id/EOL红测**

Adapter若生成无来源的 `cell_id`、EOL或RUL，测试必须失败。

- [ ] **Step 3：复用现有reader实现Bundle writer**

不得重写MAT/Excel解析数学；复用 `naumann_cycle_mat.py` 和 `naumann_calendar.py`，只增加Canonical Bundle输出、来源绑定和质量报告。

- [ ] **Step 4：重验全部Naumann文件SHA和布局**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/data/adapters/test_naumann.py tests/unit/data/adapters/test_naumann_cycle_mat.py tests/unit/data/adapters/test_naumann_calendar.py tests/unit/data/adapters/test_naumann_bundle.py -q`

Expected: PASS，所有文件的header/legend/单位符合reviewed layout。

- [ ] **Step 5：提交**

Commit: `git commit -m "feat(data): publish reviewed Naumann condition bundles"`

---

## Task 5：完成HUST隔离转换双门禁

**Files:**
- Create: `configs/data_layouts/hust_mendeley_v2_layout_v1.json`
- Create: `quarantine/hust/convert.py`
- Create: `src/quanxin_life/data/adapters/hust.py`
- Modify: `quarantine/hust/Dockerfile`
- Modify: `quarantine/hust/README.md`
- Test: `tests/integration/test_hust_quarantine_definition.py`
- Create: `tests/integration/test_hust_quarantine_converter.py`
- Create: `tests/unit/data/adapters/test_hust.py`

- [ ] **Step 1：完成Gate H1字段发现而不反序列化执行**

使用现有opcode inspector和77成员批准inventory生成字段、形状、单位候选与风险报告。人工批准的layout必须逐字段写明来源成员、单位依据和转换规则；没有依据的字段标为 `UNRESOLVED`，转换任务保持BLOCKED。

- [ ] **Step 2：写隔离环境红测**

测试Docker定义必须具备：只读输入、无网络、非root、`no-new-privileges`、CPU/内存/文件数限制、只允许写staging输出；主进程和A100路径中搜索不到 `pickle.load`、`joblib.load`、`torch.load`。

- [ ] **Step 3：写安全输出红测**

转换器只允许输出Parquet/JSON和SHA清单；77个成员逐项登记；`cell_id`唯一且稳定；字段/单位与批准layout逐字一致。

- [ ] **Step 4：实现Gate H2转换器与主进程Adapter**

隔离转换器负责受限解析；主进程Adapter只读取隔离环境发布的Parquet/JSON并再次验证SHA，不接触原pickle对象。

- [ ] **Step 5：运行HUST专项测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/integration/test_hust_quarantine_definition.py tests/integration/test_hust_quarantine_converter.py tests/unit/data/adapters/test_hust.py -q`

Expected: PASS；若人工layout未批准，Expected为明确 `BLOCKED_REVIEW`，不得生成伪Canonical。

- [ ] **Step 6：提交**

Commit: `git commit -m "feat(data): add isolated HUST canonical conversion"`

---

## Task 6：接入280Ah、180Ah和160Ah真实层级数据

**Files:**
- Create: `src/quanxin_life/data/adapters/lfp_280ah_dod.py`
- Create: `src/quanxin_life/data/adapters/lfp_280ah_tempest.py`
- Create: `src/quanxin_life/data/adapters/lfp_180ah_forklift.py`
- Create: `src/quanxin_life/data/adapters/lfp_field_160ah.py`
- Create: `configs/data_layouts/lfp_280ah_dod_v1.json`
- Create: `configs/data_layouts/lfp_280ah_tempest_v1.json`
- Create: `configs/data_layouts/lfp_180ah_forklift_v1.json`
- Create: `configs/data_layouts/lfp_field_160ah_v1.json`
- Create: `tests/unit/data/adapters/test_lfp_280ah_dod.py`
- Create: `tests/unit/data/adapters/test_lfp_280ah_tempest.py`
- Create: `tests/unit/data/adapters/test_lfp_180ah_forklift.py`
- Create: `tests/unit/data/adapters/test_lfp_field_160ah.py`

- [ ] **Step 1：先写ZIP安全与身份稳定红测**

所有ZIP Adapter拒绝路径穿越、绝对路径、重复成员、异常压缩比和未登记成员。测试固定 `cell_id/system_id` 在重复处理时不变化。

- [ ] **Step 2：实现280Ah DoD Adapter**

保留厂商、真实电芯、20/60/100% DoD和reference capacity口径；从数据中有证据时才计算SOH；不得从目录名制造寿命标签。

- [ ] **Step 3：实现280Ah TEMPEST Adapter**

将aging与15/20/25/30/35°C characterization分表；整个数据域保持单电芯/真实身份，不创建五个虚假训练样本。

- [ ] **Step 4：实现180Ah Forklift Adapter**

保留Cell/Round、RPT/Ageing类型、自然时间和工况；RPT与Ageing使用显式measurement_type；按整电芯划分。

- [ ] **Step 5：实现160Ah Field系统时序Adapter**

输出system_id、timestamp、系统电流/温度/SOC、8个单体电压、均衡状态和缺测mask；忽略 `__MACOSX/.DS_Store` 内容但在审计报告记录；划分按system_id分组。

- [ ] **Step 6：写禁止伪标签红测**

Field Adapter输出中不得出现由时间或cycle直接构造的真实SOH/RUL；TEMPEST不得跨温度分割为train/test；280Ah样本量不足时任务角色必须是calibration/external_test。

- [ ] **Step 7：运行四套Adapter测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/data/adapters/test_lfp_280ah_dod.py tests/unit/data/adapters/test_lfp_280ah_tempest.py tests/unit/data/adapters/test_lfp_180ah_forklift.py tests/unit/data/adapters/test_lfp_field_160ah.py -q`

Expected: PASS。

- [ ] **Step 8：提交**

Commit: `git commit -m "feat(data): add storage-scale LFP canonical adapters"`

---

## Task 7：生成五类冻结Model View并证明训练不访问raw

**Files:**
- Create: `src/quanxin_life/data/model_views/schemas.py`
- Create: `src/quanxin_life/data/model_views/registry.py`
- Create: `src/quanxin_life/data/model_views/builder.py`
- Create: `src/quanxin_life/data/model_views/early_life.py`
- Create: `src/quanxin_life/data/model_views/trajectory.py`
- Create: `src/quanxin_life/data/model_views/field_monitoring.py`
- Create: `src/quanxin_life/data/model_views/degradation_conditions.py`
- Create: `src/quanxin_life/data/model_views/partial_charge.py`
- Create: `configs/model_views/early_life_sequence_v1.json`
- Create: `configs/model_views/soh_trajectory_v1.json`
- Create: `configs/model_views/field_monitoring_v1.json`
- Create: `configs/model_views/degradation_condition_v1.json`
- Create: `configs/model_views/partial_charge_v1.json`
- Create: `scripts/data/build_model_views.py`
- Create: `scripts/data/verify_model_views.py`
- Create: `tests/unit/data/model_views/test_schemas.py`
- Create: `tests/unit/data/model_views/test_registry.py`
- Create: `tests/unit/data/model_views/test_early_life.py`
- Create: `tests/unit/data/model_views/test_trajectory.py`
- Create: `tests/unit/data/model_views/test_field_monitoring.py`
- Create: `tests/unit/data/model_views/test_degradation_conditions.py`
- Create: `tests/unit/data/model_views/test_partial_charge.py`
- Create: `tests/leakage/test_model_view_split_isolation.py`

- [ ] **Step 1：写Model View Manifest红测**

每个View必须绑定：Canonical SHA、split SHA、builder版本/代码SHA、配置SHA、cutoff、target语义、mask语义、normalizer和 `training_cell_ids_sha256`。

- [ ] **Step 2：写归一化泄漏红测**

构造train均值为1、test均值为100的合成cohort；断言normalizer均值仍为1。现场view按system_id验证；calibration/test不得参与fit。

- [ ] **Step 3：实现五类View**

```text
early_life_sequence_v1       → CyclePatch/BatLiNet/PBT/DITING
soh_trajectory_v1           → Hybrid/BatteryMFormer/MAGNet
field_monitoring_v1         → BattGP
degradation_condition_v1    → BLAST/Naumann情景
partial_charge_v1           → Smart Feature
```

输出仅允许 safetensors、Parquet、JSON；mask必须显式；没有真实label时target为空并携带right_censoring/evidence，不插值伪标签。

- [ ] **Step 4：写训练raw访问阻断测试**

测试monkeypatch `data/raw` 打开操作为立即失败，所有训练Dataset仍能从冻结View加载。

- [ ] **Step 5：实现幂等build/verify CLI**

Run: `./.venv/Scripts/python.exe scripts/data/build_model_views.py plan --all`

Expected: 列出每个view的数据支持和阻断原因，不写文件。

Run: `./.venv/Scripts/python.exe scripts/data/build_model_views.py build --all-ready`

Expected: 有效view生成；已存在且SHA一致的返回SKIPPED。

- [ ] **Step 6：运行View门禁**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/data/model_views tests/leakage/test_model_view_split_isolation.py -q`

Expected: PASS。

- [ ] **Step 7：提交**

Commit: `git commit -m "feat(data): build frozen model-specific views"`

---

## Task 8：建立统一训练Adapter、微批和安全上游制品策略

**Files:**
- Create: `src/quanxin_life/training/adapters/base.py`
- Create: `src/quanxin_life/training/adapters/registry.py`
- Create: `src/quanxin_life/training/batching.py`
- Create: `tests/unit/training/adapters/test_base.py`
- Create: `tests/unit/training/test_effective_batch.py`
- Create: `tests/unit/training/test_upstream_artifact_policy.py`
- Modify: `src/quanxin_life/training/engine.py`
- Modify: `src/quanxin_life/training/checkpoint.py`

- [ ] **Step 1：写Adapter协议红测**

```python
class TrainingAdapter(Protocol):
    selection_metric_name: str
    selection_metric_direction: Literal["min", "max"]
    def build_model(self, resolved_config: ResolvedTrainingConfig) -> nn.Module: ...
    def load_view(self, manifest: ModelViewManifest) -> Dataset: ...
    def train_epoch(self, state: TrainState) -> EpochMetrics: ...
    def validate(self, state: EvalState) -> EvaluationResult: ...
    def predict(self, state: EvalState) -> PredictionBatch: ...
    def export_best(self, destination: Path) -> ArtifactManifest: ...
```

测试确保训练接口接到test split立即失败；selection metric缺失或非finite时失败。

- [ ] **Step 2：写有效batch红测**

公式固定：

```text
effective_batch = micro_batch × visible_gpu_count × gradient_accumulation_steps
```

PBT首轮128×1×2=256；DITING domain adaptation 32×1×2=64；BatteryMFormer 128×1×2=256；MAGNet按具体官方入口冻结。不可整除、最后批处理策略未声明、optimizer step数不匹配均失败。

- [ ] **Step 3：写不安全制品红测**

拒绝 `.pkl/.pickle/.joblib/.pt/.pth/.ckpt`；接受且校验 safetensors/JSON/Parquet/UBJ。任何Adapter调用上游 `torch.load`、pickle或joblib loader均阻断。

- [ ] **Step 4：实现微批、梯度累积和Adapter Registry**

现有full-batch native任务也必须通过Adapter/BatchPlan运行；每次optimizer step前按accumulation缩放loss；不足批处理策略写入resolved config。

- [ ] **Step 5：扩展安全检查点上下文**

增加adapter版本、model view SHA、effective batch、selection spec和上游commit；上下文任一项变化时拒绝恢复。

- [ ] **Step 6：运行专项测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/training/adapters/test_base.py tests/unit/training/test_effective_batch.py tests/unit/training/test_upstream_artifact_policy.py tests/unit/training/test_engine.py tests/unit/training/test_checkpoint.py -q`

Expected: PASS。

- [ ] **Step 7：提交**

Commit: `git commit -m "feat(training): add governed model adapter layer"`

---

## Task 9：先让现有MATR模型完成统一Adapter闭环

**Files:**
- Create: `src/quanxin_life/training/adapters/native.py`
- Modify: `src/quanxin_life/training/advanced_tasks.py`
- Modify: `src/quanxin_life/training/tasks.py`
- Modify: `src/quanxin_life/training/advanced_orchestrator.py`
- Create: `tests/unit/training/adapters/test_native.py`
- Modify: `tests/unit/training/test_advanced_tasks.py`

- [ ] **Step 1：写四模型统一输入/输出红测**

CyclePatch Direct、CyclePatch-BatLiNet、Current Hybrid、HybridPatch-v2必须从View读取微批，输出finite loss及模型专属分量，并记录同一测试cohort。

- [ ] **Step 2：写BatLiNet参考库隔离红测**

reference IDs只来自train cells；validation/calibration/test cell加入参考库时失败；参考库SHA进入checkpoint和run manifest。

- [ ] **Step 3：写Hybrid监督边界红测**

只读取真实监督至cycle 500；轨迹结构性单调；cycle 500后标签访问立即失败；RUL只能由阈值跨越派生。

- [ ] **Step 4：实现native Adapter并保留旧入口兼容**

旧MATR脚本继续可运行，但内部委托统一Adapter；确认真正使用micro batch与gradient accumulation，不再full-batch假配置。

- [ ] **Step 5：运行native闭环测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/training/adapters/test_native.py tests/unit/training/test_advanced_tasks.py tests/unit/training/test_advanced_orchestrator.py -q`

Expected: PASS。

- [ ] **Step 6：提交**

Commit: `git commit -m "refactor(training): route native battery models through adapters"`

---

## Task 10：接入PBT和DITING多域早期寿命模型

**Files:**
- Create: `src/quanxin_life/training/adapters/pbt.py`
- Create: `src/quanxin_life/training/adapters/diting.py`
- Create: `configs/training/pbt/smoke.json`
- Create: `configs/training/pbt/selection.json`
- Create: `configs/training/pbt/final.json`
- Create: `configs/training/diting/smoke.json`
- Create: `configs/training/diting/selection.json`
- Create: `configs/training/diting/final.json`
- Create: `tests/unit/training/adapters/test_pbt.py`
- Create: `tests/unit/training/adapters/test_diting.py`
- Create: `scripts/research/audit_training_upstreams.py`
- Create: `tests/unit/research/test_training_upstream_audit.py`

- [ ] **Step 1：冻结上游commit、参数来源和许可证状态**

PBT绑定本地commit `a2df9d36db3f57ab2c5686638952ad7715ea0646`；DITING绑定 `b67f48373c591ca62c030fca257ee94910c974ce`。PBT许可证文本随审计记录；DITING无明确根许可证时状态为 `RESEARCH_ONLY_LICENSE_UNVERIFIED`，不得进入公开分发Bundle。

- [ ] **Step 2：写PBT安全输入和loss红测**

覆盖label/guidance-or-contrastive/alignment/MoE load-balancing loss、专家利用率、门控熵和Seen/Unseen指标。PCA components、mean、embedding和normalizer必须为safetensors/JSON；发现pickle即失败。

- [ ] **Step 3：写DITING域适配红测**

覆盖prediction loss、MMD domain loss、source/target loss；目标域test labels不得进入训练/早停；zero-shot、5-shot、10-shot使用独立目标域适配/测试电芯清单。

- [ ] **Step 4：实现Adapter而不运行上游I/O**

保留核心模型数学与官方超参数映射；完全绕过BatteryML loader、joblib scaler、pickle/PTH checkpoint。单卡effective batch写入manifest。

- [ ] **Step 5：证明无需Qwen部署**

测试训练和推理依赖图中不存在Qwen/Transformers运行时调用；若条件embedding缺失，任务返回 `BLOCKED_DATA_VIEW`，不得在线调用大模型补齐。

- [ ] **Step 6：运行专项测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/training/adapters/test_pbt.py tests/unit/training/adapters/test_diting.py tests/unit/research/test_training_upstream_audit.py -q`

Expected: PASS；DITING promotion保持许可证阻断状态。

- [ ] **Step 7：提交**

Commit: `git commit -m "feat(training): integrate PBT and DITING adapters"`

---

## Task 11：接入BatteryMFormer和MAGNet轨迹模型

**Files:**
- Create: `src/quanxin_life/training/adapters/batterymformer.py`
- Create: `src/quanxin_life/training/adapters/magnet.py`
- Create: `configs/training/batterymformer/smoke.json`
- Create: `configs/training/batterymformer/selection.json`
- Create: `configs/training/batterymformer/final.json`
- Create: `configs/training/magnet/smoke.json`
- Create: `configs/training/magnet/selection.json`
- Create: `configs/training/magnet/final.json`
- Create: `tests/unit/training/adapters/test_batterymformer.py`
- Create: `tests/unit/training/adapters/test_magnet.py`

- [ ] **Step 1：冻结上游证据**

BatteryMFormer commit `febe174032ad4861fa057b9af23f5bcee8a8fb77`；MAGNet commit `aafb90c551d20748251a35fd51a34eae2539aaca`。MAGNet许可证随Bundle；BatteryMFormer许可证未明确前仅研究使用。

- [ ] **Step 2：写BatteryMFormer多层loss红测**

记录trajectory/parameter/recovery/memory alignment/memory diversity loss、slot utilization、参数空间与SOH空间指标；上游 `Qwen3_total.pkl`、history pickle、joblib scaler、`torch.load`路径全部禁止。

- [ ] **Step 3：安全重建条件嵌入与history**

条件embedding使用 `condition_embeddings.safetensors + metadata.json`；history使用Parquet；normalizer使用JSON；训练和推理不加载或部署Qwen。

- [ ] **Step 4：写MAGNet PyTorch 2.12等价性红测**

固定小张量与种子，对比经人工审计的核心forward/loss参考值；记录raw MSE、SOC比例约束、截止电压、meta-train/meta-test、Qd/Ed指标。禁止安装Torch1.9/Numba0.56到主环境。

- [ ] **Step 5：实现兼容Adapter和安全保存**

重写数据、训练循环、scaler和checkpoint边界；核心数学结构的每个兼容修改进入 `compatibility_changes.json` 和模型卡。

- [ ] **Step 6：运行专项测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/training/adapters/test_batterymformer.py tests/unit/training/adapters/test_magnet.py -q`

Expected: PASS。

- [ ] **Step 7：提交**

Commit: `git commit -m "feat(training): integrate trajectory model adapters"`

---

## Task 12：把BattGP、BLAST和Smart Feature作为独立任务族接入

**Files:**
- Create: `src/quanxin_life/training/adapters/battgp.py`
- Create: `src/quanxin_life/training/adapters/blast.py`
- Create: `src/quanxin_life/training/adapters/smart_feature.py`
- Create: `configs/training/battgp/smoke.json`
- Create: `configs/training/battgp/selection.json`
- Create: `configs/training/battgp/final.json`
- Create: `configs/training/blast/smoke.json`
- Create: `configs/training/blast/selection.json`
- Create: `configs/training/blast/final.json`
- Create: `configs/training/smart_feature/smoke.json`
- Create: `configs/training/smart_feature/selection.json`
- Create: `configs/training/smart_feature/final.json`
- Create: `requirements/blast-cpu-py311.in`
- Create: `requirements/blast-cpu-py311.lock`
- Create: `tests/unit/training/adapters/test_battgp.py`
- Create: `tests/unit/training/adapters/test_blast.py`
- Create: `tests/unit/training/adapters/test_smart_feature.py`

- [ ] **Step 1：写任务类型隔离红测**

BattGP必须是 `field_monitor`，BLAST必须是 `scenario_fit`，Smart Feature必须是 `feature_estimation`；禁止把三者注册为普通cycle-life深度网络。

- [ ] **Step 2：实现BattGP Adapter**

记录negative log marginal likelihood、kernel参数、noise、预测均值/方差和在线更新；只有存在明确异常标签时才报告AUROC/AUPRC/FPR95，否则只报告趋势/残差和人工标注需求。

- [ ] **Step 3：建立BLAST独立CPU环境**

主环境NumPy 2.4与BLAST `numpy<2`硬冲突；BLAST使用独立 `quanxin-blast-py311` CPU环境或完成有数值等价测试的兼容移植。其输出为参数JSON、残差Parquet和证据清单，不是深度权重。

- [ ] **Step 4：实现Smart Feature Python对照**

上游只有MATLAB处理脚本且许可证不明；先用固定输入对照MATLAB导出数值，再实现Python特征。无训练头时只输出特征证据，不宣传为训练模型。

- [ ] **Step 5：运行专项测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/training/adapters/test_battgp.py tests/unit/training/adapters/test_blast.py tests/unit/training/adapters/test_smart_feature.py -q`

Expected: PASS。

- [ ] **Step 6：提交**

Commit: `git commit -m "feat(training): add field and scenario task adapters"`

---

## Task 13：统一训练/验证指标、模型专属loss和资源记录

**Files:**
- Create: `src/quanxin_life/evaluation/metric_contracts.py`
- Create: `src/quanxin_life/training/metric_logging.py`
- Create: `src/quanxin_life/training/resource_monitor.py`
- Modify: `src/quanxin_life/training/engine.py`
- Create: `tests/unit/evaluation/test_metric_contracts.py`
- Create: `tests/unit/training/test_metric_logging_contract.py`
- Create: `tests/unit/training/test_resource_monitor.py`
- Modify: `tests/unit/training/test_engine.py`

- [ ] **Step 1：写MetricDefinition/MetricRecord红测**

```text
MetricDefinition:
  name, stage, direction, unit, aggregation, required, display_name

MetricRecord:
  run_id, dataset_id, model_family, model_version, task, target,
  split, cutoff, seed, epoch, global_step, metric_name, value,
  unit, timestamp_utc
```

非finite值、单位冲突、同名不同方向必须拒绝。

- [ ] **Step 2：写模型专属loss不丢失红测**

构造HybridPatch返回 `history_loss/smooth_loss/order_loss/residual_loss`，断言JSONL和history Parquet均存在这些字段。PBT、DITING、BatteryMFormer、MAGNet同理；未启用分量写 `NOT_APPLICABLE`，不得写0伪装参与计算。

- [ ] **Step 3：取消best指标硬编码**

`TrainingTask/Adapter`必须声明selection metric、方向、单位和required。BatteryMFormer可按validation MAPE，MAGNet按冻结联合指标，BattGP按NLL；engine不得猜测 `mae`。

- [ ] **Step 4：实现长表history和资源记录**

每epoch至少保存total loss、各loss、LR、gradient norm、epoch seconds、samples/s、GPU peak MiB、checkpoint SHA、UTC。机器可读文件固定为 `training_history.parquet`、`validation_history.parquet`、`resource_usage.parquet`；终端事件另存 `console.log`、`stderr.log`、`events.jsonl`。NVML不可用时写null+warning，不得写0。

- [ ] **Step 5：增加独立推理benchmark**

`inference_benchmark.json`包含warmup、repeats、batch、device、p50/p95、throughput、peak memory；未执行则模型卡写“未测”，训练时长不得冒充推理延迟。

- [ ] **Step 6：运行指标日志测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/evaluation/test_metric_contracts.py tests/unit/training/test_metric_logging_contract.py tests/unit/training/test_resource_monitor.py tests/unit/training/test_engine.py -q`

Expected: PASS。

- [ ] **Step 7：提交**

Commit: `git commit -m "feat(evaluation): standardize training metrics and resources"`

---

## Task 14：在首次测试时原子保存逐电芯、轨迹和任务专属指标

**Files:**
- Create: `src/quanxin_life/evaluation/task_metrics.py`
- Create: `src/quanxin_life/evaluation/prediction_records.py`
- Create: `tests/unit/evaluation/test_task_metrics.py`
- Create: `tests/unit/evaluation/test_prediction_records.py`
- Modify: `src/quanxin_life/training/advanced_orchestrator.py`
- Modify: `scripts/analyze_advanced_predictions.py`
- Modify: `scripts/analyze_advanced_soh_predictions.py`

- [ ] **Step 1：写公共回归指标红测**

所有寿命/RUL/SOH/容量任务记录：MAE、RMSE、MSE、MAPE或sMAPE、R2、bias、median/P75/P90/P95 absolute error、±5/10/15/20%准确率。目标含0时普通MAPE必须拒绝并选择sMAPE。

- [ ] **Step 2：写任务专属指标红测**

- PBT/DITING：dataset/domain macro、Seen/Unseen、worst-domain。
- BatteryMFormer：参数空间逐参数RMSE/R2、SOH空间MAE/RMSE/sMAPE。
- MAGNet：Qd/Ed分开、condition/cell macro、unseen condition。
- BattGP：NLL、模型分布区间与异常指标边界。
- BLAST：capacity/impedance残差、参数边界和物理违规。
- 轨迹：cell-balanced、horizon bins、单调违规；knee/EOL只有标签定义一致时计算。

- [ ] **Step 3：定义逐样本Parquet契约**

`per_cell_predictions.parquet` 包含run/model/data/split/seed/cell/domain/condition/cutoff/target/true/pred/error/right_censored/OOD/evidence。

`trajectory_predictions.parquet` 额外包含cycle/time/horizon/true_soh/pred_soh/lower/upper/interval_method。lower/upper为空时禁止UI称预测区间。

- [ ] **Step 4：首次测试原子写入并锁定**

Final测试成功后同时写metrics、predictions、SHA和 `TEST_EVALUATED` marker；同run重跑只能校验并复用完全相同的输入/结果，不能再次用test挑候选。

- [ ] **Step 5：验证五seed cohort完全一致**

同模型五seed的cell集合必须一致；轨迹cycle坐标也必须一致。缺失seed/cell/cycle直接阻断聚合，不允许删除失败样本。

- [ ] **Step 6：运行专项测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/evaluation/test_task_metrics.py tests/unit/evaluation/test_prediction_records.py tests/unit/evaluation/test_advanced_metrics.py -q`

Expected: PASS。

- [ ] **Step 7：提交**

Commit: `git commit -m "feat(evaluation): persist atomic per-cell test evidence"`

---

## Task 15：完成五种子聚合、配对Bootstrap和图源数据

**Files:**
- Create: `src/quanxin_life/evaluation/multi_seed.py`
- Create: `src/quanxin_life/evaluation/figure_sources.py`
- Modify: `src/quanxin_life/evaluation/advanced_metrics.py`
- Modify: `src/quanxin_life/training/plots.py`
- Create: `tests/unit/evaluation/test_multi_seed.py`
- Create: `tests/unit/evaluation/test_figure_sources.py`
- Modify: `tests/unit/training/test_plots.py`

- [ ] **Step 1：写五种子统计红测**

每seed先在相同cell cohort计算，再报告mean、sample std、median、min、max、有效/失败seed数；禁止摊平所有seed预测假增样本量。

- [ ] **Step 2：写失败seed保留红测**

`failed_runs.parquet`保存run_id、失败阶段、错误码、最后epoch和上下文SHA；任何聚合删除失败seed时测试失败。

- [ ] **Step 3：实现10,000次同cell同seed配对Bootstrap**

输出effect、95% CI和win-rate；CI跨0时自动生成事实性结论“未证明显著优势”，不得声称稳定领先。

- [ ] **Step 4：生成可画图源数据**

至少生成：loss/val曲线、best epoch、真实-预测散点、误差分布、逐cell瀑布、代表电芯SOH轨迹、horizon error、Seen/Unseen、coverage-width、精度-资源Pareto。每张PNG伴同名CSV/Parquet和manifest SHA。

- [ ] **Step 5：运行聚合/绘图测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/evaluation/test_multi_seed.py tests/unit/evaluation/test_figure_sources.py tests/unit/training/test_plots.py -q`

Expected: PASS。

- [ ] **Step 6：提交**

Commit: `git commit -m "feat(evaluation): add multi-seed evidence and figure sources"`

---

## Task 16：材料化Conformal预测区间和独立OOD支持域

**Files:**
- Create: `src/quanxin_life/evaluation/calibration.py`
- Create: `src/quanxin_life/evaluation/ood.py`
- Create: `tests/unit/evaluation/test_calibration.py`
- Create: `tests/unit/evaluation/test_ood.py`
- Modify: `src/quanxin_life/uncertainty/cycle_life_conformal.py`
- Modify: `scripts/analyze_advanced_conformal.py`

- [ ] **Step 1：写split隔离红测**

Conformal quantile只能拟合calibration cells；test只评PICP/MPIW/coverage gap。材料绑定model+dataset+cutoff+target+feature+split+data版本和SHA。

- [ ] **Step 2：材料化80%和90%区间**

每个alpha独立保存；calibration cells少于20时写明确warning。多seed std只能作为预先冻结的difficulty scale，不能单独称置信区间。

- [ ] **Step 3：写轨迹区间诚实边界**

在正式轨迹Conformal未实现前，lower/upper保持null，`interval_method="NOT_AVAILABLE"`，模型卡写“无轨迹覆盖保证”。

- [ ] **Step 4：实现独立OOD支持域**

支持域由train统计温度、C-rate、DoD、SOC、容量、封装和特征分布；阈值只用train/validation冻结。留一dataset/temperature/protocol有明确ID/OOD标签时才计算AUROC/AUPRC/FPR95；无标签只报告距离分布和支持覆盖率。

- [ ] **Step 5：写禁止概念混用红测**

多模型分歧、seed std或超出min/max不得自动标成Conformal区间；OOD分数不得称预测概率。

- [ ] **Step 6：运行专项测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/evaluation/test_calibration.py tests/unit/evaluation/test_ood.py tests/unit/uncertainty -q`

Expected: PASS。

- [ ] **Step 7：提交**

Commit: `git commit -m "feat(evaluation): materialize calibrated intervals and OOD support"`

---

## Task 17：实现plan/smoke/select/final全任务编排和恢复语义

**Files:**
- Create: `scripts/run_training_matrix.py`
- Create: `scripts/a100/launch_all_training.sh`
- Modify: `scripts/a100/train_all_ready.sh`
- Modify: `src/quanxin_life/training/orchestrator.py`
- Create: `tests/integration/training/test_launch_modes.py`
- Create: `tests/unit/training/test_resume_matrix.py`
- Create: `tests/integration/training/test_a100_shell_contract.py`

- [ ] **Step 1：写四模式红测**

- `plan`：不触GPU、不训练，只验矩阵、view、split/hash、依赖、磁盘和阻断原因。
- `smoke`：每个合法组合1 seed、1–2 epoch，必须真实validate、保存checkpoint、模拟resume并安全export；不形成正式测试结论。
- `select`：只读train/validation，每5 epoch验证，10次验证无改善早停，生成selection manifest+SHA；打开test立即失败。
- `final`：只接受绑定selection SHA的冻结配置，五seed，best→calibration→test一次→promotion candidate。

- [ ] **Step 2：写恢复/跳过红测**

同task所有context SHA一致：COMPLETED→SKIPPED；RUNNING/PAUSED→从last恢复模型/optimizer/scheduler/RNG；任一SHA/seed/view变化→拒绝；损坏checkpoint隔离并失败，不静默从头训练。

- [ ] **Step 3：实现阻断任务不污染成功统计**

plan输出ready/blocked矩阵；运行模式只执行ready任务，但最终汇总同时列出blocked及原因，不能把blocked当completed。

- [ ] **Step 4：实现Shell和旧入口兼容**

`train_all_ready.sh`委托新入口；旧MATR专项Shell继续保留。固定GPU：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

进程内只允许看到 `cuda:0`，代表物理GPU1。

- [ ] **Step 5：运行本地无GPU计划和合成Smoke**

Run: `bash scripts/a100/launch_all_training.sh plan`

Expected: 不初始化CUDA训练，输出任务数、READY/BLOCKED和哈希。

Run: `./.venv/Scripts/python.exe -m pytest tests/integration/training/test_launch_modes.py tests/unit/training/test_resume_matrix.py tests/integration/training/test_a100_shell_contract.py -q`

Expected: PASS。

- [ ] **Step 6：提交**

Commit: `git commit -m "feat(training): orchestrate resumable multi-model runs"`

---

## Task 18：建立模型晋级、模型卡和离线推理Bundle

**Files:**
- Create: `src/quanxin_life/training/promotion_bundle.py`
- Create: `src/quanxin_life/evaluation/model_card.py`
- Modify: `src/quanxin_life/evaluation/model_promotion.py`
- Create: `scripts/export_promoted_model_bundle.py`
- Create: `tests/unit/training/test_promotion_bundle.py`
- Modify: `tests/unit/evaluation/test_model_promotion.py`
- Create: `tests/unit/evaluation/test_model_card.py`
- Create: `tests/integration/evaluation/test_promoted_bundle_export.py`

- [ ] **Step 1：写晋级顺序红测**

门禁顺序固定：完整性→无泄漏→validation-only选模→五seed完整→一次性test→逐cell/轨迹→校准coverage→OOD边界→安全制品→许可证→人工批准。

测试用于验证冻结晋级门槛和最终证据，不得再次挑超参、候选或代表seed。

- [ ] **Step 2：写best/last导出红测**

推理Bundle只能从best导出；last较差时不得误导出；best缺失或SHA不匹配立即拒绝。optimizer/scheduler/RNG只留训练恢复包，不进入推理Bundle。

- [ ] **Step 3：实现闭世界离线Bundle**

```text
runs/a100/promoted_models/<task>/<family>/<version>/
├── model.safetensors              # 有神经/GP权重时
├── model_config.json
├── input_schema.json
├── preprocessing_manifest.json
├── normalization.json
├── calibration.json
├── supported_domain.json
├── metrics_validation.json
├── metrics_test.json
├── per_cell_summary.parquet
├── inference_benchmark.json       # 未测时不伪造
├── model_card.md
└── artifact_manifest.json
```

BLAST类参数模型使用 `parameters.json` 替代神经权重，但仍需同等证据和SHA。

- [ ] **Step 4：生成UTF-8模型卡**

必含任务/目标、数据与cell split、上游commit/许可证、训练参数、validation规则、test一次性指标、五seed统计、逐cell尾部、Conformal、OOD支持域、失败实验、推理环境/延迟、拒绝条件、SHA和 `VERIFIED_NOT_ACTIVATED` 状态。MATR official cycle life不得写成EOL80或15–25年。

- [ ] **Step 5：人工批准前禁止激活**

导出成功只得到 `VERIFIED_NOT_ACTIVATED`；人工批准后由产品仓库另行注册active route。训练脚本不得自动激活模型。

- [ ] **Step 6：运行Bundle测试**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/training/test_promotion_bundle.py tests/unit/evaluation/test_model_promotion.py tests/unit/evaluation/test_model_card.py tests/integration/evaluation/test_promoted_bundle_export.py -q`

Expected: PASS；危险扩展名、缺SHA、乱码或缺章节全部失败。

- [ ] **Step 7：提交**

Commit: `git commit -m "feat(training): export governed offline model bundles"`

---

## Task 19：建立A100离线依赖、训练包、回传包和最终端到端门禁

**Files:**
- Modify: `requirements/a100-linux-py311.in`
- Modify: `requirements/a100-linux-py311.lock`
- Create: `requirements/data-prep-py311.in`
- Create: `requirements/data-prep-py311.lock`
- Modify: `scripts/a100/create_env.sh`
- Modify: `scripts/a100/preflight.sh`
- Create: `scripts/build_offline_training_package.py`
- Modify: `src/quanxin_life/training/a100_package.py`
- Modify: `src/quanxin_life/training/advanced_outputs.py`
- Modify: `scripts/verify_a100_training_output.py`
- Create: `tests/integration/training/test_offline_package.py`
- Create: `tests/integration/training/test_training_pipeline_e2e.py`
- Modify: `docs/a100-advanced-model-training-guide.md`

- [ ] **Step 1：审计统一环境而不直接安装上游requirements**

主环境固定Python 3.11.13、Torch 2.12。PBT/BatteryMFormer/MAGNet的旧Torch/NumPy版本不直接安装；只加入经Adapter实际使用、兼容且锁定哈希的依赖。BattGP增加GPyTorch/Botorch兼容锁；BLAST继续独立CPU锁。

- [ ] **Step 2：建立真正离线wheelhouse模式红测**

`create_env.sh`必须支持 `--offline-wheelhouse <path>`，运行时禁止联网；缺wheel/hash时失败。不得在A100运行中临时pip下载。

- [ ] **Step 3：构建最小训练包**

包含源码、配置、processed、model_views、wheelhouse、来源/数据/代码/包内清单和外部SHA。排除raw、interim、前端、Agent、密钥、旧结果和危险二进制。

- [ ] **Step 4：写闭世界训练包红测**

任何 `.pkl/.joblib/.pt/.pth/.ckpt`、`.env`、密钥模式、未知文件或未登记SHA均失败；归档外部生成archive size/SHA/source commit/created UTC。

- [ ] **Step 5：完成合成数据CPU端到端测试**

覆盖：Canonical→View→plan→smoke→中断恢复→select隔离→final一次test→逐cell→calibration→promotion bundle→output index逐字节验证。

Run: `./.venv/Scripts/python.exe -m pytest tests/integration/training/test_offline_package.py tests/integration/training/test_training_pipeline_e2e.py -q`

Expected: PASS。

- [ ] **Step 6：运行项目最低门禁**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/data tests/unit/features tests/unit/models tests/unit/training tests/unit/evaluation tests/unit/uncertainty tests/leakage tests/integration/data tests/integration/training tests/integration/evaluation -q`

Run: `./.venv/Scripts/python.exe -m ruff check src scripts tests`

Run: `./.venv/Scripts/python.exe -m mypy src/quanxin_life`

Expected: 全部PASS；没有最新输出不得宣称完成。

- [ ] **Step 7：生成并验证A100包**

Run: `./.venv/Scripts/python.exe scripts/build_offline_training_package.py --mode final --output dist/quanxin-a100-offline.zip`

Run: `Get-FileHash dist/quanxin-a100-offline.zip -Algorithm SHA256`

Expected: 包内清单验证通过，外部SHA sidecar存在。

- [ ] **Step 8：提交**

Commit: `git commit -m "feat(release): package reproducible offline A100 training"`

---

## 2. A100执行顺序（所有本地门禁通过后）

### 2.0 无SSH条件下经百度网盘传输

Windows端生成以下两个文件后，人工上传同一个百度网盘目录：

```text
dist/quanxin-a100-offline.zip
dist/quanxin-a100-offline.zip.sha256
```

A100端通过已有浏览器或百度网盘客户端下载到 `/data/abd/z/`。压缩包和SHA sidecar必须分别下载；不得只凭网盘“传输完成”提示信任文件。训练结果回传同样使用 `runs/a100/exports/<UTC时间>-results.tar.gz` 与对应 `.sha256` 双文件，Windows下载后再次执行 `Get-FileHash` 对账。

### 2.1 上传后核验

```bash
sha256sum -c quanxin-a100-offline.zip.sha256
unzip quanxin-a100-offline.zip -d /data/abd/z/AI-B
cd /data/abd/z/AI-B
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quanxin-a100
bash scripts/a100/preflight.sh
```

预检必须确认：物理GPU1为A100 80GB、占用不超过门槛、PyTorch只看到一张GPU且为 `cuda:0`、数据/View/配置/代码SHA正确、磁盘充足。

### 2.2 Plan与Smoke

```bash
bash scripts/a100/launch_all_training.sh plan
tmux new-session -d -s quanxin-smoke \
  'cd /data/abd/z/AI-B && source ~/miniconda3/etc/profile.d/conda.sh && conda activate quanxin-a100 && bash scripts/a100/launch_all_training.sh smoke 2>&1 | tee logs/a100/smoke.log'
tmux attach -t quanxin-smoke
```

Smoke只验证链路，不进入参赛指标。

### 2.3 Select

```bash
tmux new-session -d -s quanxin-select \
  'cd /data/abd/z/AI-B && source ~/miniconda3/etc/profile.d/conda.sh && conda activate quanxin-a100 && bash scripts/a100/launch_all_training.sh select 2>&1 | tee logs/a100/select.log'
tail -F logs/a100/select.log
```

Select完成后必须核验 `selection_manifest.json`、选择trace、final resolved config及SHA；测试集访问计数必须为0。

### 2.4 Final

```bash
tmux new-session -d -s quanxin-final \
  'cd /data/abd/z/AI-B && source ~/miniconda3/etc/profile.d/conda.sh && conda activate quanxin-a100 && bash scripts/a100/launch_all_training.sh final 2>&1 | tee logs/a100/final.log'
tail -F logs/a100/final.log
watch -n 2 'nvidia-smi -i 1'
```

断网不影响tmux任务；断电后重新执行同一Final命令，已完成任务跳过，未完成任务从相同上下文的last恢复。

### 2.5 训练结束验收与回传

```bash
python scripts/verify_a100_training_output.py runs/a100
python scripts/export_promoted_model_bundle.py --all-approved-candidates
sha256sum runs/a100/exports/*.tar.gz > runs/a100/exports/SHA256SUMS
```

回传包至少包含任务状态、失败实验、console/events/history、逐电芯/轨迹、validation/test、资源、五种子聚合、校准、OOD、图源/图、best/last恢复证据、promotion候选、环境/源码/数据/配置SHA和output index。

---

## 3. 阶段完成定义

### Gate D：数据完成

- 八个raw域全部登记来源、许可状态、下载UTC和SHA；处理前后raw哈希集合一致。
- MATR路径修复后140 cells、标签语义和既有split不变。
- 所有已支持域产生Canonical Bundle；HUST未经双门批准保持BLOCKED。
- 重复处理全部SKIPPED且产物SHA不变。
- 无假cell_id、假SOH、假RUL、假EOL；右删失显式。
- 所有监督split按cell_id，Field额外按system_id。

### Gate V：Model View完成

- 五类View均有manifest、mask、target语义、normalizer、Canonical/split/config/code SHA。
- normalizer只拟合train；测试证明训练不访问raw。
- 不同模型只读取适合其物理任务的View。

### Gate M：模型训练能力完成

- 现有native模型不再full-batch假配置，微批与累积真实生效。
- PBT/DITING/BatteryMFormer/MAGNet绕过不安全上游I/O。
- PBT/BatteryMFormer训练和推理不部署Qwen。
- BattGP/BLAST/Smart Feature按真实任务类型管理。
- 许可证不清模型不得进入公开激活Bundle。

### Gate E：证据完成

- loss证明收敛；validation证明选模；test证明冻结后的泛化。
- 五seed完整，失败显式，逐电芯和逐轨迹不缺失。
- 公共指标和模型专属指标完整；未启用写NOT_APPLICABLE。
- Conformal只用calibration；OOD阈值不读test；模型分歧不冒充区间。
- 图表全部有同名源数据和SHA。

### Gate P：离线模型完成

- 推理Bundle只来自best；last仅用于恢复。
- Bundle只含安全格式并闭世界SHA校验。
- 模型卡包含目标、数据、split、配置、五seed、test、尾部风险、区间、OOD、许可证和限制。
- 状态先为 `VERIFIED_NOT_ACTIVATED`；人工批准前不得由智能体加载。

### Gate A：A100正式训练完成

- Plan、Smoke、Select、Final顺序完整。
- GPU1 fail-closed；内部cuda:0映射可解释。
- Select不访问test；Final绑定selection SHA，test只落盘一次。
- 中断恢复、完成跳过和损坏checkpoint拒绝均有证据。
- 回传包和外部SHA在Windows端复验一致。

---

## 4. 实施优先级与首个可交付里程碑

首个里程碑不等待所有模型和数据：

```text
Task 1–3（契约与数据总控）
→ Task 7–9（MATR View + 统一Adapter + native闭环）
→ Task 13–19（完整证据、四模式、离线Bundle和A100包）
```

该里程碑必须先交付MATR三批上的：

- CyclePatch Direct；
- CyclePatch-BatLiNet；
- Current Hybrid；
- HybridPatch-v2；
- 完整训练/验证/test/逐电芯/校准/模型卡/离线Bundle。

随后按依赖顺序推进：

```text
HUST安全转换 → PBT/DITING
多域轨迹View → BatteryMFormer
多工况View → MAGNet
160Ah Field → BattGP
Naumann/280Ah → BLAST
可核验充电片段 → Smart Feature
```

这样即使后续某个上游模型受许可证、依赖或数据阻断，项目仍然拥有一条科学完整、能够进入飞书智能体的离线模型主线，而不会因“全模型一次做完”导致没有可交付结果。
