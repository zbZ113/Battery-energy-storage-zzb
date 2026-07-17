# MATR Three-Batch A100 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 MATR 的三个已核验批次接入同一套可恢复 A100 训练流水线，同时保持原有 2018 单批次入口和结果语义不变。

**Architecture:** 三批原始数据分别进行来源绑定、截止循环 150 转换和监督资格审计，通过联合清单引用各批内容寻址制品。联合划分由三个批内电芯级分层划分合并而成；标量模型使用 138 个有效官方标签，Hybrid 只使用 120 个完整真实循环 500 轨迹。新增 `matr-three-batch` 训练和 ZIP64 打包入口，不覆盖现有 `matr` 单批入口。

**Tech Stack:** Python 3.11、Pydantic、h5py、PyArrow/Parquet、PyTorch、XGBoost、safetensors、Bash、ZIP64、pytest、Ruff、mypy。

---

### Task 1: 登记并隔离两个 2017 原始批次

**Files:**
- Create: `configs/data_manifests/matr_2017_05_12_batch_v1.json`
- Create: `configs/data_manifests/matr_2017_06_30_batch_v1.json`
- Modify: `.gitignore`
- Test: `tests/unit/data/test_source_catalog.py`

- [ ] **Step 1: 写入失败测试**

```python
def test_three_reviewed_matr_raw_manifests_are_registered() -> None:
    expected = {
        "2017-05-12": "9d928ab978f0e3c70b31cb833a749fedd35094d01af76475d69b40aa3497f5ba",
        "2017-06-30": "63ab200d09ecb237fee5ef3a5c5db76e3212e3206a0bd92f769e1427fed338b8",
        "2018-04-12": "62c30e413b63e6144720e016deed3661fac8468641794a5807b123fe84717998",
    }
    assert load_reviewed_matr_manifests() == expected
```

- [ ] **Step 2: 确认测试因两个清单缺失而失败**

Run: `python -m pytest tests/unit/data/test_source_catalog.py -q`

Expected: FAIL，指出 2017 清单缺失。

- [ ] **Step 3: 写入两个来源清单并扩展忽略规则**

两个 JSON 固定保存文件名、大小、上述 SHA-256、MATR 官方项目 URI、论文 DOI 和带时区下载时间；`.gitignore` 新增：

```gitignore
/data/2017-05-12_batchdata_updated_struct_errorcorrect.mat
/data/2017-06-30_batchdata_updated_struct_errorcorrect.mat
```

- [ ] **Step 4: 运行清单与 Git 隔离验证**

Run: `python -m pytest tests/unit/data/test_source_catalog.py -q && git status --short`

Expected: PASS，`git status` 不再显示两个原始 `.mat`。

- [ ] **Step 5: 提交**

```bash
git add .gitignore configs/data_manifests tests/unit/data/test_source_catalog.py
git commit -m "data: register two reviewed MATR batches"
```

### Task 2: 支持批次级转换与循环 500 资格过滤

**Files:**
- Modify: `src/quanxin_life/data/matr_pipeline.py`
- Create: `src/quanxin_life/data/matr_multibatch.py`
- Create: `tests/unit/data/test_matr_multibatch.py`

- [ ] **Step 1: 写入失败测试，锁定真实轨迹资格**

```python
def test_supervision_eligibility_keeps_only_cells_with_real_cycle_500() -> None:
    audit = audit_matr_supervision_eligibility(
        observed_cycle_counts={"MATR_b2c0": 170, "MATR_b2c1": 501},
        horizon_cycle=500,
    )
    assert audit.eligible_cell_ids == ("MATR_b2c1",)
    assert audit.excluded[0].reason == "INSUFFICIENT_REAL_TRAJECTORY_500"
```

- [ ] **Step 2: 运行测试并确认缺少资格审计器**

Run: `python -m pytest tests/unit/data/test_matr_multibatch.py -q`

Expected: FAIL with import error.

- [ ] **Step 3: 实现冻结的资格审计契约**

```python
class MatrTrajectoryEligibilityAudit(ContractModel):
    schema_version: Literal["matr-trajectory-eligibility-v1"]
    batch_index: int
    horizon_cycle: int
    eligible_cell_ids: tuple[str, ...]
    excluded: tuple[MatrTrajectoryExclusion, ...]
```

资格条件固定为 `observed_cycle_count > horizon_cycle`。扩展 `build_matr_supervision_artifact`，新增可选的 `selected_cell_ids`，只允许转换报告中存在且通过资格审计的电芯；默认 `None` 保持旧 2018 行为。

- [ ] **Step 4: 验证旧单批兼容与新过滤行为**

Run: `python -m pytest tests/unit/data/test_matr_pipeline.py tests/unit/data/test_matr_multibatch.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/quanxin_life/data/matr_pipeline.py src/quanxin_life/data/matr_multibatch.py tests/unit/data
git commit -m "feat: audit real MATR trajectory eligibility"
```

### Task 3: 构建三批联合清单与联合划分

**Files:**
- Modify: `src/quanxin_life/data/matr_multibatch.py`
- Modify: `tests/unit/data/test_matr_multibatch.py`
- Create: `scripts/prepare_matr_three_batch_data.py`

- [ ] **Step 1: 写入联合划分失败测试**

```python
def test_combined_split_covers_each_cell_once_and_preserves_batches() -> None:
    combined = combine_matr_batch_splits((batch1, batch2, batch3))
    all_ids = combined.train + combined.validation + combined.calibration + combined.test
    assert len(all_ids) == 140
    assert len(set(all_ids)) == 140
    for partition in (combined.train, combined.validation, combined.calibration, combined.test):
        assert {cell_id.split("c", 1)[0] for cell_id in partition} == {
            "MATR_b1", "MATR_b2", "MATR_b3"
        }
```

- [ ] **Step 2: 运行测试观察联合函数缺失**

Run: `python -m pytest tests/unit/data/test_matr_multibatch.py -q`

Expected: FAIL。

- [ ] **Step 3: 实现联合清单、划分和准备入口**

联合清单保存三个批次的原始、转换、监督、资格和划分哈希。`prepare_matr_three_batch_data.py` 按 batch1/2/3 执行：原始校验、缓存复验或转换、批内划分、资格审计、监督生成，最后写入联合划分与联合清单。batch2 监督只传入28个合格电芯。

- [ ] **Step 4: 用合成批次验证所有拒绝边界**

Run: `python -m pytest tests/unit/data/test_matr_multibatch.py tests/leakage -q`

Expected: PASS，重复电芯、漏电芯、批次缺失和资格上下文不一致均被拒绝。

- [ ] **Step 5: 提交**

```bash
git add src/quanxin_life/data/matr_multibatch.py scripts/prepare_matr_three_batch_data.py tests/unit/data/test_matr_multibatch.py
git commit -m "feat: prepare governed three-batch MATR data"
```

### Task 4: 支持多处理根目录训练加载

**Files:**
- Modify: `src/quanxin_life/training/matr_data.py`
- Create: `tests/unit/training/test_matr_three_batch_data.py`

- [ ] **Step 1: 写入多根加载失败测试**

```python
def test_scalar_loader_merges_verified_batches_and_excludes_only_censored_labels() -> None:
    cohorts = load_matr_multibatch_cycle_life_cohorts(source, cutoff_cycle=20)
    assert set(cohorts.all_cell_ids) == expected_138_event_cells

def test_hybrid_loader_uses_only_eligible_real_trajectory_cells() -> None:
    cohorts = load_matr_multibatch_hybrid_cohorts(source, cutoff_cycle=20)
    assert set(cohorts.all_cell_ids) == expected_120_eligible_cells
```

- [ ] **Step 2: 确认测试因加载器不存在失败**

Run: `python -m pytest tests/unit/training/test_matr_three_batch_data.py -q`

Expected: FAIL。

- [ ] **Step 3: 实现多根加载并复用单批张量逻辑**

新增只负责定位批次根目录、复验组件哈希和合并批次样本的薄层；曲线插值、掩码、标签对象和 Hybrid 张量构造继续复用现有函数。加载器必须验证联合 `data_version`、`split_version` 和清单 SHA-256。

- [ ] **Step 4: 运行训练数据与泄漏门禁**

Run: `python -m pytest tests/unit/training/test_matr_data.py tests/unit/training/test_matr_three_batch_data.py tests/leakage -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/quanxin_life/training/matr_data.py tests/unit/training/test_matr_three_batch_data.py
git commit -m "feat: load verified MATR batches for joint training"
```

### Task 5: 新增联合训练配置与编排

**Files:**
- Modify: `src/quanxin_life/training/suite.py`
- Modify: `src/quanxin_life/training/orchestrator.py`
- Create: `configs/training/matr_three_batch_smoke.json`
- Create: `configs/training/matr_three_batch_final.json`
- Modify: `tests/unit/training/test_suite.py`

- [ ] **Step 1: 写入矩阵失败测试**

```python
def test_three_batch_final_expands_four_cutoffs_five_models_five_seeds() -> None:
    config = MatrRunConfig.model_validate_json(
        Path("configs/training/matr_three_batch_final.json").read_bytes()
    )
    assert len(build_run_matrix(config.suite)) == 100
    assert config.suite.data_version.startswith("matr-three-batch-")
```

- [ ] **Step 2: 运行测试确认配置缺失**

Run: `python -m pytest tests/unit/training/test_suite.py -q`

Expected: FAIL with missing config.

- [ ] **Step 3: 新增联合配置并参数化编排器的数据源**

Smoke 保持20个串行任务，Final保持100个串行任务；训练上限、验证频率、早停、种子、FP32和GPU1与单批配置一致。编排器根据配置中的 `source_kind` 选择单批或三批加载器，运行目录固定为 `runs/a100/matr-three-batch/{mode}`。

- [ ] **Step 4: 运行配置和编排测试**

Run: `python -m pytest tests/unit/training/test_suite.py tests/unit/training tests/integration/test_a100_scripts.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add configs/training src/quanxin_life/training tests/unit/training
git commit -m "feat: orchestrate three-batch MATR training"
```

### Task 6: 接入一键 Shell 命令

**Files:**
- Modify: `scripts/run_training_suite.py`
- Modify: `scripts/a100/train_dataset.sh`
- Modify: `scripts/a100/train_all_ready.sh`
- Modify: `tests/integration/test_a100_scripts.py`

- [ ] **Step 1: 写入命令定义失败测试**

```python
def test_three_batch_shell_prepares_then_checks_gpu_then_trains() -> None:
    script = Path("scripts/a100/train_dataset.sh").read_text()
    assert script.index("prepare_matr_three_batch_data.py") < script.index("preflight.sh")
    assert 'run_training_suite.py matr-three-batch "${MODE}"' in script
```

- [ ] **Step 2: 运行测试确认新命令未接入**

Run: `python -m pytest tests/integration/test_a100_scripts.py -q`

Expected: FAIL。

- [ ] **Step 3: 实现命令路由**

`train_dataset.sh matr-three-batch smoke|final` 先准备三批数据，再设置 `CUDA_VISIBLE_DEVICES=1`、运行A100预检，最后串行执行联合套件。非就绪HUST/Naumann语义保持原状。

- [ ] **Step 4: 验证计划展开而不启动GPU训练**

Run: `python scripts/run_training_suite.py matr-three-batch smoke --plan-only`

Expected: `PLAN_READY` and `task_count=20`。

- [ ] **Step 5: 提交**

```bash
git add scripts tests/integration/test_a100_scripts.py
git commit -m "feat: add one-command three-batch A100 suite"
```

### Task 7: 构建三原始文件 A100 ZIP64 包

**Files:**
- Modify: `src/quanxin_life/training/a100_package.py`
- Create: `scripts/build_matr_three_batch_a100_package.py`
- Create: `scripts/verify_matr_three_batch_a100_package.py`
- Modify: `tests/unit/training/test_a100_package.py`

- [ ] **Step 1: 写入三原始文件包失败测试**

```python
def test_three_batch_package_requires_exactly_three_bound_raw_matr_files() -> None:
    manifest = build_matr_three_batch_a100_archive(
        project_root=root,
        output_archive=tmp_path / "three-batch.zip",
        tracked_files=("src/train.py",),
        source_commit="a" * 40,
        raw_batches=(batch1, batch2, batch3),
        processed_relative_paths=("data/processed/MATR/combined",),
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
    )
    assert {item.batch_index for item in manifest.raw_batches} == {1, 2, 3}
    assert len([item for item in manifest.files if item.role == "raw-matr"]) == 3
```

- [ ] **Step 2: 运行测试确认 v2 包构建器缺失**

Run: `python -m pytest tests/unit/training/test_a100_package.py -q`

Expected: FAIL。

- [ ] **Step 3: 实现 v2 清单和独立验收脚本**

新增 `matr-three-batch-a100-package-v2`，要求 batch index 1/2/3、批准文件名、大小、SHA-256和HDF5头全部匹配；复用现有关闭式文件清单、危险路径拒绝、ZIP64、整包外部索引和原子发布逻辑。旧单批 v1 函数不改语义。

- [ ] **Step 4: 运行合成包、篡改和旧包回归**

Run: `python -m pytest tests/unit/training/test_a100_package.py tests/integration/test_a100_scripts.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/quanxin_life/training/a100_package.py scripts tests/unit/training/test_a100_package.py
git commit -m "feat: package three reviewed MATR batches for A100"
```

### Task 8: 处理真实两批数据、生成联合包并交接

**Files:**
- Modify: `docs/a100-secure-training-handoff-v1.md`
- Generated, ignored: `data/processed/MATR/2017-05-12-*`
- Generated, ignored: `data/processed/MATR/2017-06-30-*`
- Generated, ignored: `dist/quanxin-matr-three-batch-a100.zip`
- Generated, ignored: `dist/quanxin-matr-three-batch-a100.sha256.json`

- [ ] **Step 1: 执行真实三批准备**

Run: `python scripts/prepare_matr_three_batch_data.py final`

Expected: 140 cells，138 scalar labels，120 Hybrid-500 eligible，20 explicit exclusions。

- [ ] **Step 2: 验证联合矩阵和关键数据报告**

Run: `python scripts/run_training_suite.py matr-three-batch smoke --plan-only`

Expected: 20 serial tasks。

- [ ] **Step 3: 运行相关门禁**

```bash
python -m pytest tests/unit/data/test_matr_multibatch.py tests/unit/training tests/integration/test_a100_scripts.py -q
ruff check src/quanxin_life/data src/quanxin_life/training scripts tests
mypy src/quanxin_life/data src/quanxin_life/training
python -m compileall -q src scripts
```

Expected: all commands exit 0。

- [ ] **Step 4: 提交代码与文档**

```bash
git add .gitignore configs docs scripts src tests
git commit -m "feat: complete three-batch MATR A100 pipeline"
```

- [ ] **Step 5: 从干净提交生成并复验实际包**

```bash
python scripts/build_matr_three_batch_a100_package.py dist/quanxin-matr-three-batch-a100.zip
python scripts/verify_matr_three_batch_a100_package.py \
  dist/quanxin-matr-three-batch-a100.zip \
  dist/quanxin-matr-three-batch-a100.sha256.json
```

Expected: `MATR_THREE_BATCH_A100_PACKAGE_VERIFIED`，报告整包 SHA-256、大小、源提交和文件数。

- [ ] **Step 6: 更新服务器交接文档**

文档必须给出 Conda 环境创建、整包哈希复验、解压、GPU1预检、tmux Smoke、恢复演练、正式五种子运行和回传文件清单；不得宣称本地已完成A100训练。
