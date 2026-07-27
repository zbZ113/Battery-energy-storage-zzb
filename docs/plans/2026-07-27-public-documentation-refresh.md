# 公开文档纠偏实施计划

**目标：** 用已验收证据重建仓库公开入口，并明确研究、产品和部署成熟度。

**方案：** README 作为精简入口，Benchmark、Status、Model Card、Limitations、
Reproducibility 和 Architecture 分别承担单一职责。运行文档只描述真实入口；
完整部署链留到下一阶段单独设计。

**技术栈：** Markdown、pytest 文档契约测试、Git 链接检查、Ruff、mypy、compileall。

---

## 任务 1：冻结文档契约

- 修改 `tests/integration/test_runtime_docs.py`，要求 README 链接到六份公开证据文档。
- 要求 README 使用“受约束专业智能体工作流”，并包含 Implemented / Validated / Planned。
- 要求 README 不再把 A100 Smoke、五种子训练写成当前工作。
- 先运行定向测试，确认旧 README 因缺少新契约而失败。

## 任务 2：重写 README

- 保留真实安装、foundation API、前端、A100 和扩展入口。
- 将正式 Advanced 结果和边界前置。
- 将旧基线、当前正式模型和扩展能力分层。
- 把详细运行步骤下沉到 `docs/runtime-setup.md`。

## 任务 3：新增证据文档

- `docs/status.md`：成熟度矩阵和下一步。
- `docs/benchmark.md`：RUL、SOH、Conformal、晋级和证据摘要。
- `docs/model-card-advanced.md`：正式模型与路由。
- `docs/limitations.md`：限制和禁止性表述。
- `docs/reproducibility.md`：复现层级、数据、制品和验证命令。

## 任务 4：更新架构与运行边界

- 扩充 `ARCHITECTURE.md`，覆盖数值流、信任边界、Agent exact-step、
  ledger 原子提交、fenced claim、active route 和 calibration materialization。
- 在 `docs/runtime-setup.md` 首部增加当前能力矩阵和部署边界，不删除已有真实命令。

## 任务 5：验证

- 运行 `tests/integration/test_runtime_docs.py`。
- 检查 Markdown 相对链接是否存在。
- 扫描过时状态、禁用术语、占位符和本地绝对路径。
- 运行 Ruff、mypy、compileall 和最终 Git 差异审查。
