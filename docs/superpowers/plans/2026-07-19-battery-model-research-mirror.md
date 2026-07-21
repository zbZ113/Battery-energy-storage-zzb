# 顶级电芯寿命模型研究镜像实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完整下载顶级电芯寿命预测论文与官方代码，并建立可复验、不会污染主工程依赖的本地研究镜像。

**Architecture:** 以版本化JSON目录描述Git、PDF和归档来源；同步器完整克隆Git仓库、流式下载文件并生成哈希状态。大体积材料由主仓库忽略，轻量目录、脚本和中文导航由主仓库跟踪。

**Tech Stack:** Python 3.11标准库、Git、pytest、SHA-256、JSON。

---

### Task 1: 来源目录与路径安全

- [ ] 添加目录契约测试，拒绝重复标识、绝对路径与路径逃逸。
- [ ] 实现严格目录加载和验证。
- [ ] 运行定向pytest。

### Task 2: 完整Git镜像

- [ ] 测试Git命令不得包含 `--depth`、`--filter` 或 `--single-branch`。
- [ ] 实现完整克隆、已存在仓库更新、远端核验和对象完整性检查。
- [ ] 完整克隆所有可访问官方仓库。

### Task 3: 论文与公开归档

- [ ] 测试下载器校验PDF文件头并生成SHA-256。
- [ ] 下载开放获取论文和Zenodo公开归档。
- [ ] 对登录/访问控制阻塞项生成明确状态，不绕过限制。

### Task 4: 中文研究导航

- [ ] 生成每个模型的论文、代码、版本、任务、复现风险和项目改造切入点。
- [ ] 输出本地状态清单和哈希。
- [ ] 验证研究大文件未被主Git跟踪。

### Task 5: 质量门禁

- [ ] 运行研究工具单元测试、Ruff和mypy。
- [ ] 对每个Git镜像运行 `git fsck --full`。
- [ ] 对每个PDF校验 `%PDF-` 文件头、非零大小和SHA-256。
- [ ] 只提交轻量目录、脚本、文档和忽略规则，不执行push。
