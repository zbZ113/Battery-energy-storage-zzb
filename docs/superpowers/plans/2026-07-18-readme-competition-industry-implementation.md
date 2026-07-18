# 泉芯智寿竞赛与产业型 README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将根目录README重写为面向竞赛评审、产业合作方和工程使用者的项目首页，完整呈现价值、能力、进展、飞书协同和真实使用入口。

**Architecture:** README采用“价值主张—AI能力—系统闭环—工程进展—使用指南—可信机制”的单页信息架构。运行文档集成测试负责约束标题、真实入口、A100命令、飞书回调和禁止性表述；产品代码、训练配置和运行数据保持不变。

**Tech Stack:** Markdown、Mermaid、pytest、Python 3.11、现有FastAPI/Next.js/Streamlit/MCP/A100 Shell入口。

---

### Task 1: 更新README用户文档契约

**Files:**
- Modify: `tests/integration/test_runtime_docs.py`

- [ ] **Step 1: 将旧README章节断言替换为新信息架构**

将 `test_readme_is_user_facing_and_only_documents_real_entry_points` 中的章节列表替换为：

```python
for section in (
    "## 项目价值",
    "## AI核心能力",
    "## 端到端系统闭环",
    "## 技术架构",
    "## 工程进展全景",
    "## 快速开始",
    "## 三批MATR与A100真实训练",
    "## 产品与协同入口",
    "## 飞书研发协同",
    "## 可信与可追溯",
    "## 质量门禁",
    "## 仓库结构",
    "## 典型应用场景",
    "## 文档导航",
):
    assert section in readme
```

补充真实入口断言：

```python
for marker in (
    "matr-three-batch",
    "scripts/a100/train_dataset.sh",
    "frontend/package.json",
    "pnpm@10.28.1",
    "POST /v1/integrations/feishu/events",
    "src/quanxin_life/integrations/feishu",
    "ToolResult",
):
    assert marker in readme
```

保留 `deploy/foundation_api.py`、`workbench/streamlit_app.py`、`mcp_host.py` 和 `docs/runtime-setup.md` 断言。删除对旧“已实现能力”和“验证边界”标题的要求，并增加：

```python
assert "当前不足" not in readme
assert "生产系统" not in readme
```

- [ ] **Step 2: 运行契约测试并观察RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_runtime_docs.py::test_readme_is_user_facing_and_only_documents_real_entry_points -q
```

Expected: FAIL，原因是当前README尚不包含新的章节标题。

- [ ] **Step 3: 提交测试契约**

```powershell
git add tests/integration/test_runtime_docs.py
git commit -m "test: define competition-focused README contract"
```

### Task 2: 重写根目录README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 使用UTF-8重写项目首页**

README必须按批准规格建立以下内容：

```text
泉芯智寿完整标题和一句话价值
项目价值
AI核心能力
端到端系统闭环
技术架构Mermaid图
数据、模型、可信机制和多智能体分工
工程进展全景
快速开始
三批MATR与A100真实训练
产品与协同入口
飞书研发协同
可信与可追溯
质量门禁
仓库结构
典型应用场景
文档导航
```

三批MATR文件名必须准确写为：

```text
2017-05-12_batchdata_updated_struct_errorcorrect.mat
2017-06-30_batchdata_updated_struct_errorcorrect.mat
2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

A100命令必须使用：

```bash
bash scripts/a100/preflight.sh
bash scripts/a100/train_dataset.sh matr-three-batch smoke
bash scripts/a100/train_dataset.sh matr-three-batch final
```

前端命令必须使用仓库登记的pnpm版本：

```bash
corepack enable
corepack prepare pnpm@10.28.1 --activate
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend dev
```

飞书章节必须区分已形成的安全底座与协同工作流，列出：

```text
POST /v1/integrations/feishu/events
时间戳、Nonce、SHA-256签名、验证Token
幂等事件回执、租约令牌、失败重试
run_id/result_id引用卡片
告警、复检、报告、审批与任务跟踪
凭证不入库、模型数值不由消息层生成
```

工程进展按以下三层书写：

```text
已形成的可运行能力
当前重点推进
下一阶段完整配置
```

不得填写尚未由正式实验产生的MAE、覆盖率、寿命提升或产业收益数字。

- [ ] **Step 2: 运行README契约测试并观察GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_runtime_docs.py -q
```

Expected: PASS。

- [ ] **Step 3: 提交README正文**

```powershell
git add README.md
git commit -m "docs: present Quanxin Life competition platform"
```

### Task 3: 校验链接、编码和工程命令

**Files:**
- Modify when required: `README.md`
- Modify when required: `tests/integration/test_runtime_docs.py`

- [ ] **Step 1: 校验UTF-8、占位符和本地Markdown链接**

Run:

```powershell
@'
import re
from pathlib import Path

root = Path.cwd()
text = (root / "README.md").read_text(encoding="utf-8")
assert "\ufffd" not in text
assert not re.search(r"\b(?:TBD|TODO)\b", text)
for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
    if "://" in target or target.startswith("#"):
        continue
    path = target.split("#", 1)[0]
    if path:
        assert (root / path).exists(), target
print("README_UTF8_AND_LINKS_OK")
'@ | .\.venv\Scripts\python.exe -
```

Expected: `README_UTF8_AND_LINKS_OK`。

- [ ] **Step 2: 运行文档范围质量门禁**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_runtime_docs.py -q
.\.venv\Scripts\ruff.exe check tests\integration\test_runtime_docs.py
git diff --check
```

Expected: 全部退出码为0。

- [ ] **Step 3: 复核README事实来源**

确认以下文件仍存在，且README命令与其保持一致：

```text
pyproject.toml
frontend/package.json
scripts/a100/preflight.sh
scripts/a100/train_dataset.sh
deploy/foundation_api.py
workbench/streamlit_app.py
src/quanxin_life/tools/mcp_host.py
src/quanxin_life/api/feishu.py
.env.example
```

- [ ] **Step 4: 提交校验修正**

仅在Task 3产生修正时执行：

```powershell
git add README.md tests/integration/test_runtime_docs.py
git commit -m "docs: verify README entry points"
```
