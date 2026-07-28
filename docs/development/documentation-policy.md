# 文档同步策略

## 目标

README 是项目入口，不是全部技术细节。每次重大变更必须同步更新入口状态和对应的
权威详细文档，避免代码、部署与公开叙事分叉。

仓库的 [PR 模板](../../.github/PULL_REQUEST_TEMPLATE.md) 将该要求转成提交检查项；
自动化文档契约位于 `tests/integration/test_runtime_docs.py`。

## 必须触发文档更新的变更

出现以下任一变化时，提交或 PR 必须包含文档影响说明：

- 公共 API、认证、角色或错误语义；
- 数据库 schema 或 migration；
- 模型结构、正式指标、active route 或 calibration；
- `ToolResult`、AgentStep、ledger、claim/lease 契约；
- 部署拓扑、镜像、端口、secret 或恢复流程；
- 用户可见 UI 流程；
- 数据、标签、划分、许可证或适用范围；
- 已验证、已部署或计划状态变化。

## 更新矩阵

| 变更 | 至少更新 |
| --- | --- |
| 对外能力或成熟度变化 | `README.md`、`docs/status.md` |
| 软件边界或依赖变化 | `ARCHITECTURE.md`、模块设计、对应 ADR |
| API 变化 | `docs/api/README.md`、OpenAPI/测试 |
| 模型或指标变化 | 算法说明、Benchmark、模型卡、限制、复现 |
| 部署变化 | ECS 部署、运维 Runbook、安全与 Secrets、状态 |
| 用户流程变化 | 代码走读、README quickstart、UI 文档 |
| 重大取舍 | 新 ADR；旧 ADR 只标记 superseded，不改写历史 |

## 状态用语

- `Implemented`：实现存在；
- `Validated`：有测试、正式实验、哈希或受控 E2E；
- `Published`：不可变镜像或发布包已经进入目标 registry/release；
- `Deployed`：目标环境已经启动并通过服务验收；
- `Demonstrated`：真实用户路径或浏览器纵向 E2E 已完成；
- `Planned`：尚未实现或缺少外部输入。

禁止用 `Implemented` 或 `Validated` 替代 `Deployed`。

## 数值与图件

- README 中的指标必须来自 `docs/benchmark.md` 的权威口径；
- 图件必须有公开 manifest、来源 SHA-256 和统计口径；
- 不手工改写图中数值；
- 训练耗时不能写成推理延迟；
- 大结果、模型权重、原始数据和 secrets 不进入 Git。

## 提交前检查

```bash
python -m pytest tests/integration/test_runtime_docs.py -q
python -m ruff check tests/integration/test_runtime_docs.py
```

还应检查：

- Markdown 相对链接不存在失效目标；
- 文档没有本机绝对路径、密钥或临时账号；
- README、status、limitations 的状态一致；
- API 路径来自真实路由或 OpenAPI；
- 部署命令对应当前 Compose 和 migration；
- 新增图片与 manifest 的 SHA-256 一致。
