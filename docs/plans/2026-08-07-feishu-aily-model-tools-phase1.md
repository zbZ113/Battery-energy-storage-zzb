# 飞书/Aily 电池模型工具联动第一阶段实施计划

**目标：** 在不修改公共 ToolResult 契约和训练主线的前提下，完成可测试的飞书/Aily
接入基础设施、审计交付链和 Fake Sandbox。

**风险等级：** 高。涉及外部鉴权、非可信附件、持久化幂等、模型 activation gate 和
业务数值展示。

## 文件边界

新增或重点修改：

- `src/quanxin_life/integrations/feishu/client.py`：出站 transport、token、重试和 API。
- `src/quanxin_life/integrations/feishu/attachments.py`：CSV 附件安全策略。
- `src/quanxin_life/integrations/feishu/routing.py`：sanitized event 和任务路由端口。
- `src/quanxin_life/integrations/feishu/cards.py`：状态卡片和 audited card builder。
- `src/quanxin_life/integrations/feishu/bitable.py`：表结构与 run-id 幂等 writer。
- `src/quanxin_life/integrations/feishu/workflow.py`：验证前置的固定工具工作流。
- `src/quanxin_life/api/feishu.py`：兼容旧 router 的 sanitized route 调用。
- `src/quanxin_life/api/aily.py`：受保护的稳定连接器 façade。
- `scripts/run_fake_feishu_sandbox.py`、`scripts/feishu_preflight.py`：本地入口。
- `.env.example` 与 `docs/integrations/feishu-aily-model-tools.md`：配置和运维文档。
- 对应 `tests/unit/integrations/feishu/`、`tests/integration/` 测试。

本阶段不修改：

- `configs/training/pbt/`、`configs/training/magnet/`、训练 runner、checkpoint 或 A100 脚本；
- `src/quanxin_life/core/schemas.py`、`core/product.py`、`core/enums.py`；
- `server-results` 中的模型状态、模型卡、指标或制品；
- 用户当前修改中的 `src/quanxin_life/reporting/audited_artifacts.py`。

## 实施任务

### 1. 出站客户端

- [x] 写 token cache、401 refresh、429/5xx/timeout 重试和错误分类失败测试。
- [x] 写消息、回复、卡片更新、上传/下载和 Bitable 请求形状测试。
- [x] 运行目标测试并确认因模块缺失失败。
- [x] 实现最小 transport/client；Fake Transport 记录请求且不接触网络。
- [x] 运行目标测试、Ruff 和 mypy 聚焦检查。

### 2. 安全附件

- [x] 写超限、危险后缀、伪 CSV、ZIP/可执行 magic、SHA 不一致和 UTF-8 失败测试。
- [x] 写合法 canonical CSV 通过并把精确字节交给既有注册端口的测试。
- [x] 确认测试先失败，再实现 CSV-only policy。
- [x] 验证数据检查失败时 workflow 不调用预测工具。

### 3. 入站事件和固定工作流

- [x] 写消息、文件、卡片 action 的 sanitized reference 解析测试。
- [x] 写旧 router 兼容和新 router 持久 enqueue 后快速 ACK 测试。
- [x] 实现 validation-first allowlist workflow；`compare_operation_scenarios` 固定拒绝。
- [x] 写模型未激活、域外和工具拒绝的原样传播测试。

### 4. Audited Card Builder

- [x] 写不存在 result、未知 tool/version、非白名单 path、caller 数值注入和 inactive route 测试。
- [x] 实现只接受 run/result ID 的 builder；每个展示数值带 result ID 与 JSON path。
- [x] 实现 received/checking/queued/running/success/rejected/report-ready 状态卡片。

### 5. Bitable 和报告交付

- [x] 写 `run_id` 查询、create、update、重复写和并发冲突测试。
- [x] 实现固定字段映射；禁止把数组/轨迹写入表格。
- [x] 包装既有 audited report artifact，上传文件后只返回受控引用。
- [x] 写报告来源不是有效 `generate_audited_report` ToolResult 时拒绝测试。

### 6. Aily façade

- [x] 写缺失/错误 API key、创建任务、状态、结果和报告读取测试。
- [x] 实现 Bearer/API Key 鉴权和稳定外部业务名映射。
- [x] 生成 OpenAPI 文件和系统提示词；不包含任何业务示例数值。

### 7. Fake Sandbox

- [x] 组装 Fake Feishu transport、内存 Bitable、审计 ledger 和固定 workflow。
- [x] 验证上传 CSV -> 数据验证 -> 工具结果/拒绝 -> 卡片 -> Bitable -> 报告交付。
- [x] 单独验证真实候选 inactive route 不展示数值。
- [x] 提供一条启动命令和一条 preflight 命令。

### 8. 质量门禁

- [x] 运行相关 pytest，随后运行完整或范围匹配的 pytest。
- [x] 运行 `ruff check`、`mypy`、`compileall` 和包构建检查。
- [x] 检查 `git diff --check`、危险后缀、秘密值、`TODO`/`pass` 占位。
- [x] 输出自动验证、凭证后验证、PBT/MAGNet 后启用和科学不支持项的最终分类。

## 安全回退

任何 route、artifact、ToolResult、project binding、SHA 或 activation 检查失败都返回拒绝
或降级状态，不回退到未注册模型，不在卡片/Aily 中隐藏警告，不将循环数换算为年份。
