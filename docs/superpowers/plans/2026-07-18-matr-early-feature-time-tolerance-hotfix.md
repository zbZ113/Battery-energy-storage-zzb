# MATR Hybrid 早期特征时间容差 A100 热修复计划

**设计依据：**
`docs/superpowers/specs/2026-07-18-matr-early-feature-time-tolerance-design.md`

## 任务 1：锁定回归

- [x] 为早期特征增加容差内放行并保留告警的失败测试。
- [x] 为显式容差边界外反转增加拒绝测试。
- [x] 验证默认严格行为保持不变。
- [x] 为 MATR Hybrid 装载器增加容差传播测试。

## 任务 2：最小实现

- [x] 扩展 `EarlyCycleFeatureConfig` 的版本化时间容差字段。
- [x] 将容差传入共享 `validate_cycle_records`。
- [x] 将共享校验器的容差告警写入 `EarlyCycleFeatureSet.warnings`。
- [x] MATR Hybrid 装载器显式复用 `_MATR_TIME_MONOTONIC_TOLERANCE_S`。

## 任务 3：验证与提交

- [x] 运行早期特征、MATR 训练数据、曲线张量和共享校验的定向测试。
- [x] 运行 Ruff、mypy 与全量 Python 回归。
- [x] 本地提交代码，不执行 Git 推送。

## 任务 4：A100 交付

- [x] 从已提交代码生成第二个最小热修复 ZIP。
- [x] 生成带文件清单、提交版本和 SHA-256 的交付清单。
- [x] 给出旧 Smoke 审计备份、安装、来源版本更新和完整重跑命令。

## 验证记录

- 定向测试：`32 passed`；
- 全量 Python 回归：`996 passed, 3 skipped`；
- Ruff：通过；
- mypy（152 个源文件）：通过；
- `compileall`：通过。

## 交付记录

- 代码提交：`df004111684719fcdd0ba920ef9502c4c007905a`；
- 热修复包：`dist/quanxin-a100-smoke-hotfix-v2.zip`；
- ZIP 大小：`9647` 字节；
- ZIP SHA-256：
  `532102c10937eeffa6fc580ea0fb368cc7e5807516abd10e0c256323795ee408`；
- 交付清单：`dist/quanxin-a100-smoke-hotfix-v2.json`；
- ZIP 内容已逐文件与上述提交复验一致。
