# 项目状态

更新时间：2026-08-30。

本页区分 `Implemented`、`Validated`、`Published`、`Deployed`、
`Demonstrated` 和 `Planned`。代码存在、测试通过、镜像发布、目标环境启动和真实用户
演示是不同证据，不能互相替代。

## 状态摘要

| 能力 | 状态 | 当前证据与边界 |
| --- | --- | --- |
| MATR 数据治理与按 `cell_id` 固定划分 | Validated | 三批数据、来源清单、固定 split 与逐样本结果 |
| CyclePatch RUL | Validated | 四个 cutoff、五个随机种子的正式结果和安全制品 |
| HybridPatch 有限时域 SOH | Validated | 输出边界到 cycle 500；不外推完整工业寿命 |
| Split Conformal | Validated | 仅适用于冻结 MATR calibration/test；小校准队列警告保留 |
| BLAST-Lite 工况参考 | Validated | Naumann 与 280Ah 公开数据仅作物理参考，不是目标电芯专属模型 |
| ToolResult 与审计账本 | Validated | 卡片、曲线、报告、Bitable 和 Aily 读取同一登记结果 |
| 任意新 CSV 接入 | Implemented | 版本化字段映射、单位与周期校验；缺必要字段或超支持域时拒绝 |
| 飞书机器人闭环 | Demonstrated | 目标租户真实上传 CSV，并收到 RUL、SOH、工况卡和报告 |
| Aily MCP | Demonstrated | 自定义 MCP 服务已安装；完成受控任务查询和工况创建 |
| Next.js 工作台 | Validated | 类型检查、构建和 API 契约；不是本次飞书演示的唯一入口 |
| 本机完整 Compose | Validated | PostgreSQL、Redis、API、Worker、Next.js、Nginx、飞书/Aily override |
| 竞赛镜像 | Published | 发布记录与不可变 digest 保留在部署文档 |
| 稳定生产部署 | Planned | 临时 Quick Tunnel 不具备 SLA；仍需稳定域名、TLS、备份和运维验收 |
| 企业 BMS/EMS | Planned | 仅有协议沙箱，无企业凭证、现场网络和安全联锁 |

## 正式科学证据

```text
数据：MATR 2017-05-12、2017-06-30、2018-04-12
cutoff：20 / 50 / 100 / 150
模型：CyclePatch Direct / CyclePatch-BatLiNet /
      Current Hybrid / HybridPatch-v2
随机种子：38 / 39 / 40 / 41 / 42
正式运行：80
```

正式结果的指标、逐电芯结果、Conformal 和模型取舍见
[Advanced Benchmark](benchmark.md) 与 [Advanced 模型卡](model-card-advanced.md)。
仓库不把训练日志或 README 中的数字重新包装为在线结果；在线展示必须从受管模型生成的
有效 `ToolResult` 读取。

## 当前业务链

```text
飞书 CSV
→ 附件 SHA-256 与确定性字段映射
→ Canonical batch 与项目绑定
→ CyclePatch RUL / HybridPatch SOH
→ BLAST-Lite 参考工况
→ ToolResult / Audit Ledger
→ 飞书卡片、曲线、报告、Bitable
→ Aily 查询、参数修改与复检动作
```

Aily 只负责意图识别、参数收集、工具选择、任务轮询与证据解释。它不能生成或修改
SOH、RUL、区间、EOL、阈值比较和工况寿命数值。一次修改温度、倍率、SOC、DoD 等一个
或多个字段时，服务端继承未修改字段并原子校验整个新工况；任一字段无效则不创建新上下文。

## 保留的数据与模型范围

- MATR：正式深度学习 RUL 与有限时域 SOH。
- Naumann cycle/calendar：BLAST-Lite 外部参考核验。
- LFP 280Ah DoD：大型方形 LFP 的观测范围尺度核验。
- CyclePatch Direct、CyclePatch-BatLiNet、Current Hybrid、HybridPatch-v2。
- BLAST-Lite 大型方形 LFP 参考路由。

HUST、PBT、MAGNet、DITING、BatteryMFormer、BattGP 和 Smart Feature 的工程接入已从
当前代码树移除。竞赛文档中的论文名称仅用于学术对照，不代表仓库仍交付这些实现。

## 未完成事项

1. 将临时 HTTPS 隧道替换为稳定域名和受管 TLS。
2. 在目标服务器复验迁移、Worker 恢复、备份、回滚和长期运行。
3. 扩大 calibration cohort，并研究右删失与跨域校准。
4. 用经授权的企业电芯数据重新建立目标域模型、支持边界和验收指标。
5. 接入企业 BMS/EMS、身份系统和安全联锁。

具体禁止性表述见 [已知限制](limitations.md)，运行入口见
[运行与部署指南](runtime-setup.md)。
