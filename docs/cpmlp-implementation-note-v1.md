# CPMLP 实现说明 v1

## 结论

本仓库的 CPMLP 是 clean-room 独立实现，不是对 BatteryLife `models/CPMLP.py` 的复制或薄改。

BatteryLife 版本把每个循环的多变量曲线展平，经循环内 MLP 编码后，再把全部早期循环表示展平并用循环间 MLP 回归。当前实现则接受项目自身的 `CurveTensor` 契约，只编码显式观测的放电曲线，用掩码加权池化与观测覆盖率形成跨循环表示，并通过 `cutoff_cycle + softplus(distance)` 保证 EOL80 点预测不早于观测截断点。

## 安全与实验边界

- 缺失曲线在输入边界保持 `NaN`，只有在掩码验证后才变为中性张量填充。
- 训练标签必须来自 `train` 电芯且具有明确观测 EOL80；右删失标签不得进入监督训练。
- 模型只接受电芯级互斥划分、固定特征版本、数据版本和截断周期。
- 当前实现没有 pickle、joblib、`.pt` 或 `.pth` 加载入口。
- BatteryLife 仅作为方法对照与命名来源；真实实验必须在统一划分上与其他基线比较。

批准计划中原有“薄改 BatteryLife CPMLP”的备选实现路线，已由上述独立实现取代。若未来引入 BatteryLife 源码，必须重新进行来源、许可证、差异和模型制品审计。
